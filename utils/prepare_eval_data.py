# --- Standard library ---
import os
import json
import re
import shutil
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union


# --- Third-party ---
import numpy as np
import pandas as pd
import scanpy as sc

# --- HEST ---
from hest import iter_hest
from hest.utils import get_k_genes
from gutdecoder.utils.split_plan import create_legacy_splits as create_splits
from gutdecoder.utils.split_plan import round_robin_plan, write_plan

from gutdecoder.config import HEST_METADATA_CSV as _DEFAULT_METADATA_CSV
from loguru import logger


# ---------- helpers ----------


def _sanitize_tag(s: str, maxlen: int = 8) -> str:
    s2 = re.sub(r'[^A-Za-z0-9]', '', s)
    return s2.upper()[:maxlen] or "R"

def _extract_r_number(path: Path) -> Optional[int]:
    """
    Return the integer immediately following 'R' (case-insensitive).

    Examples:
      XeniumR6_cell_l0   -> 6
      XeniumPR3          -> 3
      VisiumR2_slideA    -> 2
      VisiumR12_slideA   -> 12

    Returns None if no such segment is found.
    """
    for part in Path(path).parts:
        m = re.search(r'[Rr](\d+)', part)
        if m:
            return int(m.group(1))
    return None

def _extract_slide_number(root: Path) -> Optional[str]:
    """
    Look for 'slideN' pattern in the root folder name and return the digit as string.
    If not found, try to infer from name like 'S1' or 's1' inside the folder name.
    """
    n = root.name.lower()
    m = re.search(r'slide[_\-]?(\d+)', n)
    if m:
        return m.group(1)
    return None

def _discover_samples_from_roots(
    roots: List[Path],
    prefix: str,
    ids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Path]]:
    """
    Discover samples under multiple roots and merge into a single map.

    Naming rule (applies to PR1, PR2, PR3, etc.):
      new_id = 'XeniumPR{n}S{slide_number}{ROI}'

    If slide number is not found, it falls back to root tag.
    Example:
        XeniumPR1/slide1/ROI3  → XeniumPR1S1ROI3
        XeniumPR2/slide2/ROI5  → XeniumPR2S2ROI5
        XeniumPR3/slideX/ROI7  → XeniumPR3SxROI7
    """
    roots = [Path(r) for r in roots]
    roots = [r for r in roots if r.exists() and r.is_dir()]
    collected = []

    if ids is None:
        for r in sorted(roots, key=lambda p: str(p)):
            for p in sorted([d for d in r.iterdir() if d.is_dir()], key=lambda d: d.name):
                collected.append((r, p.name))
    else:
        for sid in sorted(ids):
            for r in sorted(roots, key=lambda p: str(p)):
                if (r / sid).is_dir():
                    collected.append((r, sid))

    samples: Dict[str, Dict[str, Path]] = {}
    for root, sid in collected:
        sdir = root / sid
        adata = sdir / "aligned_adata.h5ad"
        if not adata.exists():
            continue

        # pick patch .h5
        patch_h5 = None
        patches_dir = sdir / "patches"
        if patches_dir.exists():
            cands = sorted(patches_dir.glob("*.h5"))
            if cands:
                exact = [c for c in cands if c.name == f"{sid}.h5"]
                patch_h5 = exact[0] if exact else cands[0]

        # pick vis .png
        vis_png = None
        vis_dir = sdir / "patches_vis"
        if vis_dir.exists():
            cands = sorted(vis_dir.glob("*.png"))
            if cands:
                exact = [c for c in cands if c.name == f"{sid}_patch_vis.png"]
                vis_png = exact[0] if exact else cands[0]

        pr_num = _extract_r_number(root)
        slide_num = _extract_slide_number(root) 

        if pr_num is not None:
            sample_prefix = f"{prefix}{pr_num}S{slide_num}"
        else:
            # fallback: sanitize root name (simple fallback below)
            root_tag = re.sub(r'\W+', '', root.name)[:10]
            sample_prefix = f"{root_tag}S{slide_num}"
        
        new_id = f"{sample_prefix}{sid}"
        if new_id in samples:
            raise ValueError(
                f"Duplicate renamed sample id '{new_id}' (collision between roots for sid='{sid}')."
            )

        samples[new_id] = {"adata": adata, "patch": patch_h5, "vis": vis_png}

    return samples



def _transfer(src: Optional[Path], dst: Path, label: str, symlink: bool, missing_list: list):
    if src is None or not Path(src).exists():
        missing_list.append((dst.stem, label, str(src) if src is not None else "<none>"))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    # is_symlink() also catches broken links, which exists() reports as absent.
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if symlink:
        os.symlink(src, dst)
    else:
        shutil.copy(src, dst)



def _discover_samples_from_eval_dirs(
    eval_dirs: List[Path],
    exclude_ids: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, Dict[str, List[Path]]]]:
    """Scan eval_dirs for adata/patches/vis files; apply exclusion; return (sorted IDs, sources map)."""
    discovered: set = set()
    sample_sources: Dict[str, Dict[str, List[Path]]] = {}

    for d in eval_dirs:
        adata_dir = d / "adata"
        patches_dir = d / "patches"
        vis_dir = patches_dir / "vis"

        if adata_dir.exists() and adata_dir.is_dir():
            for f in sorted(adata_dir.glob("*.h5ad")):
                discovered.add(f.stem)
                sample_sources.setdefault(f.stem, {}).setdefault("adata", []).append(f)

        if patches_dir.exists() and patches_dir.is_dir():
            for f in sorted(patches_dir.glob("*.h5")):
                discovered.add(f.stem)
                sample_sources.setdefault(f.stem, {}).setdefault("patch", []).append(f)

            if vis_dir.exists() and vis_dir.is_dir():
                for f in sorted(vis_dir.glob("*.png")):
                    sid = re.sub(r"_?patch_vis$", "", f.stem, flags=re.IGNORECASE)
                    discovered.add(sid)
                    sample_sources.setdefault(sid, {}).setdefault("vis", []).append(f)

    if exclude_ids:
        exclude_set = set(exclude_ids)
        missing_excludes = exclude_set - discovered
        if missing_excludes:
            logger.warning(f"Some exclude_ids not found: {sorted(missing_excludes)}")
        removed = len(discovered & exclude_set)
        discovered -= exclude_set
        logger.info(f"Excluded {removed} samples → remaining {len(discovered)}")

    sorted_ids = sorted(discovered)
    if not sorted_ids:
        raise ValueError("No samples discovered in provided eval_dirs (no *.h5ad or *.h5 files found).")
    logger.info(f"Discovered sample IDs ({len(sorted_ids)}): {sorted_ids}")
    return sorted_ids, sample_sources


def _load_patient_map(metadata_csv: Optional[str]) -> Tuple[Dict[str, str], Optional[pd.DataFrame]]:
    """
    Load the sample_id -> patient_id map from CSV. Returns (patient_map, meta_df_csv).

    Patient ids decide which samples share a split, so a missing or malformed CSV raises
    rather than quietly switching to inference. Pass metadata_csv=None to infer patients
    from sample names on purpose.
    """
    if metadata_csv is None:
        logger.info("No metadata_csv given; inferring patient ids from sample names")
        return {}, None
    if not Path(metadata_csv).exists():
        raise FileNotFoundError(
            f"metadata_csv not found: {metadata_csv}. Pass metadata_csv=None to infer patient ids from sample names."
        )
    meta_df_csv = pd.read_csv(metadata_csv, dtype=str)
    if not {"sample_id", "patient_id"}.issubset(meta_df_csv.columns):
        raise ValueError(
            f"metadata_csv {metadata_csv} needs 'sample_id' and 'patient_id' columns; got {list(meta_df_csv.columns)}"
        )
    meta_df_csv["sample_id"] = meta_df_csv["sample_id"].astype(str).str.strip()
    meta_df_csv["patient_id"] = meta_df_csv["patient_id"].astype(str).str.strip()
    patient_map = dict(zip(meta_df_csv["sample_id"], meta_df_csv["patient_id"]))
    logger.info(f"Loaded {len(patient_map)} entries from {metadata_csv}")
    return patient_map, meta_df_csv


def _plan_and_transfer_assets(
    discovered_ids: List[str],
    sample_sources: Dict[str, Dict[str, List[Path]]],
    save_dir: Path,
    symlink: bool,
) -> List[Tuple[str, str]]:
    """Create output dirs, execute copy/symlink operations. Returns list of (sid, type) for missing assets."""
    patches_out = save_dir / "patches"
    patches_vis_out = patches_out / "vis"
    adata_out = save_dir / "adata"
    for p in (patches_out, patches_vis_out, adata_out):
        p.mkdir(parents=True, exist_ok=True)

    missing: List[Tuple[str, str]] = []
    planned = []
    for sid in discovered_ids:
        srcs = sample_sources.get(sid, {})
        adata_src = srcs["adata"][0] if srcs.get("adata") else None
        patch_src = srcs["patch"][0] if srcs.get("patch") else None
        vis_srcs = srcs.get("vis", [])

        if adata_src:
            planned.append(("adata", adata_src, adata_out / f"{sid}.h5ad"))
        else:
            missing.append((sid, "adata"))

        if patch_src:
            planned.append(("patch", patch_src, patches_out / f"{sid}.h5"))
        else:
            missing.append((sid, "patch"))

        for i, vs in enumerate(vis_srcs, start=1):
            dst = patches_vis_out / f"{sid}.png" if i == 1 else patches_vis_out / f"{sid}_{i}.png"
            planned.append(("vis", vs, dst))

    logger.info(f"Planned actions: {len(planned)} file operations; {len(missing)} missing types.")
    lost = []
    for act, src, dst in planned:
        _transfer(src, dst, act, symlink, lost)
    # Sources that vanished between discovery and transfer.
    missing.extend((stem, label) for stem, label, _ in lost)

    return missing


def _resolve_patient_ids(
    discovered_ids: List[str],
    patient_map: Dict[str, str],
    meta_df_csv: Optional[pd.DataFrame],
) -> Tuple[List[str], List[str]]:
    """Resolve patient IDs for each sample. Returns (patient_ids list, unresolved IDs list)."""
    patient_ids = []
    unresolved = []
    for sid in discovered_ids:
        pid = patient_map.get(sid)
        if pid is None and meta_df_csv is not None:
            candidates = [s for s in meta_df_csv["sample_id"].values if str(s).endswith(str(sid))]
            if candidates:
                pid = patient_map.get(candidates[0])
        if pid is None:
            pid = sid.split("_")[0] if "_" in sid else sid
            unresolved.append(sid)
        patient_ids.append(pid)
    return patient_ids, unresolved


def write_var_k_genes_from_paths(
    adata_paths,
    k,  # can be int or "auto"
    criteria,
    min_cells_pct,
    var_out_path=None,           # can be None or provisional path
    all_genes_out_path=None,
    exclude_keywords=None,
    filtered_common_out_path=None,
):
    """
    Load all adatas, call HEST's get_k_genes() for top-k genes,
    and also save:
      - union of all genes across samples
      - all common genes (keyword-filtered, no expression threshold)
      - filtered common genes using min_cells_pct across each sample

    Returns:
        (var_k_genes, all_common_genes, filtered_common_genes, k_actual, var_out_path)
    """
    if exclude_keywords is None:
        exclude_keywords = ["NegControl", "Codeword", "Intergenic_Region", "Control", "BLANK","Intergenic"]

    # ---- Load all adatas
    adata_list = []
    for p in adata_paths:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            ad = sc.read_h5ad(str(p))
        adata_list.append(ad)

    # ---- UNION of all genes
    gene_union = set()
    for ad in adata_list:
        gene_union |= set(ad.var_names)
    gene_union = sorted(gene_union)

    # ---- ALL common genes (keyword-filtered only)
    common_genes = set(adata_list[0].var_names) if adata_list else set()
    for ad in adata_list[1:]:
        common_genes &= set(ad.var_names)

    def _keep_keyword(gene: str) -> bool:
        return not any(kw in gene for kw in exclude_keywords)

    all_common_genes = sorted([g for g in common_genes if _keep_keyword(g)])

    # ---- Filtered common genes (expression threshold per sample)
    filtered_sets = []
    for ad in adata_list:
        ad_tmp = ad[:, :].copy()
        min_cells = int(np.ceil(min_cells_pct * ad_tmp.n_obs)) if min_cells_pct else 0
        if min_cells > 0:
            sc.pp.filter_genes(ad_tmp, min_cells=min_cells)
        filtered_sets.append(set(ad_tmp.var_names))

    filtered_common = set.intersection(*filtered_sets) if filtered_sets else set()
    filtered_common_genes = sorted(
        [g for g in filtered_common if ("BLANK" not in g and "Control" not in g)]
    )

    # ---- Determine actual k when requested as "auto"
    if isinstance(k, str) and k.lower() == "auto":
        # choose min(50, number of filtered common genes)
        k_actual = min(50, max(0, len(filtered_common_genes)))
    else:
        k_actual = int(k)

    out_dir = Path(var_out_path).parent if var_out_path is not None else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    # override var_out_path filename to reflect actual k
    var_out_path = out_dir / f"var_{k_actual}genes.json"

    # ---- Top-k variable/mean genes (now using k_actual)
    # --- If no common gene, skip ---
    if k_actual <= 0 or len(filtered_common_genes) == 0:
        logger.warning(
            f"No common genes found after filtering. "
            f"Skipping get_k_genes(). k_actual={k_actual}, filtered_common={len(filtered_common_genes)}"
        )

        var_k_genes = []
        with open(var_out_path, "w") as f:
            json.dump({"genes": []}, f)

        if all_genes_out_path is None:
            all_genes_out_path = out_dir / "all_genes.json"
        with open(all_genes_out_path, "w") as f:
            json.dump({"genes": all_common_genes}, f)

        if filtered_common_out_path is None:
            filtered_common_out_path = out_dir / f"common_genes_{min_cells_pct}.json"
        with open(filtered_common_out_path, "w") as f:
            json.dump({"genes": filtered_common_genes, "min_cells_pct": min_cells_pct}, f)

        return var_k_genes, all_common_genes, filtered_common_genes, 0, str(var_out_path)

    var_k_genes = get_k_genes(
        adata_list,
        k_actual,
        criteria,
        save_dir=str(var_out_path),
        min_cells_pct=min_cells_pct,
    )

    if all_genes_out_path is None:
        all_genes_out_path = out_dir / "all_genes.json"
    with open(all_genes_out_path, "w") as f:
        json.dump({"genes": all_common_genes}, f)

    if filtered_common_out_path is None:
        filtered_common_out_path = out_dir / f"common_genes_{min_cells_pct}.json"
    with open(filtered_common_out_path, "w") as f:
        json.dump({"genes": filtered_common_genes, "min_cells_pct": min_cells_pct}, f)

    return var_k_genes, all_common_genes, filtered_common_genes, k_actual, str(var_out_path)


# ---------- main entry ----------
def create_benchmark_data_multislide(
    save_dir: str | Path,
    K: int | str,
    prefix: str,
    base_root: str | Path,
    slide_subdirs: List[str] | tuple = ("slide1", "slide2"),
    ids: Optional[List[str]] = None,
    gene_k: Union[int, str] = 50,
    gene_criteria: str = "var",
    min_cells_pct: float = 0.10,
    metadata_csv: Optional[str] = _DEFAULT_METADATA_CSV,
    symlink: bool = False,
    seed: int = 0,
    exclude_ids: Optional[List[str]] = None
):
    """
    Build a HEST benchmark package from both slide1 and slide2 under the XeniumPR1_segger tree
    (or any set of slide subfolders you pass), and write split CSVs that include patient metadata.

    See comments in conversation for expected layout and outputs.

    Returns:
        meta (pd.DataFrame) with columns: id, patient, dataset_title
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build slide roots list and discover samples across them
    base_root = Path(base_root)
    roots = [base_root / sd for sd in slide_subdirs]
    logger.info(f"Using slide roots: {roots}")

    samples = _discover_samples_from_roots(roots, prefix=prefix, ids=ids)
    if not samples:
        raise ValueError(
            f"No valid samples (with aligned_adata.h5ad) found under any of: {roots}."
        )
    discovered_ids = sorted(samples.keys())
    logger.info(f"Discovered {len(discovered_ids)} samples: {discovered_ids}")

    # -------------------------------------------------
    # 🔹 APPLY EXCLUSION
    # -------------------------------------------------
    if exclude_ids:
        exclude_set = set(exclude_ids)
        before = len(discovered_ids)

        discovered_ids = [sid for sid in discovered_ids if sid not in exclude_set]

        removed = before - len(discovered_ids)
        logger.info(f"Excluded {removed} samples → remaining {len(discovered_ids)}")

        missing_excludes = exclude_set - set(samples.keys())
        if missing_excludes:
            logger.warning(f"exclude_ids not found: {sorted(missing_excludes)}")

        if not discovered_ids:
            raise ValueError("All samples were excluded. Nothing left to process.")

    discovered_ids = sorted(discovered_ids)
    logger.info(f"Final sample list ({len(discovered_ids)}): {discovered_ids}")

    # 2) Map samples to patients from metadata_csv
    patient_map, meta_df_csv = _load_patient_map(metadata_csv)
    patients_for_ids, unresolved = _resolve_patient_ids(discovered_ids, patient_map, meta_df_csv)

    dataset_title = base_root.name or "xenium"
    meta = pd.DataFrame({
        "id": discovered_ids,
        "patient": patients_for_ids,
        "dataset_title": [dataset_title] * len(discovered_ids)
    })

    if unresolved:
        logger.warning(f"{len(unresolved)} samples used fallback patient ids (no entry in metadata_csv): {unresolved[:20]}")

    # 3) Compute var_k genes → var_{k}genes.json (calls helper that returns k and path)
    adata_paths = [samples[sid]["adata"] for sid in discovered_ids]
    provisional_var_json = save_dir / ("var_auto_genes.json" if isinstance(gene_k, str) and str(gene_k).lower() == "auto" else f"var_{gene_k}genes.json")

    var_k_genes, all_common_genes, filtered_common_genes, k_actual, var_out_path = write_var_k_genes_from_paths(
        adata_paths,
        gene_k,
        gene_criteria,
        min_cells_pct,
        provisional_var_json,
    )

    logger.info(f"Wrote {var_out_path} (top-{k_actual}, criteria={gene_criteria})")

    # 4) K-fold splits using HEST's create_splits
    # handle LOOCV (leave-one-(dataset_title,patient)-out)
    if isinstance(K, str) and str(K).lower() == "loocv":
        K = meta.groupby(["dataset_title", "patient"]).ngroups
        logger.info(f"Using leave-one-patient-per-dataset CV: K = {K}")

    # Build group dict: (dataset_title, patient) -> list of sample ids
    group = meta.groupby(["dataset_title", "patient"])["id"].agg(list).to_dict()

    # Deterministic shuffle within each group
    rng = np.random.RandomState(seed)
    for key, id_list in group.items():
        rng.shuffle(id_list)

    # Write splits directory and also CSVs that include patient metadata
    splits_dir = save_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Plan once, preserving the final CSV assignment and ordering from legacy runs.
    plan = round_robin_plan(group, int(K), rng)
    write_plan(splits_dir, plan, meta)

    # Write per-patient metadata CSV
    per_patient = (
        meta.groupby(["patient", "dataset_title"])["id"]
        .agg(lambda ids: ";".join(sorted(ids)))
        .reset_index()
        .rename(columns={"id": "samples"})
    )
    per_patient.to_csv(splits_dir / "per_patient_metadata.csv", index=False)
    logger.info(f"Wrote per-patient metadata to {splits_dir/'per_patient_metadata.csv'}")

    # 5) Copy/symlink assets
    (save_dir / "patches").mkdir(exist_ok=True, parents=True)
    (save_dir / "patches" / "vis").mkdir(exist_ok=True, parents=True)
    (save_dir / "adata").mkdir(exist_ok=True, parents=True)

    missing: List[tuple] = []
    for sid in discovered_ids:
        info = samples[sid]
        _transfer(info.get("patch"), save_dir / "patches" / f"{sid}.h5", "patch", symlink, missing)
        _transfer(info.get("vis"), save_dir / "patches" / "vis" / f"{sid}.png", "vis", symlink, missing)
        _transfer(info.get("adata"), save_dir / "adata" / f"{sid}.h5ad", "adata", symlink, missing)

    if missing:
        logger.warning("Missing files:")
        for sid, lbl, path in missing:
            logger.warning(f"  {sid} [{lbl}] → {path}")

    logger.info(f"Benchmark dataset created at {save_dir}")
    return meta


# copy directly from other eval folder
_DEFAULT_CONDITION_MAP = {"IBD": [1, 2, 3], "Coeliac": [4, 5], "NEC": [6, 7, 8, 9, 10]}


def _assign_conditions(sample_ids: List[str], condition_map: Dict[str, List[int]]) -> List[Optional[str]]:
    """Condition per sample from its XeniumPR<n> run number, else from a condition name in the id."""
    pr_to_condition = {int(pr): cond for cond, prs in condition_map.items() for pr in prs}
    conditions = []
    for sid in sample_ids:
        m = re.search(r"XeniumPR(\d+)", sid, re.IGNORECASE)
        cond = pr_to_condition.get(int(m.group(1))) if m else None
        if cond is None:
            cond = next((c for c in condition_map if c.lower() in sid.lower()), None)
        conditions.append(cond)
    return conditions


def _write_patient_kfold_splits(meta: pd.DataFrame, K: int | str, seed: int, splits_dir: Path) -> None:
    group = meta.groupby(["dataset_title", "patient"])["id"].agg(list).to_dict()
    rng = np.random.RandomState(seed)
    for id_list in group.values():
        rng.shuffle(id_list)
    if isinstance(K, str) and K.lower() == "loocv":
        K = len(group)
        logger.info(f"Using leave-one-patient-per-dataset CV: K = {K}")
    create_splits(str(splits_dir), group, K=K)
    logger.info(f"Wrote {K}-fold patient-level splits to {splits_dir}")


def _write_condition_splits(meta: pd.DataFrame, splits_dir: Path, drop_unassigned_from_splits: bool) -> None:
    """One split per condition: that condition is the test set, the other conditions are the train set."""
    def _split_df(samples: List[str]) -> pd.DataFrame:
        df = pd.DataFrame({"sample_id": sorted(samples)})
        df["patches_path"] = "patches/" + df["sample_id"] + ".h5"
        df["expr_path"] = "adata/" + df["sample_id"] + ".h5ad"
        return df

    conditions = meta["condition"].dropna().unique().tolist()
    unassigned = meta.loc[meta["condition"].isna(), "id"].tolist()
    logger.info(f"Creating {len(conditions)} external-validation splits for conditions: {conditions}")
    for i, cond in enumerate(conditions):
        test = meta.loc[meta["condition"] == cond, "id"].tolist()
        train = meta.loc[meta["condition"].notna() & (meta["condition"] != cond), "id"].tolist()
        if not drop_unassigned_from_splits:
            train += unassigned
        _split_df(train).to_csv(splits_dir / f"train_{i}.csv", index=False)
        _split_df(test).to_csv(splits_dir / f"test_{i}.csv", index=False)
        logger.info(f"Wrote train_{i}.csv ({len(train)} rows) / test_{i}.csv ({len(test)} rows) [held-out: {cond}]")

    cond_map_path = splits_dir / "condition_to_split.csv"
    pd.DataFrame({"split_index": range(len(conditions)), "condition": conditions}).to_csv(cond_map_path, index=False)
    logger.info(f"Wrote condition mapping to {cond_map_path}")


def create_benchmark_data_multirun(
    save_dir: str | Path,
    K: int | str,
    eval_dirs: List[str | Path],
    gene_k: Union[int, str] = 50,
    gene_criteria: str = "var",
    min_cells_pct: float = 0.10,
    symlink: bool = False,
    seed: int = 0,
    metadata_csv: Optional[str] = _DEFAULT_METADATA_CSV,
    exclude_ids: Optional[List[str]] = None,
    condition_map: Optional[Dict[str, List[int]]] = None,
    drop_unassigned_from_splits: bool = True,
):
    """
    Build a merged benchmark package by copying (or symlinking) assets from one or more
    'eval' dataset folders that already contain patches/, patches/vis/, and adata/ subdirs.

    Splits are patient-level K-fold (K may be 'loocv'), or, when `condition_map` is given
    (condition -> XeniumPR run numbers), external validation with one held-out condition per split.

    Returns:
        pd.DataFrame meta (columns: id, patient, dataset_title[, condition])
    """
    save_dir = Path(save_dir)
    existing = [Path(x) for x in eval_dirs if Path(x).exists() and Path(x).is_dir()]
    if not existing:
        raise ValueError(f"No valid eval_dirs found among: {eval_dirs}")
    logger.info(f"Using eval dirs: {existing}")

    discovered_ids, sample_sources = _discover_samples_from_eval_dirs(existing, exclude_ids)
    patient_map, meta_df_csv = _load_patient_map(metadata_csv)
    missing = _plan_and_transfer_assets(discovered_ids, sample_sources, save_dir, symlink)
    patient_ids, unresolved = _resolve_patient_ids(discovered_ids, patient_map, meta_df_csv)

    meta = pd.DataFrame({
        "id": discovered_ids,
        "patient": patient_ids,
        "dataset_title": ["XeniumPR"] * len(discovered_ids),
    })
    if condition_map is not None:
        meta["condition"] = _assign_conditions(discovered_ids, condition_map)
        unassigned = meta.loc[meta["condition"].isna(), "id"].tolist()
        if len(unassigned) == len(meta):
            raise ValueError(f"No sample matches any condition in {condition_map}")
        if unassigned:
            logger.warning(f"{len(unassigned)} samples could not be assigned to any condition: {unassigned[:20]}")
    logger.info(f"Built metadata: {len(meta)} samples, {meta['patient'].nunique()} unique patients.")
    if unresolved:
        logger.warning(f"{len(unresolved)} samples used fallback patient ids: {unresolved[:20]}")
    logger.debug(meta.to_string(index=False))

    adata_paths = [save_dir / "adata" / f"{sid}.h5ad" for sid in discovered_ids]
    var_json = save_dir / f"var_{gene_k}genes.json"
    write_var_k_genes_from_paths(adata_paths, gene_k, gene_criteria, min_cells_pct, var_json)
    logger.info(f"Wrote {var_json}")

    splits_dir = save_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    if condition_map is None:
        _write_patient_kfold_splits(meta, K, seed, splits_dir)
    else:
        _write_condition_splits(meta, splits_dir, drop_unassigned_from_splits)

    if missing:
        logger.warning("Some samples were missing adata/patch files:")
        for sid, typ in missing[:50]:
            logger.warning(f"  {sid}: missing {typ}")

    logger.info(f"Merged benchmark created at {save_dir}")
    return meta


def create_benchmark_data_multirun_ex_val(*args, condition_map: Optional[Dict[str, List[int]]] = None, **kwargs):
    """External-validation splits by condition; defaults to PR1-3 IBD, PR4-5 Coeliac, PR6-10 NEC."""
    return create_benchmark_data_multirun(*args, condition_map=condition_map or _DEFAULT_CONDITION_MAP, **kwargs)

