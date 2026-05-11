# --- Standard library ---
import os
import json
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union


# --- Third-party ---
import numpy as np
import pandas as pd
import scanpy as sc

# --- HEST ---
from hest import iter_hest
from hest.utils import get_k_genes
from hest.HESTData import create_splits

from gutdecoder.config import HEST_METADATA_CSV as _DEFAULT_METADATA_CSV
from loguru import logger


# ---------- helpers ----------
def coerce_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    return str(x).lower() not in ("0", "false", "no", "none", "")


def _int_or_str_lower(val):
    """Return val as int if possible, otherwise as lowercased str. Keeps None as-is."""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    try:
        return int(val)
    except (ValueError, TypeError):
        return str(val).lower()


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
    if dst.exists():
        dst.unlink()
    if symlink:
        try:
            os.symlink(src, dst)
        except FileExistsError:
            pass
    else:
        shutil.copy(src, dst)



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

    warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")

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
    metadata_csv: str = _DEFAULT_METADATA_CSV,
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

    # 2) Load metadata CSV mapping sample_id -> patient_id (if available); otherwise fallback to inference
    patient_map = {}
    meta_df_csv = None
    metadata_csv_path = Path(metadata_csv)
    if metadata_csv_path.exists():
        try:
            meta_df_csv = pd.read_csv(metadata_csv_path, dtype=str)
        except Exception as e:
            logger.warning(f"Failed to read metadata_csv {metadata_csv_path}: {e}; will fallback to inference")
            meta_df_csv = None

        if meta_df_csv is not None and {"sample_id", "patient_id"}.issubset(meta_df_csv.columns):
            # create patient_map from CSV (strip whitespace)
            meta_df_csv["sample_id"] = meta_df_csv["sample_id"].astype(str).str.strip()
            meta_df_csv["patient_id"] = meta_df_csv["patient_id"].astype(str).str.strip()
            patient_map = dict(zip(meta_df_csv["sample_id"], meta_df_csv["patient_id"]))
            logger.info(f"Loaded {len(patient_map)} entries from {metadata_csv_path}")
        else:
            if meta_df_csv is not None:
                logger.warning(f"metadata_csv found but missing 'sample_id'/'patient_id' columns; will fallback to automatic patient inference")
            else:
                logger.warning(f"metadata_csv not readable; will fallback to automatic patient inference")
    else:
        logger.warning(f"metadata_csv not found: {metadata_csv_path}; will fallback to automatic patient inference")

    # Helper: infer patient from sample id if no mapping exists
    def _infer_patient_from_sample(sid: str) -> str:
        return sid.split("_")[0] if "_" in sid else sid

    # Build meta DataFrame for exactly discovered_ids
    dataset_title = base_root.name or "xenium"
    patients_for_ids = []
    unresolved = []
    for sid in discovered_ids:
        if sid in patient_map:
            patients_for_ids.append(patient_map[sid])
        else:
            # try suffix / contains match in CSV if provided
            matched = None
            if meta_df_csv is not None:
                # find any csv sample_id that endswith sid
                candidates = [s for s in meta_df_csv["sample_id"].values if str(s).endswith(str(sid))]
                if candidates:
                    matched = candidates[0]
                    patients_for_ids.append(patient_map.get(matched))
                else:
                    inferred = _infer_patient_from_sample(sid)
                    patients_for_ids.append(inferred)
                    unresolved.append(sid)
            else:
                inferred = _infer_patient_from_sample(sid)
                patients_for_ids.append(inferred)
                unresolved.append(sid)

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

    # Use HEST create_splits for compatibility (this will write the HEST-native split files)
    try:
        create_splits(str(splits_dir), group, K=int(K))
        logger.info(f"Wrote {K}-fold splits to {splits_dir} (via create_splits)")
    except Exception as e:
        logger.warning(f"create_splits raised: {e} — continuing and will write CSV splits manually.")

    # Additionally write train_i.csv / test_i.csv with patient and paths (3/4 column format)
    # If create_splits produced fold files we could read them; to be robust we simply generate folds ourselves
    # Make an ordered list of group keys, shuffle deterministically and assign groups to folds round-robin
    group_keys = list(group.keys())
    rng.shuffle(group_keys)
    folds = [[] for _ in range(int(K))]
    for i, gk in enumerate(group_keys):
        folds[i % int(K)].extend(group[gk])

    def make_df_rows(sample_list: List[str]) -> pd.DataFrame:
        df = pd.DataFrame({"sample_id": sample_list})
        df = df.merge(meta[["id", "patient"]].rename(columns={"id": "sample_id"}), on="sample_id", how="left")
        df["patches_path"] = df["sample_id"].apply(lambda x: f"patches/{x}.h5")
        df["expr_path"] = df["sample_id"].apply(lambda x: f"adata/{x}.h5ad")
        return df[["sample_id", "patient", "patches_path", "expr_path"]]

    for i in range(int(K)):
        test_samples = sorted(set(folds[i]))
        train_samples = sorted(set(sum([folds[j] for j in range(int(K)) if j != i], [])))

        df_train = make_df_rows(train_samples)
        df_test = make_df_rows(test_samples)

        df_train.to_csv(splits_dir / f"train_{i}.csv", index=False, sep=",")
        df_test.to_csv(splits_dir / f"test_{i}.csv", index=False, sep=",")
        logger.info(f"Wrote split CSVs: train_{i}.csv ({len(df_train)} rows), test_{i}.csv ({len(df_test)} rows)")

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

    logger.info(f"✅ Benchmark dataset created at {save_dir}")
    return meta


# copy directly from other eval folder
def create_benchmark_data_multirun(
    save_dir: str | Path,
    K: int | str,
    eval_dirs: List[str | Path],
    gene_k: Union[int, str] = 50,
    gene_criteria: str = "var",
    min_cells_pct: float = 0.10,
    symlink: bool = False,
    seed: int = 0,
    metadata_csv: str = _DEFAULT_METADATA_CSV,
    exclude_ids: Optional[List[str]] = None
):
    """
    Build a merged benchmark package by copying (or symlinking) assets from one or more
    'eval' dataset folders that already contain:
        <eval_dir>/
            patches/
                *.h5
                vis/
                    *.png
            adata/
                *.h5ad

    Args:
        save_dir: destination directory to create merged dataset (will contain patches/, patches/vis/, adata/, splits/, var_*.json)
        K: number of folds (patient-level)
        eval_dirs: list of dataset root paths to copy from (e.g. XeniumPR2 eval folder)
        gene_k, gene_criteria: forwarded to get_k_genes
        symlink: if True, create symlinks instead of copying
        seed: RNG seed for deterministic fold assignment
        metadata_csv: CSV mapping sample_id -> patient_id
        dry_run: if True, only print planned actions without copying
    Returns:
        pd.DataFrame meta (columns: id, patient, dataset_title)
    """
    save_dir = Path(save_dir)
    eval_dirs = [Path(x) for x in eval_dirs]
    # sanitise and check inputs
    existing = [d for d in eval_dirs if d.exists() and d.is_dir()]
    if not existing:
        raise ValueError(f"No valid eval_dirs found among: {eval_dirs}")
    logger.info(f"Using eval dirs: {existing}")

    # discover sample ids by scanning adata/ and patches/ for filenames
    discovered_ids = set()
    sample_sources = {}  # id -> dict(sources found)
    for d in existing:
        adata_dir = d / "adata"
        patches_dir = d / "patches"
        vis_dir = patches_dir / "vis"

        # adata
        if adata_dir.exists() and adata_dir.is_dir():
            for f in sorted(adata_dir.glob("*.h5ad")):
                sid = f.stem
                discovered_ids.add(sid)
                sample_sources.setdefault(sid, {}).setdefault("adata", []).append(f)

        # patches
        if patches_dir.exists() and patches_dir.is_dir():
            for f in sorted(patches_dir.glob("*.h5")):
                sid = f.stem
                discovered_ids.add(sid)
                sample_sources.setdefault(sid, {}).setdefault("patch", []).append(f)

            # vis images
            if vis_dir.exists() and vis_dir.is_dir():
                for f in sorted(vis_dir.glob("*.png")):
                    # allow vis file names like '<sid>_patch_vis.png' or '<sid>.png' or anything; map by stem heuristics
                    stem = f.stem
                    # normalize: if stem endswith '_patch_vis', strip it
                    stem_clean = re.sub(r"_?patch_vis$", "", stem, flags=re.IGNORECASE)
                    # sometimes vis is named '<sid>_patch_vis' or '<sid>'
                    sid = stem_clean
                    discovered_ids.add(sid)
                    sample_sources.setdefault(sid, {}).setdefault("vis", []).append(f)

    # ---- Apply exclusion ----
    if exclude_ids:
        exclude_set = set(exclude_ids)
        before = len(discovered_ids)
        discovered_ids = [sid for sid in discovered_ids if sid not in exclude_set]

        missing_excludes = exclude_set - set(discovered_ids)
        if missing_excludes:
            logger.warning(f"Some exclude_ids not found: {sorted(missing_excludes)}")

        removed = before - len(discovered_ids)
        logger.info(f"Excluded {removed} samples → remaining {len(discovered_ids)}")
        if removed > 0:
            for e in sorted(exclude_set & set(discovered_ids)):
                logger.debug(f"   - excluded: {e}")

    discovered_ids = sorted(discovered_ids)
    if not discovered_ids:
        raise ValueError("No samples discovered in provided eval_dirs (no *.h5ad or *.h5 files found).")
    logger.info(f"Discovered sample IDs ({len(discovered_ids)}): {discovered_ids}")

    # Prepare save_dir layout
    patches_out = save_dir / "patches"
    patches_vis_out = patches_out / "vis"
    adata_out = save_dir / "adata"
    for p in (patches_out, patches_vis_out, adata_out):
        p.mkdir(parents=True, exist_ok=True)
            
    # Load metadata CSV mapping sample_id -> patient_id
    patient_map = {}
    meta_df_csv = None
    if Path(metadata_csv).exists():
        meta_df_csv = pd.read_csv(metadata_csv, dtype=str)
        if {"sample_id", "patient_id"}.issubset(meta_df_csv.columns):
            meta_df_csv["sample_id"] = meta_df_csv["sample_id"].astype(str).str.strip()
            meta_df_csv["patient_id"] = meta_df_csv["patient_id"].astype(str).str.strip()
            patient_map = dict(zip(meta_df_csv["sample_id"], meta_df_csv["patient_id"]))
            logger.info(f"Loaded {len(patient_map)} entries from {metadata_csv}")
        else:
            logger.warning(f"metadata_csv missing columns 'sample_id'/'patient_id'; will fallback to automatic patient inference")
    else:
        logger.warning(f"metadata_csv not found: {metadata_csv}; will fallback to automatic patient inference")

    # Copy / symlink files into save_dir using sample id as filename stem
    missing = []
    planned_actions = []
    for sid in discovered_ids:
        srcs = sample_sources.get(sid, {})
        # choose one adata: prefer first available
        adata_src = None
        if "adata" in srcs and srcs["adata"]:
            adata_src = srcs["adata"][0]
        # else fallback to none

        patch_src = None
        if "patch" in srcs and srcs["patch"]:
            patch_src = srcs["patch"][0]

        # vis: there may be multiple pngs per sample across eval_dirs — keep all but use a standardized name
        vis_srcs = srcs.get("vis", [])

        # plan copy/symlink
        if adata_src:
            dst = adata_out / f"{sid}.h5ad"
            planned_actions.append(("adata", adata_src, dst))
        else:
            # warn — adata missing for this sid
            missing.append((sid, "adata"))

        if patch_src:
            dst = patches_out / f"{sid}.h5"
            planned_actions.append(("patch", patch_src, dst))
        else:
            missing.append((sid, "patch"))

        # for vis, when multiple sources exist, copy each with a numeric suffix if needed
        for i, vs in enumerate(vis_srcs, start=1):
            # try base name '<sid>.png' then '<sid>_1.png', '<sid>_2.png'...
            if i == 1:
                dst = patches_vis_out / f"{sid}.png"
            else:
                dst = patches_vis_out / f"{sid}_{i}.png"
            planned_actions.append(("vis", vs, dst))

    # Show dry run summary
    logger.info(f"Planned actions: {len(planned_actions)} file operations; {len(missing)} missing types.")
    # perform file ops
    for act, src, dst in planned_actions:
        try:
            _transfer(src, dst, act, symlink, [])  # we pass temporary missing list per transfer
        except Exception as e:
            logger.error(f"transferring {src} -> {dst}: {e}")

    # Build metadata DataFrame: use discovered sample IDs and patient mapping (full sample id)
    patient_ids = []
    unresolved = []
    for sid in discovered_ids:
        pid = patient_map.get(sid)
        if pid is None:
            # fallback: try a stem match where original source had 'orig' info: try to find sample with full stem in filenames
            # attempt to match any filename that contains sid as suffix: useful if CSV used 'XeniumPR1S1ROI1' but discovered was 'ROI1' etc.
            # we'll try simple heuristics:
            matched = None
            if meta_df_csv is not None:
                # try find any csv sample_id that endswith sid
                candidates = [s for s in meta_df_csv["sample_id"].values if str(s).endswith(str(sid))]
                if candidates:
                    matched = candidates[0]
                    pid = patient_map.get(matched)
            if pid is None:
                # fallback to using prefix before '_' or the sid itself as patient
                pid = sid.split("_")[0] if "_" in sid else sid
                unresolved.append(sid)
        patient_ids.append(pid)

    meta = pd.DataFrame({"id": discovered_ids, "patient": patient_ids, "dataset_title": ["XeniumPR"] * len(discovered_ids)})

    logger.info(f"Built metadata: {len(meta)} samples, {meta['patient'].nunique()} unique patients.")
    logger.debug(meta.to_string(index=False))

    # write var_k genes (requires adata files to be present in save_dir or accessible)
    adata_paths = [adata_out / f"{sid}.h5ad" for sid in discovered_ids]

    var_json = save_dir / f"var_{gene_k}genes.json"
    write_var_k_genes_from_paths(adata_paths, gene_k, gene_criteria, min_cells_pct, var_json)
    logger.info(f"Wrote {var_json}")

    # patient-level splits
    group = meta.groupby(["dataset_title", "patient"])["id"].agg(list).to_dict()
    rng = np.random.RandomState(seed)
    for key, id_list in group.items():
        rng.shuffle(id_list)

    splits_dir = save_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(K, str) and K.lower() == "loocv":
        # number of (dataset_title, patient) groups — exactly matches `group` below
        K = meta.groupby(["dataset_title", "patient"]).ngroups
        logger.info(f"Using leave-one-patient-per-dataset CV: K = {K}")

    create_splits(str(splits_dir), group, K=K)
    logger.info(f"Wrote {K}-fold patient-level splits to {splits_dir}")

    # final warnings about missing files
    if missing:
        logger.warning("Some samples were missing adata/patch files:")
        for sid, typ in missing[:50]:
            logger.warning(f"  {sid}: missing {typ}")

    logger.info(f"✅ Merged benchmark created at {save_dir}")
    return meta


def create_benchmark_data_multirun_ex_val(
    save_dir: str | Path,
    K: int | str,
    eval_dirs: List[str | Path],
    gene_k: Union[int, str] = 50,
    gene_criteria: str = "var",
    min_cells_pct: float = 0.10,
    symlink: bool = False,
    seed: int = 0,
    metadata_csv: str = _DEFAULT_METADATA_CSV,
    exclude_ids: Optional[List[str]] = None,
    # NEW: map condition name -> list of PR integers (e.g. {"PR123":[1,2,3], "PR45":[4,5]})
    condition_map: Optional[Dict[str, List[int]]] = None,
    # If True, samples that can't be assigned to any condition will be excluded from splits (but copied).
    drop_unassigned_from_splits: bool = True
):
    """
    Build a merged benchmark package by copying (or symlinking) assets from one or more
    'eval' dataset folders that already contain:
        <eval_dir>/
            patches/
                *.h5
                vis/
                    *.png
            adata/
                *.h5ad

    Added behavior: if `condition_map` is provided, create external-validation splits
    by condition: for each condition produce a split where that condition is the test set
    and all other conditions are the train set.

    Args:
        save_dir: destination directory to create merged dataset (will contain patches/, patches/vis/, adata/, splits/, var_*.json)
        K: number of folds (patient-level) -- kept for backward compat. If `condition_map` is provided,
           K is ignored for split generation and condition-based external splits are created instead.
        eval_dirs: list of dataset root paths to copy from (e.g. XeniumPR2 eval folder)
        condition_map: OPTIONAL dict mapping condition_name -> list of PR integers (e.g. {"PR1-3":[1,2,3], ...}).
                       If None, function will attempt to infer PR from sample ids using regex 'XeniumPR(\d+)' and group
                       by integer ranges (1-3,4-5,6-9) as a reasonable default.
        drop_unassigned_from_splits: whether to exclude samples that can't be assigned to any condition from the split CSVs.
    Returns:
        pd.DataFrame meta (columns: id, patient, dataset_title, condition)
    """
    save_dir = Path(save_dir)
    eval_dirs = [Path(x) for x in eval_dirs]
    # sanitise and check inputs
    existing = [d for d in eval_dirs if d.exists() and d.is_dir()]
    if not existing:
        raise ValueError(f"No valid eval_dirs found among: {eval_dirs}")
    logger.info(f"Using eval dirs: {existing}")

    # discover sample ids by scanning adata/ and patches/ for filenames
    discovered_ids = set()
    sample_sources = {}  # id -> dict(sources found)
    for d in existing:
        adata_dir = d / "adata"
        patches_dir = d / "patches"
        vis_dir = patches_dir / "vis"

        # adata
        if adata_dir.exists() and adata_dir.is_dir():
            for f in sorted(adata_dir.glob("*.h5ad")):
                sid = f.stem
                discovered_ids.add(sid)
                sample_sources.setdefault(sid, {}).setdefault("adata", []).append(f)

        # patches
        if patches_dir.exists() and patches_dir.is_dir():
            for f in sorted(patches_dir.glob("*.h5")):
                sid = f.stem
                discovered_ids.add(sid)
                sample_sources.setdefault(sid, {}).setdefault("patch", []).append(f)

            # vis images
            if vis_dir.exists() and vis_dir.is_dir():
                for f in sorted(vis_dir.glob("*.png")):
                    stem = f.stem
                    stem_clean = re.sub(r"_?patch_vis$", "", stem, flags=re.IGNORECASE)
                    sid = stem_clean
                    discovered_ids.add(sid)
                    sample_sources.setdefault(sid, {}).setdefault("vis", []).append(f)

    # ---- Apply exclusion ----
    if exclude_ids:
        exclude_set = set(exclude_ids)
        before = len(discovered_ids)
        discovered_ids = sorted([sid for sid in discovered_ids if sid not in exclude_set])

        missing_excludes = exclude_set - set(discovered_ids)
        if missing_excludes:
            logger.warning(f"Some exclude_ids not found: {sorted(missing_excludes)}")

        removed = before - len(discovered_ids)
        logger.info(f"Excluded {removed} samples → remaining {len(discovered_ids)}")
        if removed > 0:
            for e in sorted(exclude_set & set(discovered_ids)):
                logger.debug(f"   - excluded: {e}")
    else:
        discovered_ids = sorted(discovered_ids)

    if not discovered_ids:
        raise ValueError("No samples discovered in provided eval_dirs (no *.h5ad or *.h5 files found).")
    logger.info(f"Discovered sample IDs ({len(discovered_ids)}): {discovered_ids}")

    # Prepare save_dir layout
    patches_out = save_dir / "patches"
    patches_vis_out = patches_out / "vis"
    adata_out = save_dir / "adata"
    for p in (patches_out, patches_vis_out, adata_out):
        p.mkdir(parents=True, exist_ok=True)

    # Load metadata CSV mapping sample_id -> patient_id
    patient_map = {}
    meta_df_csv = None
    if Path(metadata_csv).exists():
        meta_df_csv = pd.read_csv(metadata_csv, dtype=str)
        if {"sample_id", "patient_id"}.issubset(meta_df_csv.columns):
            meta_df_csv["sample_id"] = meta_df_csv["sample_id"].astype(str).str.strip()
            meta_df_csv["patient_id"] = meta_df_csv["patient_id"].astype(str).str.strip()
            patient_map = dict(zip(meta_df_csv["sample_id"], meta_df_csv["patient_id"]))
            logger.info(f"Loaded {len(patient_map)} entries from {metadata_csv}")
        else:
            logger.warning(f"metadata_csv missing columns 'sample_id'/'patient_id'; will fallback to automatic patient inference")
    else:
        logger.warning(f"metadata_csv not found: {metadata_csv}; will fallback to automatic patient inference")

    # Copy / symlink files into save_dir using sample id as filename stem
    missing = []
    planned_actions = []
    for sid in discovered_ids:
        srcs = sample_sources.get(sid, {})
        # choose one adata: prefer first available
        adata_src = None
        if "adata" in srcs and srcs["adata"]:
            adata_src = srcs["adata"][0]

        patch_src = None
        if "patch" in srcs and srcs["patch"]:
            patch_src = srcs["patch"][0]

        # vis: there may be multiple pngs per sample across eval_dirs — keep all but use a standardized name
        vis_srcs = srcs.get("vis", [])

        # plan copy/symlink
        if adata_src:
            dst = adata_out / f"{sid}.h5ad"
            planned_actions.append(("adata", adata_src, dst))
        else:
            missing.append((sid, "adata"))

        if patch_src:
            dst = patches_out / f"{sid}.h5"
            planned_actions.append(("patch", patch_src, dst))
        else:
            missing.append((sid, "patch"))

        # for vis, when multiple sources exist, copy each with a numeric suffix if needed
        for i, vs in enumerate(vis_srcs, start=1):
            if i == 1:
                dst = patches_vis_out / f"{sid}.png"
            else:
                dst = patches_vis_out / f"{sid}_{i}.png"
            planned_actions.append(("vis", vs, dst))

    # Show dry run summary
    logger.info(f"Planned actions: {len(planned_actions)} file operations; {len(missing)} missing types.")
    # perform file ops
    for act, src, dst in planned_actions:
        try:
            _transfer(src, dst, act, symlink, [])  # we pass temporary missing list per transfer (keeps original signature)
        except Exception as e:
            logger.error(f"transferring {src} -> {dst}: {e}")

    # Build metadata DataFrame: use discovered sample IDs and patient mapping (full sample id)
    patient_ids = []
    unresolved = []
    for sid in discovered_ids:
        pid = patient_map.get(sid)
        if pid is None:
            matched = None
            if meta_df_csv is not None:
                # try find any csv sample_id that endswith sid
                candidates = [s for s in meta_df_csv["sample_id"].values if str(s).endswith(str(sid))]
                if candidates:
                    matched = candidates[0]
                    pid = patient_map.get(matched)
            if pid is None:
                pid = sid.split("_")[0] if "_" in sid else sid
                unresolved.append(sid)
        patient_ids.append(pid)

    # derive condition for each sample
    def infer_pr_number(sample_id: str) -> Optional[int]:
        m = re.search(r"XeniumPR(\d+)", sample_id, re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None

    # If no condition_map provided, set sensible default groups by PR number ranges
    if condition_map is None:
        condition_map = {
            "IBD": [1, 2, 3],
            "Coeliac": [4, 5],
            "NEC": [6, 7, 8, 9, 10],
        }
        logger.info("No condition_map provided; using default PR groups: PR1-3, PR4-5, PR6-10")

    # Create reverse lookup from PR int -> condition name
    pr_to_condition = {}
    for cond_name, pr_list in condition_map.items():
        for pr in pr_list:
            pr_to_condition[int(pr)] = cond_name

    conditions_assigned = []
    unassigned_samples = []
    for sid in discovered_ids:
        pr = infer_pr_number(sid)
        if pr is not None and pr in pr_to_condition:
            conditions_assigned.append(pr_to_condition[pr])
        else:
            # fallback: try to match condition name substring in sid (case-insensitive)
            matched = None
            for cond_name in condition_map.keys():
                if cond_name.lower() in sid.lower():
                    matched = cond_name
                    break
            if matched:
                conditions_assigned.append(matched)
            else:
                conditions_assigned.append(None)
                unassigned_samples.append(sid)

    meta = pd.DataFrame({
        "id": discovered_ids,
        "patient": patient_ids,
        "dataset_title": ["XeniumPR"] * len(discovered_ids),
        "condition": conditions_assigned
    })

    logger.info(f"Built metadata: {len(meta)} samples, {meta['patient'].nunique()} unique patients.")
    if unresolved:
        logger.warning(f"{len(unresolved)} samples used fallback patient ids (no entry in metadata_csv): {unresolved[:20]}")

    if unassigned_samples:
        logger.warning(f"{len(unassigned_samples)} samples could not be assigned to any condition: {unassigned_samples[:20]}")

    logger.debug(meta.to_string(index=False))

    # write var_k genes (requires adata files to be present in save_dir or accessible)
    adata_paths = [adata_out / f"{sid}.h5ad" for sid in discovered_ids]
    var_json = save_dir / f"var_{gene_k}genes.json"
    write_var_k_genes_from_paths(adata_paths, gene_k, gene_criteria, min_cells_pct, var_json)
    logger.info(f"Wrote {var_json}")

    # SPLITS: if condition_map present, create one external-val split per condition (leave-one-condition-out)
    splits_dir = save_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # helper to make df as in your example
    def make_df(samples: List[str]) -> pd.DataFrame:
        df = pd.DataFrame({"sample_id": samples})
        df["patches_path"] = df["sample_id"].apply(lambda x: f"patches/{x}.h5")
        df["expr_path"]    = df["sample_id"].apply(lambda x: f"adata/{x}.h5ad")
        return df

    # collect unique conditions in order
    unique_conditions = [c for c in pd.Series(meta["condition"]).dropna().unique()]
    if not unique_conditions:
        # fallback: use patient-level K-fold as before
        logger.info("No conditions detected; falling back to patient-level K-fold splitting.")
        group = meta.groupby(["dataset_title", "patient"])["id"].agg(list).to_dict()
        rng = np.random.RandomState(seed)
        for key, id_list in group.items():
            rng.shuffle(id_list)
        create_splits(str(splits_dir), group, K=K)
        logger.info(f"Wrote {K}-fold patient-level splits to {splits_dir}")
    else:
        logger.info(f"Creating {len(unique_conditions)} external-validation splits for conditions: {unique_conditions}")
        # for each condition, test = samples with that condition, train = all samples with assigned conditions != that one
        for i, cond in enumerate(unique_conditions):
            test_samples = meta.loc[meta["condition"] == cond, "id"].tolist()
            train_samples = meta.loc[(meta["condition"].notna()) & (meta["condition"] != cond), "id"].tolist()

            if drop_unassigned_from_splits:
                # unassigned are excluded; otherwise, you could optionally include them in train
                pass
            else:
                # include unassigned samples in train set
                train_samples += unassigned_samples

            df_train = make_df(sorted(train_samples))
            df_test  = make_df(sorted(test_samples))

            df_train.to_csv(splits_dir / f"train_{i}.csv", index=False, sep=",")
            df_test.to_csv (splits_dir / f"test_{i}.csv",  index=False, sep=",")

            logger.info(f"Wrote train_{i}.csv ({len(df_train)} rows) / test_{i}.csv ({len(df_test)} rows) [held-out: {cond}]")

        # Also write a mapping file describing condition -> split index
        cond_map_path = splits_dir / "condition_to_split.csv"
        pd.DataFrame({"split_index": list(range(len(unique_conditions))), "condition": unique_conditions}).to_csv(cond_map_path, index=False)
        logger.info(f"Wrote condition mapping to {cond_map_path}")

    # final warnings about missing files
    if missing:
        logger.warning("Some samples were missing adata/patch files:")
        for sid, typ in missing[:50]:
            logger.warning(f"  {sid}: missing {typ}")

    logger.info(f"✅ Merged benchmark created at {save_dir}")
    return meta

