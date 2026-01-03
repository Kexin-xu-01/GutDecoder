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


# ---------- helpers ----------
def _sanitize_tag(s: str, maxlen: int = 8) -> str:
    s2 = re.sub(r'[^A-Za-z0-9]', '', s)
    return s2.upper()[:maxlen] or "R"

def _extract_pr_number(path: Path) -> Optional[int]:
    """
    Look for 'XeniumPR<digit>' pattern in the path (case-insensitive).
    Returns int digit (1..9) or None.
    """
    m = re.search(r'XeniumPR(\d)', str(path), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def _extract_slide_number(root: Path) -> Optional[str]:
    """
    Look for 'slideN' pattern in the root folder name and return the digit as string.
    If not found, try to infer from name like 'S1' or 's1' inside the folder name.
    """
    n = root.name.lower()
    m = re.search(r'slide[_\-]?(\d+)', n)
    if m:
        return m.group(1)
    m2 = re.search(r'\bS(\d+)\b', root.name, flags=re.IGNORECASE)
    if m2:
        return m2.group(1)
    return None

def _discover_samples_from_roots(
    roots: List[Path],
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

        # --- Naming rule ---
        pr_num = _extract_pr_number(root)
        slide_num = _extract_slide_number(root) or _sanitize_tag(root.name, 3)

        if pr_num is not None:
            prefix = f"XeniumPR{pr_num}S{slide_num}"
        else:
            # fallback for unknown roots
            prefix = f"{_sanitize_tag(root.name)}S{slide_num}"

        new_id = f"{prefix}{sid}"
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


# def write_var_k_genes_from_paths(
#     adata_paths,
#     k,
#     criteria,
#     min_cells_pct,
#     var_out_path,
#     all_genes_out_path=None,
#     exclude_keywords=None,
#     filtered_common_out_path=None
# #     union_genes_out_path=None,
# ):
#     """
#     Load all adatas, call HEST's get_k_genes() for top-k genes,
#     and also save:
#       - union of all genes across samples
#       - all common genes (keyword-filtered, no expression threshold)
#       - filtered common genes using min_cells_pct across each sample

#     Returns:
#         (var_k_genes, all_common_genes, filtered_common_genes, gene_union)
#     """
#     import json, warnings
#     import numpy as np
#     import scanpy as sc
#     from pathlib import Path
#     from hest.utils import get_k_genes

#     if exclude_keywords is None:
#         exclude_keywords = ["NegControl", "Codeword", "Intergenic_Region", "Control", "BLANK"]

#     warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")

#     # ---- Load all adatas
#     adata_list = []
#     for p in adata_paths:
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=FutureWarning)
#             ad = sc.read_h5ad(str(p))
#         adata_list.append(ad)

#     # ---- Top-k variable/mean genes
#     var_k_genes = get_k_genes(
#         adata_list,
#         k,
#         criteria,
#         save_dir=str(var_out_path),
#         min_cells_pct=min_cells_pct,
#     )

#     # ---- UNION of all genes
#     gene_union = set()
#     for ad in adata_list:
#         gene_union |= set(ad.var_names)
#     gene_union = sorted(gene_union)

#     # ---- ALL common genes (keyword-filtered only)
#     common_genes = set(adata_list[0].var_names)
#     for ad in adata_list[1:]:
#         common_genes &= set(ad.var_names)

#     def _keep_keyword(gene: str) -> bool:
#         return not any(kw in gene for kw in exclude_keywords)

#     all_common_genes = sorted([g for g in common_genes if _keep_keyword(g)])

#     # ---- Filtered common genes (expression threshold per sample)
#     filtered_sets = []
#     for ad in adata_list:
#         ad_tmp = ad[:, :].copy()
#         min_cells = int(np.ceil(min_cells_pct * ad_tmp.n_obs)) if min_cells_pct else 0
#         if min_cells > 0:
#             sc.pp.filter_genes(ad_tmp, min_cells=min_cells)
#         filtered_sets.append(set(ad_tmp.var_names))

#     filtered_common = set.intersection(*filtered_sets) if filtered_sets else set()
#     filtered_common_genes = sorted(
#         [g for g in filtered_common if ("BLANK" not in g and "Control" not in g)]
#     )

#     # ---- Write JSONs
#     out_dir = Path(var_out_path).parent

# #     if union_genes_out_path is None:
# #         union_genes_out_path = out_dir / "union_genes.json"
# #     with open(union_genes_out_path, "w") as f:
# #         json.dump({"genes": gene_union}, f)

#     if all_genes_out_path is None:
#         all_genes_out_path = out_dir / "all_genes.json"
#     with open(all_genes_out_path, "w") as f:
#         json.dump({"genes": all_common_genes}, f)

#     if filtered_common_out_path is None:
#         filtered_common_out_path = out_dir / f"common_genes_{min_cells_pct}.json"
#     with open(filtered_common_out_path, "w") as f:
#         json.dump(
#             {"genes": filtered_common_genes, "min_cells_pct": min_cells_pct}, f
#         )

#     print(
#         f"[INFO] Wrote {var_out_path} (top-{k}, criteria={criteria}); "
# #         f"{union_genes_out_path} (union={len(gene_union)}); "
#         f"{all_genes_out_path} (all_common={len(all_common_genes)}); "
#         f"{filtered_common_out_path} (filtered_common={len(filtered_common_genes)}, "
#         f"min_cells_pct={min_cells_pct})"
#     )

#     return var_k_genes, all_common_genes, filtered_common_genes

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
    import json, warnings
    import numpy as np
    import scanpy as sc
    from pathlib import Path
    from hest.utils import get_k_genes

    if exclude_keywords is None:
        exclude_keywords = ["NegControl", "Codeword", "Intergenic_Region", "Control", "BLANK"]

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
    var_k_genes = get_k_genes(
        adata_list,
        k_actual,
        criteria,
        save_dir=str(var_out_path),
        min_cells_pct=min_cells_pct,
    )

    # ---- Write other JSONs
    if all_genes_out_path is None:
        all_genes_out_path = out_dir / "all_genes.json"
    with open(all_genes_out_path, "w") as f:
        json.dump({"genes": all_common_genes}, f)

    if filtered_common_out_path is None:
        filtered_common_out_path = out_dir / f"common_genes_{min_cells_pct}.json"
    with open(filtered_common_out_path, "w") as f:
        json.dump({"genes": filtered_common_genes, "min_cells_pct": min_cells_pct}, f)

    print(
        f"[INFO] Wrote {var_out_path} (top-{k_actual}, criteria={criteria}); "
        f"{all_genes_out_path} (all_common={len(all_common_genes)}); "
        f"{filtered_common_out_path} (filtered_common={len(filtered_common_genes)}, "
        f"min_cells_pct={min_cells_pct})"
    )

    # return the gene lists and the actual k + path used
    return var_k_genes, all_common_genes, filtered_common_genes, k_actual, str(var_out_path)


# ---------- main entry ----------

def create_benchmark_data_multislide(
    save_dir: str | Path,
    K: int | str,
    base_root: str | Path = "sftp://login1.molbiol.ox.ac.uk/ceph/project/simmons_hts/kxu/hest/xenium_data/XeniumPR1_segger",
    slide_subdirs: List[str] | tuple = ("slide1", "slide2"),
    ids: Optional[List[str]] = None,
    gene_k: Union[int, str] = 50,
    gene_criteria: str = "var",
    min_cells_pct: float = 0.10,
    symlink: bool = False,
    seed: int = 0,
):
    """
    Build a HEST benchmark package from both slide1 and slide2 under the XeniumPR1_segger tree
    (or any set of slide subfolders you pass), without relying on a prebuilt metadata DF.

    Expected layout:
        <base_root>/slide1/<sample_id>/...
        <base_root>/slide2/<sample_id>/...

    Output tree:
      <save_dir>/
        var_50genes.json
        splits/...
        patches/<id>.h5
        patches/vis/<id>.png
        adata/<id>.h5ad

    Args:
        save_dir: destination directory for the assembled benchmark package
        K: number of folds for HEST's create_splits
        base_root: base directory containing slide subfolders
        slide_subdirs: which slide folders to include (defaults to ["slide1", "slide2"])
        ids: optional list of sample IDs to include (if None, auto-discovers)
        gene_k: number of variable genes to select
        gene_criteria: criteria for get_k_genes (e.g., "var")
        symlink: if True, symlink files instead of copying
        seed: RNG seed used to deterministically shuffle within groups before splitting
    """
    
    from hest.HESTData import create_splits

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build slide roots list and discover samples across them
    base_root = Path(base_root)
    roots = [base_root / sd for sd in slide_subdirs]
    print(f"[INFO] Using slide roots: {roots}")

    samples = _discover_samples_from_roots(roots, ids=ids)
    if not samples:
        raise ValueError(
            f"No valid samples (with aligned_adata.h5ad) found under any of: {roots}."
        )
    discovered_ids = sorted(samples.keys())
    print(f"[INFO] Discovered {len(discovered_ids)} samples: {discovered_ids}")

    # 2) Minimal metadata DF for splitting (patient from prefix; dataset_title from base folder name)
    def _infer_patient(sid: str) -> str:
        return sid.split("_")[0] if "_" in sid else sid

    dataset_title = base_root.name or "xenium"
    meta = pd.DataFrame(
        {
            "id": discovered_ids,
            "patient": [_infer_patient(s) for s in discovered_ids],
            "dataset_title": [dataset_title] * len(discovered_ids),
        }
    )

    # 3) Compute var_k genes → var_50genes.json
    # adata_paths = [samples[sid]["adata"] for sid in discovered_ids]
    # var_json = save_dir / f"var_{gene_k}genes.json"
    # write_var_k_genes_from_paths(adata_paths, gene_k, gene_criteria, min_cells_pct,var_json)
    # print(f"[INFO] Wrote {var_json}")

    # 3) Compute var_k genes → var_{k}genes.json
    adata_paths = [samples[sid]["adata"] for sid in discovered_ids]

    # pass a provisional path; function will choose the final filename and return it
    provisional_var_json = save_dir / ("var_auto_genes.json" if isinstance(gene_k, str) and gene_k.lower() == "auto" else f"var_{gene_k}genes.json")

    # now call helper and capture the actual k and path returned
    var_k_genes, all_common_genes, filtered_common_genes, k_actual, var_out_path = write_var_k_genes_from_paths(
        adata_paths,
        gene_k,
        gene_criteria,
        min_cells_pct,
        provisional_var_json,
    )

    print(f"[INFO] Wrote {var_out_path} (top-{k_actual}, criteria={gene_criteria})")


    # 4) K-fold splits using HEST's create_splits
    #    Group by (dataset_title, patient)
    # --- handle LOOCV (leave-one-(dataset_title,patient)-out) ---
    if isinstance(K, str) and K.lower() == "loocv":
        # number of (dataset_title, patient) groups — exactly matches `group` below
        K = meta.groupby(["dataset_title", "patient"]).ngroups
        print(f"[INFO] Using leave-one-patient-per-dataset CV: K = {K}")

    group = meta.groupby(["dataset_title", "patient"])["id"].agg(list).to_dict()

    # Deterministic shuffle within each group
    rng = np.random.RandomState(seed)
    for key, id_list in group.items():
        rng.shuffle(id_list)

    splits_dir = save_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    create_splits(str(splits_dir), group, K=K)
    print(f"[INFO] Wrote {K}-fold splits to {splits_dir}")

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
        print("[WARN] Missing files:")
        for sid, lbl, path in missing:
            print(f"  - {sid} [{lbl}] → {path}")

    print(f"✅ Benchmark dataset created at {save_dir}")


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
    metadata_csv: str = "/project/simmons_hts/kxu/hest/hest_directory.csv",
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
    from hest.HESTData import create_splits

    save_dir = Path(save_dir)
    eval_dirs = [Path(x) for x in eval_dirs]
    # sanitise and check inputs
    existing = [d for d in eval_dirs if d.exists() and d.is_dir()]
    if not existing:
        raise ValueError(f"No valid eval_dirs found among: {eval_dirs}")
    print(f"[INFO] Using eval dirs: {existing}")

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
            print(f"[WARN] Some exclude_ids not found: {sorted(missing_excludes)}")

        removed = before - len(discovered_ids)
        print(f"[INFO] Excluded {removed} samples → remaining {len(discovered_ids)}")
        if removed > 0:
            for e in sorted(exclude_set & set(discovered_ids)):
                print(f"   - excluded: {e}")

    discovered_ids = sorted(discovered_ids)
    if not discovered_ids:
        raise ValueError("No samples discovered in provided eval_dirs (no *.h5ad or *.h5 files found).")
    print(f"[INFO] Discovered sample IDs ({len(discovered_ids)}): {discovered_ids}")

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
            print(f"[INFO] Loaded {len(patient_map)} entries from {metadata_csv}")
        else:
            print(f"[WARN] metadata_csv missing columns 'sample_id'/'patient_id'; will fallback to automatic patient inference")
    else:
        print(f"[WARN] metadata_csv not found: {metadata_csv}; will fallback to automatic patient inference")

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
    print(f"[INFO] Planned actions: {len(planned_actions)} file operations; {len(missing)} missing types.")
    # perform file ops
    for act, src, dst in planned_actions:
        try:
            _transfer(src, dst, act, symlink, [])  # we pass temporary missing list per transfer
        except Exception as e:
            print(f"[ERROR] transferring {src} -> {dst}: {e}")

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

    print(f"[INFO] Built metadata: {len(meta)} samples, {meta['patient'].nunique()} unique patients.")
    print(meta.head(20).to_string(index=False))

    # write var_k genes (requires adata files to be present in save_dir or accessible)
    adata_paths = [adata_out / f"{sid}.h5ad" for sid in discovered_ids]

    var_json = save_dir / f"var_{gene_k}genes.json"
    write_var_k_genes_from_paths(adata_paths, gene_k, gene_criteria, min_cells_pct, var_json)
    print(f"[INFO] Wrote {var_json}")

    # patient-level splits
    group = meta.groupby(["dataset_title", "patient"])["id"].agg(list).to_dict()
    rng = np.random.RandomState(seed)
    for key, id_list in group.items():
        rng.shuffle(id_list)

    splits_dir = save_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    create_splits(str(splits_dir), group, K=K)
    print(f"[INFO] Wrote {K}-fold patient-level splits to {splits_dir}")

    # final warnings about missing files
    if missing:
        print("[WARN] Some samples were missing adata/patch files (listing up to 50):")
        for sid, typ in missing[:50]:
            print(f"  - {sid}: missing {typ}")

    print(f"✅ Merged benchmark created at {save_dir}")
    return meta
