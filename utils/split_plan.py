"""Pure split planning and validation, independent of filesystem operations."""
from pathlib import Path

import numpy as np
import pandas as pd


def sample_frame(samples, meta=None):
    frame = pd.DataFrame({"sample_id": samples})
    if meta is not None:
        frame = frame.merge(meta[["id", "patient"]].rename(columns={"id": "sample_id"}),
                            on="sample_id", how="left")
    frame["patches_path"] = frame["sample_id"].apply(lambda sid: f"patches/{sid}.h5")
    frame["expr_path"] = frame["sample_id"].apply(lambda sid: f"adata/{sid}.h5ad")
    return frame


def validate_plan(plan, available, groups=None):
    available = set(available)
    for train, test in plan:
        if len(set(train)) != len(train) or len(set(test)) != len(test):
            raise ValueError("Duplicate sample IDs in split")
        if set(train) & set(test):
            raise ValueError("Train and test samples overlap")
        if set(train) | set(test) != available:
            raise ValueError("Split sample coverage differs from available samples")
        if groups and {groups[s] for s in train} & {groups[s] for s in test}:
            raise ValueError("Train and test groups overlap")
    return plan


def round_robin_plan(group, k, rng):
    """Match the existing multislide CSV assignment, including its RNG state."""
    k = int(k)
    if k < 1:
        raise ValueError("K must be positive")
    keys = list(group)
    rng.shuffle(keys)
    folds = [[] for _ in range(k)]
    for index, key in enumerate(keys):
        folds[index % k].extend(group[key])
    plan = [(sorted(set(s for j, fold in enumerate(folds) if j != i for s in fold)),
             sorted(set(folds[i]))) for i in range(k)]
    memberships = {sid: key for key, ids in group.items() for sid in ids}
    return validate_plan(plan, memberships, memberships)


def legacy_hest_plan(group, k):
    """Preserve HEST's sequential group packing for existing multirun datasets."""
    folds = list(group.values())
    if k != len(group):
        packed = {}
        per_fold = sum(map(len, folds)) // k
        index = 0
        for fold in folds:
            packed.setdefault(index, []).extend(fold)
            if len(packed[index]) >= per_fold:
                index += 1
        folds = list(packed.values())
    plan = [([sid for j, fold in enumerate(folds) if j != i for sid in fold],
             np.asarray(folds[i]).flatten().tolist()) for i in range(len(folds))]
    memberships = {sid: key for key, ids in group.items() for sid in ids}
    return validate_plan(plan, memberships, memberships)


def write_plan(directory, plan, meta=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for index, (train, test) in enumerate(plan):
        sample_frame(train, meta).to_csv(directory / f"train_{index}.csv", index=False)
        sample_frame(test, meta).to_csv(directory / f"test_{index}.csv", index=False)


def create_legacy_splits(directory, group, K):
    """Compatibility adapter preserving existing HEST CSV columns and row order."""
    write_plan(directory, legacy_hest_plan(group, K))
