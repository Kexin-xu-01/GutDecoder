"""
Parse rows of the sample metadata sheets (metadata/xenium_directory.xlsx, visium_directory.xlsx)
into run / slide / ROI and build the HEST output folder of a sample:

    <root>/<prefix><run><suffix>/slide<slide>/ROI<roi>      e.g. xenium_data/XeniumPR1/slide1/ROI3
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = ["clean_sample_id", "SampleRow"]


def clean_sample_id(value) -> str:
    """Normalise a sample ID: NFKC Unicode form, all whitespace removed; NaN becomes ""."""
    text = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def _to_int(value) -> int | None:
    """Integer from 3, 3.0, "3"; None for NaN or blank."""
    if pd.isna(value) or str(value).strip() == "":
        return None
    return int(float(value))


def _slide_number(value) -> int | None:
    """First integer in the Slide field ("slide1", "S2", 1.0 -> 1, 2, 1); None if there is none."""
    if pd.isna(value):
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


@dataclass(frozen=True)
class SampleRow:
    sample_id: str
    run: str
    slide: int | None
    roi: int | None

    @classmethod
    def from_row(cls, row, sample_col: str = "Sample_ID", run_col: str = "run",
                 slide_col: str = "Slide", roi_col: str = "Roi") -> "SampleRow":
        """Build from one metadata row (dict or pandas Series). A missing run column gives run=""."""
        run = row.get(run_col, "")
        return cls(
            sample_id=clean_sample_id(row[sample_col]),
            run="" if pd.isna(run) else str(run).strip(),
            slide=_slide_number(row[slide_col]),
            roi=_to_int(row[roi_col]),
        )

    def hest_dir(self, root, prefix: str = "Xenium", suffix: str = "") -> Path:
        """<root>/<prefix><run><suffix>/slide<slide>/ROI<roi>; ValueError if slide or ROI is missing."""
        if self.slide is None or self.roi is None:
            raise ValueError(f"{self.sample_id}: slide={self.slide}, roi={self.roi}; cannot build the sample folder")
        return Path(root) / f"{prefix}{self.run}{suffix}" / f"slide{self.slide}" / f"ROI{self.roi}"
