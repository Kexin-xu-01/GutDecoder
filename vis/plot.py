#!/usr/bin/env python3
"""
plot.py — Backward-compatible façade — imports everything from sub-modules.

The implementation has been split across:
  _helpers.py    — pure utilities, no internal deps
  _io.py         — data loading / IO
  _per_gene.py   — per-gene plots
  _per_sample.py — per-sample plots
  _summary.py    — summary / aggregate plots
  _grid.py       — grid layout plots

All public names are re-exported here so that existing callers such as:
    from gutdecoder.vis.plot import extract_best_model_gene_corrs, get_test_splits
continue to work without modification.
"""

from gutdecoder.vis._helpers import *
from gutdecoder.vis._io import *
from gutdecoder.vis._per_gene import *
from gutdecoder.vis._per_sample import *
from gutdecoder.vis._summary import *
from gutdecoder.vis._grid import *
