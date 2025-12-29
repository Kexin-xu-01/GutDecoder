# HEST-Wrapper

A lightweight wrapper and extension around the **HEST** codebase (Mahmood Lab) that fixes practical bugs, adds support for new H&E foundation models, and provides utilities for visualisation, evaluation, and summary table generation.

This repository is **not a fork of HEST**, but a companion package that *calls into* an existing HEST installation and augments it with additional functionality and robustness.

---

## Motivation

While HEST provides a strong foundation for spatial transcriptomics modelling, in practice, I encountered several limitations when running large experiments across multiple datasets:

- Silent or hard failures in edge cases (small gene counts, missing vars, inconsistent metadata)
- Limited support for newer H&E foundation models

This wrapper package addresses these gaps while keeping the original HEST code untouched.

---

## Key Features

### 🛠 Bug Fixes & Robustness
- Monkey patches for known Scanpy / AnnData edge cases (e.g. QC metrics with small `n_vars`)
- Safer handling of missing genes, empty variables, and inconsistent feature ordering
- CPU/GPU auto-detection for models (e.g. cuML RandomForest fallback to CPU)

---

### 🧠 Extended Model Support
- Integrated **additional H&E foundation models** beyond the default HEST set

---

### 📊 Evaluation & Summary Utilities
- Automatic discovery and aggregation of encoder results across datasets to fix the NaN error in prediction metrics
- Automatic generation of:
  - Mean / std Pearson correlation tables
  - Encoder ranking summaries
  - Cross-dataset union and intersection gene sets
- Consistent output formats for downstream analysis

---

### 🎨 Visualisation
- Spatial plots for gene expression, predictions, and QC metrics
- Histogram and distribution plots for transcript- and sample-level statistics
- Patch-level visualisation helpers for WSI-based features

---

## Notes

- This repository is intended for **research use only**.
- Model weights, raw datasets, and the HEST source code are **not included**.
- GPU acceleration (e.g. cuML, PyTorch) is **optional** and automatically enabled when available.

---

## Acknowledgements

- **Mahmood Lab** for developing and maintaining the original **HEST** framework.
- The open-source communities behind **Scanpy**, **AnnData**, **PyTorch**, and related scientific Python tools.

---

## Disclaimer

This project is an **independent wrapper and extension** built on top of HEST.

It is **not an official fork**, and it is **not affiliated with, maintained by, or endorsed by** the Mahmood Lab or its contributors. All modifications and extensions are provided as-is for research and experimental purposes.
