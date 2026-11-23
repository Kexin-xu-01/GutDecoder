# GutDecoder

GutDecoder predicts spatial gene expression in intestinal tissue from routine H&E whole-slide images. Pathology foundation models embed image patches, and regression models trained on matched spatial transcriptomics (10x Xenium and Visium) map those embeddings to gene expression.

The code builds on [HEST](https://github.com/mahmoodlab/HEST) and [TRIDENT](https://github.com/mahmoodlab/TRIDENT) from the Mahmood Lab. It calls into both rather than modifying them.

## Pipeline

| Step | Module | What it does |
|---|---|---|
| 1. Read and align samples | `reader/` | Builds HEST objects from Xenium/Visium outputs and H&E images, applies spot and region exclusion rules, and pools cell-level annotations onto a spot grid |
| 2. Stain normalisation | `stain_norm/` | Reinhard, Macenko and Vahadane normalisation of whole slides (OME-TIFF, TIFF, CZI, NDPI, SVS), in memory or tile by tile with identical results |
| 3. Tissue segmentation and patching | `wrappers/` | Segments tissue and extracts 224 px patches at a fixed resolution with HEST |
| 4. Benchmark datasets | `utils/prepare_eval_data.py` | Assembles per-dataset folders of patches and expression, selects genes, and writes patient-level or condition-held-out cross-validation splits |
| 5. Embedding and regression | `bench/` | Embeds patches with a foundation model and trains ridge, random forest, XGBoost or MLP regressors per split |
| 6. Inference | `inference/` | Applies trained models to new slides and attaches predictions and H&E images to AnnData objects |
| 7. Figures | `vis/` | Summary tables and plots across runs, per-gene and per-sample correlations, spatial prediction maps and slide-embedding UMAPs |

## Installation

Requires Python 3.10 or later. A CUDA GPU is recommended for embedding extraction but is not required.

```bash
git clone https://github.com/Kexin-xu-01/GutDecoder.git
cd GutDecoder
pip install -e .
```

`pyproject.toml` pins HEST and TRIDENT to the commits used for this work, and pins `timm` because model definitions change between releases. Local changes made to the HEST and TRIDENT checkouts are kept in `patches/` and are applied with `git -C <checkout> apply patches/<name>.patch`.

Gated models (UNI, CONCH, Virchow and others) require a Hugging Face account with access granted by the model authors.

## Configuration

Data and result locations are set in `config.py`. Every path is derived from `GUTDECODER_HEST_ROOT`, and each can also be set on its own:

```bash
export GUTDECODER_HEST_ROOT=/path/to/hest          # moves all paths below
export GUTDECODER_RESULTS_ROOT=/path/to/results    # overrides one of them
```

## Usage

### Benchmark encoders and regressors

```bash
python -m gutdecoder.bench.benchmark --config benchmark.yaml
```

```yaml
bench_data_root: /path/to/eval/data
embed_dataroot: /path/to/eval/ST_data_emb
weights_root: /path/to/eval/fm_v1
results_dir: /path/to/eval/ST_pred_results
datasets: [XeniumPR1]
encoders: [uni_v2, hoptimus1, virchow2]
gene_list: var_50genes.json
method: ridge
log_normalize: true
library_size_normalize: true
```

### Predict expression on new slides

```python
from gutdecoder.inference.inference import infer

infer(test_embed_dir="embeddings/", model_directory_path="run_<id>/XeniumPR1/uni_v2", out_dir="predictions/")
```

For models trained with slide-level embeddings, pass `slide_emb_dir` as well. To attach the H&E image and plot the predictions:

```bash
python -m gutdecoder.inference.spatial_plot --run_dir predictions/ --wsi_root slides/
```

### Pool annotated cells onto spots

```bash
python -m gutdecoder.reader.spot_pooling <SampleID> --feature-col cell_type
```

### Stain normalisation

```python
from gutdecoder.stain_norm import build_reference, batch_normalize

build_reference(slide_paths=["ref1.ome.tiff", "ref2.ome.tiff"], output_path="reference.tiff")
batch_normalize("raw_slides/", "normalised_slides/", reference="reference.tiff", method="macenko")
```

## Supported encoders

`conch_v1`, `conch_v1_5`, `uni_v1`, `uni_v2`, `ctranspath`, `phikon`, `phikon_v2`, `h0_mini`, `hibou_large`, `kaiko_base_8`, `remedis`, `resnet50`, `gigapath`, `virchow`, `virchow2`, `hoptimus0`, `hoptimus1`, `gpfm`, `gemma4-e4b`, `gemma4-26b`, and through TRIDENT `kaiko-vitb16`, `kaiko-vits8`, `kaiko-vits16`, `kaiko-vitl14`, `lunit-vits8`, `musk`, `midnight12k`, `openmidnight`, `genbio-pathfm` and `keep`.


## Acknowledgements

We thank the Mahmood Lab for HEST and TRIDENT.
