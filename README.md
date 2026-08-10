# Marine Plastic Detection

Master's thesis project: validation and comparison of three models for detecting marine plastic debris in Sentinel-2 imagery using the [MARIDA Benchmark Dataset](https://github.com/marine-debris/marine-debris.github.io).

## Models compared

1. **Random Forest** — traditional machine learning baseline (pixel-wise classification)
2. **U-Net / ResNet** — standard deep learning model (segmentation or classification)
3. **Custom CNN** — bespoke architecture for patch classification

## Repository structure

```
marine-plastic-detection/
├── data/
│   ├── raw/             # MARIDA dataset (not in git)
│   │   ├── patches/     # Per-patch sub-folders with .tif bands + labels
│   │   └── splits/      # train_X.txt, val_X.txt, test_X.txt
│   └── processed/       # Cached features, preprocessed arrays
│
├── notebooks/
│   └── data_exploration.ipynb
│
├── src/
│   ├── data/
│   │   └── dataset_loader.py    # Loads Sentinel-2 patches and labels
│   ├── features/
│   │   └── spectral_indices.py  # NDVI, FDI, NDWI computation
│   ├── models/
│   │   ├── random_forest.py     # RF baseline (Model 1)
│   │   ├── unet.py              # U-Net (Model 2) — placeholder
│   │   └── custom_cnn.py        # Custom CNN (Model 3) — placeholder
│   ├── training/
│   │   └── train.py             # Training entrypoint
│   ├── validation/
│   │   ├── metrics.py           # Precision, Recall, F1, IoU, accuracy + confusion matrix
│   │   ├── report.py            # One-page PDF evaluation report (matplotlib)
│   │   └── evaluate.py          # Test-set evaluation entrypoint
│   └── inference/
│       └── export_predictions.py  # Export classified GeoTIFFs for QGIS
│
├── experiments/         # Saved models and results (gitignored)
├── configs/             # YAML / JSON config files (future use)
├── environment.yml      # Conda env definition (sole source of truth for deps)
├── .gitignore
└── README.md
```

## Setup

Dependencies are managed via a single conda environment.

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate marine-debris

# Later, to pick up updated pins:
conda env update -f environment.yml --prune
```

The env includes the geospatial stack (rasterio/GDAL/geopandas) on conda-forge,
PyTorch 2.3 with CUDA 12.1, and segmentation-models-pytorch / albumentations /
torchmetrics for the deep learning models. Python is pinned to 3.11.

## Data preparation

1. Download the MARIDA dataset from [Zenodo](https://zenodo.org/record/5151941).
2. Unzip into `data/raw/` so the structure is:
   ```
   data/raw/patches/S2_<DATE>_<TILE>/
       S2_<DATE>_<TILE>_<CROP>.tif        # 256x256, 11 bands stacked
       S2_<DATE>_<TILE>_<CROP>_cl.tif     # 256x256 class mask (0..15)
       S2_<DATE>_<TILE>_<CROP>_conf.tif   # 256x256 confidence mask
   data/raw/splits/train_X.txt            # one patch ID per line; no "S2_" prefix
   data/raw/splits/val_X.txt
   data/raw/splits/test_X.txt
   ```
   1,381 patches across 63 scenes (694 train / 328 val / 359 test).
3. MARIDA's stacked GeoTIFF has 11 Sentinel-2 bands in this order:
   `B01, B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12` (B09 and B10 omitted).
   `dataset_loader.load_patch` defaults to the 6-band subset
   `[B02, B03, B04, B08, B11, B12]` used by the RF baseline; pass
   `bands=MARIDA_BAND_ORDER` for the full stack.

## MARIDA class IDs

| ID | Class                    |
|----|--------------------------|
| 0  | Nodata (excluded)        |
| 1  | Marine Debris            |
| 2  | Dense Sargassum          |
| 3  | Sparse Sargassum         |
| 4  | Natural Organic Material |
| 5  | Ship                     |
| 6  | Clouds                   |
| 7  | Marine Water             |
| 8  | Sediment-Laden Water     |
| 9  | Foam                     |
| 10 | Turbid Water             |
| 11 | Shallow Water            |
| 12 | Waves                    |
| 13 | Cloud Shadows            |
| 14 | Wakes                    |
| 15 | Mixed Water              |

For binary classification, class `1` is treated as the positive class (debris) by default.

## Running the Random Forest baseline

From the repository root:

```bash
python -m src.training.train
```

Outputs:
- Trained model:      `experiments/rf_baseline_rf.joblib`
- Fitted scaler:      `experiments/rf_baseline_scaler.joblib`
- Validation metrics: `experiments/rf_baseline_val_metrics.json` (+ `_report.txt`)
- PDF report:         `experiments/rf_baseline_val_report.pdf`
- Console report:     precision / recall / F1 on validation set

Validation metrics are computed on **all valid pixels** of the val split (no
subsampling), so they are directly comparable to the test-set numbers.

## Test-set evaluation

Evaluate the trained model on the held-out test split (all valid pixels, no
subsampling):

```bash
python -m src.validation.evaluate
```

Outputs the sklearn classification report and debris-class precision / recall /
F1 / IoU to the console, and saves them to
`experiments/rf_baseline_test_metrics.json` (+ `_report.txt`), plus a one-page
PDF summary (metrics table, confusion-matrix heatmap, classification report) at
`experiments/rf_baseline_test_report.pdf`.

## Inspecting predictions in QGIS

Export per-patch classification maps as georeferenced GeoTIFFs that overlay the
source Sentinel-2 imagery:

```bash
python -m src.inference.export_predictions --split test
# or choose a split / output location:
python -m src.inference.export_predictions --split val --out-dir experiments/predictions
```

Each `<patch>_pred.tif` is written to `experiments/predictions/` as a single-band
`uint8` raster carrying the source patch's CRS and transform. Pixel values:
`0` = non-debris, `1` = debris, `255` = nodata. An embedded colormap renders
debris in red with everything else transparent, so the layer drops cleanly on
top of a basemap in QGIS.

## References

- Kikaki et al. (2022). *MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data.* PLOS ONE.
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262247
- MARIDA repository: https://github.com/marine-debris/marine-debris.github.io
