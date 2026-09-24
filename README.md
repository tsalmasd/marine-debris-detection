# Marine Plastic Detection

Master's thesis project: validation and comparison of three models for detecting marine plastic debris in Sentinel-2 imagery using the [MARIDA Benchmark Dataset](https://github.com/marine-debris/marine-debris.github.io).

> **Next work is planned in [`LOSS_ABLATION_PLAN.md`](LOSS_ABLATION_PLAN.md)** —
> loss-objective ablation, reporting repairs, and Model 3. Read it before
> changing the U-Net objective or the reported metrics; it records the
> derivations behind those decisions.

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
│   │   ├── dataset_loader.py    # Loads Sentinel-2 patches and labels
│   │   └── patch_dataset.py     # PyTorch Dataset (dense masks) for U-Net
│   ├── features/
│   │   └── spectral_indices.py  # NDVI, FDI, NDWI computation
│   ├── models/
│   │   ├── random_forest.py     # RF baseline (Model 1)
│   │   ├── unet.py              # U-Net (Model 2) + checkpoint save/load
│   │   └── custom_cnn.py        # Custom CNN (Model 3) — placeholder
│   ├── training/
│   │   ├── train.py             # RF training entrypoint
│   │   ├── train_unet.py        # U-Net training entrypoint
│   │   └── sweep_unet.py        # Multi-seed U-Net sweep (mean±std reporting)
│   ├── validation/
│   │   ├── metrics.py           # Precision, Recall, F1, IoU, MCC, balanced acc + confusion matrix
│   │   ├── report.py            # One-page PDF evaluation report (matplotlib)
│   │   ├── evaluate.py          # RF test-set evaluation entrypoint
│   │   ├── evaluate_unet.py     # U-Net test-set evaluation entrypoint
│   │   ├── ablation_features.py # RQ2 feature ablation (bands vs indices vs spatial)
│   │   ├── threshold.py         # Decision-threshold sweep on val (selection TODO)
│   │   └── figures.py           # Thesis figures — run AFTER evaluate_unet
│   └── inference/
│       ├── export_predictions.py       # Export RF GeoTIFFs for QGIS
│       └── export_predictions_unet.py  # Export U-Net GeoTIFFs for QGIS
│
├── test/data/           # Run artifacts (gitignored): model/, val/, outputs/
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
- Trained model:      `test/data/model/rf_baseline_rf.joblib`
- Fitted scaler:      `test/data/model/rf_baseline_scaler.joblib`
- Validation metrics: `test/data/val/rf_baseline_val_metrics.json` (+ `_report.txt`)
- PDF report:         `test/data/val/rf_baseline_val_report.pdf`
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
`test/data/outputs/rf_baseline_test_metrics.json` (+ `_report.txt`), plus a one-page
PDF summary (metrics table, confusion-matrix heatmap, classification report) at
`test/data/outputs/rf_baseline_test_report.pdf`.

## Inspecting predictions in QGIS

Export per-patch classification maps as georeferenced GeoTIFFs that overlay the
source Sentinel-2 imagery:

```bash
python -m src.inference.export_predictions --split test
# or choose a split / output location:
python -m src.inference.export_predictions --split val --out-dir test/data/outputs/predictions
```

Each `<patch>_pred.tif` is written to `test/data/outputs/predictions/` as a single-band
`uint8` raster carrying the source patch's CRS and transform. Pixel values:
`0` = non-debris, `1` = debris, `255` = nodata. An embedded colormap renders
debris in red with everything else transparent, so the layer drops cleanly on
top of a basemap in QGIS.

## U-Net segmentation model

A second baseline: a from-scratch U-Net that solves the **same binary
debris/non-debris task** on the same MARIDA splits, but end-to-end from the raw
6-band patches (it learns spatial context, unlike the pixel-wise RF).

```bash
# train (writes checkpoint + validation metrics/report/PDF)
python -m src.training.train_unet                 # 80 epochs, early-stop on val F1
python -m src.training.train_unet --epochs 30     # shorter run

# evaluate on the test split
python -m src.validation.evaluate_unet

# export test predictions for QGIS
python -m src.inference.export_predictions_unet --split test
```

Outputs (mirroring the RF layout):
- Checkpoint:          `test/data/model/unet_baseline.pt` (weights + normalization stats)
- Validation metrics:  `test/data/val/unet_val_metrics.json` (+ `_report.txt`, `_report.pdf`)
- Test metrics:        `test/data/outputs/unet_test_metrics.json` (+ `_report.txt`, `_report.pdf`)
- Predictions:         `test/data/outputs/predictions_unet/` (same GeoTIFF format as the RF exporter)

Training details: per-channel normalization (stats computed on the train split),
masked `BCEWithLogitsLoss` + soft-Dice (nodata pixels excluded; `pos_weight`
capped to keep the extreme debris imbalance stable), Adam + gradient clipping,
early stopping on validation debris-F1. A handful of MARIDA patches contain
NaN pixels; these are sanitized to zero after normalization so they cannot
poison the convolutions.

### Reproducibility: report seeds, not single runs

The test split carries **381 debris pixels among 194,863 valid ones**, so a
single training run measures the seed as much as the model. U-Net results are
therefore reported as mean ± SD over five seeds:

```bash
python -m src.training.sweep_unet          # 5 seeds; writes unet_seed_sweep.json
python -m src.training.sweep_rf            # same, for the RF (CPU, minutes)
```

The RF sweep exists so both sides of the comparison are stated the same way — a
single fit compared against a distribution is not a comparison. It varies the
per-patch training subsample *and* the forest's own randomness, with
hyperparameters frozen at the CV winner.

Each seed early-stops on validation F1; the promoted checkpoint (used for the
figures and the prediction exports) is chosen on **validation** F1 alone. The
test split is never consulted in that choice, and validation metrics are
selection scores — not generalization estimates — so they are not quoted here.

Results (debris class, test split — all valid pixels):

Both models over 5 seeds (mean ± SD):

| Model | Precision | Recall | F1 | IoU |
|-------|-----------|--------|------|------|
| Random Forest (7-band + NDVI/FDI, CV-tuned) | 0.808 ± 0.006 | 0.828 ± 0.003 | **0.818 ± 0.003** | 0.692 ± 0.004 |
| U-Net (6-band, from scratch)                | 0.714 ± 0.087 | 0.910 ± 0.021 | **0.798 ± 0.060** | 0.668 ± 0.083 |

Per-seed U-Net test F1 ranges **0.71 – 0.87** (`unet_seed_sweep.json`), while its
validation F1 is near-constant at 0.898 ± 0.006 — the signature of a small
evaluation set, not of unstable training. The RF's spread is **±0.003**, twenty
times tighter (`rf_seed_sweep.json`).

**The two models finish level, and the RF is far more stable.** On F1 and IoU the
gap is smaller than the U-Net's seed spread. What separates them is an
*operating point*, not accuracy: the U-Net trades precision for recall (0.91 vs
0.82, bought at 0.71 vs 0.81), and it does so in every run.

### Statistical comparison (patch-level bootstrap)

```bash
python -m src.validation.bootstrap         # writes bootstrap_ci.json
```

Resamples the 359 test **patches** with replacement (2,000 times) and recomputes
the metrics. Patches, not pixels: debris occurs in contiguous slicks, so pixels
within a patch are not independent and resampling them would return an interval
several times too narrow. Both models are scored on the same resampled sets, so
the difference is *paired*.

| Metric | Random Forest | U-Net (promoted seed) | Δ (U-Net − RF), 95% CI | P(U-Net better) |
|---|---|---|---|---|
| Precision | 0.805 [0.717, 0.873] | 0.713 [0.600, 0.819] | **[−0.162, −0.015]** | 0.9% |
| Recall | 0.824 [0.755, 0.878] | 0.895 [0.845, 0.940] | **[+0.016, +0.138]** | 99.5% |
| F1 | 0.815 [0.749, 0.863] | 0.794 [0.714, 0.862] | [−0.076, +0.036] | 26.8% |
| IoU | 0.687 [0.598, 0.760] | 0.658 [0.555, 0.757] | [−0.102, +0.052] | 26.8% |

Reading: the precision and recall intervals **exclude zero** in opposite
directions — those differences are real. The F1 and IoU intervals **include
zero**, so no overall accuracy difference is detectable on this benchmark. The
operating-point story is not a hedge; it is the only difference the data
supports.

Note the two uncertainties are distinct and neither subsumes the other. The
bootstrap measures how precisely *one trained model's* score is known on *this*
test set; the seed sweeps measure how much the score moves when you retrain. For
the U-Net the second is the larger of the two (±0.060 vs a ±0.074-wide half-CI);
for the RF it is negligible.

The comparison is in fact slightly worse than a tie for the U-Net. Sweeping the
promoted seed's decision threshold traces its full precision-recall frontier,
and the RF operating point sits **above** that curve: matched at the RF's own
recall (0.824) the U-Net reaches 0.777 precision against the RF's 0.805, and
even at its oracle-best threshold it reaches F1 0.806 against the RF's 0.815.
Threshold-free average precision is 0.815. So for this seed there is no
threshold at which the U-Net dominates the baseline.

The Random Forest uses the red-edge band B06 and the Biermann (2020) FDI, with
hyperparameters selected by cross-validation (best: `max_depth=None,
min_samples_leaf=2, n_estimators=200`). It is fitted once from a fixed seed, so
it contributes a single point rather than a distribution.

Two caveats bound all of the above. Evaluation rests on those 381 debris pixels,
so differences of a few hundredths of F1 are not measurable here. And MARIDA
labels only ~0.99% of the pixels in a patch (213,102 of 21,495,808 on the val
split); the rest are *unlabelled*, not verified debris-free, and are excluded
from both loss and metrics. The imbalance these models actually confront is
≈197:1 **among annotated pixels**, and the false-positive rate over open water
is not measured by these experiments at all.

### Feature ablation (RQ2)

```bash
python -m src.validation.ablation_features
```

Isolates the marginal value of the spectral indices and of spatial context, on
the test split (debris class). RF hyperparameters are held fixed at the CV
winner; only the feature set varies.

| Configuration | Features | F1 | IoU |
|---|---|---|---|
| RF — bands only | 7 | 0.77 | 0.63 |
| RF — indices only (NDVI, FDI) | 2 | 0.02 | 0.01 |
| RF — bands + indices | 9 | 0.81 | 0.69 |
| U-Net — bands + spatial context (5 seeds) | 6 (+conv) | 0.80 ± 0.06 | 0.67 ± 0.08 |

Reading: the raw bands carry most of the signal (0.77 F1 alone); NDVI+FDI add a
modest but real **+0.045 F1** on top; the indices *alone* are near-useless (they
discard absolute brightness and only sharpen a decision made from the bands).
Adding **spatial context** (RF → U-Net) contributes **nothing measurable** once
the seed spread is accounted for — the U-Net mean sits marginally *below* the RF
on both metrics, well inside ±1 SD.

Note that a single-run U-Net figure would have reversed that conclusion: the
promoted-seed value of 0.88 F1 previously reported here is one draw from a
0.71–0.87 spread, selected by validation F1. The `U-Net` row above is the
five-seed mean; the row written by `ablation_features.py` into
`test/data/outputs/ablation_features.json` is still the single promoted
checkpoint, and should be read as one seed, not as the result.

## References

- Kikaki et al. (2022). *MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data.* PLOS ONE.
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262247
- MARIDA repository: https://github.com/marine-debris/marine-debris.github.io
