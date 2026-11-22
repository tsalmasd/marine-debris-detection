# Marine Plastic Detection

Master's thesis project: validation and comparison of three models for detecting marine plastic debris in Sentinel-2 imagery using the [MARIDA Benchmark Dataset](https://github.com/marine-debris/marine-debris.github.io).

> **Remaining work is planned in [`REVISED_PLAN.md`](REVISED_PLAN.md)** (which
> supersedes the order of work in [`NEXT_STEPS.md`](NEXT_STEPS.md)) — a 7-band
> U-Net retrain, a loss-objective ablation, and the three-model bootstrap. All
> three models are now implemented. Read both before changing the U-Net
> objective or the reported metrics; they record the derivations behind those
> decisions.

## Models compared

1. **Random Forest** — traditional machine learning baseline (pixel-wise classification)
2. **U-Net / ResNet** — standard deep learning model (segmentation or classification)
3. **Custom CNN** — bespoke architecture; a W x W window classifier that labels the
   window's centre pixel, so it is scored on the same pixels as the other two

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
│   │   ├── patch_dataset.py     # PyTorch Dataset (dense masks) for U-Net
│   │   └── window_dataset.py    # PyTorch Dataset (W x W window per labelled pixel) for the CNN
│   ├── features/
│   │   └── spectral_indices.py  # NDVI, FDI, NDWI computation
│   ├── models/
│   │   ├── random_forest.py     # RF baseline (Model 1)
│   │   ├── unet.py              # U-Net (Model 2) + checkpoint save/load
│   │   └── custom_cnn.py        # WindowCNN centre-pixel classifier (Model 3) + checkpoint I/O
│   ├── training/
│   │   ├── train.py             # RF training entrypoint
│   │   ├── train_unet.py        # U-Net training entrypoint
│   │   ├── sweep_unet.py        # Multi-seed U-Net sweep (mean±std reporting)
│   │   ├── sweep_rf.py          # Multi-seed RF sweep (error bars for the baseline)
│   │   ├── train_cnn.py         # Window CNN training entrypoint
│   │   └── sweep_cnn.py         # Multi-seed window CNN sweep (one per window size)
│   ├── validation/
│   │   ├── metrics.py           # Precision, Recall, F1, IoU, MCC, balanced acc + confusion matrix
│   │   ├── report.py            # One-page PDF evaluation report (matplotlib)
│   │   ├── evaluate.py          # RF test-set evaluation entrypoint
│   │   ├── evaluate_unet.py     # U-Net test-set evaluation entrypoint
│   │   ├── evaluate_cnn.py      # Window CNN test-set evaluation entrypoint
│   │   ├── ablation_features.py # RQ2 feature ablation (bands vs indices vs spatial)
│   │   ├── threshold.py         # Decision-threshold sweep on val (selection TODO)
│   │   ├── bootstrap.py         # Patch-level bootstrap CIs + paired model contrast
│   │   └── figures.py           # Thesis figures — run AFTER evaluate_unet
│   └── inference/
│       ├── export_predictions.py       # Export RF GeoTIFFs for QGIS
│       └── export_predictions_unet.py  # Export U-Net GeoTIFFs for QGIS
│
├── tests/               # pytest suite (window augmentation correctness)
├── pytest.ini           # testpaths = tests  (NOT test/, the artifacts dir)
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
python -m src.validation.bootstrap                 # all four models
python -m src.validation.bootstrap --cnn-ckpt ""   # RF vs U-Net only
```

Resamples the 359 test **patches** with replacement (2,000 times) and recomputes
the metrics. Patches, not pixels: debris occurs in contiguous slicks, so pixels
within a patch are not independent and resampling them would return an interval
several times too narrow. All four models are scored on the same resampled sets,
so every difference is *paired*. Each model's own preprocessing is applied to the
same patch and the same `valid` mask, so a CNN is scored on precisely the pixels
the other models are scored on (verified bit-identical against the training-time
window dataset).

Promoted checkpoints, 95% percentile intervals:

| Model | Context | Precision | Recall | F1 | IoU |
|---|---|---|---|---|---|
| Random Forest | 1 px | 0.805 [0.717, 0.873] | 0.824 [0.755, 0.878] | 0.815 [0.749, 0.863] | 0.687 [0.598, 0.760] |
| CNN-16 | 16 px | 0.786 [0.705, 0.859] | 0.908 [0.819, 0.978] | 0.843 [0.780, 0.897] | 0.728 [0.639, 0.813] |
| CNN-32 | 32 px | 0.760 [0.670, 0.834] | 0.929 [0.865, 0.981] | 0.836 [0.775, 0.887] | 0.718 [0.633, 0.797] |
| U-Net | 256 px | 0.713 [0.600, 0.819] | 0.895 [0.845, 0.940] | 0.794 [0.714, 0.862] | 0.658 [0.555, 0.757] |

Paired contrasts (`paired_deltas` in the JSON holds all six pairs; the three
against the baseline are shown here):

| Contrast | Precision Δ, 95% CI | Recall Δ, 95% CI | F1 Δ, 95% CI |
|---|---|---|---|
| CNN-16 − RF | −0.019 [−0.085, +0.053] | +0.084 [−0.026, +0.184] | +0.028 [−0.040, +0.097] |
| CNN-32 − RF | −0.045 [−0.105, +0.016] | **+0.105 [+0.020, +0.190]** | +0.021 [−0.033, +0.084] |
| U-Net − RF | **−0.092 [−0.162, −0.015]** | **+0.071 [+0.016, +0.138]** | −0.021 [−0.076, +0.036] |

Reading: **no F1 or IoU difference between any pair of the four models is
resolvable on this benchmark** — all six paired F1 intervals include zero,
including CNN-32 vs the U-Net (−0.042 [−0.097, +0.012], P = 7.3%). What context
does buy is visible and monotone in the operating point: as the receptive field
grows 1 → 16 → 32 → 256 px, precision falls 0.805 → 0.786 → 0.760 → 0.713 while
recall rises 0.824 → 0.908 → 0.929 → 0.895. Spatial context trades precision for
recall; it does not raise overall accuracy. Only two of those trades clear the
noise: CNN-32 beats the RF on recall, and the U-Net's precision deficit against
the RF.

The single-model intervals are also wide enough to swallow the differences — each
spans roughly ±0.06 F1 on 381 debris pixels — so this is a statement about what
the benchmark can measure, not a claim that the models are identical.

Note these rows are the **val-selected promoted checkpoints**, which is why they
sit above the 5-seed means above (CNN-16 reads 0.843 here against a 0.806 ± 0.022
mean). Promotion picks the best validation seed, so a promoted row is an
optimistic draw from the seed distribution; quote the sweep means for the models'
performance and these intervals for how precisely *one* checkpoint is pinned.

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
on both metrics, well inside ±1 SD. The window CNN tests this at two intermediate
receptive fields (16 and 32 px) rather than only at 256, and reaches the same
conclusion: see the context curve under *Custom CNN window classifier* below.

Note that a single-run U-Net figure would have reversed that conclusion: the
promoted-seed value of 0.88 F1 previously reported here is one draw from a
0.71–0.87 spread, selected by validation F1. The `U-Net` row above is the
five-seed mean; the row written by `ablation_features.py` into
`test/data/outputs/ablation_features.json` is still the single promoted
checkpoint, and should be read as one seed, not as the result.

## Custom CNN window classifier (Model 3)

```bash
python -m src.training.train_cnn --window 16          # single run
python -m src.training.sweep_cnn --window 16          # 5 seeds -> cnn_w16_seed_sweep.json
python -m src.training.sweep_cnn --window 32          # 5 seeds -> cnn_w32_seed_sweep.json
python -m src.validation.evaluate_cnn --ckpt test/data/model/cnn_w16.pt
pytest                                                # augmentation correctness tests
```

The model takes a `W x W` window of 7 bands (`RF_BANDS`, including the red-edge
B06 the FDI needs) and predicts the class of the window's **centre pixel**. One
example per annotated pixel; class 0 (nodata) emits none. That is what makes a
three-model table legitimate: the RF sees 1 pixel, this CNN sees `W`, the U-Net
sees 256, and **all three are scored on the same labelled test pixels**.
Varying `W` traces a context curve with the other two models as its endpoints.

Architecture: three conv blocks, then `Flatten -> Linear(128) -> Dropout ->
Linear(1)` (357k parameters at W=16, 1.14M at W=32). Flatten rather than global
pooling on purpose — global pooling is translation-invariant, which is exactly
wrong when the label belongs to one specific position in the window.

Training matches the U-Net's recipe so the comparison is about receptive field
and not about tuning effort: masked `BCEWithLogitsLoss` with
`pos_weight = min(raw, 20)` (raw is 220:1 here), gradient clipping at 1.0,
threshold 0.5, dihedral augmentation (rot90 + flips), early stopping and model
selection on **validation** F1 alone, 5 seeds, 80 epochs, patience 15.

Results (debris class, test split, 5 seeds, mean ± SD):

| Model | Context | Precision | Recall | F1 | IoU | AP |
|---|---|---|---|---|---|---|
| Random Forest | 1 px | 0.808 ± 0.006 | 0.828 ± 0.003 | **0.818 ± 0.003** | 0.692 ± 0.004 | — |
| CNN-16 | 16 px | 0.720 ± 0.037 | 0.915 ± 0.008 | **0.806 ± 0.022** | 0.675 ± 0.031 | 0.894 ± 0.016 |
| CNN-32 | 32 px | 0.768 ± 0.014 | 0.899 ± 0.022 | **0.828 ± 0.009** | 0.707 ± 0.014 | 0.890 ± 0.028 |
| U-Net (6-band) | 256 px | 0.714 ± 0.087 | 0.910 ± 0.021 | **0.798 ± 0.060** | 0.668 ± 0.083 | — |

Reading: **the curve is flat from 1 px to 32 px.** CNN-32 is nominally the
highest F1 of any model here, but it leads the RF by 0.010 with a seed spread of
±0.009 — about one SD, which this test split cannot resolve (see the 381-pixel
caveat above). The honest statement is that 1-, 16- and 32-pixel context all
finish level, and the 256-pixel U-Net is nominally lowest with six times the
spread. The paired bootstrap above covers all four models and agrees: no F1 or
IoU difference between any pair is resolvable, and what context buys is recall
at the cost of precision.

Both CNNs sit at high recall against lower precision, which is the
`pos_weight=20` operating point, not an accuracy result. Their threshold-free
**average precision of ~0.89** is the strongest ranking of any model measured so
far, against a random-ranking floor of 0.00196.

Two confounds to state plainly. The U-Net row is the **6-band** model without
B06, so the CNN-vs-U-Net gap mixes architecture with band availability until
that is retrained (`NEXT_STEPS.md`). And runs are not bit-reproducible:
`cudnn.benchmark` is on and no deterministic-algorithm flag is set, so a seed
fixes the data order and initialisation but not the kernels. Re-running W=32
seed 0 at a larger epoch budget gave val F1 0.9099 against 0.9107 — the same
plateau, a different trajectory.

### Epoch budget: a plateau is not convergence failure

`sweep_cnn.py` warns when a seed's best epoch lands within `patience` of the cap,
because early stopping never got the chance to fire and the budget, not
convergence, ended the run. Raising the cap from 40 to 80 epochs was worth
+0.049 F1 at W=16 and +0.075 at W=32, and cut W=32's seed spread from ±0.063 to
±0.009 — most of the early instability was undertrained seeds, not capacity.

The warning then fired again at 80, and that one was a false alarm. Validation F1
had flattened (mean 0.8896 over epochs 41–60 against 0.8940 over 61–80, with
±0.02 noise), and **on a plateau the argmax of a noisy series drifts late by
chance**, so the test would have fired at any budget while each re-run bought
only a luckier argmax. A 160-epoch probe confirmed it: W=32 seed 0 peaked at
epoch 63 and early-stopped well inside the budget. The guard now compares the
last `patience` epochs against the `patience` before them, scaled by the noise
over both, and only warns when a real climb clears its own noise.

Superseded results are kept, never overwritten: `cnn_w{16,32}_e40_seed_sweep.json`
(40-epoch budget) and `cnn_w32_probe_seed0_e160_seed_sweep.json` (the probe).

## References

- Kikaki et al. (2022). *MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data.* PLOS ONE.
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262247
- MARIDA repository: https://github.com/marine-debris/marine-debris.github.io
