# Revised plan — read this first

> Written 2026-11-20 after a full review of the repo (commit `8b236b0`).
> It **supersedes the "Order of work" section of `NEXT_STEPS.md`** but not the
> rest of it: the fix-ups, the Dice derivation and the pre-registration rules in
> `NEXT_STEPS.md` still hold. Where the two disagree, this file wins.
>
> Put this file in the repo root next to `README.md` and `NEXT_STEPS.md`.

---

## 0. TL;DR

The statistics are already solid: multi-seed results, a patch-level bootstrap,
and a pre-registered loss experiment. Five things are still wrong or missing:

1. **The RF-vs-U-Net comparison is confounded.** The RF sees 7 bands (including
   **B06**) plus NDVI and FDI. The U-Net sees 6 bands, with no B06 and no indices.
   "Spatial context adds nothing" cannot be concluded yet.
2. **Model 3 does not exist**, and the plan for it (one label per 256×256 patch)
   cannot be scored on the same pixels as the other two.
3. **There is no augmentation** in the deep models.
4. **There is no error analysis.** The multi-class MARIDA labels are available
   but go unused, and nothing explains the 0.71–0.87 spread of U-Net test F1
   across seeds.
5. **The loss experiment is too big** (4 variants × 5 seeds) for an expected null
   result, and it is planned on a U-Net recipe that is about to change.

The aim is a three-model table in which the **only things that vary are spatial
context and input features**:

| Model | Spatial context | Inputs |
|---|---|---|
| Random Forest | none (1 pixel) | 7 bands (+ NDVI, FDI) |
| Custom CNN | local (16×16 or 32×32 window) | 7 bands (+ NDVI, FDI) |
| U-Net | wide (full 256×256 patch) | 7 bands (+ NDVI, FDI) |

---

## 1. Background: decisions made and why

These are the conclusions of the design discussion, written down so they don't
get relitigated.

### 1.1 Scope stays at three models: RF, U-Net, custom CNN
SegFormer, DeepLabV3+, Swin-UNet and TransUNet were considered and **rejected**.
- They turn a clean three-model study into a five- or six-model one.
- Pretrained Transformer weights expect RGB. Adapting them to 7–11 Sentinel-2
  bands weakens the main reason to use them.
- SegFormer predicts at 1/4 resolution. Debris objects are often only 1–3 pixels
  wide at 10 m, so that resolution hurts exactly the class that matters.
- If a supervisor insists on a Transformer, add **SegFormer-B0/B2 as an
  appendix-only extra**, not as part of the main comparison.

### 1.2 No "PINN", and no physics loss in the main thesis
A Physics-Informed Neural Network enforces a PDE residual. The ideas that came up:
- **Advection–diffusion of debris:** needs time series plus ocean current and
  wind data. MARIDA is single-date patches. **Rejected.**
- **Linear spectral unmixing loss** `R̂_b = f·R_debris,b + (1−f)·R_water,b`:
  **dropped**, for three reasons:
  1. The network outputs a debris **probability** p. The equation needs a
     debris **fractional cover** f. They are different quantities, and using p
     as f is not physically meaningful.
  2. No single "plastic spectrum" exists. It varies with polymer, biofouling,
     submergence and pixel fill. A fixed endmember pushes the model toward the
     wrong answer.
  3. Water reflectance varies by scene (sediment, glint, depth), so a global
     water endmember is poor.
- **What the thesis uses instead:** physics enters as **input features** (FDI,
  NDVI), tested in all three models (step 4). The right label is
  "physics-guided features", not "physics-informed network".
- Unmixing or PINN ideas go in **Future work**, one paragraph.

### 1.3 Facts about the current code that drive the plan
Checked on commit `8b236b0`:

| Fact | Where |
|---|---|
| U-Net bands = `DEFAULT_BANDS` = `[B02,B03,B04,B08,B11,B12]` (6, **no B06**) | `src/data/dataset_loader.py:32` |
| RF bands = `RF_BANDS` = `[B02,B03,B04,B06,B08,B11,B12]` (7) | `src/data/dataset_loader.py:38` |
| `fit_unet` hard-codes `in_channels=6` | `src/training/train_unet.py` (inside `fit_unet`) |
| `train_unet.py` / `sweep_unet.py` have **no `--bands` flag** | argparse in both |
| The ablation docstring wrongly says the U-Net has "the same spectral bands" | `src/validation/ablation_features.py:18` |
| No augmentation of any kind | `src/data/patch_dataset.py` (`MaridaSegmentationDataset.__getitem__`) |
| The dataset returns only the binary target + valid mask; the class label is discarded | `patch_dataset.py`, `target = (label == 1)`, `valid = (label > 0)` |
| `custom_cnn.py` is a docstring only | `src/models/custom_cnn.py` |
| FDI is **linear** in the bands; NDVI is a ratio (non-linear) | `src/features/spectral_indices.py` |

**Why B06 matters more than the indices.** FDI = B08 − [B06 + (B11 − B06)·k],
a fixed linear combination of three bands. A CNN can compute it exactly with one
1×1 convolution, **but only if it can see B06.** So the U-Net's real handicap is
the missing band. The missing index is secondary.

**Why augmentation may not fix the seed spread.** U-Net validation F1 is flat
(0.898 ± 0.006) while test F1 ranges 0.71–0.87. Seeds that look the same on
validation generalise differently to test, which points to a **val/test
distribution shift** (probably a few hard test scenes), not to noisy training.
Augmentation is still correct to add, but **do not promise** it shrinks the
spread. Step 5 (per-scene analysis) is what will explain the spread.

### 1.4 MARIDA comparability (check before claiming it)
The README keeps threshold 0.5 "for comparability with MARIDA". As far as we
know, MARIDA's published baselines (Kikaki et al., 2022) are **multi-class** and
the RF there uses extra (e.g. texture) features. A binary debris-vs-rest F1 may
not compare directly with their per-class debris F1. **Re-read the paper's
evaluation section** and phrase the claim accordingly: either "comparable
protocol" with caveats, or drop the claim.

---

## 2. The plan, step by step

Every step lists: **goal → code changes → how to run → done when**.
Order matters. Steps 2–4 must finish before step 6, so the loss experiment runs
on the recipe that will actually be reported.

### Step 1 — Fix-ups and loss unit tests (no GPU)
Unchanged from `NEXT_STEPS.md` ("Fix-ups first"). Short version:
- [ ] Rewrite the `soft_dice_loss` docstring (`train_unet.py:58-72`) to describe
      the `S/(S+1)` collapse on empty patches.
- [ ] Label validation numbers as **selection scores** in `README.md` and the
      PDF metadata (`train_unet.py:264-279`). Drop the accuracy row, or show it
      next to the all-negative baseline (0.9950).
- [ ] **New:** fix `ablation_features.py:18`. The U-Net does **not** have the
      same bands as the RF.
- [ ] Create `tests/test_losses.py` in **`tests/` (plural)**. `test/` is the
      gitignored artifacts folder; pytest must not walk it. Add a `pytest.ini`
      or `pyproject.toml` with `testpaths = tests`.
      Tests to write:
      - invalid pixels (`valid == 0`) cannot change either loss term;
      - an empty-target patch is finite forwards and backwards;
      - batch-aggregated Dice == per-sample Dice at batch size 1;
      - the mixed batch from `NEXT_STEPS.md` reproduces **0.7357** (per-sample)
        and **0.1991** (batch-aggregated);
      - `sweep_thresholds` finds a known max-IoU cut on a hand-built array.

**Done when:** `pytest` passes locally, and README, docstrings and PDF metadata
no longer overclaim.

---

### Step 2 — Lock the final deep-model recipe and retrain the U-Net
**Goal:** a U-Net that sees the same bands as the RF, trained with standard
geometric augmentation. It becomes the **new U-Net result and the new control**
for every later experiment.

**Decide before running (pre-registration):**
- Bands: **`RF_BANDS` (7 bands)**. The +NDVI/FDI variant is step 4.
- Augmentation: random horizontal flip, vertical flip, and rotation by k·90°
  (k ∈ {0,1,2,3}), applied **only to the train split**, identically to image,
  target and valid mask. No colour or brightness jitter: reflectance values
  carry physical meaning.
- Everything else unchanged: loss, optimiser, LR 1e-4, batch 8, 80 epochs,
  patience 15, `pos_weight` cap 20, seeds 0–4, selection on validation F1,
  threshold 0.5.
- Both changes (bands + augmentation) are made **together**. This is the
  recipe, not an ablation; we don't attribute the gain between them. Say so in
  the thesis.

**Code changes:**
1. `src/data/patch_dataset.py`: add `augment: bool = False` to
   `MaridaSegmentationDataset.__init__`. In `__getitem__`, after normalisation
   and `nan_to_num`, when `self.augment` is set:
   ```python
   k = np.random.randint(4)
   img   = np.rot90(img,   k, axes=(1, 2))
   label = np.rot90(label, k, axes=(0, 1))
   if np.random.rand() < 0.5:
       img, label = img[:, :, ::-1], label[:, ::-1]
   if np.random.rand() < 0.5:
       img, label = img[:, ::-1, :], label[::-1, :]
   ```
   Augment `label` **before** deriving `target` and `valid`, so all three stay
   aligned. Keep the existing `np.ascontiguousarray`.
   Use a worker-safe RNG: set `worker_init_fn` in the DataLoader, or use
   `torch.randint`, so `num_workers > 0` doesn't repeat the same augmentation
   in every worker.
2. `src/training/train_unet.py`:
   - add `--bands` (choices: `default`, `rf`, `all`) and `--augment` flags;
   - replace `in_channels=6` in `fit_unet` with `in_channels=len(bands)`;
   - compute normalisation stats with the chosen bands;
   - **save `bands` and `augment` in the checkpoint** next to mean/std.
3. `src/training/sweep_unet.py`: pass the same two flags through. Write the
   output to a **new** file, e.g. `unet_seed_sweep_rf7_aug.json`, so the old
   sweep stays as a record.
4. `src/validation/evaluate_unet.py`, `src/inference/export_predictions_unet.py`,
   `src/validation/bootstrap.py`, `src/validation/figures.py`: **read `bands`
   from the checkpoint** instead of assuming 6. Fall back to `DEFAULT_BANDS`
   for old checkpoints that lack the key.
5. Give each recipe its own checkpoint names, e.g.
   `unet_rf7_aug_seed{n}.pt`, so `sweep_unet.py` deleting the losing
   checkpoints cannot destroy the old baseline.

**Run:**
```bash
python -m src.training.sweep_unet --bands rf --augment      # 5 seeds
python -m src.validation.evaluate_unet  --ckpt <promoted>   # promoted = best VAL F1
python -m src.validation.bootstrap                          # re-run the paired CI vs RF
python -m src.validation.figures                            # ALWAYS after evaluate_unet
```

**Done when:** a mean ± SD row exists for the 7-band augmented U-Net, plus a new
bootstrap table against the RF. Keep the old 6-band row in the thesis as
"original U-Net"; it now becomes part of the story.

**Sanity check:** a one-epoch run with `--bands default` and without
`--augment` must reproduce the old loss curve.

---

### Step 3 — Custom CNN (Model 3) as a window classifier
**Goal:** a third model with **local** context that is scored on **exactly the
same pixels** as the RF and the U-Net. This removes the "shared target" blocker
from `NEXT_STEPS.md`.

**Idea:** keep the architecture already sketched in `custom_cnn.py`, but change
the input. Instead of one label per 256×256 patch, cut a small **window centred
on each annotated pixel** and predict **that centre pixel's class**. It is still
"patch classification" (README promise kept), yet the output is per-pixel and
uses the same `valid` mask.

**Architecture** (`src/models/custom_cnn.py`), for a W×W window with C channels:
```
Input (C, W, W)
Conv3x3(32) - BN - ReLU - MaxPool2
Conv3x3(64) - BN - ReLU - MaxPool2
Conv3x3(128) - BN - ReLU
Global average pool  (or Flatten if W is fixed)
Dense(128) - ReLU - Dropout(0.3)
Dense(1)              -> raw logit, use BCEWithLogitsLoss
```
- **No Softmax at the end.** `BCEWithLogitsLoss` / `CrossEntropyLoss` apply it
  internally (already noted in `NEXT_STEPS.md`).
- Use padding so W = 16 still leaves a 4×4 map after two pools.

**Window sizes (pre-registered):** W = **16** and W = **32**. Optional, if time
allows: W = 8. Together with the RF (W = 1) and the U-Net (W = 256) this gives a
**context curve**: debris F1 against receptive field. That curve is the most
interesting figure the thesis can have.

**Data pipeline** (`src/data/window_dataset.py`, new):
- Reuse `load_patch` and the same normalisation stats as the U-Net (train split,
  same bands).
- Index = list of `(patch_idx, row, col)` for every pixel with `label > 0`.
- Pad each patch by W/2 with **reflect** padding, so windows near the edge
  exist; the centre pixel always comes from real data.
- Target = `label[row, col] == 1`.
- Same flip/rot90 augmentation as step 2 (it keeps the centre pixel fixed).
- **Imbalance:** about 197:1 negatives to positives among labelled pixels. Use
  the same `pos_weight = min(raw, 20)` as the U-Net, for consistency. Do **not**
  also oversample; pick one mechanism. Optional: subsample negatives per epoch
  for speed, but then recompute `pos_weight` on what is actually seen.
- Inference: the same window extraction over every valid pixel of val/test.
  Batch it (e.g. 4096 windows per batch). The model is small, so this is fast.
  Optional speed-up: run the CNN fully convolutionally over the padded patch.

**Training script** (`src/training/train_cnn.py` + `sweep_cnn.py`): copy the
structure of `train_unet.py` / `sweep_unet.py`. Same seeds 0–4, early stopping
on val debris F1, promotion on val F1 only, threshold 0.5. Reuse
`src/validation/metrics.py` and the `report.py` PDF so all three models have
identical outputs.

**Hook into the existing tooling:**
- `bootstrap.py`: add the CNN as a third model, with paired patch-level
  resampling on the same resampled sets.
- `export_predictions_cnn.py`: optional. Same GeoTIFF format as the others.

**Run:**
```bash
python -m src.training.sweep_cnn --window 16 --bands rf --augment
python -m src.training.sweep_cnn --window 32 --bands rf --augment
```

**Done when:** mean ± SD rows exist for CNN-16 and CNN-32, and the bootstrap
covers all three models.

**Talk to the supervisor first** if the thesis proposal explicitly promised a
whole-patch classifier. The window version is the same network applied
differently, and it is defensible because it makes the comparison fair.

---

### Step 4 — Physics-guided features in the deep models
**Goal:** answer the RQ2 question ("do spectral indices help?") for all three
models, not just the RF.

**Pre-registered prediction:** little or no gain for the CNNs, because FDI is
linear in the bands they already see (section 1.3). A null result here is a
**finding**: hand-crafted indices help a pixel-wise RF but are redundant for
convolutional models. Say it in advance, in writing, before running.

**Code:**
- Add a `--indices` flag to both deep pipelines. When set, compute NDVI and FDI
  **from raw reflectance, before normalisation**, using the existing
  `spectral_indices.compute_ndvi` / `compute_fdi`, append them as 2 extra
  channels (7 → 9), then normalise all 9 with train-split stats.
- Store `indices: true` in the checkpoint; evaluation and export must honour it.

**Runs:** U-Net 9-ch (5 seeds) and the **better** CNN window from step 3 with
9 channels (5 seeds). Choose "better" on **validation** F1, never test.

**Done when:** the ablation table looks like this:

| Model | bands only (7) | bands + indices (9) | Δ |
|---|---|---|---|
| RF | 0.77 | 0.81 | +0.045 (already have) |
| Custom CNN-W | … ± … | … ± … | … |
| U-Net | … ± … | … ± … | … |

Update `ablation_features.py` (or a new `ablation_summary.py`) to build this
table from the sweep JSONs, using **seed means**, not the single promoted
checkpoint.

---

### Step 5 — Error analysis (no retraining)
**Goal:** explain *how* the models fail and *why* U-Net test F1 swings from 0.71
to 0.87.

**5a. False positives by true MARIDA class.**
- `MaridaSegmentationDataset` (and the window dataset) gain a
  `return_label=True` option that also returns the raw class mask.
- A new `src/validation/error_analysis.py` gathers, for each model's promoted
  checkpoint on **test**, the true class of every false-positive pixel and of
  every missed debris pixel.
- Output: a table plus a bar chart of FP counts by class (Dense/Sparse
  Sargassum, Natural Organic Material, Ship, Foam, Waves, Wakes, water types…)
  for RF, CNN and U-Net side by side. Normalise by class size, too (FP rate per
  class), since the classes differ hugely in pixel count.
- Expected story: the U-Net's extra false alarms come from a few confuser
  classes. The thesis can name them.

**5b. Per-scene and per-patch breakdown.**
- MARIDA patch IDs encode the scene (`S2_<DATE>_<TILE>`). Group test pixels by
  scene and compute debris F1 per scene, per model and **per seed**.
- Plot the seeds × scenes matrix of F1 for the U-Net. If a handful of scenes
  account for most of the spread, that is the explanation for ±0.06, and it
  belongs in the discussion chapter.
- Also list the 10 test patches with the most FP and the most FN, and render
  them with the existing GeoTIFF exporters for a qualitative figure.

**5c. Full-scene sanity check** (already in `NEXT_STEPS.md`): look at whole
exported patches in QGIS and confirm open water doesn't light up. Qualitative
only; say so.

**Done when:** one FP-by-class figure, one per-scene figure, and one panel of
worst-case patches exist.

---

### Step 6 — Loss experiment, trimmed
**Goal:** test the one real defect found in the code (the per-sample Dice
collapse). Nothing more.

**Changes from `NEXT_STEPS.md`:**
- **2 variants, not 4:** Control vs **Batch-aggregated Dice**. "No cap" and
  Focal are dropped (optional only if GPU time is left over).
- Run on the **step 2 recipe** (7 bands + augmentation), or the step 4 recipe
  if indices won. The control is then the step 2/4 sweep itself, so only
  **5 new runs** are needed.
- The code tasks stay the same: make `loss_fn` swappable in `fit_unet`, add
  batch-aggregated Dice, and log the fraction of minibatches with no positives.
- Decision rule unchanged: promote batch Dice only if paired **validation IoU**
  improves on **every** seed **and** by ≥ 1 pp on average. Otherwise report the
  null.

---

### Step 7 — Threshold as a second operating point (optional, last)
Unchanged from `NEXT_STEPS.md`: implement `select_threshold()` in
`src/validation/threshold.py`, choose on validation, freeze, then report as a
secondary row. Keep 0.5 as the headline. Do it for all three models, or for
none.

---

## 3. GPU budget

| Step | Runs | Notes |
|---|---|---|
| 2 — U-Net 7-band + aug | 5 | the U-Net is the slow one |
| 3 — CNN W=16, W=32 | 10 | small model, fast |
| 4 — U-Net 9-ch, CNN 9-ch | 10 | |
| 6 — batch Dice | 5 | control reused from step 2/4 |
| **Total** | **≈ 30** | (+5 for optional CNN W=8) |

RF sweeps are CPU-only and already done for 7 and 9 features. Re-run
`sweep_rf.py` only if the RF pipeline changes, which it shouldn't.

---

## 4. Rules that apply to every step

- **Test is never used for any choice:** not early stopping, seed promotion,
  window size, band set, threshold or loss variant. Validation only.
- **Report seed means ± SD** for every deep model. A single promoted checkpoint
  is used only for figures, exports and the paired bootstrap, and is labelled
  as one seed.
- **Pre-register** each experiment's settings and expected outcome in this file
  (or in `NEXT_STEPS.md`) **before** running it. Add the date.
- **Never overwrite** old result JSONs or checkpoints. New recipe, new filename.
- **`figures.py` runs after `evaluate_*`**, never before (known stale-file bug).
- Primary metric is **debris IoU**. F1 is a monotone transform of IoU at a fixed
  threshold, so don't present both as separate evidence. Average precision is
  the threshold-free secondary.
- Unlabelled pixels (class 0, ~99% of each patch) are **unlabelled, not water**.
  Every metric is over labelled pixels only; scene-level FPR is not measured.

---

## 5. What the final thesis tables should look like

**Table A — main comparison** (test, debris class, mean ± SD over 5 seeds):

| Model | Context | Inputs | Precision | Recall | F1 | IoU | AP |
|---|---|---|---|---|---|---|---|
| RF | 1 px | 9 | | | | | |
| Custom CNN | W px | 7 or 9 | | | | | |
| U-Net | 256 px | 7 or 9 | | | | | |
| *(U-Net, original 6-band, no aug)* | 256 px | 6 | 0.714 ± 0.087 | 0.910 ± 0.021 | 0.798 ± 0.060 | 0.668 ± 0.083 | |

**Table B — paired bootstrap:** every model pair, Δ with 95% CI, P(better).

**Table C — feature ablation:** step 4 table.

**Figure 1 — context curve:** F1/IoU against receptive field (1, 16, 32, 256),
with error bars.

**Figure 2 — false positives by MARIDA class**, three models side by side.

**Figure 3 — per-scene F1 × seed** heatmap for the U-Net.

**Optional appendix:** loss experiment (control vs batch Dice); threshold-tuned
operating points; SegFormer, if a supervisor asks for it.

---

## 6. Open questions to settle early

- [ ] **Supervisor:** is a window-based Model 3 acceptable in place of a
      whole-patch classifier?
- [ ] **MARIDA paper:** is the binary debris-vs-rest protocol comparable to
      their reported numbers? (section 1.4)
- [ ] **Confidence masks:** MARIDA ships `_conf.tif` (annotation confidence),
      and `load_patch(..., return_conf=True)` already reads it. Decide whether
      to use it (e.g. as a loss weight, or a "high-confidence-only" evaluation
      subset) or to state explicitly that it is ignored. Don't decide after
      seeing results.
- [ ] **Bands for everyone?** Using all 11 bands in every model would also be
      fair, but it means re-running the RF. Current choice: align the deep
      models to the RF's 7 bands. Mention the 11-band option as a limitation.

---

## 7. Progress checklist

- [ ] Step 1 — fix-ups + `tests/test_losses.py` passing
- [ ] Step 2 — `--bands` / `--augment` flags; bands in checkpoint; U-Net 7-band+aug sweep; new bootstrap
- [ ] Step 3 — `custom_cnn.py`, `window_dataset.py`, `train_cnn.py`, `sweep_cnn.py`; W=16 and W=32 sweeps; bootstrap with 3 models
- [ ] Step 4 — `--indices` flag; 9-channel sweeps; ablation table
- [ ] Step 5 — `error_analysis.py`; FP-by-class figure; per-scene figure; worst-patch panel
- [ ] Step 6 — swappable loss; batch Dice; 5 runs; decision recorded
- [ ] Step 7 — `select_threshold()` (optional)
- [ ] README updated: three-model table, corrected claims, new commands
- [ ] Thesis text updated in the same pass as the README (as before)
