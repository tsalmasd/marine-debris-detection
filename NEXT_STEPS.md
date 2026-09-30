# Next steps — remaining work

Planning document for the outstanding work on this thesis: reporting repairs, a
loss-objective ablation, and the third model. It records the derivations behind
each decision so that the reasoning survives independently of whoever carries the
work out.

> **Status, 2026-09-30.** R3 is closed as unnecessary (its premise was tested and
> found false) and R4 is done. R1, R2 and R5 remain open; Sprints A, P and C are
> untouched. Figures elsewhere in this repository that predate 2026-09-24 are
> superseded by the table in §1.

Everything below requires the `marine-debris` conda environment, and Sprints A
and C require a GPU:

```bash
conda env create -f environment.yml   # if not already present
conda activate marine-debris
```

---

## 1. Where the project stands

Detecting marine plastic in Sentinel-2 imagery on the MARIDA benchmark, comparing
three models.

| Model | Status |
|---|---|
| 1. Random Forest | done — test F1 **0.818 ± 0.003** / IoU **0.692 ± 0.004** over 5 seeds |
| 2. U-Net (from scratch, 6-band) | done — test F1 **0.798 ± 0.060** / IoU **0.668 ± 0.083** over 5 seeds (per-seed F1 0.71–0.87) |
| 3. Custom CNN | **not implemented** — `src/models/custom_cnn.py` is a docstring |

**The two completed models finish level.** The gap on F1 and IoU is smaller than
the seed spread; what separates them is an operating point (U-Net recall
0.91 ± 0.02 against 0.82, at a precision of 0.71 ± 0.09 against 0.81), not
accuracy. The Random Forest operating point sits *above* the promoted U-Net's
entire precision-recall frontier: at the forest's own recall (0.824) the U-Net
reaches 0.777 precision against 0.805, and at its oracle-best threshold it
reaches F1 0.806 against 0.815 (AP 0.8149). There is no threshold at which that
U-Net dominates the baseline.

This reframes the ablation below. It is no longer a search for a better score but
the candidate *explanation* for the U-Net's precision instability — see F-A.

The U-Net's objective is masked `BCEWithLogitsLoss(pos_weight=min(raw, 20))` plus
a soft-Dice term, summed unweighted (`src/training/train_unet.py:51-77`). The
question that motivated this document was whether an imbalance-aware objective
(Focal, Tversky, Lovász) would improve it. The conclusion reached below is to run
that as a pre-registered ablation, expect a null result, and report the null: its
value is as evidence in the thesis, not as a performance gain.

---

## 2. Findings that motivate the plan

Each of the following was derived from, or read directly out of, the source.

### F-A. The Dice term degenerates on debris-free patches

`soft_dice_loss` (`train_unet.py:58-72`) reduces over `dim=(1,2)` — **per sample**
— and then takes the mean. On a patch containing no debris, `tgt.sum() == 0`, so
with `eps=1.0`:

```
loss = S / (S + 1)        where S = total predicted debris probability in the patch
∂loss/∂p = 1 / (S+1)^2                 (probability gradient)
∂loss/∂z = p(1-p) / (S+1)^2            (logit gradient)
```

Verified numerically: the gradient is **1.000** at S=0, **0.0278** at S=5 and
**0.000384** at S=50. This is not linear suppression — it collapses
quadratically. As a hard-negative mining signal the shape is precisely wrong: the
maximum corrective push falls on empty patches that are already clean, and
essentially none on empty patches the model floods with false positives.

For a realistic batch (8 patches: one with 100 debris pixels of which 90 are
caught, seven empty with S=5 each):

```
per-sample Dice, .mean()  = 0.7357     <- what the code currently does
batch-aggregated Dice     = 0.1991

loss-value share from the 7 patches with NO debris : 99.1%
loss-value share from the 1 patch that HAS debris  :  0.9%
```

The term whose docstring claims it "optimizes overlap directly"
(`train_unet.py:62-65`) is 99% driven by patches with no overlap to optimise.
That docstring sentence is false; correcting it is task R1.

Two corrections to earlier characterisations of this defect:

- It does **not** oppose `pos_weight` within a patch. `pos_weight` scales only
  `target == 1` terms (`train_unet.py:133-136`), and an empty patch has none, so
  `pos_weight` is inert there. The real problem is *cross-patch*: identical
  negative pixels receive different gradients depending on whether their patch
  happens to contain debris.
- The BCE term is **batch**-normalised (`train_unet.py:54`) while Dice is
  **per-sample**-then-meaned (`:72`), and the two are summed with no weighting
  coefficient (`:77`). Two differently normalised terms added as equals is an
  unmeasured design choice.

### F-B. Annotation coverage, stated correctly

`patch_dataset.py:147-148`: `target = (label == 1)`, `valid = (label > 0)`. Class
0 is nodata or unlabelled and is excluded from **both the loss and the metrics**.

```
val split: 328 patches x 256 x 256 = 21,495,808 pixels
scored (valid)                     =    213,102   -> 0.99%
excluded                           = 21,282,706   -> 99.01%
```

The defensible claim is therefore: **197:1 imbalance among annotated pixels, at
about 0.99% annotation coverage.** The stronger-sounding claim that the true
scene ratio is around 20,000:1 is not defensible, because it presumes every
excluded pixel is non-debris when they are *unlabelled* rather than verified
negatives.

Consequence for the thesis: every loss arm optimises the 197:1 annotated
imbalance. No loss can claim to address scene-level imbalance, and scene-level
false-positive behaviour is unmeasured. This belongs in the limitations, not in
an experiment.

### F-C. `segmentation-models-pytorch` is already a dependency and unused

`environment.yml` pins `segmentation-models-pytorch==0.3.4`, but
`grep -rn "smp\.\|segmentation_models" --include=*.py` returns nothing — the
U-Net is hand-written in `src/models/unet.py`. `smp.losses` ships `FocalLoss`,
`TverskyLoss`, `LovaszLoss`, `DiceLoss` and `JaccardLoss`. `torchmetrics=1.4.*`
is likewise pinned and unused, and provides `AveragePrecision`.

No loss function needs implementing from scratch; everything required is already
installed.

### F-D. Validation F1 is selection-biased

`train_unet.py:144-155` early-stops on validation F1 and checkpoints on
improvement; `:239-248` reloads that checkpoint and re-evaluates it on the *same*
validation loader to produce the PDF. The resulting figure is therefore a maximum
over roughly 34 noisy evaluations of the selection metric. The test number is the
honest one, and validation figures are not quoted as generalisation estimates
anywhere in the thesis.

### F-E. The multi-seed instrument predates its use in the headline

`src/training/sweep_unet.py` trains N seeds, selects by validation F1 and never
by test (`:97-99`), and reports test mean ± std (`:104-117`), deleting the
non-promoted checkpoints (`:138-142`). Its own docstring notes that single-run
test metrics are high-variance because the test split holds roughly 381 debris
pixels (`:4-5`). The README nonetheless reported single-point numbers until R4
corrected it on 2026-09-24. The finding stands; it is why R4 existed.

### F-F. Focal loss has less to act on here than usual

MARIDA negatives are classes 2–15. They include genuinely hard negatives
(Sargassum, ships, foam, wakes, sediment) **and** easy water (7 Marine Water,
10 Turbid Water, 11 Shallow Water, 15 Mixed Water). Easy negatives are therefore
not absent, but the pool is enriched for hard classes relative to a full scene.
Focal's down-weighting of easy negatives should be expected to buy little. This
prediction is made in advance and should be reported alongside the outcome.

---

## 3. Out of scope, and why

Each of the following costs thesis time for no defensible gain:

- A loss zoo — Lovász, asymmetric losses, unified or class-balanced focal,
  boundary losses.
- Grid-searching γ, α, β or the BCE/Dice coefficient λ. These are fixed in
  advance instead.
- Choosing Tversky's asymmetry from test precision and recall already seen.
- Running the ablation more than once per configuration set, or tuning anything
  on the test split.
- Migrating the hand-written U-Net to `smp` encoders: that changes the *model*
  rather than the loss, and breaks the three-model comparison.
- Treating unlabelled pixels as negatives, or claiming that any loss addresses
  scene-level imbalance.
- Chasing a number that beats the baseline. The comparison table is the
  contribution.

---

## 4. Sprint R — repairs

No GPU or data required. Each ticket is atomic and independently committable.

### R1 — correct the Dice docstring

**File:** `src/training/train_unet.py:58-72`

The docstring claims Dice "cannot produce exploding gradients" — a bounded loss
value does not bound its gradients — and that it "optimizes overlap directly",
which is false on debris-free patches, which contribute 99.1% of the term.
Rewrite it to describe what the code does, including the `S/(S+1)` degeneration.

**Validation:** `python -m src.training.train_unet --help` still works; the
docstring matches the derivation in F-A.

### R2 — qualify the validation metrics in the reports

**Files:** `README.md`, `src/training/train_unet.py:264-279` (PDF metadata)

Record in the validation PDF metadata and the README that validation metrics are
model-selection scores rather than generalisation estimates (F-D). Contextualise
`accuracy: 0.9990` against the all-negative baseline of 0.9950, or drop the
accuracy row.

**Validation:** regenerate the validation PDF; the caveat is present.

### ~~R3 — align the evaluation precision path~~ — closed 2026-09-24, no change needed

The concern was that `use_amp = device.startswith("cuda")`
(`evaluate_unet.py:52`) autocasts on any CUDA device, so evaluation would run in
fp16 while training defaults to fp32, and that fp16 quantisation near sigmoid
saturation would move the reported numbers.

Measured directly on `test/data/model/unet_baseline.pt`, running the test split
twice under each path: fp16 autocast and fp32 produce **identical metrics** —
P 0.7134, R 0.8950, F1 0.7939, IoU 0.6583, TP 341, FP 137 across all four runs.
Evaluation is reproducible, and a threshold chosen in `threshold.py` reproduces
in `evaluate_unet`. No reported figure needs re-running on these grounds.

### ~~R4 — report the sweep's mean ± std rather than single points~~ — done 2026-09-24

- **Seed set:** 0–4, `python -m src.training.sweep_unet` defaults; results in
  `test/data/outputs/unet_seed_sweep.json`.
- **Promoted seed:** 4, selected on maximum validation F1 (0.9081); its test F1
  is 0.7939, close to the five-seed mean rather than the best of the five.
- **Threshold:** 0.5 throughout, kept for comparability with MARIDA. P4 remains
  open.
- **Commits:** `97e5d55`, `dcda58e`, `3377bf6`.

The thesis document was updated in the same pass, so the README and the thesis
carry identical numbers.

One further defect was found and fixed during this work: the operating point
plotted in `fig_pr_curve.png` had been read from a superseded metrics file and
sat off its own curve. `src/validation/figures.py` must run **after**
`evaluate_unet`, never before.

### R5 — unit tests for the loss terms

**File:** new `tests/test_losses.py`

> **Directory naming.** A `test/` directory (singular) already exists and is
> **not** a test suite: it holds run artifacts (`test/data/model`,
> `test/data/val`, `test/data/outputs`) and is gitignored. Put the pytest suite
> in `tests/` (plural) and do not merge the two, or pytest will walk trained
> checkpoints and prediction rasters. Renaming the artifact directory instead is
> defensible, but it touches every path constant in `src/training/` and
> `src/validation/` and is its own ticket.

Cover:

1. invalid (`valid == 0`) pixels cannot change either loss term;
2. an empty-target patch's forward and backward passes are finite;
3. batch-aggregated Dice is identical to per-sample Dice at batch size 1;
4. a mixed positive/empty batch matches the hand-derived values in F-A
   (0.7357 per-sample, 0.1991 batch-aggregated);
5. `sweep_thresholds` finds a known maximum-IoU cut on a hand-built array.

**Validation:** `pytest tests/ -v` green. This is the prerequisite for trusting
any ablation number.

---

## 5. Sprint A — the loss ablation

### A1 — add a `loss_fn` seam to `fit_unet`

**File:** `src/training/train_unet.py:105-167`

`fit_unet` hardcodes the criterion at `:133-136`. Add a `loss_fn` parameter
defaulting to the current behaviour so `sweep_unet.py` is unaffected — roughly
20 lines.

**Validation:** R5 tests still green; a one-epoch smoke run reproduces the
existing loss curve with the default.

### A2 — implement `batch_dice_loss`

**File:** `src/training/train_unet.py`

Sum intersection and denominator across all valid pixels in the minibatch
*before* taking the ratio, rather than per-sample and then meaning. This is a
one-line conceptual change. Two caveats: an all-empty batch still reduces to
`S/(S+1)`, and at batch size 1 the new form is identical to the old, so the
fraction of positive-free minibatches should be logged during training.

**Validation:** R5 tests 3 and 4.

### A3 — define the four arms

Fix γ=2, Dice ε=1 and the BCE/Dice coefficient at 1 **in advance**. Splits,
augmentation, optimiser, schedule, stopping rule and batch sampler are identical
across arms; only the objective varies.

| Arm | Objective | Isolates |
|---|---|---|
| **L0** | `WBCE(pos_weight=min(raw,20))` + per-sample Dice | control (current) |
| **L1** | same BCE + **batch-aggregated** Dice | the F-A defect |
| **L2** | `BCE(pos_weight=1)` + batch Dice | whether the never-measured cap of 20 does anything |
| **L3** | focal-modulated BCE + batch Dice | the original question |

For L3, multiply the unreduced BCE by `(1-p_t)^2`, keep the same positive cap,
and mask **before** reduction. `smp.losses.FocalLoss` is available (F-C), but
check its reduction and masking semantics against the `valid` mask before
substituting it.

### A4 — run the ablation

Use the same seed count as the L0 baseline so the comparison is paired
(`sweep_unet.py` defaults to 5; 3 is the acceptable floor). Existing L0 seed
results can be reused if the code path is unchanged.

**Validation:** each run's `best_epoch`, convergence status and positive-free
batch fraction recorded; no run silently diverged.

### A5 — the decision rule, recorded before results are seen

**Primary metric: debris IoU**, not F1. At a fixed threshold
`F1 = 2*IoU/(1 + IoU)`, so reporting both is one number twice. **Secondary:
average precision**, threshold-free, via `torchmetrics`.

> An arm is promoted over L0 only if it improves paired validation IoU in the
> same direction across **every** seed **and** by at least one absolute
> percentage point on average. Otherwise L0 is kept and the null is reported.

---

## 6. Sprint P — reporting

### P1 — the ablation table

Per arm: per-seed values and mean ± std for precision, recall, F1, IoU and AP on
test, plus paired per-seed deltas against L0. Selection by validation, never by
test.

### P2 — two explanatory paragraphs

1. The F-A degeneration, the batch-Dice correction, and the empirical verdict.
2. The F-B framing: all results are computed on roughly 0.99% of pixels, and
   scene-level false-positive rates are unmeasured.

### P3 — a full-scene sanity check

Use `src/inference/export_predictions_unet.py` to render predictions over whole
patches and confirm that the model does not light up open water. This is the only
available evidence about the 99% that F-B cannot score, and it is qualitative.

### P4 — threshold as a secondary operating point

Implement `select_threshold()` in `src/validation/threshold.py` — the criterion
is a domain judgment, discussed in its docstring — select on **validation**,
freeze, then apply via `evaluate_unet --threshold <v>`. Report it as a secondary
operating point and keep 0.5 for comparability with MARIDA. Note in writing that
validation is then triple-used (early stopping, seed promotion, threshold), so
validation metrics at the tuned cut are not a generalisation estimate.

---

## 7. Sprint C — the third model

This is a three-model thesis and the third model does not exist. It should not be
displaced by loss micro-optimisation.

### C1 — implement the patch-level classifier

**File:** `src/models/custom_cnn.py`

The planned architecture ends `Dense(2) -> Softmax`. Softmax must not be applied
before `CrossEntropyLoss`, which applies its own. Use two raw logits with
`CrossEntropyLoss`, or one raw logit with `BCEWithLogitsLoss`.

### C2 — measure its own imbalance, then choose its loss

The task is *patch-level* classification, so the imbalance is how many patches
contain any debris, measured in patches rather than the pixel-level 197:1. None
of Sprint A transfers: Dice, Tversky or Lovász on a single patch label is a
category error. Measure the train-split patch ratio, start with weighted
cross-entropy, and add focal only as a third pre-registered arm if the ratio is
severe.

### C3 — define a common evaluation target

A patch classifier cannot be compared against pixel segmentation metrics without
an explicit shared target. This must be decided and documented before any
three-model table is reported.

---

## 8. Order of work

1. **R1, R2, R5** — no GPU required, and R5 gates Sprint A.
2. **A statistical comparison worth the name.** Paired per-seed deltas plus a
   patch-level bootstrap confidence interval on the test split. *(Done
   2026-09-24: `src/validation/bootstrap.py` and `src/training/sweep_rf.py`;
   the F1 difference interval includes zero, the precision and recall intervals
   do not.)*
3. **A1, A2**, guarded by R5, then **A3, A4, A5**, then **P1, P2**. Note the
   reframing in §1: the ablation is diagnostic rather than a performance chase,
   and F-A is its hypothesis.
4. **C1–C3**, in parallel with the ablation runs if GPU time is the bottleneck.
5. **P3, P4** last — genuinely secondary.

One decision not in the original plan and worth pre-registering: the U-Net
dataset applies **no augmentation** — there are no flips or rotations in
`patch_dataset.MaridaSegmentationDataset`. This is a standard and inexpensive
lever against the ±0.06 seed spread, but it changes the model being compared, so
it should be decided before running rather than after seeing the number.

Sprint R alone is a demonstrable improvement: corrected documentation, honest
headline numbers, and a test suite where there was none. Sprint A is
demonstrable as a comparison table whatever its outcome.
