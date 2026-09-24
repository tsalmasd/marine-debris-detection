# Plan — loss ablation, reporting repair, and model 3

**Target audience: a fresh Claude Code CLI session with no prior conversation
context.** Everything needed is in this file. Read it end to end before starting.

> **Status as of 2026-09-24.** R3 is **resolved as unnecessary** (the premise was
> false — see the ticket) and R4 is **done** (README now carries the five-seed
> mean±std; thesis document updated to match). R1, R2 and R5 remain open, and
> Sprints A, P and C are untouched. Numbers below that predate this date are
> superseded by the table in §1.

**Prerequisite this session must have that the authoring session did not:** the
`marine-debris` conda env and a GPU. Nothing below can be validated without them.

```bash
conda env create -f environment.yml   # if not already present
conda activate marine-debris
```

---

## 1. Where the project stands

Master's thesis: detecting marine plastic in Sentinel-2 imagery on the MARIDA
benchmark, comparing three models.

| Model | Status |
|---|---|
| 1. Random Forest | done — test F1 0.81 / IoU 0.69 (single seeded fit) |
| 2. U-Net (from scratch, 6-band) | done — test F1 **0.80 ± 0.06** / IoU **0.67 ± 0.08** over 5 seeds (per-seed F1 0.71–0.87) |
| 3. Custom CNN | **does not exist** — `src/models/custom_cnn.py` is 14 lines of docstring |

**The two models finish level.** The gap on F1 and IoU is smaller than the seed
spread; what separates them is an operating point (U-Net recall 0.91 ± 0.02 vs
0.82, bought at precision 0.71 ± 0.09 vs 0.81), not accuracy. Stronger still: the
RF operating point sits *above* the promoted U-Net's entire precision-recall
frontier — at the RF's own recall (0.824) the U-Net reaches 0.777 precision vs
0.805, and at its oracle-best threshold it reaches F1 0.806 vs 0.815 (AP 0.8149).
There is no threshold at which that U-Net dominates the baseline. This reframes
the whole plan: the ablation below is no longer a performance chase but the
candidate *explanation* for the U-Net's precision instability (F-A).

The U-Net's objective is masked `BCEWithLogitsLoss(pos_weight=min(raw, 20))` plus
a soft-Dice term, summed unweighted (`src/training/train_unet.py:51-77`).

The originating question was: *"why not change the loss and use class-imbalance
losses (Focal / Tversky / Lovász)?"* A seven-seat multi-model review panel
(4 returned: Claude Opus, Claude Fable, OpenAI codex, Z-AI GLM) converged on:
**run it as a pre-registered ablation, expect a null, and report the null.** The
ablation's value is as thesis evidence, not as a performance chase.

---

## 2. Verified findings that motivate this plan

Each was derived or read directly from source. Re-derive anything you doubt.

### F-A. The Dice term degenerates on debris-free patches

`soft_dice_loss` (`train_unet.py:58-72`) reduces over `dim=(1,2)` — **per sample**
— then `.mean()`s. On a patch with no debris, `tgt.sum()==0`, so with `eps=1.0`:

```
loss = S / (S + 1)        where S = total predicted debris probability in the patch
∂loss/∂p = 1 / (S+1)^2                 (probability gradient)
∂loss/∂z = p(1-p) / (S+1)^2            (logit gradient)
```

Verified numerically: gradient is **1.000** at S=0, **0.0278** at S=5, **0.000384**
at S=50. So it is *not* linear suppression — it collapses quadratically. As
hard-negative mining this is the worst possible shape: maximum corrective push on
empty patches that are already clean, essentially none on empty patches the model
floods with false positives.

Realistic batch (8 patches: one with 100 debris px / 90 caught, seven empty with
S=5 each):

```
per-sample Dice, .mean()  = 0.7357     <- what the code does now
batch-aggregated Dice     = 0.1991

loss-value share from the 7 patches with NO debris : 99.1%
loss-value share from the 1 patch that HAS debris  :  0.9%
```

The term whose docstring claims it "optimizes overlap directly"
(`train_unet.py:62-65`) is 99% driven by patches with no overlap to optimize.
**That docstring sentence is false and must be fixed** (task R1).

Two related corrections to earlier characterisations — do not repeat them:

- It does **not** "oppose `pos_weight`" within a patch. `pos_weight` scales only
  `target==1` terms (`train_unet.py:133-136`); an empty patch has none, so
  `pos_weight` is inert there. The real problem is *cross-patch*: identical
  negative pixels get different gradients depending on whether their patch
  happens to contain debris.
- The BCE term is **batch**-normalised (`train_unet.py:54`) while Dice is
  **per-sample**-then-meaned (`:72`), and they are summed with no λ (`:77`). Two
  differently-normalised terms added as equals — an unmeasured design choice.

### F-B. Annotation coverage — state this correctly

`patch_dataset.py:147-148`: `target = (label == 1)`, `valid = (label > 0)`. Class 0
is nodata/unlabelled and is excluded from **both loss and metrics**.

```
val split: 328 patches x 256 x 256 = 21,495,808 pixels
scored (valid)                     =    213,102   -> 0.99%
excluded                           = 21,282,706   -> 99.01%
```

**The defensible claim is: 197:1 imbalance among annotated pixels, at ~0.99%
annotation coverage.** Do **not** write "the true scene ratio is ~20,000:1" — that
presumes every excluded pixel is non-debris, and they are *unlabelled*, not
verified negatives. An earlier draft of this analysis made exactly that error.

Consequence for the thesis: every loss arm optimises the 197:1 *annotated*
imbalance. No loss can claim to address scene-level imbalance, and scene-level
false-positive behaviour is unmeasured. This is a limitation paragraph, not an
experiment.

### F-C. `segmentation-models-pytorch==0.3.4` is already a dependency and unused

`environment.yml` pins it; `grep -rn "smp\.\|segmentation_models" --include=*.py`
returns nothing (the U-Net is hand-written in `src/models/unet.py`). `smp.losses`
ships `FocalLoss`, `TverskyLoss`, `LovaszLoss`, `DiceLoss`, `JaccardLoss`.
`torchmetrics=1.4.*` is also pinned and unused (provides `AveragePrecision`).

**Implement no loss from scratch.** Everything needed is installed.

### F-D. The reported val F1 is selection-biased

`train_unet.py:144-155` early-stops on val F1 and checkpoints on improvement;
`:239-248` reloads that checkpoint and re-evaluates on the *same* val loader to
produce the PDF. So val F1 0.9032 is a max over ~34+ noisy evaluations of the
selection metric. **The test number is the honest one.** Never headline the val
figure.

### F-E. The multi-seed instrument already exists and is unused for the headline

`src/training/sweep_unet.py` trains N seeds, selects by val F1 (never test,
`:97-99`), reports test **mean±std** (`:104-117`) — then deletes the non-promoted
checkpoints (`:138-142`). Its own docstring says single-run test metrics are
high-variance because the test split has ≈381 debris pixels (`:4-5`). Yet
`README.md:196-199` still reports single-point numbers.

### F-F. Focal has less to bite on here than usual

MARIDA negatives are classes 2..15. They include genuinely hard negatives
(sargassum, ships, foam, wakes, sediment) **and** easy water (`README.md:92-110`:
7 Marine Water, 10 Turbid Water, 11 Shallow Water, 15 Mixed Water). So easy
negatives are *not* absent — but the pool is enriched for hard classes relative to
a full scene. Expect Focal's easy-negative down-weighting to buy little. Predict
this in advance and report it.

---

## 3. Non-goals — do not do these

Unanimous across the panel. Each of these burns thesis time for no defensible gain:

- A loss zoo: Lovász, asymmetric losses, unified/class-balanced focal, boundary losses.
- Grid-searching γ, α, β, or the BCE/Dice coefficient λ. Fix them in advance.
- Choosing Tversky's asymmetry from test precision/recall you have already seen.
- Running the ablation more than once per configuration set, or tuning anything
  on the test split.
- Migrating the hand-written U-Net to `smp` encoders — that changes the *model*,
  not the loss, and breaks the three-model comparison.
- Treating unlabelled pixels as negatives, or claiming any loss addresses
  scene-level imbalance.
- Chasing a number that beats 0.88. The comparison table is the contribution.

---

## 4. Sprint R — repairs (no GPU, no data; do first, ship same day)

Each ticket is atomic and independently committable.

### R1 — correct the false Dice docstring
**File:** `src/training/train_unet.py:58-72`
The docstring claims Dice "cannot produce exploding gradients" (a bounded loss
value does not bound its gradients) and "optimizes overlap directly" (false on
debris-free patches, which contribute 99.1% of the term). Rewrite it to describe
what the code does, including the `S/(S+1)` degeneration.
**Validation:** `python -m src.training.train_unet --help` still works; docstring
matches the derivation in F-A.

### R2 — stop headlining the selection-biased val F1
**Files:** `README.md`, `src/training/train_unet.py:264-279` (PDF meta)
Add to the val PDF meta and the README that val metrics are model-selection
scores, not generalisation estimates (F-D). Contextualise `accuracy: 0.9990`
against the all-negative baseline of 0.9950, or drop the accuracy row.
**Validation:** regenerate the val PDF; the caveat is present.

### ~~R3 — align the evaluation precision path~~ — RESOLVED 2026-09-24, NO CHANGE NEEDED

**Do not do this ticket. Its premise was tested and is false.**

The concern was that `use_amp = device.startswith("cuda")`
(`evaluate_unet.py:52`) autocasts on any CUDA device, so evaluation would run
fp16 while training defaults to fp32, and that fp16 quantisation near sigmoid
saturation would move the reported numbers.

Measured directly on `test/data/model/unet_baseline.pt`, running the test split
twice under each path: fp16 autocast and fp32 produce **byte-identical metrics**
— P 0.7134, R 0.8950, F1 0.7939, IoU 0.6583, TP 341, FP 137 in all four runs.
Evaluation is bit-stable and reproducible; a threshold chosen in `threshold.py`
*will* reproduce in `evaluate_unet`. No re-run of any reported figure is needed
on these grounds.

The docstring in `threshold.py:collect_flat_probabilities` still warns about this
and still points at R3. That warning is now wrong and should be corrected when
that file is next touched (it is not otherwise load-bearing).

### ~~R4 — replace README single-point results with the sweep's mean±std~~ — DONE 2026-09-24

Record, as the ticket asked:

- **Seed set:** 0–4, `python -m src.training.sweep_unet` defaults, results in
  `test/data/outputs/unet_seed_sweep.json`.
- **Promoted seed:** 4, selected on max validation F1 (0.9081); its test F1 is
  0.7939 — close to the five-seed mean, not the best of the five.
- **Threshold:** 0.5 throughout (kept for MARIDA comparability; P4 remains open).
- **Commits:** `97e5d55` (results table), `dcda58e` (RQ2 conclusion),
  `3377bf6` (module listing).

The thesis document `../Marine_Debris_Detection_Thesis.docx` was updated in the
same pass, so README and thesis now carry identical numbers. Its §8.2 also
records the precision-recall frontier result noted in §1 above.

One further defect found and fixed while doing this: the operating point plotted
in `fig_pr_curve.png` was read from a superseded metrics JSON and sat *off* its
own curve. `src/validation/figures.py` must run **after** `evaluate_unet`, never
before. Figures regenerated.

### R5 — synthetic loss unit tests *(no test suite exists yet)*
**File:** new `tests/test_losses.py`

> **Naming collision — read this first.** A `test/` directory (singular) already
> exists and is **not** a test suite: it holds run artifacts (`test/data/model`,
> `test/data/val`, `test/data/outputs`) and `test/data/` is gitignored. Put the
> pytest suite in `tests/` (plural) and do not merge the two, or `pytest` will
> start walking trained checkpoints and prediction rasters. If that ambiguity is
> unacceptable, rename the artifact directory instead — but that touches every
> path constant in `src/training/` and `src/validation/`, so it is its own ticket.
The authoring session could not run these (no numpy/sklearn) and deliberately did
not ship unverified test code. Write and run them here. Cover:
1. invalid (`valid==0`) pixels cannot change either loss term;
2. an empty-target patch's forward and backward passes are finite;
3. batch-aggregated Dice ≡ per-sample Dice when batch size is 1;
4. a mixed positive/empty batch matches the hand-derived values in F-A
   (0.7357 per-sample, 0.1991 batch-aggregated);
5. `sweep_thresholds` finds a known max-IoU cut on a hand-built array.
**Validation:** `pytest tests/ -v` green. This is the prerequisite for trusting
any ablation number.

---

## 5. Sprint A — the loss ablation

### A1 — add a `loss_fn` seam to `fit_unet`
**File:** `src/training/train_unet.py:105-167`
`fit_unet` hardcodes the criterion at `:133-136`. Add a `loss_fn` parameter
defaulting to current behaviour so `sweep_unet.py` is unaffected. ~20 lines.
**Validation:** R5 tests still green; a 1-epoch smoke run reproduces the existing
loss curve with the default.

### A2 — implement `batch_dice_loss`
**File:** `src/training/train_unet.py`
Sum intersection and denominator across all valid pixels in the minibatch *before*
taking the ratio, instead of per-sample-then-mean. One-line conceptual change.
Note the caveats: an all-empty batch still reduces to `S/(S+1)`, and batch size 1
is identical to the current form — so **log the fraction of positive-free
minibatches** during training.
**Validation:** R5 test 3 and 4.

### A3 — define the four arms
Fix γ=2, Dice ε=1, and the BCE/Dice coefficient at 1 **in advance**. Identical
splits, augmentation, optimizer, schedule, stopping rule, and batch sampler across
arms — only the objective varies.

| Arm | Objective | Isolates |
|---|---|---|
| **L0** | `WBCE(pos_weight=min(raw,20))` + per-sample Dice | control (current) |
| **L1** | same BCE + **batch-aggregated** Dice | the F-A defect |
| **L2** | `BCE(pos_weight=1)` + batch Dice | whether the never-measured cap of 20 does anything |
| **L3** | focal-modulated BCE + batch Dice | the original question |

For L3, multiply the unreduced BCE by `(1-p_t)^2`, keep the same positive cap, and
mask **before** reduction. `smp.losses.FocalLoss` is available (F-C) but check its
reduction and masking semantics against the `valid` mask before substituting it.

### A4 — run the ablation
Seeds: **match whatever seed count the L0 baseline used** so the comparison is
paired (`sweep_unet.py` defaults to 5; 3 is the acceptable floor). Reuse existing
L0 seed results if the code path is unchanged.
**Validation:** every run's `best_epoch`, convergence status, and positive-free
batch fraction recorded; no run silently diverged.

### A5 — pre-registered decision rule *(write this down before looking at results)*
**Primary metric: debris IoU** — not F1. At a fixed threshold
`F1 = 2*IoU/(1+IoU)`, so reporting both is one number twice.
**Secondary: Average Precision** (threshold-free, via `torchmetrics`).

> Promote an arm over L0 only if it improves paired validation IoU in the same
> direction across **every** seed **and** by ≥1 absolute percentage point on
> average. Otherwise keep L0 and report the null.

---

## 6. Sprint P — reporting

### P1 — the ablation table
Per arm: per-seed values **and** mean±std for P / R / F1 / IoU / AP on test, plus
paired per-seed deltas against L0. Selection by val, never test.

### P2 — two explanatory paragraphs
1. The F-A degeneration, the batch-Dice fix, and the empirical verdict.
2. The F-B framing: all results are on ~0.99% annotated pixels; scene-level
   false-positive rates are unmeasured.

### P3 — a full-scene sanity check
Use the existing `src/inference/export_predictions_unet.py` to render predictions
over whole patches and confirm the model does not light up open water. This is the
only available evidence about the 99% F-B cannot score. Qualitative figure.

### P4 — threshold as a secondary operating point
`src/validation/threshold.py` is now runnable. Implement `select_threshold()`
(the criterion is a domain judgment — see its docstring), select on **val**,
freeze, then apply via `evaluate_unet --threshold <v>`. Report it as a secondary
operating point; keep 0.5 for MARIDA comparability. Note in writing that val is
now triple-used (early stopping, seed promotion, threshold), so val metrics at the
tuned cut are not a generalisation estimate.

---

## 7. Sprint C — model 3 (on the critical path)

**This is a three-model thesis and model 3 does not exist. Do not let loss
micro-optimisation displace it.**

### C1 — implement the patch-level classifier
**File:** `src/models/custom_cnn.py`
The planned architecture ends `Dense(2) -> Softmax`. **Do not apply Softmax before
`CrossEntropyLoss`** — it double-applies. Use two raw logits with
`CrossEntropyLoss`, or one raw logit with `BCEWithLogitsLoss`.

### C2 — measure its own imbalance, then choose its loss
Its task is *patch-level* classification: the imbalance is "how many patches
contain any debris", measured in patches, not the pixel-level 197:1. **None of
Sprint A transfers.** Dice/Tversky/Lovász on a single patch label is a category
error. Measure the train-split patch ratio, start with weighted CE, and add focal
only as a third pre-registered arm if the ratio is severe.

### C3 — define a common evaluation target
A patch classifier cannot be compared to pixel segmentation metrics without an
explicit shared target. Decide and document it before reporting a three-model
table.

---

## 8. Suggested order

1. **R1, R2, R5** — free, no GPU, removes viva liabilities immediately.
2. **R4** — one sweep run; fixes the headline numbers.
3. **R3** — then re-run evaluation; corrected numbers become the baseline.
4. **A1, A2** (guarded by R5) → **A3, A4, A5** → **P1, P2**.
5. **C1-C3** in parallel with the ablation runs if GPU time is the bottleneck.
6. **P3, P4** last — genuinely secondary.

Sprint R alone is a demoable improvement: corrected docs, honest headline numbers,
a test suite where there was none. Sprint A is demoable as a comparison table
whatever the outcome.

---

## 9. Reference — the review record

Two multi-model panel runs produced the findings above. Full verbatim panelist
blocks and the judge's adjudication:

- `.fusion/runs/20260812-132003/LEDGER.md` — review of the imbalance diagnosis
- `.fusion/runs/20260812-150341/LEDGER.md` — this plan's originating question

`.fusion/` is excluded via `.git/info/exclude` (deliberately not `.gitignore`, to
avoid polluting the diff under review). It is local-only, is **not** committed, and
will not survive a fresh clone — copy anything worth keeping into the repo before
relying on it. This plan lives at the repository root precisely so it is visible
without knowing that history exists.
