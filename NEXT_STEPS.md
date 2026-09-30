# Next steps — remaining work

What is left to do on this thesis, and the reasoning behind each decision. Some
of these look like obvious wins until you work through the numbers, and two of
them turned out to be unnecessary once measured — so the derivations are kept
here rather than only in someone's head.

> **Status, 2026-09-30.** Reporting has been corrected and the statistical
> comparison is done. Still open: the docstring and reporting fix-ups, the loss
> unit tests, the loss experiment, and the third model. Any numbers elsewhere in
> the repo dated before 2026-09-24 are superseded by the table below.

Everything here needs the conda environment; the loss experiment and the third
model also need a GPU.

```bash
conda env create -f environment.yml   # if not already present
conda activate marine-debris
```

---

## Where things stand

| Model | Status |
|---|---|
| 1. Random Forest | done — test F1 **0.818 ± 0.003**, IoU **0.692 ± 0.004** over 5 seeds |
| 2. U-Net (from scratch, 6-band) | done — test F1 **0.798 ± 0.060**, IoU **0.668 ± 0.083** over 5 seeds (per-seed F1 0.71–0.87) |
| 3. Custom CNN | not implemented — `src/models/custom_cnn.py` is still a docstring |

**The two finished models are level.** The gap in F1 and IoU is smaller than the
spread between seeds, so it is not a real difference. What is real is the trade:
the U-Net finds more debris (recall 0.91 against 0.82) and raises more false
alarms doing it (precision 0.71 against 0.81).

It goes further than a tie. Sweeping the U-Net's threshold across its whole range
traces every precision-recall pair it can reach, and the Random Forest sits
*above* that curve. At the forest's own recall the U-Net manages 0.777 precision
against 0.805, and even at its best possible threshold it reaches F1 0.806
against 0.815. There is no setting at which this U-Net beats the baseline.

That changes what the loss experiment is for. It will not make the U-Net win. It
is the best available explanation for *why the U-Net's precision is so unstable*.

The U-Net currently optimises masked `BCEWithLogitsLoss(pos_weight=min(raw, 20))`
plus soft Dice, added together with no weighting
(`src/training/train_unet.py:51-77`). The question that started this was whether
an imbalance-aware loss (Focal, Tversky, Lovász) would do better. The plan is to
run it properly, expect nothing, and report that. A clean null is worth having.

---

## What we know so far

All of this came from reading or measuring the code.

### The Dice term does almost nothing useful

`soft_dice_loss` (`train_unet.py:58-72`) reduces per sample and then averages. On
a patch with no debris in it, `tgt.sum() == 0`, so with `eps=1.0` the term
collapses to:

```
loss = S / (S + 1)        S = total predicted debris probability in that patch
∂loss/∂z = p(1-p) / (S+1)^2
```

The gradient is **1.000** at S=0, **0.0278** at S=5, **0.000384** at S=50. That
is backwards: the model is pushed hardest on patches that are already clean, and
barely at all on the patches it is flooding with false positives.

For a realistic batch — 8 patches, one with 100 debris pixels of which 90 are
caught, seven empty with S=5 each:

```
per-sample Dice, averaged = 0.7357     <- what the code does now
batch-aggregated Dice     = 0.1991

share of the loss from the 7 patches with NO debris : 99.1%
share from the 1 patch that actually has debris     :  0.9%
```

So the term whose docstring claims it "optimizes overlap directly" is 99% driven
by patches with no overlap to optimise. That docstring needs correcting.

Two things that are easy to get wrong about this, worth stating so they do not
get repeated:

- It does **not** fight `pos_weight` within a patch. `pos_weight` only scales
  `target == 1` pixels (`train_unet.py:133-136`), and an empty patch has none.
  The problem is *between* patches: two identical water pixels get different
  gradients depending on whether their patch happens to contain debris.
- BCE is normalised over the batch (`:54`), Dice per sample then averaged
  (`:72`), and the two are simply added (`:77`). Two differently scaled terms
  added as equals, which nobody has ever measured.

### Only about 1% of pixels are labelled

In `patch_dataset.py:147-148`, `target = (label == 1)` and `valid = (label > 0)`.
Class 0 is unlabelled and is dropped from both the loss and the metrics.

```
val split: 328 patches x 256 x 256 = 21,495,808 pixels
labelled                           =    213,102   -> 0.99%
dropped                            = 21,282,706   -> 99.01%
```

The honest way to describe the imbalance is therefore **197:1 among labelled
pixels, at about 1% coverage**. It is tempting to say the true scene ratio is
around 20,000:1, but that assumes every dropped pixel is water, and they are
*unlabelled* rather than checked and found empty.

What this means for the thesis: every loss we try is optimising the 197:1
labelled imbalance. None can claim to address scene-level imbalance, and nothing
here measures how any model behaves over open water. That is a limitation to
write down, not an experiment to run.

### The loss functions are already installed

`environment.yml` pins `segmentation-models-pytorch==0.3.4` and nothing imports
it — the U-Net is hand-written in `src/models/unet.py`. `smp.losses` ships
`FocalLoss`, `TverskyLoss`, `LovaszLoss`, `DiceLoss` and `JaccardLoss`.
`torchmetrics` is pinned and unused too, and provides `AveragePrecision`.

Nothing needs writing from scratch.

### The validation F1 flatters itself

`train_unet.py:144-155` early-stops on validation F1 and saves on improvement,
then `:239-248` reloads that checkpoint and scores it on the *same* validation
set for the PDF. The result is the best of roughly 34 noisy attempts at the very
metric being used to choose. The test number is the honest one, and validation
figures are never quoted as generalisation anywhere in the thesis.

### The multi-seed tooling existed before it was used

`src/training/sweep_unet.py` trains N seeds, selects by validation F1 and never
by test (`:97-99`), reports mean ± std (`:104-117`), then deletes the losing
checkpoints (`:138-142`). Its own docstring warns that single-run test numbers
are noisy because the test split holds about 381 debris pixels. The README
nonetheless reported single points until this was corrected on 2026-09-24.

### Focal loss has less to work with here than usual

MARIDA's negatives are classes 2–15. Some are genuinely hard (Sargassum, ships,
foam, wakes, sediment) but plenty are easy water (Marine, Turbid, Shallow,
Mixed). Easy negatives do exist, but there are proportionally fewer than in a
full scene, and down-weighting them is Focal's whole trick. Expect it to buy very
little — and say so in advance.

---

## Not worth doing

Each of these costs real time and buys nothing defensible:

- Trying every loss available — Lovász, asymmetric, class-balanced focal,
  boundary losses. Four variants is enough.
- Grid-searching γ, α, β or the BCE/Dice weight. Fix them in advance instead.
- Choosing Tversky's asymmetry after looking at test precision and recall.
- Running a variant more than once, or tuning anything at all on the test split.
- Swapping the hand-written U-Net for an `smp` encoder. That changes the model,
  not the loss, and breaks the three-way comparison.
- Treating unlabelled pixels as water, or claiming any loss fixes scene-level
  imbalance.
- Chasing a number that beats the baseline. The comparison is the point.

---

## Fix-ups first

No GPU or data needed, and each can go in on its own.

**Correct the Dice docstring** (`src/training/train_unet.py:58-72`). It claims
Dice "cannot produce exploding gradients" — a bounded loss does not bound its
gradients — and that it "optimizes overlap directly", which is wrong on
debris-free patches. Rewrite it to describe what the code actually does,
including the `S/(S+1)` collapse.

**Label the validation numbers as selection scores** in `README.md` and the PDF
metadata at `train_unet.py:264-279`. Either put `accuracy: 0.9990` next to the
all-negative baseline of 0.9950, or drop the accuracy row entirely.

**Write unit tests for the loss terms**, in a new `tests/test_losses.py`.

> Careful with the directory name. `test/` (singular) already exists and is
> **not** a test suite — it holds run artifacts and is gitignored. Put pytest in
> `tests/` (plural), or it will start walking trained checkpoints and prediction
> rasters. Renaming the artifact directory instead is defensible, but it touches
> every path constant in `src/training/` and `src/validation/`, so treat that as
> its own job.

Cover: invalid pixels cannot change either loss term; an empty-target patch stays
finite forwards and backwards; batch-aggregated Dice equals per-sample Dice at
batch size 1; a mixed batch reproduces the numbers above (0.7357 and 0.1991); and
`sweep_thresholds` finds a known maximum-IoU cut on a hand-built array. These
need to exist before any ablation number can be trusted.

### Two items already closed

**Matching the evaluation precision path — not needed.** The worry was that
`use_amp = device.startswith("cuda")` (`evaluate_unet.py:52`) runs evaluation in
fp16 while training runs fp32, and that this would shift the reported numbers. It
was measured: fp16 and fp32 give **identical** results — P 0.7134, R 0.8950,
F1 0.7939, IoU 0.6583, TP 341, FP 137, across four runs. A threshold chosen in
`threshold.py` reproduces exactly in `evaluate_unet`.

**Reporting mean ± std instead of single runs — done 2026-09-24.** Seeds 0–4 via
`sweep_unet.py` defaults, saved in `test/data/outputs/unet_seed_sweep.json`. Seed
4 was promoted on validation F1 (0.9081); its test F1 is 0.7939, near the mean
rather than the best. Threshold kept at 0.5 for comparability with MARIDA. The
thesis was updated in the same pass, so it and the README agree.

One bug surfaced while doing this: the operating point in `fig_pr_curve.png` had
been read from a superseded metrics file and was sitting off its own curve.
**`src/validation/figures.py` must run after `evaluate_unet`, never before.**

---

## The loss experiment

**Make the loss swappable.** `fit_unet` hardcodes the criterion at
`train_unet.py:133-136`; add a `loss_fn` argument defaulting to current behaviour
so `sweep_unet.py` is unaffected. About 20 lines. A one-epoch run should
reproduce the existing loss curve with the default.

**Add a batch-aggregated Dice.** Sum intersection and denominator across the
whole minibatch *before* dividing, rather than per sample and then averaging.
Conceptually one line. Two catches: an all-empty batch still collapses to
`S/(S+1)`, and at batch size 1 it is identical to the current form — so log how
many minibatches contain no positives at all.

**The four variants.** Fix γ=2, Dice ε=1 and the BCE/Dice weight at 1 before
starting. Splits, augmentation, optimiser, schedule, stopping rule and sampler
stay identical; only the objective changes.

| Variant | Objective | What it isolates |
|---|---|---|
| Control | `WBCE(pos_weight=min(raw,20))` + per-sample Dice | what we have now |
| Batch Dice | same BCE + batch-aggregated Dice | the defect described above |
| No cap | `BCE(pos_weight=1)` + batch Dice | whether the cap of 20 ever did anything |
| Focal | focal-modulated BCE + batch Dice | the original question |

For the focal variant, multiply the unreduced BCE by `(1-p_t)^2`, keep the same
positive cap, and mask **before** reducing. `smp.losses.FocalLoss` is available,
but check how it reduces and masks against our `valid` mask before substituting
it.

**Running it.** Use the same number of seeds as the control so the comparison is
paired — five by default, three at an absolute minimum. Existing control results
can be reused if that code path has not changed. Record `best_epoch`, whether the
run converged, and the positive-free batch fraction for every run, so a silently
diverged run cannot slip through.

**Decide the rule before looking at results.** The primary metric is **debris
IoU**, not F1: at a fixed threshold `F1 = 2*IoU/(1+IoU)`, so reporting both is
the same number twice. Average precision is the threshold-free secondary.

> Promote a variant over the control only if it improves paired validation IoU in
> the same direction on **every** seed, **and** by at least one percentage point
> on average. Otherwise keep the control and report the null.

---

## Writing it up

**The ablation table.** Per variant: every seed plus mean ± std for precision,
recall, F1, IoU and average precision on test, and the paired per-seed
differences against the control. Selection by validation, never by test.

**Two paragraphs.** One on the Dice collapse, the batch-Dice fix and what
actually happened. One on annotation coverage: everything is measured on about 1%
of pixels, and scene-level false positives are not measured at all.

**A full-scene sanity check.** Use `src/inference/export_predictions_unet.py` to
render whole patches and confirm the model does not light up open water. It is
the only evidence available about the 99% that cannot be scored, and it is
qualitative — say so.

**A threshold as a second operating point.** Implement `select_threshold()` in
`src/validation/threshold.py` — the criterion is a judgement call, and the
docstring lays out the options — choose on **validation**, freeze it, then run
`evaluate_unet --threshold <v>`. Keep 0.5 as the headline for comparability with
MARIDA. Note in the write-up that validation is by then doing three jobs (early
stopping, seed promotion, threshold), so validation numbers at the tuned cut mean
even less than before.

---

## The third model

This is a three-model thesis with two models in it. Do not let loss tinkering
push this down the list.

**Write the patch classifier** in `src/models/custom_cnn.py`. The sketched
architecture ends `Dense(2) -> Softmax`. Do not put Softmax before
`CrossEntropyLoss` — it applies its own and you would be doing it twice. Use two
raw logits with `CrossEntropyLoss`, or one with `BCEWithLogitsLoss`.

**Measure its imbalance before choosing a loss.** This model works on whole
patches, so the imbalance is "how many patches contain any debris", counted in
patches, not the 197:1 pixel figure. None of the loss experiment carries over —
Dice or Tversky on a single patch label is a category error. Count the
train-split ratio first, start with weighted cross-entropy, and only add focal as
a third pre-registered variant if the ratio turns out to be severe.

**Decide what all three models are scored on.** A patch classifier cannot be
compared against per-pixel metrics without agreeing a shared target first. Settle
this and write it down before producing any three-model table. It is the real
blocker, not the network.

---

## Order of work

1. The three fix-ups above. No GPU needed, and the unit tests have to exist
   before any ablation number can be trusted.
2. ~~Proper statistics on the comparison~~ — **done 2026-09-24.** Paired per-seed
   differences plus a patch-level bootstrap, in `src/validation/bootstrap.py` and
   `src/training/sweep_rf.py`. The F1 difference interval includes zero; the
   precision and recall intervals do not.
3. The loss experiment, in the order given: swappable loss, batch Dice, then the
   four variants and the write-up. Remember it is diagnostic now, not a hunt for
   a better score.
4. The third model, alongside the ablation runs if the GPU is the bottleneck.
5. The full-scene check and the threshold selection last. Genuinely secondary.

One thing not in the original plan, worth deciding deliberately: the U-Net
dataset applies **no augmentation** at all — no flips, no rotations
(`patch_dataset.MaridaSegmentationDataset`). That is the cheapest lever available
against the ±0.06 seed spread and might shrink it considerably. But it changes
the model being compared, so decide before running it, not after seeing the
number.

The fix-ups alone are a real improvement: corrected docs, honest headline
numbers, and a test suite where there was none. The loss experiment is worth
presenting whichever way it comes out.
