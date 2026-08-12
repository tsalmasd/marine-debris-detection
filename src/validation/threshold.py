"""
Decision-threshold selection for the binary debris segmentation models.

The U-Net emits a per-pixel debris *probability*; a hard label only appears once
that probability is compared against a threshold. Both evaluation entrypoints use
a hardcoded 0.5 (``evaluate_unet.collect_flat_predictions``).

Why 0.5 is the wrong cut here — note this is *not* because the classes are
imbalanced. For a calibrated posterior under symmetric 0-1 loss, 0.5 is
Bayes-optimal at any prevalence; imbalance alone does not move it. The two real
reasons are:

  1. ``pos_weight`` (BCE) and the Dice term deliberately de-calibrate the output,
     so ``sigmoid(logit)`` is not a posterior probability at all; and
  2. the reported objective is F1 / IoU, not 0-1 loss, and the F1-optimal cut is
     generally not 0.5 even for a perfectly calibrated model.

Methodology — selection must happen on the **validation** split and the chosen
value must be frozen before the test split is touched. Picking the threshold that
maximises a test metric is test-set leakage.

Note the cost this adds, and state it in the write-up: validation is *already*
load-bearing for early stopping (``train_unet.py:150``) and cross-seed checkpoint
promotion (``sweep_unet.py:97-99``). Adding threshold selection makes it a third
use. That does not contaminate test, but it does mean the validation metrics *at
the tuned threshold* are not a generalisation estimate and must never be reported
as one. Only the frozen-threshold test metrics are quotable.

Run from the repository root:
    python -m src.validation.threshold                 # sweep on val, print table
    python -m src.validation.threshold --split test    # inspect only; do NOT select here
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.validation.metrics import compute_binary_metrics

PATCHES_ROOT = "data/raw/patches"
SPLITS_DIR = "data/raw/splits"
MODEL_DIR = "test/data/model"
VAL_OUTPUT_DIR = "test/data/val"
MODEL_PATH = os.path.join(MODEL_DIR, "unet_baseline.pt")

# Candidate cuts. ``linspace`` rather than ``arange``: arange accumulates
# floating-point drift over 99 steps.
DEFAULT_THRESHOLDS = np.linspace(0.01, 0.99, 99)


@torch.no_grad()
def collect_flat_probabilities(
    model,
    loader: DataLoader,
    device: str,
    use_amp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model over a loader and return flat (y_true, y_prob) over valid pixels.

    The probability twin of ``evaluate_unet.collect_flat_predictions`` -- identical
    masking semantics, but it stops short of thresholding so one forward pass can
    be reused across an entire threshold sweep.

    ``use_amp`` defaults to **False**, deliberately. Training runs fp32 by default
    (``train_unet.py`` ``--amp`` is opt-in because mixed precision with the large
    ``pos_weight`` is numerically unstable), and fp16 quantises the sigmoid output
    near saturation -- exactly the quantity a 0.01-spaced grid is trying to
    resolve. ``evaluate_unet.collect_flat_predictions`` currently autocasts on any
    CUDA device, so a threshold chosen here will not exactly reproduce there until
    that path is aligned. See LOSS_ABLATION_PLAN.md, task R3.

    Returns:
        (y_true, y_prob) as 1-D arrays over labelled (non-nodata) pixels only.
    """
    model.eval()
    ys, ps = [], []
    for img, target, valid in loader:
        img = img.to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], enabled=use_amp):
            logits = model(img)
        prob = torch.sigmoid(logits.squeeze(1).float()).cpu().numpy()  # (N,H,W)
        target = target.numpy().astype(np.uint8)
        valid = valid.numpy() > 0
        for b in range(prob.shape[0]):
            m = valid[b]
            ys.append(target[b][m])
            ps.append(prob[b][m])
    return np.concatenate(ys), np.concatenate(ps)


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict]:
    """
    Evaluate the debris-class metrics at every candidate threshold.

    Args:
        y_true: 1-D binary ground truth over valid pixels.
        y_prob: 1-D debris probabilities, same length as ``y_true``.
        thresholds: candidate cuts; defaults to :data:`DEFAULT_THRESHOLDS`.

    Returns:
        One dict per threshold: ``{threshold, precision, recall, f1, iou,
        accuracy, tp, fp, fn, tn}``, in ascending threshold order.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.uint8)
        row = {"threshold": float(t), **compute_binary_metrics(y_true, y_pred)}
        pred_pos, true_pos = y_pred == 1, y_true == 1
        row["tp"] = int((pred_pos & true_pos).sum())
        row["fp"] = int((pred_pos & ~true_pos).sum())
        row["fn"] = int((~pred_pos & true_pos).sum())
        row["tn"] = int((~pred_pos & ~true_pos).sum())
        rows.append(row)
    return rows


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict]:
    """
    Choose the operating threshold from a validation-split sweep.

    TODO(nick): implement the selection criterion.

    :func:`sweep_thresholds` scores every candidate; all that is left is deciding
    which row wins. That is a domain judgment about the relative cost of a missed
    debris pixel versus a false alarm, and it belongs in the thesis as an argued
    choice rather than an implicit 0.5.

    Two genuinely distinct criteria:

      - **max IoU** -- the conventional, easily defended default; treats a false
        negative and a false positive as equally costly.

        Note this is the *same* choice as "max F1": at a fixed threshold
        ``F1 = 2*IoU/(1 + IoU)``, which is strictly increasing in IoU, so both
        criteria select the identical row. Pick IoU and say so; offering them as
        alternatives would be presenting one option as two.

      - **recall-constrained** -- highest precision subject to ``recall >= R``
        (e.g. 0.95). The honest choice if the downstream use is tasking cleanup
        vessels or flagging scenes for an analyst, where a missed slick costs more
        than a false alarm someone dismisses in seconds. Requires an explicit,
        defensible R.

    Add whatever parameters your criterion needs (e.g. ``min_recall``).

    Returns:
        ``(threshold, row)`` -- the chosen cut and its full metrics dict.
    """
    raise NotImplementedError(
        "select_threshold: choose the winning row from sweep_thresholds(...). "
        "See the docstring for the two criteria worth considering."
    )


def _print_sweep(rows: list[dict], every: int = 5) -> None:
    """Print the sweep as a table, plus the max-IoU row for reference."""
    print(f"{'thr':>6}{'prec':>9}{'recall':>9}{'f1':>8}{'iou':>8}"
          f"{'tp':>8}{'fp':>8}{'fn':>8}")
    for i, r in enumerate(rows):
        if i % every and r is not rows[-1]:
            continue
        print(f"{r['threshold']:>6.2f}{r['precision']:>9.4f}{r['recall']:>9.4f}"
              f"{r['f1']:>8.4f}{r['iou']:>8.4f}{r['tp']:>8d}{r['fp']:>8d}{r['fn']:>8d}")
    best = max(rows, key=lambda r: r["iou"])
    print(f"\nmax-IoU row (== max-F1 row): threshold {best['threshold']:.2f} | "
          f"P {best['precision']:.4f} R {best['recall']:.4f} "
          f"F1 {best['f1']:.4f} IoU {best['iou']:.4f}")
    print("This is a reference point, not a selection. Implement select_threshold() "
          "and argue the criterion.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="Split to sweep. Select ONLY on val; --split test is "
                             "for inspection after the threshold is frozen.")
    parser.add_argument("--checkpoint", default=MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", default=None,
                        help="Optional path to write the full sweep as JSON.")
    args = parser.parse_args()

    # Imported here so `--help` works without torch/rasterio present.
    from src.data.patch_dataset import MaridaSegmentationDataset
    from src.models.unet import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | split: {args.split}")

    model, mean, std = load_checkpoint(args.checkpoint, device=device)
    ds = MaridaSegmentationDataset(
        PATCHES_ROOT, os.path.join(SPLITS_DIR, f"{args.split}_X.txt"),
        mean=mean, std=std,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    y_true, y_prob = collect_flat_probabilities(model, loader, device)
    print(f"Valid pixels: {len(y_true):,} | debris: {int(y_true.sum()):,}")

    rows = sweep_thresholds(y_true, y_prob)
    _print_sweep(rows)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"split": args.split, "sweep": rows}, f, indent=2)
        print(f"\nSweep saved to: {args.out}")


if __name__ == "__main__":
    main()
