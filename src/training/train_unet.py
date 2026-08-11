"""
Training entrypoint for the U-Net segmentation baseline.

Trains a binary debris / non-debris U-Net on the same MARIDA patches and splits
as the Random Forest baseline, using:

    - 6-band Sentinel-2 input, per-channel normalized on the train split
    - masked BCEWithLogitsLoss (nodata pixels excluded; pos_weight counteracts
      the heavy debris class imbalance)
    - Adam + automatic mixed precision (AMP) on GPU
    - early stopping on validation F1 (debris class)

Outputs (mirroring the RF baseline layout):
    test/data/model/unet_baseline.pt        best checkpoint (weights + norm stats)
    test/data/val/unet_val_metrics.json     validation metrics (+ _report.txt)
    test/data/val/unet_val_report.pdf       one-page PDF report

Run from the repository root:
    python -m src.training.train_unet
    python -m src.training.train_unet --epochs 30 --batch-size 16
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from src.data.patch_dataset import (
    MaridaSegmentationDataset,
    compute_normalization_stats,
)
from src.models.unet import build_unet, save_checkpoint
from src.validation.evaluate_unet import collect_flat_predictions
from src.validation.metrics import (
    compute_binary_metrics,
    compute_confusion_matrix,
    save_metrics,
)
from src.validation.report import generate_pdf_report

PATCHES_ROOT = "data/raw/patches"
SPLITS_DIR = "data/raw/splits"
MODEL_DIR = "test/data/model"     # trained checkpoint
VAL_OUTPUT_DIR = "test/data/val"  # validation metrics + reports


def masked_bce(criterion, logits, target, valid):
    """Per-pixel BCE averaged over valid (labelled) pixels only."""
    per_px = criterion(logits.squeeze(1), target)  # (N, H, W)
    denom = valid.sum().clamp(min=1.0)
    return (per_px * valid).sum() / denom


def soft_dice_loss(logits, target, valid, eps=1.0):
    """
    Soft Dice loss over valid pixels only.

    Dice is bounded in [0, 1] regardless of class imbalance, so (unlike a
    heavily pos-weighted BCE) it cannot produce exploding gradients. It also
    optimizes overlap directly, which aligns with the debris-class F1/IoU we
    report. Combined with BCE it gives a stable, imbalance-robust objective.
    """
    prob = torch.sigmoid(logits.squeeze(1)) * valid  # (N, H, W)
    tgt = target * valid
    inter = (prob * tgt).sum(dim=(1, 2))
    denom = prob.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
    dice = (2 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()


def seg_loss(criterion, logits, target, valid):
    """Combined objective: masked BCE (weighted) + soft Dice."""
    return masked_bce(criterion, logits, target, valid) + soft_dice_loss(logits, target, valid)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, max_norm=1.0):
    model.train()
    running = 0.0
    for img, target, valid in loader:
        img = img.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(img)
            loss = seg_loss(criterion, logits, target, valid)

        scaler.scale(loss).backward()
        # Unscale before clipping so max_norm is applied to the true gradients.
        # Gradient clipping is essential here: the large pos_weight (rare debris
        # class) otherwise produces exploding gradients that diverge to NaN.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * img.size(0)
    return running / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--max-pos-weight", type=float, default=20.0,
                        help="Cap on the BCE pos_weight. The raw imbalance ratio (~220) "
                             "destabilizes training; Dice loss handles the rest of the imbalance.")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early-stop after this many epochs without val-F1 improvement.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision training. Off by default: with the large "
                             "pos_weight the GradScaler path is numerically unstable (NaN loss), "
                             "and fp32 fits comfortably for this dataset size.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = args.amp and device == "cuda"
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device} | AMP: {use_amp}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VAL_OUTPUT_DIR, exist_ok=True)

    train_split = os.path.join(SPLITS_DIR, "train_X.txt")
    val_split = os.path.join(SPLITS_DIR, "val_X.txt")

    print("Computing normalization stats on train split...")
    mean, std = compute_normalization_stats(PATCHES_ROOT, train_split)
    print(f"  per-channel mean: {np.round(mean, 2)}")
    print(f"  per-channel std : {np.round(std, 2)}")

    print("Loading datasets (cached in RAM)...")
    train_ds = MaridaSegmentationDataset(PATCHES_ROOT, train_split, mean=mean, std=std)
    val_ds = MaridaSegmentationDataset(PATCHES_ROOT, val_split, mean=mean, std=std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    raw_pos_weight = train_ds.compute_pos_weight()
    pos_weight = min(raw_pos_weight, args.max_pos_weight)
    print(f"Train patches: {len(train_ds)} | Val patches: {len(val_ds)} | "
          f"pos_weight: {pos_weight:.1f} (raw {raw_pos_weight:.1f}, capped at {args.max_pos_weight:.0f})")

    model = build_unet(in_channels=6, n_classes=1, base_channels=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor([pos_weight], device=device),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_path = os.path.join(MODEL_DIR, "unet_baseline.pt")
    best_f1, best_epoch, epochs_no_improve = -1.0, -1, 0

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)

        y_val, y_pred = collect_flat_predictions(model, val_loader, device)
        val_f1 = f1_score(y_val, y_pred, zero_division=0)
        print(f"Epoch {epoch:3d}/{args.epochs} | loss {loss:.4f} | val F1 {val_f1:.4f}", end="")

        if val_f1 > best_f1:
            best_f1, best_epoch, epochs_no_improve = val_f1, epoch, 0
            save_checkpoint(
                model, ckpt_path, mean, std,
                extra={"best_epoch": epoch, "val_f1": float(val_f1)},
            )
            print("  <- best (saved)")
        else:
            epochs_no_improve += 1
            print(f"  (no improve {epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (best F1 {best_f1:.4f} @ epoch {best_epoch}).")
                break

    print(f"\nBest val F1 {best_f1:.4f} at epoch {best_epoch}. Checkpoint: {ckpt_path}")

    # Final validation report from the best checkpoint.
    from src.models.unet import load_checkpoint
    model, mean, std = load_checkpoint(ckpt_path, device=device)
    y_val, y_pred = collect_flat_predictions(model, val_loader, device)

    report = classification_report(
        y_val, y_pred, target_names=["Non-Debris", "Debris"], digits=4,
        labels=[0, 1], zero_division=0,
    )
    print(report)
    metrics = compute_binary_metrics(y_val, y_pred)

    metrics_prefix = os.path.join(VAL_OUTPUT_DIR, "unet_val")
    save_metrics(
        metrics_prefix, metrics, report_text=report,
        extra={
            "model": "unet",
            "split": "val",
            "best_epoch": best_epoch,
            "n_samples": int(len(y_val)),
            "n_debris": int(y_val.sum()),
            "n_non_debris": int((y_val == 0).sum()),
        },
    )
    print(f"Validation metrics saved to: {metrics_prefix}_metrics.json")

    pdf_path = f"{metrics_prefix}_report.pdf"
    generate_pdf_report(
        pdf_path,
        title="U-Net Baseline — Validation Set",
        metrics=metrics,
        confusion=compute_confusion_matrix(y_val, y_pred),
        report_text=report,
        meta={
            "model": f"U-Net (6-band, base_channels={args.base_channels})",
            "split": "validation",
            "best epoch": best_epoch,
            "samples": f"{len(y_val):,} pixels (all valid, no subsampling)",
            "debris pixels": f"{int(y_val.sum()):,}",
            "non-debris pixels": f"{int((y_val == 0).sum()):,}",
        },
    )
    print(f"Validation PDF report saved to: {pdf_path}")


if __name__ == "__main__":
    main()
