"""
Training entrypoint for the custom CNN centre-pixel classifier (REVISED_PLAN.md step 3).

Trains model 3 of the thesis: a small CNN that sees a W x W window and predicts
the class of its **centre pixel**. Because the prediction is per-pixel, it is
scored on exactly the same labelled pixels as the Random Forest and the U-Net,
which is what makes the three-model table mean anything. W is the experimental
variable: with RF at 1 px and U-Net at 256 px, W = 16 and W = 32 fill in a
**context curve** of debris F1 against receptive field.

Recipe, pre-registered (REVISED_PLAN.md sections 2 and 4):
    - bands: RF_BANDS (7, including B06) so the spectra match the RF
    - augmentation: random flips + k*90 rotations, train split only
    - masked-free BCEWithLogitsLoss with pos_weight = min(raw, 20) — the same
      cap and the same reasoning as the U-Net (raw ratio is ~220:1)
    - Adam, early stopping on validation debris F1, threshold 0.5
    - selection on validation only; test is never used for any choice

Why there is no Dice term and no loss masking: the target is one scalar per
window, so there is nothing to mask, and an overlap loss on a scalar label is a
category error. The U-Net loss experiment does not carry over.

Validation F1 is the *selection* score — it early-stops and saves the
checkpoint, so the reported value is the best of many noisy attempts at the very
metric used to choose. Quote the test numbers from
``src.validation.evaluate_cnn`` as the honest ones.

Outputs:
    test/data/model/cnn_w{W}.pt              best checkpoint (weights + norm stats + bands)
    test/data/val/cnn_w{W}_history.csv       per-epoch loss / val F1 / val AP
    test/data/val/cnn_w{W}_val_metrics.json  validation metrics (+ _report.txt)
    test/data/val/cnn_w{W}_val_report.pdf    one-page PDF report

Run from the repository root:
    python -m src.training.train_cnn --window 16
    python -m src.training.train_cnn --window 32 --epochs 40
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, classification_report, f1_score
from torch.utils.data import DataLoader

from src.data.dataset_loader import DEFAULT_BANDS, MARIDA_BAND_ORDER, RF_BANDS
from src.data.window_dataset import MaridaWindowDataset, compute_normalization_stats
from src.models.custom_cnn import build_window_cnn, save_checkpoint
from src.validation.evaluate_cnn import collect_window_predictions
from src.validation.metrics import (
    compute_binary_metrics,
    compute_confusion_matrix,
    save_metrics,
)
from src.validation.report import generate_pdf_report

PATCHES_ROOT = "data/raw/patches"
SPLITS_DIR = "data/raw/splits"
MODEL_DIR = "test/data/model"
VAL_OUTPUT_DIR = "test/data/val"

BAND_SETS = {"default": DEFAULT_BANDS, "rf": RF_BANDS, "all": MARIDA_BAND_ORDER}


def train_one_epoch(model, loader, criterion, optimizer, device, max_norm=1.0):
    model.train()
    running, n = 0.0, 0
    for win, label in loader:
        win = win.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(win).squeeze(1), label)
        loss.backward()
        # Same guard as train_unet: a large pos_weight on a rare class can spike
        # the gradient norm. Cheap insurance, and it keeps the two deep models'
        # optimisation as similar as possible.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        running += loss.item() * win.size(0)
        n += win.size(0)
    return running / max(n, 1)


def fit_window_cnn(
    train_loader,
    val_loader,
    mean,
    std,
    bands,
    pos_weight,
    device,
    *,
    epochs,
    lr,
    window,
    base_channels,
    hidden_dim,
    dropout,
    patience,
    ckpt_path,
    threshold=0.5,
    seed=42,
    verbose=True,
) -> dict:
    """
    Train one window CNN with early stopping on validation F1, saving the best
    checkpoint to ``ckpt_path``. Returns ``{best_f1, best_epoch, best_ap, history}``.

    Reusable across seeds (``sweep_cnn`` calls this with prebuilt loaders so the
    cached patches are built only once), mirroring
    :func:`src.training.train_unet.fit_unet`.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_window_cnn(
        in_channels=len(bands),
        window_size=window,
        n_classes=1,
        base_channels=base_channels,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1, best_epoch, best_ap, no_improve = -1.0, -1, float("nan"), 0
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        y_val, y_prob = collect_window_predictions(model, val_loader, device)
        y_pred = (y_prob >= threshold).astype(np.uint8)
        val_f1 = f1_score(y_val, y_pred, zero_division=0)
        val_ap = average_precision_score(y_val, y_prob)
        history.append((epoch, loss, float(val_f1), float(val_ap)))
        if verbose:
            print(
                f"Epoch {epoch:3d}/{epochs} | loss {loss:.4f} | "
                f"val F1 {val_f1:.4f} | val AP {val_ap:.4f}",
                end="",
            )

        if val_f1 > best_f1:
            best_f1, best_epoch, best_ap, no_improve = val_f1, epoch, float(val_ap), 0
            save_checkpoint(
                model, ckpt_path, mean, std, bands=bands,
                extra={
                    "best_epoch": epoch,
                    "val_f1": float(val_f1),
                    "val_average_precision": float(val_ap),
                    "threshold": float(threshold),
                    "pos_weight": float(pos_weight),
                    "window": int(window),
                    "seed": seed,
                },
            )
            if verbose:
                print("  <- best (saved)")
        else:
            no_improve += 1
            if verbose:
                print(f"  (no improve {no_improve}/{patience})")
            if no_improve >= patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(best F1 {best_f1:.4f} @ epoch {best_epoch})."
                    )
                break

    return {
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "best_ap": best_ap,
        "history": history,
    }


def add_common_args(parser):
    """Flags shared by train_cnn and sweep_cnn, so the two cannot drift apart."""
    parser.add_argument("--window", type=int, default=16,
                        help="Window side W. Pre-registered sizes are 16 and 32.")
    parser.add_argument("--bands", choices=sorted(BAND_SETS), default="rf",
                        help="Band set. 'rf' (7, with B06) matches the Random Forest "
                             "and is the pre-registered choice.")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable train-split flips/rotations (they are on by default, "
                             "as pre-registered).")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-pos-weight", type=float, default=20.0,
                        help="Cap on the BCE pos_weight; raw is ~220:1. Same cap as the U-Net.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers. >0 is a large win here (windows are cut "
                             "on demand); the cached patches are shared copy-on-write.")
    return parser


def build_loaders(args, device):
    """Shared dataset/loader construction for train_cnn and sweep_cnn."""
    bands = BAND_SETS[args.bands]
    train_split = os.path.join(SPLITS_DIR, "train_X.txt")
    val_split = os.path.join(SPLITS_DIR, "val_X.txt")

    print(f"Window {args.window} | bands ({len(bands)}): {bands} | "
          f"augment: {not args.no_augment}")
    print("Computing normalization stats on train split...")
    mean, std = compute_normalization_stats(PATCHES_ROOT, train_split, bands=bands)
    print(f"  per-channel mean: {np.round(mean, 3)}")
    print(f"  per-channel std : {np.round(std, 3)}")

    print("Caching patches (windows are cut on demand)...")
    train_ds = MaridaWindowDataset(
        PATCHES_ROOT, train_split, window=args.window, bands=bands,
        mean=mean, std=std, augment=not args.no_augment,
    )
    val_ds = MaridaWindowDataset(
        PATCHES_ROOT, val_split, window=args.window, bands=bands,
        mean=mean, std=std, augment=False,
    )
    common = dict(
        batch_size=args.batch_size, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0, pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)

    raw_pw = train_ds.compute_pos_weight(cap=None)
    pos_weight = min(raw_pw, args.max_pos_weight)
    print(f"Train windows: {len(train_ds):,} ({int(train_ds.targets.sum()):,} debris) | "
          f"Val windows: {len(val_ds):,}")
    print(f"pos_weight: {pos_weight:.1f} (raw {raw_pw:.1f}, capped at {args.max_pos_weight:.0f})")
    return bands, mean, std, train_ds, val_ds, train_loader, val_loader, pos_weight


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VAL_OUTPUT_DIR, exist_ok=True)

    (bands, mean, std, _train_ds, val_ds,
     train_loader, val_loader, pos_weight) = build_loaders(args, device)

    tag = f"cnn_w{args.window}"
    n_params = sum(
        p.numel() for p in build_window_cnn(
            in_channels=len(bands), window_size=args.window,
            base_channels=args.base_channels, hidden_dim=args.hidden_dim,
        ).parameters()
    )
    print(f"WindowCNN parameters: {n_params:,}")

    ckpt_path = os.path.join(MODEL_DIR, f"{tag}.pt")
    result = fit_window_cnn(
        train_loader, val_loader, mean, std, bands, pos_weight, device,
        epochs=args.epochs, lr=args.lr, window=args.window,
        base_channels=args.base_channels, hidden_dim=args.hidden_dim,
        dropout=args.dropout, patience=args.patience, ckpt_path=ckpt_path,
        threshold=args.threshold, seed=args.seed,
    )
    best_epoch, history = result["best_epoch"], result["history"]
    print(f"\nBest val F1 {result['best_f1']:.4f} (AP {result['best_ap']:.4f}) "
          f"at epoch {best_epoch}. Checkpoint: {ckpt_path}")

    history_path = os.path.join(VAL_OUTPUT_DIR, f"{tag}_history.csv")
    with open(history_path, "w") as f:
        f.write("epoch,train_loss,val_f1,val_average_precision\n")
        for e, l, v, a in history:
            f.write(f"{e},{l:.6f},{v:.6f},{a:.6f}\n")
    print(f"Training history saved to: {history_path}")

    # Final validation report from the best checkpoint.
    from src.models.custom_cnn import load_checkpoint
    model, mean, std = load_checkpoint(ckpt_path, device=device)
    y_val, y_prob = collect_window_predictions(model, val_loader, device)
    y_pred = (y_prob >= args.threshold).astype(np.uint8)

    report = classification_report(
        y_val, y_pred, target_names=["Non-Debris", "Debris"], digits=4,
        labels=[0, 1], zero_division=0,
    )
    print(report)
    metrics = compute_binary_metrics(y_val, y_pred)
    metrics["average_precision"] = float(average_precision_score(y_val, y_prob))

    metrics_prefix = os.path.join(VAL_OUTPUT_DIR, f"{tag}_val")
    save_metrics(
        metrics_prefix, metrics, report_text=report,
        extra={
            "model": "custom_cnn_window",
            "split": "val",
            "unit": "pixel",
            "window": int(args.window),
            "bands": bands,
            "augment": not args.no_augment,
            "note": "selection scores, not generalisation estimates — "
                    "early stopping and checkpointing both used this split",
            "best_epoch": best_epoch,
            "threshold": float(args.threshold),
            "n_samples": int(len(y_val)),
            "n_debris": int(y_val.sum()),
            "n_non_debris": int((y_val == 0).sum()),
        },
    )
    print(f"Validation metrics saved to: {metrics_prefix}_metrics.json")

    pdf_path = f"{metrics_prefix}_report.pdf"
    generate_pdf_report(
        pdf_path,
        title=f"Custom CNN (window {args.window}) — Validation Set",
        metrics=metrics,
        confusion=compute_confusion_matrix(y_val, y_pred),
        report_text=report,
        meta={
            "model": f"Custom CNN, {args.window}x{args.window} window, {len(bands)}-band",
            "split": "validation (SELECTION scores, not generalisation)",
            "unit": "labelled pixel",
            "best epoch": best_epoch,
            "samples": f"{len(y_val):,} pixels",
            "debris pixels": f"{int(y_val.sum()):,}",
            "threshold": f"{args.threshold:g}",
        },
    )
    print(f"Validation PDF report saved to: {pdf_path}")


if __name__ == "__main__":
    main()
