"""
Training entrypoint for the custom CNN patch classifier.

Trains the third model of the thesis on the same MARIDA patches and splits as
the other two, answering a patch-level question ("does this patch contain any
marine debris?") rather than a per-pixel one:

    - 6-band Sentinel-2 input, per-channel normalized on the train split
      (identical preprocessing to the U-Net)
    - BCEWithLogitsLoss with ``pos_weight`` measured on the train split
    - Adam, early stopping on validation F1

Deliberately simpler than ``train_unet``, because the problem is:

    - **No loss masking.** The U-Net masks nodata pixels out of its loss; here
      the label is one scalar per patch and every patch has one, so there is
      nothing to mask.
    - **No Dice term.** An overlap loss on a single scalar label is a category
      error. None of the U-Net loss experiment carries over.
    - **No pos_weight cap, no gradient clipping, no AMP.** Those exist in
      ``train_unet`` to survive a pos_weight of ~220. Counted in patches the
      imbalance is 2.65:1, so the weighted BCE is unremarkable and fp32 on a
      ~111k-parameter model costs seconds per epoch.

Validation F1 is the *selection* score — it early-stops and saves the
checkpoint, so the reported value is the best of many noisy attempts at the
very metric used to choose. Quote the test numbers from
``src.validation.evaluate_custom_cnn`` as the honest ones.

Outputs (mirroring the U-Net layout):
    test/data/model/custom_cnn.pt               best checkpoint (weights + norm stats)
    test/data/val/custom_cnn_history.csv        per-epoch loss / val F1 / val AP
    test/data/val/custom_cnn_val_metrics.json   validation metrics (+ _report.txt)
    test/data/val/custom_cnn_val_report.pdf     one-page PDF report

Run from the repository root:
    python -m src.training.train_custom_cnn
    python -m src.training.train_custom_cnn --epochs 30 --batch-size 32
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, classification_report, f1_score
from torch.utils.data import DataLoader

from src.data.patch_dataset import (
    MaridaPatchClassificationDataset,
    compute_normalization_stats,
)
from src.models.custom_cnn import build_custom_cnn, save_checkpoint
from src.validation.evaluate_custom_cnn import collect_patch_predictions
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


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for img, label in loader:
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(img).squeeze(1)  # (N,)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        running += loss.item() * img.size(0)
    return running / len(loader.dataset)


def fit_custom_cnn(
    train_loader,
    val_loader,
    mean,
    std,
    pos_weight,
    device,
    *,
    epochs,
    lr,
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
    Train one classifier with early stopping on validation F1, saving the best
    checkpoint to ``ckpt_path``. Returns ``{best_f1, best_epoch, best_ap, history}``.

    Reusable across seeds (a multi-seed sweep calls this directly with prebuilt
    loaders, so the cached datasets are built only once), mirroring
    :func:`src.training.train_unet.fit_unet`.

    Selection is on F1 rather than IoU to match the U-Net's rule. At a fixed
    threshold ``F1 = 2*IoU/(1+IoU)``, so the two orderings are identical and the
    choice only affects which number gets printed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_custom_cnn(
        in_channels=6,
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
        y_val, y_prob = collect_patch_predictions(model, val_loader, device)
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
                model, ckpt_path, mean, std,
                extra={
                    "best_epoch": epoch,
                    "val_f1": float(val_f1),
                    "val_average_precision": float(val_ap),
                    "threshold": float(threshold),
                    "pos_weight": float(pos_weight),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Higher than the U-Net's 1e-4: ~111k parameters and a "
                             "single scalar target per patch tolerate it.")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pos-weight", type=float, default=None,
                        help="BCE pos_weight. Default: measured on the train split "
                             "(2.65). No cap is applied — unlike the U-Net's pixel-level "
                             "ratio, this one is mild enough to use as measured.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold used for the validation F1 that drives "
                             "early stopping. Kept at 0.5 for comparability.")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early-stop after this many epochs without val-F1 improvement.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VAL_OUTPUT_DIR, exist_ok=True)

    train_split = os.path.join(SPLITS_DIR, "train_X.txt")
    val_split = os.path.join(SPLITS_DIR, "val_X.txt")

    print("Computing normalization stats on train split...")
    mean, std = compute_normalization_stats(PATCHES_ROOT, train_split)
    print(f"  per-channel mean: {np.round(mean, 2)}")
    print(f"  per-channel std : {np.round(std, 2)}")

    print("Loading datasets (cached in RAM)...")
    train_ds = MaridaPatchClassificationDataset(PATCHES_ROOT, train_split, mean=mean, std=std)
    val_ds = MaridaPatchClassificationDataset(PATCHES_ROOT, val_split, mean=mean, std=std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    pos_weight = args.pos_weight if args.pos_weight is not None else train_ds.compute_pos_weight()
    train_labels = train_ds.patch_labels()
    print(
        f"Train patches: {len(train_ds)} ({int(train_labels.sum())} with debris) | "
        f"Val patches: {len(val_ds)} | pos_weight: {pos_weight:.2f}"
    )

    n_params = sum(
        p.numel() for p in build_custom_cnn(
            base_channels=args.base_channels, hidden_dim=args.hidden_dim
        ).parameters()
    )
    print(f"Custom CNN parameters: {n_params:,}")

    ckpt_path = os.path.join(MODEL_DIR, "custom_cnn.pt")
    result = fit_custom_cnn(
        train_loader, val_loader, mean, std, pos_weight, device,
        epochs=args.epochs, lr=args.lr, base_channels=args.base_channels,
        hidden_dim=args.hidden_dim, dropout=args.dropout, patience=args.patience,
        ckpt_path=ckpt_path, threshold=args.threshold, seed=args.seed,
    )
    best_epoch, history = result["best_epoch"], result["history"]
    print(
        f"\nBest val F1 {result['best_f1']:.4f} (AP {result['best_ap']:.4f}) "
        f"at epoch {best_epoch}. Checkpoint: {ckpt_path}"
    )

    # Persist the training history so the loss / val-F1 curve is reproducible.
    history_path = os.path.join(VAL_OUTPUT_DIR, "custom_cnn_history.csv")
    with open(history_path, "w") as f:
        f.write("epoch,train_loss,val_f1,val_average_precision\n")
        for e, l, v, a in history:
            f.write(f"{e},{l:.6f},{v:.6f},{a:.6f}\n")
    print(f"Training history saved to: {history_path}")

    # Final validation report from the best checkpoint.
    from src.models.custom_cnn import load_checkpoint
    model, mean, std = load_checkpoint(ckpt_path, device=device)
    y_val, y_prob = collect_patch_predictions(model, val_loader, device)
    y_pred = (y_prob >= args.threshold).astype(np.uint8)

    report = classification_report(
        y_val, y_pred, target_names=["Clean", "Has Debris"], digits=4,
        labels=[0, 1], zero_division=0,
    )
    print(report)
    metrics = compute_binary_metrics(y_val, y_pred)
    metrics["average_precision"] = float(average_precision_score(y_val, y_prob))

    metrics_prefix = os.path.join(VAL_OUTPUT_DIR, "custom_cnn_val")
    save_metrics(
        metrics_prefix, metrics, report_text=report,
        extra={
            "model": "custom_cnn",
            "split": "val",
            "unit": "patch",
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
        title="Custom CNN (patch classifier) — Validation Set",
        metrics=metrics,
        confusion=compute_confusion_matrix(y_val, y_pred),
        report_text=report,
        meta={
            "model": f"Custom CNN (6-band, base_channels={args.base_channels})",
            "split": "validation (SELECTION scores, not generalisation)",
            "unit": "patch (NOT comparable to the per-pixel RF / U-Net tables)",
            "best epoch": best_epoch,
            "samples": f"{len(y_val):,} patches",
            "patches with debris": f"{int(y_val.sum()):,}",
            "clean patches": f"{int((y_val == 0).sum()):,}",
            "threshold": f"{args.threshold:g}",
        },
    )
    print(f"Validation PDF report saved to: {pdf_path}")


if __name__ == "__main__":
    main()
