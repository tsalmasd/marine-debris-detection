"""
Test-set evaluation for the custom CNN centre-pixel classifier (REVISED_PLAN.md step 3).

Loads the trained checkpoint, classifies the centre pixel of a window around
every labelled pixel in the test split, and reports the same binary
debris-vs-non-debris metrics as the Random Forest and the U-Net through the
shared helpers.

Unlike the superseded whole-patch classifier, these numbers **are** directly
comparable with the RF and U-Net tables: the unit is a labelled pixel, the mask
is the same ``valid`` mask, and the positive class is the same. That is the
entire point of the window formulation.

The band list and window size come from the checkpoint, never from a default, so
a model trained on 7 bands cannot silently be evaluated as if it were 6.

Run from the repository root:
    python -m src.validation.evaluate_cnn
    python -m src.validation.evaluate_cnn --ckpt test/data/model/cnn_w32.pt
"""

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, classification_report
from torch.utils.data import DataLoader

from src.data.window_dataset import MaridaWindowDataset
from src.models.custom_cnn import load_checkpoint, load_checkpoint_bands
from src.validation.metrics import (
    compute_binary_metrics,
    compute_confusion_matrix,
    save_metrics,
)
from src.validation.report import generate_pdf_report

PATCHES_ROOT = "data/raw/patches"
SPLITS_DIR = "data/raw/splits"
MODEL_DIR = "test/data/model"
TEST_OUTPUT_DIR = "test/data/outputs"


@torch.no_grad()
def collect_window_predictions(
    model,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model over a loader and return ``(y_true, y_prob)``, one entry per pixel.

    Probabilities rather than hard labels, so the caller owns the threshold and
    threshold-free scores (average precision) come from the same pass.
    """
    model.eval()
    ys, ps = [], []
    for win, label in loader:
        win = win.to(device, non_blocking=True)
        prob = model.predict_proba(win)
        ys.append(label.numpy().astype(np.uint8))
        ps.append(prob.float().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def build_test_loader(ckpt_path: str, batch_size: int, num_workers: int, device: str):
    """Rebuild the exact dataset a checkpoint was trained for, and load it."""
    model, mean, std = load_checkpoint(ckpt_path, device=device)
    bands = load_checkpoint_bands(ckpt_path)
    if bands is None:
        raise ValueError(
            f"{ckpt_path} has no 'bands' key, so the band set it was trained on is "
            f"unknown. Retrain with src.training.train_cnn, which records it."
        )
    test_ds = MaridaWindowDataset(
        PATCHES_ROOT,
        os.path.join(SPLITS_DIR, "test_X.txt"),
        window=model.window_size,
        bands=bands,
        mean=mean,
        std=std,
        augment=False,  # never augment an evaluation split
    )
    loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True,
    )
    return model, test_ds, loader, bands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=os.path.join(MODEL_DIR, "cnn_w16.pt"),
                        help="Checkpoint to evaluate. Window size and bands are read from it.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold. Default 0.5 for comparability with the "
                             "other two models. Pass a value chosen on VALIDATION only.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tag", default=None,
                        help="Output filename tag (default: derived from the checkpoint name).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | checkpoint: {args.ckpt} | threshold: {args.threshold}")

    model, test_ds, test_loader, bands = build_test_loader(
        args.ckpt, args.batch_size, args.num_workers, device
    )
    tag = args.tag or os.path.splitext(os.path.basename(args.ckpt))[0]
    print(f"Window: {model.window_size} | bands ({len(bands)}): {bands}")
    print(f"Test windows (= labelled pixels): {len(test_ds):,}")

    print("Running inference...")
    y_test, y_prob = collect_window_predictions(model, test_loader, device)
    y_pred = (y_prob >= args.threshold).astype(np.uint8)

    print(f"Debris: {int(y_test.sum()):,} | Non-debris: {int((y_test == 0).sum()):,}")

    report = classification_report(
        y_test, y_pred, target_names=["Non-Debris", "Debris"], digits=4,
        labels=[0, 1], zero_division=0,
    )
    print("\n=== sklearn classification report ===")
    print(report)

    metrics = compute_binary_metrics(y_test, y_pred)
    metrics["average_precision"] = float(average_precision_score(y_test, y_prob))
    print("=== test metrics (debris = positive class) ===")
    for name, value in metrics.items():
        print(f"{name:>18s}: {value:.4f}")

    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    metrics_prefix = os.path.join(TEST_OUTPUT_DIR, f"{tag}_test")
    save_metrics(
        metrics_prefix, metrics, report_text=report,
        extra={
            "model": "custom_cnn_window",
            "split": "test",
            "unit": "pixel",
            "window": int(model.window_size),
            "bands": bands,
            "threshold": float(args.threshold),
            "n_samples": int(len(y_test)),
            "n_debris": int(y_test.sum()),
            "n_non_debris": int((y_test == 0).sum()),
        },
    )
    print(f"\nTest metrics saved to: {metrics_prefix}_metrics.json")

    pdf_path = f"{metrics_prefix}_report.pdf"
    generate_pdf_report(
        pdf_path,
        title=f"Custom CNN (window {model.window_size}) — Test Set",
        metrics=metrics,
        confusion=compute_confusion_matrix(y_test, y_pred),
        report_text=report,
        meta={
            "model": f"Custom CNN, {model.window_size}x{model.window_size} window, "
                     f"{len(bands)}-band centre-pixel classifier",
            "split": "test",
            "unit": "labelled pixel (directly comparable to RF / U-Net)",
            "samples": f"{len(y_test):,} pixels",
            "debris pixels": f"{int(y_test.sum()):,}",
            "non-debris pixels": f"{int((y_test == 0).sum()):,}",
            "threshold": f"{args.threshold:g}",
        },
    )
    print(f"Test PDF report saved to: {pdf_path}")


if __name__ == "__main__":
    main()
