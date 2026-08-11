"""
Test-set evaluation for the U-Net segmentation model.

Loads the trained checkpoint, runs dense inference over every patch in the test
split, and reports the same binary debris-vs-non-debris metrics as the Random
Forest baseline (precision / recall / F1 / IoU / accuracy) so the two models are
directly comparable. Nodata pixels (MARIDA class 0) are excluded from the metrics.

Run from the repository root:
    python -m src.validation.evaluate_unet
"""

import os

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.data.patch_dataset import MaridaSegmentationDataset
from src.models.unet import load_checkpoint
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
MODEL_PATH = os.path.join(MODEL_DIR, "unet_baseline.pt")


@torch.no_grad()
def collect_flat_predictions(
    model,
    loader: DataLoader,
    device: str,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model over a loader and return flat (y_true, y_pred) over valid pixels.

    Only pixels with ``valid == 1`` (labelled, non-nodata) are kept, so the
    returned 1-D arrays are directly consumable by the shared metric helpers.
    """
    model.eval()
    ys, ps = [], []
    use_amp = device.startswith("cuda")
    for img, target, valid in loader:
        img = img.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(img)
        prob = torch.sigmoid(logits.squeeze(1).float()).cpu().numpy()  # (N,H,W)
        pred = (prob >= threshold).astype(np.uint8)
        target = target.numpy().astype(np.uint8)
        valid = valid.numpy() > 0
        for b in range(pred.shape[0]):
            m = valid[b]
            ys.append(target[b][m])
            ps.append(pred[b][m])
    return np.concatenate(ys), np.concatenate(ps)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading trained U-Net checkpoint...")
    model, mean, std = load_checkpoint(MODEL_PATH, device=device)

    print("Loading test data...")
    test_ds = MaridaSegmentationDataset(
        PATCHES_ROOT,
        os.path.join(SPLITS_DIR, "test_X.txt"),
        mean=mean,
        std=std,
    )
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    print("Running inference (all valid pixels)...")
    y_test, y_pred = collect_flat_predictions(model, test_loader, device)

    print(
        f"Test samples: {len(y_test)} | "
        f"Debris: {int(y_test.sum())} | "
        f"Non-debris: {int((y_test == 0).sum())}"
    )

    report = classification_report(
        y_test, y_pred, target_names=["Non-Debris", "Debris"], digits=4,
        labels=[0, 1], zero_division=0,
    )
    print("\n=== sklearn classification report ===")
    print(report)

    metrics = compute_binary_metrics(y_test, y_pred)
    print("=== validation metrics (debris = positive class) ===")
    for name, value in metrics.items():
        print(f"{name:>10s}: {value:.4f}")

    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    metrics_prefix = os.path.join(TEST_OUTPUT_DIR, "unet_test")
    save_metrics(
        metrics_prefix,
        metrics,
        report_text=report,
        extra={
            "model": "unet",
            "split": "test",
            "n_samples": int(len(y_test)),
            "n_debris": int(y_test.sum()),
            "n_non_debris": int((y_test == 0).sum()),
        },
    )
    print(f"\nTest metrics saved to: {metrics_prefix}_metrics.json")

    pdf_path = f"{metrics_prefix}_report.pdf"
    generate_pdf_report(
        pdf_path,
        title="U-Net Baseline — Test Set",
        metrics=metrics,
        confusion=compute_confusion_matrix(y_test, y_pred),
        report_text=report,
        meta={
            "model": "U-Net (6-band input, base_channels=64)",
            "split": "test",
            "samples": f"{len(y_test):,} pixels (all valid, no subsampling)",
            "debris pixels": f"{int(y_test.sum()):,}",
            "non-debris pixels": f"{int((y_test == 0).sum()):,}",
        },
    )
    print(f"Test PDF report saved to: {pdf_path}")


if __name__ == "__main__":
    main()
