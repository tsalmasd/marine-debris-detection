"""
Test-set evaluation for the custom CNN patch classifier.

Loads the trained checkpoint, scores every patch in the test split, and reports
the same binary debris-vs-non-debris metrics as the other two models
(precision / recall / F1 / IoU / MCC) through the shared metric helpers.

The unit of evaluation is a **patch**, not a pixel. These numbers are therefore
*not* comparable line-for-line with the Random Forest or U-Net test numbers,
which are per-pixel: 359 decisions here against ~3.9M there, on a different
question ("does this patch contain debris" vs "is this pixel debris"). Putting
all three in one table requires agreeing a shared target first — see the
three-model scoring note in NEXT_STEPS.md.

Run from the repository root:
    python -m src.validation.evaluate_custom_cnn
    python -m src.validation.evaluate_custom_cnn --threshold 0.42
"""

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, classification_report
from torch.utils.data import DataLoader

from src.data.patch_dataset import MaridaPatchClassificationDataset
from src.models.custom_cnn import load_checkpoint
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
MODEL_PATH = os.path.join(MODEL_DIR, "custom_cnn.pt")


@torch.no_grad()
def collect_patch_predictions(
    model,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model over a loader and return ``(y_true, y_prob)``, one entry per patch.

    Probabilities rather than hard labels, so the caller picks the threshold and
    so threshold-free scores (average precision) can be computed from the same
    pass. Both arrays are 1-D of length ``len(loader.dataset)``.

    Unlike the U-Net's collector this runs in fp32: the model is ~111k
    parameters and a whole split is a few seconds, so autocast would only add a
    precision question for no measurable gain.
    """
    model.eval()
    ys, ps = [], []
    for img, label in loader:
        img = img.to(device, non_blocking=True)
        logits = model(img)
        if model.n_classes == 1:
            prob = torch.sigmoid(logits.squeeze(1))
        else:
            prob = torch.softmax(logits, dim=1)[:, 1]
        ys.append(label.numpy().astype(np.uint8))
        ps.append(prob.float().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold for the debris class. Default 0.5 keeps previously "
             "reported numbers reproducible. Pass a value selected on the VALIDATION "
             "split; never one chosen on test.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | threshold: {args.threshold}")

    print("Loading trained custom CNN checkpoint...")
    model, mean, std = load_checkpoint(MODEL_PATH, device=device)

    print("Loading test data...")
    test_ds = MaridaPatchClassificationDataset(
        PATCHES_ROOT,
        os.path.join(SPLITS_DIR, "test_X.txt"),
        mean=mean,
        std=std,
    )
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    print("Running inference (one score per patch)...")
    y_test, y_prob = collect_patch_predictions(model, test_loader, device)
    y_pred = (y_prob >= args.threshold).astype(np.uint8)

    print(
        f"Test patches: {len(y_test)} | "
        f"With debris: {int(y_test.sum())} | "
        f"Clean: {int((y_test == 0).sum())}"
    )

    report = classification_report(
        y_test, y_pred, target_names=["Clean", "Has Debris"], digits=4,
        labels=[0, 1], zero_division=0,
    )
    print("\n=== sklearn classification report ===")
    print(report)

    metrics = compute_binary_metrics(y_test, y_pred)
    # Threshold-free, so it survives any later change to the operating point.
    metrics["average_precision"] = float(average_precision_score(y_test, y_prob))
    print("=== test metrics (patch contains debris = positive class) ===")
    for name, value in metrics.items():
        print(f"{name:>18s}: {value:.4f}")

    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    metrics_prefix = os.path.join(TEST_OUTPUT_DIR, "custom_cnn_test")
    save_metrics(
        metrics_prefix,
        metrics,
        report_text=report,
        extra={
            "model": "custom_cnn",
            "split": "test",
            "unit": "patch",
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
        title="Custom CNN (patch classifier) — Test Set",
        metrics=metrics,
        confusion=compute_confusion_matrix(y_test, y_pred),
        report_text=report,
        meta={
            "model": "Custom CNN (6-band input, patch-level)",
            "split": "test",
            "unit": "patch (NOT comparable to the per-pixel RF / U-Net tables)",
            "samples": f"{len(y_test):,} patches",
            "patches with debris": f"{int(y_test.sum()):,}",
            "clean patches": f"{int((y_test == 0).sum()):,}",
            "threshold": f"{args.threshold:g}",
        },
    )
    print(f"Test PDF report saved to: {pdf_path}")


if __name__ == "__main__":
    main()
