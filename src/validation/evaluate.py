"""
Test-set evaluation for the Random Forest baseline.

Loads the trained model + scaler, runs pixel-wise inference over every
patch in the test split, and reports the validation metrics defined in
``src.validation.metrics`` (precision, recall, F1, IoU) for the debris class.

Run from the repository root:
    python -m src.validation.evaluate
"""

import os

import numpy as np
from sklearn.metrics import classification_report

from src.data.dataset_loader import load_split
from src.models.random_forest import prepare_rf_data, load_model
from src.validation.metrics import (
    compute_binary_metrics,
    compute_confusion_matrix,
    save_metrics,
)
from src.validation.report import generate_pdf_report

# Adjust these paths to match your local layout
PATCHES_ROOT = "data/raw/patches"
SPLITS_DIR   = "data/raw/splits"
EXPERIMENTS_DIR = "experiments"           # trained model + scaler artifacts
TEST_OUTPUT_DIR = "test/data/outputs"     # test metrics + reports
MODEL_PREFIX = os.path.join(EXPERIMENTS_DIR, "rf_baseline")

# 256x256 patch => at most 65536 pixels; using this as the per-patch cap in
# prepare_rf_data guarantees NO subsampling, so we evaluate on every valid pixel.
ALL_PIXELS = 256 * 256


def main():
    print("Loading trained model...")
    clf, scaler = load_model(MODEL_PREFIX)

    print("Loading test data...")
    test_bands, test_labels = load_split(
        PATCHES_ROOT,
        os.path.join(SPLITS_DIR, "test_X.txt"),
    )

    print("Preparing features (all valid pixels, no subsampling)...")
    X_test, y_test = prepare_rf_data(
        test_bands, test_labels, max_pixels_per_patch=ALL_PIXELS
    )

    print(
        f"Test samples: {len(X_test)} | "
        f"Debris: {int(y_test.sum())} | "
        f"Non-debris: {int((y_test == 0).sum())}"
    )

    print("Running inference...")
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)

    print("\n=== sklearn classification report ===")
    report = classification_report(
        y_test, y_pred, target_names=["Non-Debris", "Debris"], digits=4
    )
    print(report)

    print("=== validation metrics (debris = positive class) ===")
    metrics = compute_binary_metrics(y_test, y_pred)
    for name, value in metrics.items():
        print(f"{name:>10s}: {value:.4f}")

    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    metrics_prefix = os.path.join(TEST_OUTPUT_DIR, "rf_baseline_test")
    save_metrics(
        metrics_prefix,
        metrics,
        report_text=report,
        extra={
            "split": "test",
            "n_samples": int(len(X_test)),
            "n_debris": int(y_test.sum()),
            "n_non_debris": int((y_test == 0).sum()),
        },
    )
    print(f"\nTest metrics saved to: {metrics_prefix}_metrics.json")

    pdf_path = f"{metrics_prefix}_report.pdf"
    generate_pdf_report(
        pdf_path,
        title="Random Forest Baseline — Test Set",
        metrics=metrics,
        confusion=compute_confusion_matrix(y_test, y_pred),
        report_text=report,
        meta={
            "split": "test",
            "samples": f"{len(X_test):,} pixels (all valid, no subsampling)",
            "debris pixels": f"{int(y_test.sum()):,}",
            "non-debris pixels": f"{int((y_test == 0).sum()):,}",
            "model": "RandomForest (200 trees, max_depth=20, balanced)",
        },
    )
    print(f"Test PDF report saved to: {pdf_path}")


if __name__ == "__main__":
    main()
