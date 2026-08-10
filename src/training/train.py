"""
Training entrypoint for the Random Forest baseline.

Run from the repository root:
    python -m src.training.train
"""

import os
from sklearn.metrics import classification_report

from src.data.dataset_loader import load_split
from src.models.random_forest import (
    prepare_rf_data,
    train_random_forest,
    save_model,
)
from src.validation.metrics import (
    compute_binary_metrics,
    compute_confusion_matrix,
    save_metrics,
)
from src.validation.report import generate_pdf_report

# Adjust these paths to match your local layout
PATCHES_ROOT = "data/raw/patches"
SPLITS_DIR   = "data/raw/splits"
EXPERIMENTS_DIR = "experiments"

# 256x256 patch => at most 65536 valid pixels. Using this as the per-patch cap
# guarantees NO subsampling, so the validation metrics are computed on every
# valid pixel -- directly comparable to the test-set evaluation.
ALL_PIXELS = 256 * 256


def main():
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    print("Loading training data...")
    train_bands, train_labels = load_split(
        PATCHES_ROOT,
        os.path.join(SPLITS_DIR, "train_X.txt")
    )

    print("Loading validation data...")
    val_bands, val_labels = load_split(
        PATCHES_ROOT,
        os.path.join(SPLITS_DIR, "val_X.txt")
    )

    print("Preparing features...")
    # Training features are subsampled per patch (default cap) to stay tractable;
    # validation features use ALL valid pixels so the reported metrics are exact.
    X_train, y_train = prepare_rf_data(train_bands, train_labels)
    X_val,   y_val   = prepare_rf_data(
        val_bands, val_labels, max_pixels_per_patch=ALL_PIXELS
    )

    print(
        f"Train samples: {len(X_train)} | "
        f"Debris: {y_train.sum()} | "
        f"Non-debris: {(y_train == 0).sum()}"
    )

    print("Training Random Forest...")
    clf, scaler = train_random_forest(X_train, y_train)

    print("Evaluating on validation set...")
    X_val_scaled = scaler.transform(X_val)
    y_pred = clf.predict(X_val_scaled)

    report = classification_report(
        y_val, y_pred, target_names=["Non-Debris", "Debris"], digits=4
    )
    print(report)

    metrics = compute_binary_metrics(y_val, y_pred)
    print(f"F1 (debris class): {metrics['f1']:.4f}")

    save_path = os.path.join(EXPERIMENTS_DIR, "rf_baseline")
    save_model(clf, scaler, save_path)
    print(f"Model saved to: {save_path}_rf.joblib")

    metrics_prefix = os.path.join(EXPERIMENTS_DIR, "rf_baseline_val")
    save_metrics(
        metrics_prefix,
        metrics,
        report_text=report,
        extra={
            "split": "val",
            "n_samples": int(len(X_val)),
            "n_debris": int(y_val.sum()),
            "n_non_debris": int((y_val == 0).sum()),
        },
    )
    print(f"Validation metrics saved to: {metrics_prefix}_metrics.json")

    pdf_path = f"{metrics_prefix}_report.pdf"
    generate_pdf_report(
        pdf_path,
        title="Random Forest Baseline — Validation Set",
        metrics=metrics,
        confusion=compute_confusion_matrix(y_val, y_pred),
        report_text=report,
        meta={
            "split": "validation",
            "samples": f"{len(X_val):,} pixels (all valid, no subsampling)",
            "debris pixels": f"{int(y_val.sum()):,}",
            "non-debris pixels": f"{int((y_val == 0).sum()):,}",
            "model": "RandomForest (200 trees, max_depth=20, balanced)",
        },
    )
    print(f"Validation PDF report saved to: {pdf_path}")


if __name__ == "__main__":
    main()
