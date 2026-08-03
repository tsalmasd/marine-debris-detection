"""
Training entrypoint for the Random Forest baseline.

Run from the repository root:
    python -m src.training.train
"""

import os
from sklearn.metrics import classification_report, f1_score

from src.data.dataset_loader import load_split
from src.models.random_forest import (
    prepare_rf_data,
    train_random_forest,
    save_model,
)

# Adjust these paths to match your local layout
PATCHES_ROOT = "data/raw/patches"
SPLITS_DIR   = "data/raw/splits"
EXPERIMENTS_DIR = "experiments"


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
    X_train, y_train = prepare_rf_data(train_bands, train_labels)
    X_val,   y_val   = prepare_rf_data(val_bands,   val_labels)

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

    print(classification_report(
        y_val, y_pred, target_names=["Non-Debris", "Debris"]
    ))
    print(f"F1 (debris class): {f1_score(y_val, y_pred):.4f}")

    save_path = os.path.join(EXPERIMENTS_DIR, "rf_baseline")
    save_model(clf, scaler, save_path)
    print(f"Model saved to: {save_path}_rf.joblib")


if __name__ == "__main__":
    main()
