"""
Feature ablation for research question RQ2:
    "How important are spectral indices vs. raw bands, and spatial context vs.
     spectral information?"

Two controlled comparisons, both on the held-out test split (all valid pixels):

  (1) Spectral indices vs. raw bands -- the Random Forest is retrained on three
      feature subsets while the (CV-selected) hyperparameters are held FIXED, so
      the only thing that varies is the feature set:
        - bands only            : [B02,B03,B04,B06,B08,B11,B12]
        - indices only          : [NDVI, FDI]
        - bands + indices (full): the thesis RF feature vector
      The gap between "bands only" and "bands + indices" is the marginal
      contribution of the spectral indices.

  (2) Spatial context vs. spectral -- the best spectral model (RF, full features)
      is compared with the U-Net (same spectral bands PLUS spatial context). The
      U-Net row is read from its saved test-metrics file; the gap isolates the
      contribution of spatial information.

Run from the repository root (after training the RF and U-Net):
    python -m src.validation.ablation_features
"""

import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.data.dataset_loader import load_split, RF_BANDS
from src.features.spectral_indices import RF_FEATURE_NAMES
from src.models.random_forest import prepare_rf_data
from src.validation.metrics import compute_binary_metrics

PATCHES_ROOT = "data/raw/patches"
SPLITS_DIR = "data/raw/splits"
OUTPUT_DIR = "test/data/outputs"
UNET_METRICS = os.path.join(OUTPUT_DIR, "unet_test_metrics.json")

ALL_PIXELS = 256 * 256  # per-patch cap that guarantees no subsampling on test

# Hyperparameters held fixed across feature variants (the CV winner from
# train_random_forest_cv), so the comparison isolates the feature set alone.
BEST_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

# Column indices into the 9-feature vector produced by extract_features_from_patch:
# [B02, B03, B04, B06, B08, B11, B12, NDVI, FDI]  (see RF_FEATURE_NAMES).
BAND_COLS = list(range(0, 7))    # the 7 raw bands
INDEX_COLS = [7, 8]              # NDVI, FDI
VARIANTS = {
    "Bands only": BAND_COLS,
    "Indices only": INDEX_COLS,
    "Bands + indices (full)": BAND_COLS + INDEX_COLS,
}


def _fit_eval(cols, X_train, y_train, X_test, y_test) -> dict:
    """Train an RF on the selected feature columns and evaluate on the test set."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train[:, cols])
    Xte = scaler.transform(X_test[:, cols])
    clf = RandomForestClassifier(**BEST_PARAMS)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    return compute_binary_metrics(y_test, y_pred)


def main():
    print("Loading data (RF 7-band set)...")
    train_bands, train_labels = load_split(
        PATCHES_ROOT, os.path.join(SPLITS_DIR, "train_X.txt"), bands=RF_BANDS
    )
    test_bands, test_labels = load_split(
        PATCHES_ROOT, os.path.join(SPLITS_DIR, "test_X.txt"), bands=RF_BANDS
    )

    print("Preparing features...")
    X_train, y_train = prepare_rf_data(train_bands, train_labels)
    X_test, y_test = prepare_rf_data(
        test_bands, test_labels, max_pixels_per_patch=ALL_PIXELS
    )
    print(f"Features: {RF_FEATURE_NAMES}")
    print(f"Train pixels: {len(X_train):,} | Test pixels: {len(X_test):,} "
          f"(debris {int(y_test.sum())})\n")

    # ---- (1) spectral indices vs. raw bands -------------------------------
    print("=== RQ2a: spectral indices vs. raw bands (RF, fixed hyperparameters) ===")
    header = f"{'feature set':<26}{'#feat':>6}{'precision':>11}{'recall':>9}{'f1':>8}{'iou':>8}"
    print(header)
    print("-" * len(header))
    rq2a = {}
    for name, cols in VARIANTS.items():
        m = _fit_eval(cols, X_train, y_train, X_test, y_test)
        rq2a[name] = {"n_features": len(cols), **m}
        print(f"{name:<26}{len(cols):>6}{m['precision']:>11.4f}"
              f"{m['recall']:>9.4f}{m['f1']:>8.4f}{m['iou']:>8.4f}")

    full_f1 = rq2a["Bands + indices (full)"]["f1"]
    bands_f1 = rq2a["Bands only"]["f1"]
    print(f"\nMarginal contribution of NDVI+FDI (F1): "
          f"{bands_f1:.4f} -> {full_f1:.4f}  (Δ {full_f1 - bands_f1:+.4f})")

    # ---- (2) spatial context vs. spectral ---------------------------------
    print("\n=== RQ2b: spatial context vs. spectral (test split) ===")
    rq2b = {"RF (spectral, full features)": rq2a["Bands + indices (full)"]}
    if os.path.exists(UNET_METRICS):
        with open(UNET_METRICS) as f:
            unet = json.load(f)["metrics"]
        rq2b["U-Net (spectral + spatial)"] = unet
        for name, m in rq2b.items():
            print(f"{name:<32} F1={m['f1']:.4f}  IoU={m['iou']:.4f}")
        d_f1 = unet["f1"] - full_f1
        d_iou = unet["iou"] - rq2a["Bands + indices (full)"]["iou"]
        print(f"\nContribution of spatial context (RF -> U-Net): "
              f"F1 {d_f1:+.4f}, IoU {d_iou:+.4f}")
    else:
        print(f"(U-Net metrics not found at {UNET_METRICS}; run evaluate_unet first.)")

    out_path = os.path.join(OUTPUT_DIR, "ablation_features.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "features": RF_FEATURE_NAMES,
                "fixed_rf_params": {k: v for k, v in BEST_PARAMS.items()
                                    if k in ("n_estimators", "max_depth", "min_samples_leaf")},
                "rq2a_indices_vs_bands": rq2a,
                "rq2b_spatial_vs_spectral": rq2b,
            },
            f,
            indent=2,
        )
    print(f"\nAblation results saved to: {out_path}")


if __name__ == "__main__":
    main()
