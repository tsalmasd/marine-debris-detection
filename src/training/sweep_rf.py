"""
Multi-seed Random Forest sweep -- error bars for the RF baseline.

The U-Net is reported as a mean +- std over five seeds (``sweep_unet.py``)
because a single run of it measures the seed as much as the model. The RF was
reported as a single ``random_state=42`` fit, so the headline "the two models
finish level" was comparing a point against a distribution. This module closes
that asymmetry: it refits the RF across the same seeds and reports the same
statistics.

Two sources of randomness are varied together, mirroring what a U-Net seed
changes:

    1. the per-patch training subsample (``prepare_rf_data(random_state=seed)``),
       analogous to the U-Net's batch order; and
    2. the forest itself (bootstrap samples + per-split feature subsets).

Hyperparameters are held FIXED at the cross-validated winner -- this measures
run-to-run spread, not a hyperparameter search, and nothing here is selected on
the test split. The saved ``rf_baseline_rf.joblib`` is left untouched; this
module only reports.

Run from the repository root (a few minutes on CPU):
    python -m src.training.sweep_rf
    python -m src.training.sweep_rf --seeds 0 1 2
"""

import argparse
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.data.dataset_loader import load_split, RF_BANDS
from src.models.random_forest import prepare_rf_data
from src.validation.metrics import compute_binary_metrics

PATCHES_ROOT = "data/raw/patches"
SPLITS_DIR = "data/raw/splits"
OUTPUT_DIR = "test/data/outputs"

ALL_PIXELS = 256 * 256  # per-patch cap that guarantees no subsampling on test

# The CV winner from train_random_forest_cv, held fixed across seeds.
BEST_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
)

METRIC_KEYS = ["precision", "recall", "f1", "iou", "mcc", "balanced_accuracy"]


def fit_eval_seed(X_train_raw, y_train, X_test, y_test, seed: int) -> dict:
    """Fit the RF at one seed and score it on the test split."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train_raw)
    Xte = scaler.transform(X_test)
    clf = RandomForestClassifier(random_state=seed, **BEST_PARAMS)
    clf.fit(Xtr, y_train)
    return compute_binary_metrics(y_test, clf.predict(Xte))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "rf_seed_sweep.json"))
    args = parser.parse_args()

    print("Loading training data...")
    train_bands, train_labels = load_split(
        PATCHES_ROOT, os.path.join(SPLITS_DIR, "train_X.txt"), bands=RF_BANDS
    )
    print("Loading test data...")
    test_bands, test_labels = load_split(
        PATCHES_ROOT, os.path.join(SPLITS_DIR, "test_X.txt"), bands=RF_BANDS
    )

    # Test features are identical across seeds (no subsampling), so prepare once.
    X_test, y_test = prepare_rf_data(
        test_bands, test_labels, max_pixels_per_patch=ALL_PIXELS
    )
    print(f"Test samples: {len(X_test):,} | debris: {int(y_test.sum()):,}")

    runs = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        # The training subsample is re-drawn per seed; see prepare_rf_data.
        X_train, y_train = prepare_rf_data(train_bands, train_labels, random_state=seed)
        print(f"  train samples: {len(X_train):,} | debris: {int(y_train.sum()):,}")
        m = fit_eval_seed(X_train, y_train, X_test, y_test, seed)
        runs.append({"seed": seed, "test": m})
        print(f"  test F1 {m['f1']:.4f} IoU {m['iou']:.4f} "
              f"P {m['precision']:.4f} R {m['recall']:.4f}")

    agg = {k: {"mean": float(np.mean([r["test"][k] for r in runs])),
               "std": float(np.std([r["test"][k] for r in runs]))}
           for k in METRIC_KEYS}

    print("\n=== per-seed test metrics ===")
    print(f"{'seed':>5}{'precision':>11}{'recall':>9}{'f1':>8}{'iou':>8}{'mcc':>8}")
    for r in runs:
        t = r["test"]
        print(f"{r['seed']:>5}{t['precision']:>11.4f}{t['recall']:>9.4f}"
              f"{t['f1']:>8.4f}{t['iou']:>8.4f}{t['mcc']:>8.4f}")
    print("\n=== test mean ± std across seeds ===")
    for k in METRIC_KEYS:
        print(f"  {k:>18}: {agg[k]['mean']:.3f} ± {agg[k]['std']:.3f}")

    summary = {
        "seeds": args.seeds,
        "fixed_rf_params": {k: v for k, v in BEST_PARAMS.items() if k != "n_jobs"},
        "varies": "per-patch training subsample AND forest randomness",
        "test_mean_std": agg,
        "per_seed": runs,
        "n_samples": int(len(X_test)),
        "n_debris": int(y_test.sum()),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSweep summary saved to: {args.out}")


if __name__ == "__main__":
    main()
