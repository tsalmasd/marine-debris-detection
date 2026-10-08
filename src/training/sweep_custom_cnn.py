"""
Multi-seed training sweep for the custom CNN patch classifier.

Single-run test metrics for this model are not reportable. The test split holds
101 patches with debris out of 359, and the single-seed run that motivated this
sweep scored validation AP 0.33 against test AP 0.55 — a gap far too wide to
quote one number from. Same remedy as ``sweep_unet``, same rules:

  - **selects** the final model by validation F1 (never by test — that would leak
    the test set), promoting its checkpoint to ``custom_cnn.pt`` and its
    training history to ``custom_cnn_history.csv``;
  - **reports** the test metrics as mean ± std across seeds, so the comparison
    states its own uncertainty.

Average precision is aggregated alongside the thresholded metrics: it is
threshold-free, so it separates "the model ranks patches well" from "0.5 happens
to be a good cut", and the two can disagree.

Datasets are cached once and reused across seeds. Run from the repo root:
    python -m src.training.sweep_custom_cnn                 # seeds 0..4, 80 epochs
    python -m src.training.sweep_custom_cnn --seeds 0 1 2   # custom seeds
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from src.data.patch_dataset import (
    MaridaPatchClassificationDataset,
    compute_normalization_stats,
)
from src.models.custom_cnn import load_checkpoint
from src.training.train_custom_cnn import fit_custom_cnn
from src.validation.evaluate_custom_cnn import collect_patch_predictions
from src.validation.metrics import compute_binary_metrics

PATCHES_ROOT = "data/raw/patches"
SPLITS_DIR = "data/raw/splits"
MODEL_DIR = "test/data/model"
VAL_OUTPUT_DIR = "test/data/val"
OUTPUT_DIR = "test/data/outputs"

METRIC_KEYS = ["precision", "recall", "f1", "iou", "mcc", "balanced_accuracy",
               "average_precision"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VAL_OUTPUT_DIR, exist_ok=True)
    print(f"Device: {device} | seeds: {args.seeds}")

    train_split = os.path.join(SPLITS_DIR, "train_X.txt")
    val_split = os.path.join(SPLITS_DIR, "val_X.txt")
    test_split = os.path.join(SPLITS_DIR, "test_X.txt")

    print("Computing normalization stats (train) and caching datasets once...")
    mean, std = compute_normalization_stats(PATCHES_ROOT, train_split)
    train_ds = MaridaPatchClassificationDataset(PATCHES_ROOT, train_split, mean=mean, std=std)
    val_ds = MaridaPatchClassificationDataset(PATCHES_ROOT, val_split, mean=mean, std=std)
    test_ds = MaridaPatchClassificationDataset(PATCHES_ROOT, test_split, mean=mean, std=std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    pos_weight = train_ds.compute_pos_weight()
    test_prevalence = float(test_ds.patch_labels().mean())
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)} | "
          f"pos_weight {pos_weight:.2f}")
    # The floors any honest reading has to clear: predicting "debris" for every
    # patch, and ranking patches at random.
    print(f"Test prevalence {test_prevalence:.4f} -> trivial all-positive F1 "
          f"{2 * test_prevalence / (1 + test_prevalence):.4f}, random-ranking AP "
          f"{test_prevalence:.4f}\n")

    runs = []
    for seed in args.seeds:
        ckpt = os.path.join(MODEL_DIR, f"custom_cnn_seed{seed}.pt")
        print(f"=== seed {seed} ===")
        res = fit_custom_cnn(
            train_loader, val_loader, mean, std, pos_weight, device,
            epochs=args.epochs, lr=args.lr, base_channels=args.base_channels,
            hidden_dim=args.hidden_dim, dropout=args.dropout, patience=args.patience,
            ckpt_path=ckpt, threshold=args.threshold, seed=seed, verbose=False,
        )
        model, _, _ = load_checkpoint(ckpt, device=device)
        y_test, y_prob = collect_patch_predictions(model, test_loader, device)
        test_m = compute_binary_metrics(y_test, (y_prob >= args.threshold).astype(np.uint8))
        test_m["average_precision"] = float(average_precision_score(y_test, y_prob))
        runs.append({
            "seed": seed, "ckpt": ckpt,
            "val_f1": res["best_f1"], "val_ap": res["best_ap"],
            "best_epoch": res["best_epoch"], "history": res["history"], "test": test_m,
        })
        print(f"  val F1 {res['best_f1']:.4f} @ epoch {res['best_epoch']} | "
              f"test F1 {test_m['f1']:.4f} AP {test_m['average_precision']:.4f} "
              f"MCC {test_m['mcc']:.4f}\n")

    # ---- select by validation F1 (never by test) --------------------------
    best = max(runs, key=lambda r: r["val_f1"])
    shutil.copyfile(best["ckpt"], os.path.join(MODEL_DIR, "custom_cnn.pt"))
    with open(os.path.join(VAL_OUTPUT_DIR, "custom_cnn_history.csv"), "w") as f:
        f.write("epoch,train_loss,val_f1,val_average_precision\n")
        for e, l, v, a in best["history"]:
            f.write(f"{e},{l:.6f},{v:.6f},{a:.6f}\n")

    # ---- aggregate test metrics across seeds ------------------------------
    agg = {k: {"mean": float(np.mean([r["test"][k] for r in runs])),
               "std": float(np.std([r["test"][k] for r in runs]))}
           for k in METRIC_KEYS}

    print("=== per-seed test metrics ===")
    print(f"{'seed':>5}{'val_f1':>9}{'precision':>11}{'recall':>9}{'f1':>8}"
          f"{'mcc':>8}{'ap':>8}")
    for r in runs:
        t = r["test"]
        print(f"{r['seed']:>5}{r['val_f1']:>9.4f}{t['precision']:>11.4f}"
              f"{t['recall']:>9.4f}{t['f1']:>8.4f}{t['mcc']:>8.4f}"
              f"{t['average_precision']:>8.4f}")
    print("\n=== test mean ± std across seeds ===")
    for k in METRIC_KEYS:
        print(f"  {k:>18}: {agg[k]['mean']:.3f} ± {agg[k]['std']:.3f}")
    print(f"\nSelected model: seed {best['seed']} (best val F1 {best['val_f1']:.4f}) "
          f"-> promoted to custom_cnn.pt")

    summary = {
        "seeds": args.seeds,
        "selected_seed": best["seed"],
        "selection_rule": "max validation F1",
        "unit": "patch",
        "threshold": float(args.threshold),
        "pos_weight": float(pos_weight),
        "test_prevalence": test_prevalence,
        "trivial_all_positive_f1": 2 * test_prevalence / (1 + test_prevalence),
        "random_ranking_ap": test_prevalence,
        "test_mean_std": agg,
        "per_seed": [
            {"seed": r["seed"], "val_f1": r["val_f1"], "val_ap": r["val_ap"],
             "best_epoch": r["best_epoch"], "test": r["test"]}
            for r in runs
        ],
    }
    out_path = os.path.join(OUTPUT_DIR, "custom_cnn_seed_sweep.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Sweep summary saved to: {out_path}")

    # tidy up per-seed checkpoints (the promoted one lives as custom_cnn.pt)
    for r in runs:
        try:
            os.remove(r["ckpt"])
        except OSError:
            pass


if __name__ == "__main__":
    main()
