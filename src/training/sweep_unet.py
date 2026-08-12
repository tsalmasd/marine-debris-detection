"""
Multi-seed U-Net training sweep for a robust, honest RF-vs-U-Net comparison.

The debris class is tiny (≈381 test pixels), so a single training run's test
metrics are high-variance. This sweep trains the U-Net under several random seeds,
then:

  - **selects** the final model by validation F1 (never by test — that would leak
    the test set), promoting its checkpoint to ``unet_baseline.pt`` and its
    training history to ``unet_history.csv``;
  - **reports** the test metrics as mean ± std across seeds, so the comparison
    states its own uncertainty.

Datasets are cached once and reused across seeds. Run from the repo root:
    python -m src.training.sweep_unet                 # seeds 0..4, 80 epochs
    python -m src.training.sweep_unet --seeds 0 1 2   # custom seeds
"""

import argparse
import csv
import json
import os
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.patch_dataset import MaridaSegmentationDataset, compute_normalization_stats
from src.models.unet import load_checkpoint
from src.training.train_unet import fit_unet
from src.validation.evaluate_unet import collect_flat_predictions
from src.validation.metrics import compute_binary_metrics

PATCHES_ROOT = "data/raw/patches"
SPLITS_DIR = "data/raw/splits"
MODEL_DIR = "test/data/model"
VAL_OUTPUT_DIR = "test/data/val"
OUTPUT_DIR = "test/data/outputs"

METRIC_KEYS = ["precision", "recall", "f1", "iou", "mcc", "balanced_accuracy"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Device: {device} | seeds: {args.seeds}")

    train_split = os.path.join(SPLITS_DIR, "train_X.txt")
    val_split = os.path.join(SPLITS_DIR, "val_X.txt")
    test_split = os.path.join(SPLITS_DIR, "test_X.txt")

    print("Computing normalization stats (train) and caching datasets once...")
    mean, std = compute_normalization_stats(PATCHES_ROOT, train_split)
    train_ds = MaridaSegmentationDataset(PATCHES_ROOT, train_split, mean=mean, std=std)
    val_ds = MaridaSegmentationDataset(PATCHES_ROOT, val_split, mean=mean, std=std)
    test_ds = MaridaSegmentationDataset(PATCHES_ROOT, test_split, mean=mean, std=std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    pos_weight = min(train_ds.compute_pos_weight(), args.max_pos_weight)
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)} | pos_weight {pos_weight:.1f}\n")

    runs = []
    for seed in args.seeds:
        ckpt = os.path.join(MODEL_DIR, f"unet_seed{seed}.pt")
        print(f"=== seed {seed} ===")
        res = fit_unet(
            train_loader, val_loader, mean, std, pos_weight, device,
            epochs=args.epochs, lr=args.lr, base_channels=args.base_channels,
            patience=args.patience, use_amp=False, ckpt_path=ckpt, seed=seed, verbose=False,
        )
        model, _, _ = load_checkpoint(ckpt, device=device)
        y_test, y_pred = collect_flat_predictions(model, test_loader, device)
        test_m = compute_binary_metrics(y_test, y_pred)
        runs.append({
            "seed": seed, "ckpt": ckpt,
            "val_f1": res["best_f1"], "best_epoch": res["best_epoch"],
            "history": res["history"], "test": test_m,
        })
        print(f"  val F1 {res['best_f1']:.4f} @ epoch {res['best_epoch']} | "
              f"test F1 {test_m['f1']:.4f} IoU {test_m['iou']:.4f}\n")

    # ---- select by validation F1 (never by test) --------------------------
    best = max(runs, key=lambda r: r["val_f1"])
    shutil.copyfile(best["ckpt"], os.path.join(MODEL_DIR, "unet_baseline.pt"))
    with open(os.path.join(VAL_OUTPUT_DIR, "unet_history.csv"), "w") as f:
        f.write("epoch,train_loss,val_f1\n")
        for e, l, v in best["history"]:
            f.write(f"{e},{l:.6f},{v:.6f}\n")

    # ---- aggregate test metrics across seeds ------------------------------
    agg = {k: {"mean": float(np.mean([r["test"][k] for r in runs])),
               "std": float(np.std([r["test"][k] for r in runs]))}
           for k in METRIC_KEYS}

    print("=== per-seed test metrics ===")
    print(f"{'seed':>5}{'val_f1':>9}{'precision':>11}{'recall':>9}{'f1':>8}{'iou':>8}{'mcc':>8}")
    for r in runs:
        t = r["test"]
        print(f"{r['seed']:>5}{r['val_f1']:>9.4f}{t['precision']:>11.4f}"
              f"{t['recall']:>9.4f}{t['f1']:>8.4f}{t['iou']:>8.4f}{t['mcc']:>8.4f}")
    print("\n=== test mean ± std across seeds ===")
    for k in METRIC_KEYS:
        print(f"  {k:>9}: {agg[k]['mean']:.3f} ± {agg[k]['std']:.3f}")
    print(f"\nSelected model: seed {best['seed']} (best val F1 {best['val_f1']:.4f}) "
          f"-> promoted to unet_baseline.pt")

    summary = {
        "seeds": args.seeds,
        "selected_seed": best["seed"],
        "selection_rule": "max validation F1",
        "test_mean_std": agg,
        "per_seed": [
            {"seed": r["seed"], "val_f1": r["val_f1"], "best_epoch": r["best_epoch"], "test": r["test"]}
            for r in runs
        ],
    }
    out_path = os.path.join(OUTPUT_DIR, "unet_seed_sweep.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Sweep summary saved to: {out_path}")

    # tidy up per-seed checkpoints (the promoted one lives as unet_baseline.pt)
    for r in runs:
        try:
            os.remove(r["ckpt"])
        except OSError:
            pass


if __name__ == "__main__":
    main()
