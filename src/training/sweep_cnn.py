"""
Multi-seed sweep for the custom CNN window classifier (REVISED_PLAN.md step 3).

Trains the W x W centre-pixel CNN under several seeds, then:

  - **selects** the final model by validation F1 (never by test — that would leak
    the test set), promoting its checkpoint to ``cnn_w{W}.pt`` and its training
    history to ``cnn_w{W}_history.csv``;
  - **reports** test metrics as mean +/- std across seeds, so the comparison
    states its own uncertainty.

Run one sweep per pre-registered window size; together with the RF (1 px) and
the U-Net (256 px) the results form the context curve of REVISED_PLAN.md
figure 1. Each sweep writes its own JSON, so no result file is ever overwritten.

Datasets are cached once and reused across seeds. Run from the repo root:
    python -m src.training.sweep_cnn --window 16
    python -m src.training.sweep_cnn --window 32 --seeds 0 1 2
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from src.data.window_dataset import MaridaWindowDataset
from src.models.custom_cnn import load_checkpoint
from src.training.train_cnn import (
    MODEL_DIR,
    PATCHES_ROOT,
    SPLITS_DIR,
    VAL_OUTPUT_DIR,
    add_common_args,
    build_loaders,
    fit_window_cnn,
)
from src.validation.evaluate_cnn import collect_window_predictions
from src.validation.metrics import compute_binary_metrics

OUTPUT_DIR = "test/data/outputs"

METRIC_KEYS = ["precision", "recall", "f1", "iou", "mcc", "balanced_accuracy",
               "average_precision"]


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VAL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {device} | seeds: {args.seeds}")

    (bands, mean, std, train_ds, val_ds,
     train_loader, val_loader, pos_weight) = build_loaders(args, device)

    test_ds = MaridaWindowDataset(
        PATCHES_ROOT, os.path.join(SPLITS_DIR, "test_X.txt"),
        window=args.window, bands=bands, mean=mean, std=std, augment=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    prevalence = float(test_ds.targets.mean())
    print(f"Test windows: {len(test_ds):,} ({int(test_ds.targets.sum()):,} debris)")
    print(f"Test prevalence {prevalence:.5f} -> trivial all-positive F1 "
          f"{2 * prevalence / (1 + prevalence):.4f}, random-ranking AP {prevalence:.5f}\n")

    tag = f"cnn_w{args.window}"
    runs = []
    for seed in args.seeds:
        ckpt = os.path.join(MODEL_DIR, f"{tag}_seed{seed}.pt")
        print(f"=== seed {seed} ===", flush=True)
        res = fit_window_cnn(
            train_loader, val_loader, mean, std, bands, pos_weight, device,
            epochs=args.epochs, lr=args.lr, window=args.window,
            base_channels=args.base_channels, hidden_dim=args.hidden_dim,
            dropout=args.dropout, patience=args.patience, ckpt_path=ckpt,
            threshold=args.threshold, seed=seed, verbose=False,
        )
        model, _, _ = load_checkpoint(ckpt, device=device)
        y_test, y_prob = collect_window_predictions(model, test_loader, device)
        test_m = compute_binary_metrics(y_test, (y_prob >= args.threshold).astype(np.uint8))
        test_m["average_precision"] = float(average_precision_score(y_test, y_prob))
        runs.append({
            "seed": seed, "ckpt": ckpt,
            "val_f1": res["best_f1"], "val_ap": res["best_ap"],
            "best_epoch": res["best_epoch"], "history": res["history"], "test": test_m,
        })
        print(f"  val F1 {res['best_f1']:.4f} @ epoch {res['best_epoch']}/{args.epochs} | "
              f"test F1 {test_m['f1']:.4f} IoU {test_m['iou']:.4f} "
              f"AP {test_m['average_precision']:.4f}\n", flush=True)

    # ---- select by validation F1 (never by test) --------------------------
    best = max(runs, key=lambda r: r["val_f1"])
    shutil.copyfile(best["ckpt"], os.path.join(MODEL_DIR, f"{tag}.pt"))
    with open(os.path.join(VAL_OUTPUT_DIR, f"{tag}_history.csv"), "w") as f:
        f.write("epoch,train_loss,val_f1,val_average_precision\n")
        for e, l, v, a in best["history"]:
            f.write(f"{e},{l:.6f},{v:.6f},{a:.6f}\n")

    # ---- aggregate test metrics across seeds ------------------------------
    agg = {k: {"mean": float(np.mean([r["test"][k] for r in runs])),
               "std": float(np.std([r["test"][k] for r in runs]))}
           for k in METRIC_KEYS}

    print("=== per-seed test metrics ===")
    print(f"{'seed':>5}{'val_f1':>9}{'epoch':>7}{'precision':>11}{'recall':>9}"
          f"{'f1':>8}{'iou':>8}{'ap':>8}")
    for r in runs:
        t = r["test"]
        print(f"{r['seed']:>5}{r['val_f1']:>9.4f}{r['best_epoch']:>7}"
              f"{t['precision']:>11.4f}{t['recall']:>9.4f}{t['f1']:>8.4f}"
              f"{t['iou']:>8.4f}{t['average_precision']:>8.4f}")
    print("\n=== test mean ± std across seeds ===")
    for k in METRIC_KEYS:
        print(f"  {k:>18}: {agg[k]['mean']:.3f} ± {agg[k]['std']:.3f}")
    print(f"\nSelected model: seed {best['seed']} (best val F1 {best['val_f1']:.4f}) "
          f"-> promoted to {tag}.pt")

    # Early stopping only fires after `patience` epochs with no improvement, so
    # any seed peaking after (epochs - patience) was cut off by the budget
    # rather than by convergence -- it never had the grace period to stop
    # itself. Testing `best_epoch >= epochs` alone misses those: it is the
    # exact trap the first patch-classifier sweep fell into.
    converged_before = args.epochs - args.patience
    at_cap = [r["seed"] for r in runs if r["best_epoch"] > converged_before]
    if at_cap:
        print(f"WARNING: seeds {at_cap} peaked after epoch {converged_before} "
              f"of {args.epochs}, so early stopping (patience {args.patience}) "
              f"never fired: the budget stopped them, not convergence. "
              f"Re-run with more epochs before reporting.")

    summary = {
        "seeds": args.seeds,
        "selected_seed": best["seed"],
        "selection_rule": "max validation F1",
        "unit": "pixel",
        "window": int(args.window),
        "bands": bands,
        "augment": not args.no_augment,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "threshold": float(args.threshold),
        "pos_weight": float(pos_weight),
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "n_test_windows": len(test_ds),
        "test_prevalence": prevalence,
        "trivial_all_positive_f1": 2 * prevalence / (1 + prevalence),
        "random_ranking_ap": prevalence,
        "seeds_at_epoch_cap": at_cap,
        "converged_before_epoch": converged_before,
        "test_mean_std": agg,
        "per_seed": [
            {"seed": r["seed"], "val_f1": r["val_f1"], "val_ap": r["val_ap"],
             "best_epoch": r["best_epoch"], "test": r["test"]}
            for r in runs
        ],
    }
    out_path = os.path.join(OUTPUT_DIR, f"{tag}_seed_sweep.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Sweep summary saved to: {out_path}")

    # tidy up per-seed checkpoints (the promoted one lives as {tag}.pt)
    for r in runs:
        try:
            os.remove(r["ckpt"])
        except OSError:
            pass


if __name__ == "__main__":
    main()
