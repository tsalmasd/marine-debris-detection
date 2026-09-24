"""
Patch-level bootstrap confidence intervals for the RF vs. U-Net comparison.

The test split carries only 381 annotated debris pixels, so a point estimate of
F1 or IoU says nothing on its own about how precisely it is known. This module
quantifies that: it resamples the 359 test **patches** with replacement, recomputes
the debris-class metrics on each resample, and reports percentile intervals.

Why resample patches rather than pixels. Pixels within a patch are not
independent -- debris occurs in contiguous slicks, and a model that finds one
slick gets all of its pixels right at once. Bootstrapping pixels would treat
those as independent draws and return an interval several times too narrow. The
patch is the unit that was independently sampled into the benchmark, so the
patch is the unit to resample.

The comparison is **paired**: both models are scored on the same resampled patch
sets, so the interval on the difference accounts for the fact that an easy draw
of patches is easy for both. This is what licenses a claim about whether the two
models differ, rather than two separate intervals that happen to overlap.

What this does NOT measure: training variance. The U-Net numbers here come from
the single promoted checkpoint, so these intervals describe how precisely *that
model's* score is known on *this* test set. Run-to-run spread across seeds is a
separate and, for the U-Net, larger source of uncertainty -- see
``sweep_unet.py`` (+-0.06 F1) and ``sweep_rf.py``. Both must be quoted; neither
subsumes the other.

Run from the repository root (needs the trained RF and U-Net):
    python -m src.validation.bootstrap
    python -m src.validation.bootstrap --n-boot 10000 --threshold 0.4
"""

import argparse
import json
import os

import numpy as np
import torch

from src.data.dataset_loader import RF_BANDS, load_patch
from src.data.patch_dataset import MaridaSegmentationDataset
from src.features.spectral_indices import extract_features_from_patch
from src.models.random_forest import load_model as load_rf
from src.models.unet import load_checkpoint as load_unet

PATCHES_ROOT = "data/raw/patches"
SPLITS_DIR = "data/raw/splits"
MODEL_DIR = "test/data/model"
OUTPUT_DIR = "test/data/outputs"
RF_PREFIX = os.path.join(MODEL_DIR, "rf_baseline")
UNET_PATH = os.path.join(MODEL_DIR, "unet_baseline.pt")

METRIC_KEYS = ["precision", "recall", "f1", "iou"]


def metrics_from_counts(tp, fp, fn) -> dict:
    """
    Debris-class metrics from confusion counts.

    Accepts scalars or equal-shaped arrays (one entry per bootstrap resample),
    which is what makes the resampling loop vectorizable. Divisions are guarded:
    a resample containing no debris pixels at all leaves the affected metric at
    0.0 rather than producing a NaN that would silently poison the percentiles.
    """
    tp, fp, fn = np.asarray(tp, float), np.asarray(fp, float), np.asarray(fn, float)

    def _div(num, den):
        return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)

    return {
        "precision": _div(tp, tp + fp),
        "recall": _div(tp, tp + fn),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "iou": _div(tp, tp + fp + fn),
    }


@torch.no_grad()
def collect_patch_counts(threshold: float = 0.5, device: str | None = None):
    """
    Per-patch TP/FP/FN counts for both models over the test split.

    Both models are scored on exactly the same pixels: the loop is driven by the
    U-Net dataset's (post-filtering) patch id list, and each patch's RF features
    are read for that same id, so the two predictions stay aligned patch by patch
    and pixel by pixel. The per-patch valid-pixel counts are asserted equal --
    if the two preprocessing paths ever diverge, this fails loudly rather than
    silently comparing different pixel sets.

    Returns:
        (counts, patch_ids, n_debris) where counts maps "rf"/"unet" to an
        (n_patches, 3) int array of [tp, fp, fn].
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    clf, scaler = load_rf(RF_PREFIX)
    model, mean, std = load_unet(UNET_PATH, device=device)
    model.eval()

    ds = MaridaSegmentationDataset(
        PATCHES_ROOT, os.path.join(SPLITS_DIR, "test_X.txt"), mean=mean, std=std
    )
    print(f"Device: {device} | patches: {len(ds)} | threshold: {threshold}")

    rf_counts, un_counts = [], []
    for i in range(len(ds)):
        img, target, valid = ds[i]
        mask = valid.numpy() > 0
        y_true = target.numpy()[mask].astype(np.uint8)

        # --- U-Net ---------------------------------------------------------
        logits = model(img.unsqueeze(0).to(device))
        prob = torch.sigmoid(logits.squeeze().float()).cpu().numpy()
        un_pred = (prob[mask] >= threshold).astype(np.uint8)

        # --- Random Forest (same patch, same valid pixels) -----------------
        bands, _ = load_patch(PATCHES_ROOT, ds.patch_ids[i], bands=RF_BANDS)
        feats = extract_features_from_patch(bands)[mask.reshape(-1)]
        assert len(feats) == len(y_true), f"pixel mismatch on {ds.patch_ids[i]}"
        rf_pred = clf.predict(scaler.transform(feats)).astype(np.uint8)

        for pred, store in ((rf_pred, rf_counts), (un_pred, un_counts)):
            store.append([
                int(((pred == 1) & (y_true == 1)).sum()),
                int(((pred == 1) & (y_true == 0)).sum()),
                int(((pred == 0) & (y_true == 1)).sum()),
            ])

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ds)} patches")

    counts = {"rf": np.array(rf_counts), "unet": np.array(un_counts)}
    n_debris = int(counts["rf"][:, 0].sum() + counts["rf"][:, 2].sum())
    return counts, ds.patch_ids, n_debris


def bootstrap(counts: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """
    Paired patch-level bootstrap.

    One set of resampled patch indices is drawn and applied to *both* models, so
    the per-resample difference is a paired contrast.
    """
    n_patches = len(counts["rf"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_patches, size=(n_boot, n_patches))

    dists = {}
    for name, c in counts.items():
        # c[idx] -> (n_boot, n_patches, 3); sum over the patch axis.
        totals = c[idx].sum(axis=1)
        dists[name] = metrics_from_counts(totals[:, 0], totals[:, 1], totals[:, 2])

    point = {name: {k: float(v) for k, v in
                    metrics_from_counts(*c.sum(axis=0)).items()}
             for name, c in counts.items()}

    out = {"point_estimate": point, "ci95": {}, "paired_delta_unet_minus_rf": {}}
    for name, dist in dists.items():
        out["ci95"][name] = {
            k: [float(np.percentile(dist[k], 2.5)),
                float(np.percentile(dist[k], 97.5))]
            for k in METRIC_KEYS
        }
    for k in METRIC_KEYS:
        delta = dists["unet"][k] - dists["rf"][k]
        out["paired_delta_unet_minus_rf"][k] = {
            "point": point["unet"][k] - point["rf"][k],
            "ci95": [float(np.percentile(delta, 2.5)),
                     float(np.percentile(delta, 97.5))],
            "p_unet_better": float((delta > 0).mean()),
        }
    return out


def _print_report(res: dict, n_patches: int, n_debris: int, n_boot: int) -> None:
    print(f"\n=== patch-level bootstrap ({n_boot:,} resamples of "
          f"{n_patches} patches; {n_debris} debris pixels) ===\n")
    print(f"{'metric':>10} | {'Random Forest':>24} | {'U-Net (promoted)':>24}")
    print("-" * 66)
    for k in METRIC_KEYS:
        row = []
        for m in ("rf", "unet"):
            lo, hi = res["ci95"][m][k]
            row.append(f"{res['point_estimate'][m][k]:.3f} [{lo:.3f}, {hi:.3f}]")
        print(f"{k:>10} | {row[0]:>24} | {row[1]:>24}")

    print(f"\n=== paired difference (U-Net - RF), same resampled patches ===\n")
    print(f"{'metric':>10} | {'delta':>7} | {'95% CI':>18} | {'P(U-Net better)':>16}")
    print("-" * 62)
    for k in METRIC_KEYS:
        d = res["paired_delta_unet_minus_rf"][k]
        lo, hi = d["ci95"]
        straddles = "" if (lo > 0 or hi < 0) else "   <- CI includes 0"
        print(f"{k:>10} | {d['point']:>+7.3f} | [{lo:>+6.3f}, {hi:>+6.3f}] | "
              f"{d['p_unet_better']:>15.1%}{straddles}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="U-Net decision threshold; 0.5 matches the reported "
                             "test metrics.")
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "bootstrap_ci.json"))
    args = parser.parse_args()

    counts, patch_ids, n_debris = collect_patch_counts(threshold=args.threshold)
    res = bootstrap(counts, n_boot=args.n_boot, seed=args.seed)
    _print_report(res, len(patch_ids), n_debris, args.n_boot)

    res.update({
        "n_patches": len(patch_ids),
        "n_debris_pixels": n_debris,
        "n_boot": args.n_boot,
        "bootstrap_seed": args.seed,
        "unet_threshold": args.threshold,
        "unit_of_resampling": "patch (pixels within a patch are not independent)",
        "note": "Evaluation uncertainty for the promoted checkpoint only; "
                "training/seed variance is reported separately by sweep_unet.py "
                "and sweep_rf.py.",
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nBootstrap results saved to: {args.out}")


if __name__ == "__main__":
    main()
