"""
Patch-level bootstrap confidence intervals for the three-model comparison.

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

The comparison is **paired**: every model is scored on the same resampled patch
sets, so the interval on a difference accounts for the fact that an easy draw of
patches is easy for all of them. This is what licenses a claim about whether two
models differ, rather than separate intervals that happen to overlap.

Models covered: the Random Forest (1-pixel context), the U-Net (256), and any
window CNN checkpoints passed with ``--cnn-ckpt`` (16 and 32 by default) -- the
context curve of ``REVISED_PLAN.md``. Every model predicts the same labelled
test pixels, in the same order: the loop is driven by one ``valid`` mask per
patch and each model's own preprocessing is applied to that patch, so a CNN is
scored on precisely the pixels the other two are scored on.

What this does NOT measure: training variance. The U-Net and CNN numbers here
come from single promoted checkpoints, so these intervals describe how precisely
*that* model's score is known on *this* test set. Run-to-run spread across seeds
is a separate source of uncertainty -- larger than this one for the U-Net
(+-0.06 F1) -- reported by ``sweep_unet.py``, ``sweep_rf.py`` and
``sweep_cnn.py``. Both must be quoted; neither subsumes the other.

Run from the repository root (needs the trained RF, U-Net and window CNNs):
    python -m src.validation.bootstrap
    python -m src.validation.bootstrap --n-boot 10000 --threshold 0.4
    python -m src.validation.bootstrap --cnn-ckpt test/data/model/cnn_w16.pt
    python -m src.validation.bootstrap --cnn-ckpt ""      # RF vs U-Net only
"""

import argparse
import json
import os

import numpy as np
import torch

from src.data.dataset_loader import RF_BANDS, load_patch
from src.data.patch_dataset import MaridaSegmentationDataset
from src.data.window_dataset import extract_windows, prepare_padded_patch
from src.features.spectral_indices import extract_features_from_patch
from src.models.custom_cnn import load_checkpoint as load_cnn
from src.models.custom_cnn import load_checkpoint_bands
from src.models.random_forest import load_model as load_rf
from src.models.unet import load_checkpoint as load_unet

PATCHES_ROOT = "data/raw/patches"
SPLITS_DIR = "data/raw/splits"
MODEL_DIR = "test/data/model"
OUTPUT_DIR = "test/data/outputs"
RF_PREFIX = os.path.join(MODEL_DIR, "rf_baseline")
UNET_PATH = os.path.join(MODEL_DIR, "unet_baseline.pt")
DEFAULT_CNN_CKPTS = [os.path.join(MODEL_DIR, "cnn_w16.pt"),
                     os.path.join(MODEL_DIR, "cnn_w32.pt")]

METRIC_KEYS = ["precision", "recall", "f1", "iou"]

# Printed and contrasted in this order: increasing spatial context, which is the
# axis the comparison is about. Models absent from a run are skipped.
MODEL_ORDER = ["rf", "cnn_w16", "cnn_w32", "unet"]
MODEL_LABELS = {"rf": "Random Forest (1 px)", "unet": "U-Net (256 px)",
                "cnn_w16": "CNN-16 (16 px)", "cnn_w32": "CNN-32 (32 px)"}


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


def load_window_cnns(ckpt_paths: list[str], device: str) -> dict:
    """
    Load each window CNN checkpoint into a scoring spec keyed ``cnn_w{W}``.

    The window size, band list and normalization stats all come *from the
    checkpoint*, never from defaults: a CNN scored with the wrong bands or the
    wrong stats would silently produce a meaningless row in the table. A
    checkpoint written before bands were recorded is refused for that reason.
    """
    specs = {}
    for path in ckpt_paths:
        if not os.path.exists(path):
            print(f"  skipping {path} (not found)")
            continue
        model, mean, std = load_cnn(path, device=device)
        model.eval()
        bands = load_checkpoint_bands(path)
        if bands is None:
            raise SystemExit(
                f"{path} does not record its band list, so it cannot be scored "
                f"on the same pixels as the other models. Retrain with "
                f"src.training.train_cnn, which records it."
            )
        window = int(model.window_size)
        name = f"cnn_w{window}"
        specs[name] = {"model": model, "mean": mean, "std": std,
                       "bands": list(bands), "window": window, "ckpt": path}
        print(f"  {name}: window {window}, {len(bands)} bands, from {path}")
    return specs


@torch.no_grad()
def cnn_patch_probs(spec: dict, img: np.ndarray, rows: np.ndarray,
                    cols: np.ndarray, device: str, chunk: int = 4096) -> np.ndarray:
    """
    Debris probability for each (row, col) of one patch, from a window CNN.

    ``rows``/``cols`` are *unpadded* patch coordinates; the half-window offset is
    added here because :func:`prepare_padded_patch` shifts the patch by that
    much. Windows are cut in chunks rather than all at once: a patch can carry
    several thousand annotated pixels, and at W=32 the full stack would be a
    needless hundreds of MB.
    """
    padded = prepare_padded_patch(img, spec["window"], spec["mean"], spec["std"])
    half = spec["window"] // 2
    rows, cols = np.asarray(rows) + half, np.asarray(cols) + half
    probs = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), chunk):
        sl = slice(start, min(start + chunk, len(rows)))
        batch = extract_windows(padded, rows[sl], cols[sl], spec["window"])
        logits = spec["model"](torch.from_numpy(batch).to(device))
        probs[sl] = torch.sigmoid(logits.reshape(-1).float()).cpu().numpy()
    return probs


@torch.no_grad()
def collect_patch_counts(threshold: float = 0.5, device: str | None = None,
                         cnn_ckpts: list[str] | None = None):
    """
    Per-patch TP/FP/FN counts for every model over the test split.

    All models are scored on exactly the same pixels: the loop is driven by the
    U-Net dataset's (post-filtering) patch id list and by that patch's single
    ``valid`` mask. The RF features and the CNN windows are both read for that
    same patch id, and ``np.nonzero(mask)`` yields the annotated coordinates in
    the same row-major order in which ``target[mask]`` flattens the labels --
    the same order ``MaridaWindowDataset`` builds its index in. Per-patch pixel
    counts are asserted equal, so if the preprocessing paths ever diverge this
    fails loudly rather than silently comparing different pixel sets.

    Returns:
        (counts, patch_ids, n_debris) where counts maps a model name to an
        (n_patches, 3) int array of [tp, fp, fn].
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    clf, scaler = load_rf(RF_PREFIX)
    model, mean, std = load_unet(UNET_PATH, device=device)
    model.eval()
    cnns = load_window_cnns(cnn_ckpts or [], device)

    ds = MaridaSegmentationDataset(
        PATCHES_ROOT, os.path.join(SPLITS_DIR, "test_X.txt"), mean=mean, std=std
    )
    names = ["rf", "unet"] + list(cnns)
    print(f"Device: {device} | patches: {len(ds)} | threshold: {threshold} | "
          f"models: {names}")

    counts: dict[str, list] = {name: [] for name in names}
    for i in range(len(ds)):
        img, target, valid = ds[i]
        mask = valid.numpy() > 0
        y_true = target.numpy()[mask].astype(np.uint8)
        rows, cols = np.nonzero(mask)       # row-major: matches target[mask]

        # One read per band set per patch, shared by the RF and the CNNs (whose
        # default band set is the RF's), instead of one read per model.
        band_cache: dict[tuple, np.ndarray] = {}

        def bands_for(names_: list[str]) -> np.ndarray:
            key = tuple(names_)
            if key not in band_cache:
                band_cache[key] = load_patch(
                    PATCHES_ROOT, ds.patch_ids[i], bands=list(names_))[0]
            return band_cache[key]

        preds = {}

        # --- U-Net ---------------------------------------------------------
        logits = model(img.unsqueeze(0).to(device))
        prob = torch.sigmoid(logits.squeeze().float()).cpu().numpy()
        preds["unet"] = (prob[mask] >= threshold).astype(np.uint8)

        # --- Random Forest (same patch, same valid pixels) -----------------
        feats = extract_features_from_patch(bands_for(RF_BANDS))[mask.reshape(-1)]
        preds["rf"] = clf.predict(scaler.transform(feats)).astype(np.uint8)

        # --- Window CNNs (same patch, same valid pixels) -------------------
        for name, spec in cnns.items():
            cnn_prob = cnn_patch_probs(
                spec, bands_for(spec["bands"]), rows, cols, device)
            preds[name] = (cnn_prob >= threshold).astype(np.uint8)

        for name, pred in preds.items():
            assert len(pred) == len(y_true), \
                f"pixel mismatch for {name} on {ds.patch_ids[i]}"
            counts[name].append([
                int(((pred == 1) & (y_true == 1)).sum()),
                int(((pred == 1) & (y_true == 0)).sum()),
                int(((pred == 0) & (y_true == 1)).sum()),
            ])

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ds)} patches")

    counts = {name: np.array(c) for name, c in counts.items()}
    n_debris = int(counts["rf"][:, 0].sum() + counts["rf"][:, 2].sum())
    return counts, ds.patch_ids, n_debris


def bootstrap(counts: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """
    Paired patch-level bootstrap over every model in ``counts``.

    One set of resampled patch indices is drawn and applied to *all* models, so
    every per-resample difference is a paired contrast. Contrasts are reported
    for each ordered pair in :data:`MODEL_ORDER` (increasing spatial context),
    as ``"{b}_minus_{a}"`` with ``p_better`` = P(b beats a) over the resamples.
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

    out = {"point_estimate": point, "ci95": {}, "paired_deltas": {}}
    for name, dist in dists.items():
        out["ci95"][name] = {
            k: [float(np.percentile(dist[k], 2.5)),
                float(np.percentile(dist[k], 97.5))]
            for k in METRIC_KEYS
        }

    ordered = [m for m in MODEL_ORDER if m in counts]
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            entry = {}
            for k in METRIC_KEYS:
                delta = dists[b][k] - dists[a][k]
                entry[k] = {
                    "point": point[b][k] - point[a][k],
                    "ci95": [float(np.percentile(delta, 2.5)),
                             float(np.percentile(delta, 97.5))],
                    "p_better": float((delta > 0).mean()),
                }
            out["paired_deltas"][f"{b}_minus_{a}"] = entry
    return out


def _print_report(res: dict, n_patches: int, n_debris: int, n_boot: int) -> None:
    models = [m for m in MODEL_ORDER if m in res["point_estimate"]]
    print(f"\n=== patch-level bootstrap ({n_boot:,} resamples of "
          f"{n_patches} patches; {n_debris} debris pixels) ===\n")
    # One row per model, one column per metric: the table has to stay readable
    # as models are added, and context (the row order) is the axis of interest.
    header = f"{'model':>20} | " + " | ".join(f"{k:>22}" for k in METRIC_KEYS)
    print(header)
    print("-" * len(header))
    for m in models:
        cells = []
        for k in METRIC_KEYS:
            lo, hi = res["ci95"][m][k]
            cells.append(f"{res['point_estimate'][m][k]:.3f} [{lo:.3f}, {hi:.3f}]")
        print(f"{MODEL_LABELS.get(m, m):>20} | " + " | ".join(f"{c:>22}" for c in cells))

    print("\n=== paired differences on the same resampled patches ===\n")
    head = (f"{'contrast':>22} | {'metric':>9} | {'delta':>7} | {'95% CI':>18} | "
            f"{'P(better)':>9}")
    print(head)
    print("-" * len(head))
    for contrast, entry in res["paired_deltas"].items():
        b, a = contrast.split("_minus_")
        label = f"{b} - {a}"
        for k in METRIC_KEYS:
            d = entry[k]
            lo, hi = d["ci95"]
            straddles = "" if (lo > 0 or hi < 0) else "   <- CI includes 0"
            print(f"{label:>22} | {k:>9} | {d['point']:>+7.3f} | "
                  f"[{lo:>+6.3f}, {hi:>+6.3f}] | {d['p_better']:>8.1%}{straddles}")
            label = ""
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="decision threshold for the U-Net and the CNNs; 0.5 "
                             "matches the reported test metrics.")
    parser.add_argument("--cnn-ckpt", nargs="*", default=DEFAULT_CNN_CKPTS,
                        help="window CNN checkpoints to include; missing files "
                             "are skipped, and an empty list reproduces the "
                             "original RF vs U-Net comparison.")
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "bootstrap_ci.json"))
    args = parser.parse_args()

    cnn_ckpts = [c for c in args.cnn_ckpt if c]
    counts, patch_ids, n_debris = collect_patch_counts(
        threshold=args.threshold, cnn_ckpts=cnn_ckpts)
    res = bootstrap(counts, n_boot=args.n_boot, seed=args.seed)
    _print_report(res, len(patch_ids), n_debris, args.n_boot)

    res.update({
        "n_patches": len(patch_ids),
        "n_debris_pixels": n_debris,
        "n_boot": args.n_boot,
        "bootstrap_seed": args.seed,
        "threshold": args.threshold,
        "models": [m for m in MODEL_ORDER if m in counts],
        "cnn_checkpoints": cnn_ckpts,
        "unit_of_resampling": "patch (pixels within a patch are not independent)",
        "note": "Evaluation uncertainty for the promoted checkpoints only; "
                "training/seed variance is reported separately by sweep_unet.py, "
                "sweep_rf.py and sweep_cnn.py.",
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nBootstrap results saved to: {args.out}")


if __name__ == "__main__":
    main()
