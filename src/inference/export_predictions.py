"""
Export per-patch Random Forest predictions as georeferenced GeoTIFFs.

Each output is a single-band ``uint8`` classification map written with the same
CRS and affine transform as its source patch, so it overlays the Sentinel-2
imagery correctly when dropped into QGIS. Pixel values:

    0 = non-debris
    1 = debris (positive class)
  255 = nodata (source label == 0, excluded from training/eval)

An embedded colormap renders debris in red and leaves everything else
transparent, so the raster is immediately readable on top of a basemap.

Run from the repository root:
    python -m src.inference.export_predictions --split test
    python -m src.inference.export_predictions --split test --out-dir test/data/outputs/predictions
"""

import argparse
import os
from pathlib import Path

import numpy as np
import rasterio

from src.data.dataset_loader import (
    _resolve_patch,
    load_patch,
    load_patch_profile,
)
from src.models.random_forest import load_model, predict

# Adjust these paths to match your local layout
PATCHES_ROOT = "data/raw/patches"
SPLITS_DIR = "data/raw/splits"
EXPERIMENTS_DIR = "experiments"
MODEL_PREFIX = os.path.join(EXPERIMENTS_DIR, "rf_baseline")
DEFAULT_OUT_DIR = os.path.join("test", "data", "outputs", "predictions")

NODATA = 255  # value written where the source label is nodata (0)

# QGIS-friendly colormap: transparent non-debris, opaque red debris.
COLORMAP = {
    0: (0, 0, 0, 0),
    1: (227, 26, 28, 255),
    NODATA: (0, 0, 0, 0),
}


def _read_patch_ids(split_file: str) -> list[str]:
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def write_prediction_geotiff(
    pred: np.ndarray,
    profile: dict,
    label: np.ndarray,
    out_path: str,
) -> None:
    """
    Write a single prediction mask to a georeferenced GeoTIFF.

    Args:
        pred: (H, W) binary prediction (0/1).
        profile: rasterio profile of the source patch (supplies CRS/transform).
        label: (H, W) source class mask; pixels == 0 are marked nodata (255).
        out_path: destination .tif path.
    """
    out = pred.astype(np.uint8)
    out[label == 0] = NODATA  # keep true nodata out of the visual

    profile = profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=NODATA,
        compress="lzw",
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.write_colormap(1, COLORMAP)


def export_split_predictions(
    patches_root: str,
    split_file: str,
    model_prefix: str,
    out_dir: str,
) -> int:
    """
    Run RF inference over every patch in a split and write GeoTIFFs.

    Returns the number of prediction rasters written.
    """
    os.makedirs(out_dir, exist_ok=True)
    clf, scaler = load_model(model_prefix)

    patch_ids = _read_patch_ids(split_file)
    written, missing = 0, 0

    for pid in patch_ids:
        scene_dir, full_pid = _resolve_patch(Path(patches_root), pid)
        if not (scene_dir / f"{full_pid}.tif").exists():
            missing += 1
            continue

        bands, label = load_patch(patches_root, pid)
        profile = load_patch_profile(patches_root, pid)
        pred = predict(clf, scaler, bands)

        out_path = os.path.join(out_dir, f"{full_pid}_pred.tif")
        write_prediction_geotiff(pred, profile, label, out_path)
        written += 1

    if missing:
        print(f"Warning: {missing}/{len(patch_ids)} patches not found under {patches_root}")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Which split to run inference over (default: test).",
    )
    parser.add_argument(
        "--patches-root",
        default=PATCHES_ROOT,
        help=f"Path to the patches directory (default: {PATCHES_ROOT}).",
    )
    parser.add_argument(
        "--model-prefix",
        default=MODEL_PREFIX,
        help=f"Trained model path prefix (default: {MODEL_PREFIX}).",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Where to write prediction GeoTIFFs (default: {DEFAULT_OUT_DIR}).",
    )
    args = parser.parse_args()

    split_file = os.path.join(SPLITS_DIR, f"{args.split}_X.txt")

    print(f"Loading model from: {args.model_prefix}")
    print(f"Exporting '{args.split}' predictions to: {args.out_dir}")
    written = export_split_predictions(
        args.patches_root, split_file, args.model_prefix, args.out_dir
    )
    print(f"Done. Wrote {written} prediction GeoTIFF(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
