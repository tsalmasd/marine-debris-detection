"""
Export per-patch U-Net predictions as georeferenced GeoTIFFs for QGIS.

Dense sibling of ``src.inference.export_predictions`` (the RF exporter): loads the
trained U-Net checkpoint, runs one forward pass per patch, and writes a single-band
``uint8`` classification map carrying the source patch's CRS + transform. Output
values and colormap match the RF exporter (0 = non-debris, 1 = debris, 255 = nodata),
and it reuses the same ``write_prediction_geotiff`` writer.

Run from the repository root:
    python -m src.inference.export_predictions_unet --split test
    python -m src.inference.export_predictions_unet --split val --out-dir test/data/outputs/predictions_unet
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from src.data.dataset_loader import _resolve_patch, load_patch, load_patch_profile
from src.inference.export_predictions import write_prediction_geotiff
from src.models.unet import load_checkpoint

PATCHES_ROOT = "data/raw/patches"
SPLITS_DIR = "data/raw/splits"
MODEL_DIR = "test/data/model"
MODEL_PATH = os.path.join(MODEL_DIR, "unet_baseline.pt")
DEFAULT_OUT_DIR = os.path.join("test", "data", "outputs", "predictions_unet")


def _read_patch_ids(split_file: str) -> list[str]:
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


@torch.no_grad()
def _predict_patch(model, img: np.ndarray, mean, std, device: str, threshold: float = 0.5) -> np.ndarray:
    """(C,H,W) float32 patch -> (H,W) uint8 binary mask, normalized as in training."""
    norm = (img - np.asarray(mean, dtype=np.float32)[:, None, None]) / np.asarray(std, dtype=np.float32)[:, None, None]
    # Match the training-time sanitization: NaN/inf pixels -> 0 (channel mean).
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.from_numpy(np.ascontiguousarray(norm)).unsqueeze(0).to(device)  # (1,C,H,W)
    with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
        logits = model(x)
    prob = torch.sigmoid(logits.squeeze().float()).cpu().numpy()  # (H,W)
    return (prob >= threshold).astype(np.uint8)


def export_split_predictions(patches_root, split_file, model_path, out_dir, device) -> int:
    os.makedirs(out_dir, exist_ok=True)
    model, mean, std = load_checkpoint(model_path, device=device)

    patch_ids = _read_patch_ids(split_file)
    written, missing = 0, 0
    for pid in patch_ids:
        scene_dir, full_pid = _resolve_patch(Path(patches_root), pid)
        if not (scene_dir / f"{full_pid}.tif").exists():
            missing += 1
            continue

        img, label = load_patch(patches_root, pid)
        profile = load_patch_profile(patches_root, pid)
        pred = _predict_patch(model, img.astype(np.float32), mean, std, device)

        out_path = os.path.join(out_dir, f"{full_pid}_pred.tif")
        write_prediction_geotiff(pred, profile, label, out_path)
        written += 1

    if missing:
        print(f"Warning: {missing}/{len(patch_ids)} patches not found under {patches_root}")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--patches-root", default=PATCHES_ROOT)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split_file = os.path.join(SPLITS_DIR, f"{args.split}_X.txt")

    print(f"Device: {device}")
    print(f"Loading model from: {args.model_path}")
    print(f"Exporting '{args.split}' predictions to: {args.out_dir}")
    written = export_split_predictions(
        args.patches_root, split_file, args.model_path, args.out_dir, device
    )
    print(f"Done. Wrote {written} prediction GeoTIFF(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
