"""
MARIDA dataset loader.

Loads Sentinel-2 multispectral patches and their class-label masks from the
MARIDA benchmark dataset as published on Zenodo.

Layout on disk:
    data/raw/patches/
        S2_<DATE>_<TILE>/                           # one folder per scene
            S2_<DATE>_<TILE>_<CROP>.tif             # 256x256, 11 bands stacked
            S2_<DATE>_<TILE>_<CROP>_cl.tif          # class mask (0..15)
            S2_<DATE>_<TILE>_<CROP>_conf.tif        # confidence mask
    data/raw/splits/
        train_X.txt / val_X.txt / test_X.txt        # one patch ID per line
                                                    # IDs omit the "S2_" prefix
"""

from pathlib import Path

import numpy as np
import rasterio

# Band order inside MARIDA's stacked GeoTIFF (1-indexed, as rasterio reads).
# Source: MARIDA paper / repo. B09 and B10 are not included.
MARIDA_BAND_ORDER = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12",
]
MARIDA_BAND_INDEX = {name: i + 1 for i, name in enumerate(MARIDA_BAND_ORDER)}

# Default 6-band subset used by the RF baseline and spectral-index pipeline.
DEFAULT_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def _resolve_patch(patches_root: Path, patch_id: str) -> tuple[Path, str]:
    """
    Resolve a split-file patch_id to (scene_dir, full_pid).

    Split files list IDs like "1-12-19_48MYU_0" (no S2_ prefix), but the
    on-disk folders and files use "S2_1-12-19_48MYU_0".
    """
    full_pid = patch_id if patch_id.startswith("S2_") else f"S2_{patch_id}"
    scene = full_pid.rsplit("_", 1)[0]  # strip trailing "_<CROP>"
    return Path(patches_root) / scene, full_pid


def load_patch(
    patches_root: str,
    patch_id: str,
    bands: list[str] | None = None,
    return_conf: bool = False,
):
    """
    Load a MARIDA patch: multi-band image + class mask.

    Args:
        patches_root: path to data/raw/patches.
        patch_id: ID as written in a split file (e.g. "1-12-19_48MYU_0").
            The "S2_" prefix is added automatically if missing.
        bands: band names to read (e.g. ["B02", "B08"]).
            Defaults to the 6-band subset used by the RF baseline.
            Pass MARIDA_BAND_ORDER to return all 11 bands.
        return_conf: also read and return the confidence mask.

    Returns:
        arr:   np.ndarray of shape (C, H, W), float32
        label: np.ndarray of shape (H, W), int32, MARIDA class IDs (0..15)
        conf:  optional np.ndarray (H, W), float32, if return_conf=True
    """
    scene_dir, full_pid = _resolve_patch(Path(patches_root), patch_id)
    img_path  = scene_dir / f"{full_pid}.tif"
    cl_path   = scene_dir / f"{full_pid}_cl.tif"
    conf_path = scene_dir / f"{full_pid}_conf.tif"

    if bands is None:
        bands = DEFAULT_BANDS
    try:
        band_idxs = [MARIDA_BAND_INDEX[b] for b in bands]
    except KeyError as e:
        raise ValueError(f"Unknown band {e}; valid: {MARIDA_BAND_ORDER}")

    with rasterio.open(img_path) as src:
        arr = src.read(band_idxs).astype(np.float32)  # (C, H, W)

    with rasterio.open(cl_path) as src:
        label = src.read(1).astype(np.int32)  # (H, W)

    if not return_conf:
        return arr, label

    with rasterio.open(conf_path) as src:
        conf = src.read(1).astype(np.float32)
    return arr, label, conf


def load_split(
    patches_root: str,
    split_file: str,
    bands: list[str] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Load every patch listed in a MARIDA split file (train_X / val_X / test_X).

    Returns parallel lists of (C, H, W) image arrays and (H, W) label masks.
    """
    with open(split_file, "r") as f:
        patch_ids = [line.strip() for line in f if line.strip()]

    all_bands, all_labels = [], []
    missing = 0
    for pid in patch_ids:
        scene_dir, full_pid = _resolve_patch(Path(patches_root), pid)
        if not (scene_dir / f"{full_pid}.tif").exists():
            missing += 1
            continue
        arr, label = load_patch(patches_root, pid, bands=bands)
        all_bands.append(arr)
        all_labels.append(label)

    if missing:
        print(f"Warning: {missing}/{len(patch_ids)} patches not found under {patches_root}")
    return all_bands, all_labels
