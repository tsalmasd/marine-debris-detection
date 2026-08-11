"""
PyTorch dataset for U-Net semantic segmentation on MARIDA patches.

Wraps the same MARIDA patches / split files used by the Random Forest baseline,
but yields dense per-pixel tensors instead of flattened pixel samples:

    image  : (C, H, W) float32, per-channel normalized Sentinel-2 bands
    target : (H, W)    float32, binary debris (1) / non-debris (0)
    valid  : (H, W)    float32, 1 where the pixel is labelled, 0 for nodata

Binary task, identical semantics to the RF baseline:
    class 1        -> debris   (target 1)
    classes 2..15  -> non-debris (target 0)
    class 0        -> nodata    (excluded via ``valid`` mask, never in the loss)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset_loader import _resolve_patch, load_patch
from pathlib import Path

DEBRIS_CLASS = 1  # positive class in MARIDA (Marine Debris)


def _read_patch_ids(split_file: str) -> list[str]:
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def compute_normalization_stats(
    patches_root: str,
    split_file: str,
    bands: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-channel mean/std over every valid pixel of a split (typically train).

    Returned as float32 arrays of shape (C,). Nodata pixels (label == 0) are
    excluded so the statistics reflect the real spectral distribution.
    """
    ids = _read_patch_ids(split_file)
    sums = None
    sq_sums = None
    count = 0
    for pid in ids:
        scene_dir, full_pid = _resolve_patch(Path(patches_root), pid)
        if not (scene_dir / f"{full_pid}.tif").exists():
            continue
        img, label = load_patch(patches_root, pid, bands=bands)  # (C,H,W), (H,W)
        valid = label.reshape(-1) > 0
        flat = img.reshape(img.shape[0], -1)[:, valid]  # (C, n_valid)
        # Exclude any non-finite pixels so a few NaN patches don't poison the stats.
        flat = flat[:, np.isfinite(flat).all(axis=0)]
        if flat.shape[1] == 0:
            continue
        if sums is None:
            sums = np.zeros(img.shape[0], dtype=np.float64)
            sq_sums = np.zeros(img.shape[0], dtype=np.float64)
        sums += flat.sum(axis=1)
        sq_sums += (flat.astype(np.float64) ** 2).sum(axis=1)
        count += flat.shape[1]

    if count == 0:
        raise RuntimeError("No valid pixels found to compute normalization stats.")

    mean = sums / count
    var = np.maximum(sq_sums / count - mean**2, 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


class MaridaSegmentationDataset(Dataset):
    """
    Dense patch dataset for U-Net training / evaluation.

    Args:
        patches_root: path to data/raw/patches.
        split_file: path to a split list (train_X.txt / val_X.txt / test_X.txt).
        bands: band names to read (defaults to the RF-baseline 6-band subset).
        mean, std: per-channel normalization stats (shape (C,)). Required for
            training/eval so inputs match what the model was trained on; compute
            them once on the train split with :func:`compute_normalization_stats`.
        cache: preload every patch into RAM in ``__init__`` (fast epochs; the
            MARIDA splits are small enough to fit comfortably).

    Missing patches listed in the split file are skipped (with a warning),
    mirroring :func:`src.data.dataset_loader.load_split`.
    """

    def __init__(
        self,
        patches_root: str,
        split_file: str,
        bands: list[str] | None = None,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        cache: bool = True,
    ):
        self.patches_root = patches_root
        self.bands = bands
        self.cache = cache

        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)[:, None, None]
        self.std = None if std is None else np.asarray(std, dtype=np.float32)[:, None, None]

        ids = _read_patch_ids(split_file)
        self.patch_ids: list[str] = []
        missing = 0
        for pid in ids:
            scene_dir, full_pid = _resolve_patch(Path(patches_root), pid)
            if not (scene_dir / f"{full_pid}.tif").exists():
                missing += 1
                continue
            self.patch_ids.append(pid)
        if missing:
            print(f"Warning: {missing}/{len(ids)} patches not found under {patches_root}")

        self._cache: dict[int, tuple] = {}
        if cache:
            for i in range(len(self.patch_ids)):
                self._cache[i] = self._load_raw(i)

    def __len__(self) -> int:
        return len(self.patch_ids)

    def _load_raw(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        img, label = load_patch(self.patches_root, self.patch_ids[idx], bands=self.bands)
        return img.astype(np.float32), label.astype(np.int32)

    def __getitem__(self, idx: int):
        img, label = self._cache[idx] if self.cache else self._load_raw(idx)

        if self.mean is not None:
            img = (img - self.mean) / self.std

        # A few MARIDA patches carry NaN/inf pixels (sensor/nodata artifacts).
        # A single non-finite pixel propagates through the convolutions and
        # poisons the entire output (NaN loss), so replace them with 0 (the
        # per-channel mean after normalization). Loss is masked by ``valid``
        # anyway, so these pixels never contribute a gradient directly.
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        target = (label == DEBRIS_CLASS).astype(np.float32)  # (H, W)
        valid = (label > 0).astype(np.float32)               # (H, W)

        return (
            torch.from_numpy(np.ascontiguousarray(img)),      # (C, H, W)
            torch.from_numpy(target),                         # (H, W)
            torch.from_numpy(valid),                          # (H, W)
        )

    def compute_pos_weight(self) -> float:
        """
        BCE ``pos_weight`` = (# non-debris valid pixels) / (# debris valid pixels),
        computed over the whole split. Counteracts the heavy class imbalance.
        """
        n_pos, n_neg = 0, 0
        for i in range(len(self)):
            _, label = self._cache[i] if self.cache else self._load_raw(i)
            valid = label > 0
            pos = int(((label == DEBRIS_CLASS) & valid).sum())
            n_pos += pos
            n_neg += int(valid.sum()) - pos
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos
