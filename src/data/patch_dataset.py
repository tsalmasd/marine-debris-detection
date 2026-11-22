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
        augment: dihedral augmentation (k*90-degree rotations and the two
            flips), applied identically to the image and the class mask.
            **Train split only** -- on val/test it would make the metrics
            non-deterministic. Geometry only, no photometric jitter: the pixel
            values are reflectances.
        seed: seed for this dataset's augmentation RNG.

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
        augment: bool = False,
        seed: int = 0,
    ):
        self.patches_root = patches_root
        self.bands = bands
        self.cache = cache
        self.augment = augment
        self.seed = seed
        self._rng: np.random.Generator | None = None

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

    def _rng_for_worker(self) -> np.random.Generator:
        """
        Augmentation RNG, created lazily so each DataLoader worker gets its own.

        A ``Generator`` built in ``__init__`` would be *copied* into every worker
        process, so all of them would replay an identical augmentation stream and
        the effective diversity would drop by a factor of ``num_workers``.
        ``torch.initial_seed()`` is already distinct per worker (PyTorch derives
        it as base seed + worker id), so seeding from it gives each worker an
        independent stream while staying reproducible. Same approach as
        :class:`src.data.window_dataset.MaridaWindowDataset`.
        """
        if self._rng is None:
            self._rng = np.random.default_rng(
                (int(torch.initial_seed()) + self.seed) % (2 ** 32)
            )
        return self._rng

    def _augment_pair(self, img: np.ndarray, label: np.ndarray):
        """
        Dihedral augmentation of one (image, class mask) pair.

        Rotations by k*90 degrees and the two flips, applied with **identical
        parameters** to the image and to the raw class mask -- and to the mask
        *before* ``target`` and ``valid`` are derived from it, so all three stay
        aligned by construction rather than by three separate transforms
        agreeing. Geometry only: no brightness or colour jitter, because
        reflectance values carry physical meaning here.

        Unlike the window dataset there is no centre-pixel subtlety: the whole
        mask moves with the whole image, so every pixel keeps its own label
        wherever it lands.
        """
        rng = self._rng_for_worker()
        k = int(rng.integers(4))
        if k:
            img = np.rot90(img, k, axes=(1, 2))
            label = np.rot90(label, k, axes=(0, 1))
        if rng.random() < 0.5:
            img, label = img[:, :, ::-1], label[:, ::-1]
        if rng.random() < 0.5:
            img, label = img[:, ::-1, :], label[::-1, :]
        return img, label

    def __getitem__(self, idx: int):
        img, label = self._cache[idx] if self.cache else self._load_raw(idx)

        # Augment before normalization and before target/valid are derived. The
        # transforms are geometric, so the order relative to normalization does
        # not matter numerically -- but doing it here keeps the cached arrays
        # untouched, which matters because the cache is shared across epochs.
        if self.augment:
            img, label = self._augment_pair(img, label)

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
            torch.from_numpy(np.ascontiguousarray(target)),   # (H, W)
            torch.from_numpy(np.ascontiguousarray(valid)),    # (H, W)
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


class MaridaPatchClassificationDataset(MaridaSegmentationDataset):
    """
    Patch-level dataset for the custom CNN classifier.

    Reuses everything from :class:`MaridaSegmentationDataset` — patch
    resolution, caching, normalization, NaN sanitation, missing-patch skipping
    — but collapses the per-pixel mask into one scalar label per patch:

        image : (C, H, W) float32, per-channel normalized Sentinel-2 bands
        label : () float32, 1.0 if the patch contains >= 1 debris pixel else 0.0

    What the negative class actually means
    -------------------------------------
    The label is ``(mask == 1).any()``, which reads as "is any pixel in this
    patch *annotated* as debris". It is not "this patch contains no debris":
    MARIDA annotates a median of 0.10% of pixels per patch (as little as a
    single pixel in the thinnest case), so a 0 means *no debris among the
    labelled pixels*, with the other 99.9% unchecked. Every negative therefore
    carries some chance of unannotated debris, which puts a ceiling on
    achievable precision that no model can see. State this as a limitation
    rather than treating patch labels as ground truth about whole patches.

    No patch in any split is fully unlabelled, so none has to be dropped.
    """

    def __getitem__(self, idx: int):
        img, label = self._cache[idx] if self.cache else self._load_raw(idx)

        if self.mean is not None:
            img = (img - self.mean) / self.std
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        has_debris = np.float32((label == DEBRIS_CLASS).any())

        return (
            torch.from_numpy(np.ascontiguousarray(img)),  # (C, H, W)
            torch.tensor(has_debris),                     # scalar
        )

    def patch_labels(self) -> np.ndarray:
        """
        Every patch label as a (N,) float32 array, without a forward pass.

        Used for the positive-class weight, for stratified reporting and to
        line predictions up against truth in evaluation.
        """
        out = np.zeros(len(self), dtype=np.float32)
        for i in range(len(self)):
            _, label = self._cache[i] if self.cache else self._load_raw(i)
            out[i] = float((label == DEBRIS_CLASS).any())
        return out

    def compute_pos_weight(self) -> float:
        """
        BCE ``pos_weight`` = (# clean patches) / (# debris patches) for this split.

        Counted in patches, not pixels — this deliberately shadows the parent's
        pixel-level version, which would return the 197:1 figure and is the
        wrong quantity for a patch classifier. Measures 2.65 on the train split.
        """
        labels = self.patch_labels()
        n_pos = int(labels.sum())
        if n_pos == 0:
            return 1.0
        return (len(labels) - n_pos) / n_pos
