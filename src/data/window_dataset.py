"""
Window dataset for the custom CNN centre-pixel classifier (REVISED_PLAN.md step 3).

Yields one small window per *annotated pixel*, labelled with that pixel's class:

    window : (C, W, W) float32, per-channel normalized Sentinel-2 bands
    target : ()       float32, 1.0 if the centre pixel is debris else 0.0

This is what makes the three-model comparison fair. The Random Forest sees one
pixel, this model sees a W x W neighbourhood, the U-Net sees the whole 256x256
patch — and all three are scored on the *same* labelled pixels, because every
one of them predicts per-pixel. Varying W (16, 32) traces a context curve.

Binary task, identical semantics to the other two models:
    class 1        -> debris     (target 1)
    classes 2..15  -> non-debris (target 0)
    class 0        -> nodata     (no window is emitted for it at all)

Scale
-----
One window per labelled pixel, counted on disk:

    split   windows    debris   neg:pos
    train   429,412     1,943   220.0:1
    val     213,102     1,075   197.2:1
    test    194,863       381   510.5:1

So the pixel-level imbalance is back — the 2.65:1 figure in ``custom_cnn.py``
belongs to the superseded whole-patch formulation and does not apply here. Use
``pos_weight = min(raw, 20)``, the same cap and the same reasoning as the U-Net.

Caching 429k windows densely would cost 3.1 GB at W=16 and 12.3 GB at W=32, so
the patches are cached once (reflect-padded) and windows are cut on demand. A
window is a *view* into that cache until the DataLoader collates it, which keeps
extraction to under a microsecond.

Augmentation, and a correction to the plan
------------------------------------------
``REVISED_PLAN.md`` step 3 says the flip/rot90 augmentation "keeps the centre
pixel fixed". **That is only true for odd W.** An even window has no centre
pixel: for W=16 the target sits at offset 8, and a flip sends offset 8 to
offset 15-8 = 7. Augmenting the cropped window would therefore relabel every
augmented sample with its neighbour's class — and since debris objects are often
1-3 pixels wide, that is real label noise, not a rounding detail.

The fix keeps the pre-registered W = 16 and 32 *and* the pre-registered
augmentation: transform the **padded patch** (cheap, these are stride views),
map the target's coordinate through the same transform, and only then crop, so
the target always lands at offset ``W // 2``. See :func:`_augment_and_crop`,
which is verified by ``tests/test_window_dataset.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset_loader import RF_BANDS, _resolve_patch, load_patch

DEBRIS_CLASS = 1  # positive class in MARIDA (Marine Debris)


def _read_patch_ids(split_file: str) -> list[str]:
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def crop_window(padded: np.ndarray, r: int, c: int, window: int) -> np.ndarray:
    """
    The ``window``-sided crop of ``padded`` centred on padded coordinate (r, c).

    Single definition of "centred", used by the dataset, by the augmentation
    path and by the bootstrap, so the three cannot drift apart: the target pixel
    always lands at offset ``window // 2``.
    """
    half = window // 2
    return padded[:, r - half: r - half + window, c - half: c - half + window]


def prepare_padded_patch(
    img: np.ndarray,
    window: int,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """
    Normalize, sanitize and reflect-pad one patch so windows can be cut from it.

    Three steps, in this order:

    1. per-channel normalization with the *training* stats;
    2. ``nan_to_num`` -- same reasoning as the segmentation dataset, a single
       non-finite pixel propagates through the convolutions and poisons the
       output (MARIDA has a few such patches);
    3. reflect-pad by ``window // 2`` so windows centred on a border pixel still
       exist and the centre pixel always comes from real data.

    Shared with :mod:`src.validation.bootstrap` so a model scored there sees
    pixels preprocessed exactly as they were in training.
    """
    img = img.astype(np.float32)
    if mean is not None:
        mean_arr = np.asarray(mean, dtype=np.float32)[:, None, None]
        std_arr = np.asarray(std, dtype=np.float32)[:, None, None]
        img = (img - mean_arr) / std_arr
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    half = window // 2
    return np.pad(img, ((0, 0), (half, half), (half, half)), mode="reflect")


def extract_windows(
    padded: np.ndarray, rows: np.ndarray, cols: np.ndarray, window: int
) -> np.ndarray:
    """Stack the windows centred on each padded coordinate -> (N, C, W, W)."""
    out = np.empty((len(rows), padded.shape[0], window, window), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        out[i] = crop_window(padded, int(r), int(c), window)
    return out


def _augment_and_crop(
    padded: np.ndarray,
    r: int,
    c: int,
    window: int,
    k: int,
    flip_v: bool,
    flip_h: bool,
) -> np.ndarray:
    """
    Cut a ``window`` x ``window`` crop around a target pixel, under a dihedral transform.

    Args:
        padded: (C, P, P) reflect-padded patch.
        r, c: the target pixel's coordinates **in padded space**.
        window: W, the output side length.
        k: number of 90-degree rotations (0..3).
        flip_v, flip_h: whether to flip vertically / horizontally.

    The transform is applied to the whole padded patch and the target coordinate
    is carried through it, so the returned crop always has the target at offset
    ``(W // 2, W // 2)`` — which is what the label refers to. Doing it the other
    way round (crop, then flip the crop) moves the target off centre whenever W
    is even; see the module docstring.

    ``np.rot90`` and reverse slicing both return views, so this is cheap; the
    single copy happens in the caller via ``np.ascontiguousarray``.
    """
    size = padded.shape[1]  # padded patches are square
    for _ in range(k):
        # np.rot90 on (H, W) gives out[i, j] = in[j, N-1-i], so (r, c) -> (N-1-c, r).
        padded = np.rot90(padded, 1, axes=(1, 2))
        r, c = size - 1 - c, r
    if flip_v:
        padded = padded[:, ::-1, :]
        r = size - 1 - r
    if flip_h:
        padded = padded[:, :, ::-1]
        c = size - 1 - c

    return crop_window(padded, r, c, window)


def compute_normalization_stats(
    patches_root: str,
    split_file: str,
    bands: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-channel mean/std over every valid pixel of a split (typically train).

    Thin wrapper over the segmentation dataset's version so the window model and
    the U-Net normalize identically when given the same band list — the whole
    point of the comparison is that only context and features differ.
    """
    from src.data.patch_dataset import compute_normalization_stats as _stats

    return _stats(patches_root, split_file, bands=bands)


class MaridaWindowDataset(Dataset):
    """
    One W x W window per annotated pixel, labelled with the centre pixel's class.

    Args:
        patches_root: path to data/raw/patches.
        split_file: path to a split list (train_X.txt / val_X.txt / test_X.txt).
        window: W, the window side length (16 and 32 are the pre-registered sizes).
        bands: band names to read. Defaults to ``RF_BANDS`` (7 bands, including
            B06) so the model sees the same spectra as the Random Forest.
        mean, std: per-channel normalization stats, shape (C,). Compute once on
            the train split with :func:`compute_normalization_stats`.
        augment: random flips and 90-degree rotations. **Train split only** —
            passing it for val/test would make the metrics non-deterministic.
        seed: seed for this dataset's augmentation RNG.

    Missing patches listed in the split file are skipped (with a warning),
    mirroring :class:`src.data.patch_dataset.MaridaSegmentationDataset`.
    """

    def __init__(
        self,
        patches_root: str,
        split_file: str,
        window: int = 16,
        bands: list[str] | None = None,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        augment: bool = False,
        seed: int = 0,
    ):
        if window < 4:
            raise ValueError(f"window must be >= 4, got {window}")
        self.patches_root = patches_root
        self.window = window
        self.bands = list(bands) if bands is not None else list(RF_BANDS)
        self.augment = augment
        self.half = window // 2

        # Catch a mismatched stats/bands pair here rather than as an opaque
        # broadcast error 200 patches into loading: the stats must have been
        # computed with the same band list this dataset is about to read.
        for name, arr in (("mean", mean), ("std", std)):
            if arr is not None and len(np.asarray(arr)) != len(self.bands):
                raise ValueError(
                    f"{name} has {len(np.asarray(arr))} channels but bands has "
                    f"{len(self.bands)} ({self.bands}). Compute the normalization "
                    f"stats with the same bands you pass here."
                )

        ids = _read_patch_ids(split_file)
        self.patch_ids: list[str] = []
        self._patches: list[np.ndarray] = []   # reflect-padded, normalized
        index: list[tuple[int, int, int]] = []  # (patch_idx, row, col) in PADDED coords
        targets: list[np.uint8] = []
        missing = 0

        for pid in ids:
            scene_dir, full_pid = _resolve_patch(Path(patches_root), pid)
            if not (scene_dir / f"{full_pid}.tif").exists():
                missing += 1
                continue
            img, label = load_patch(patches_root, pid, bands=self.bands)
            padded = prepare_padded_patch(img, window, mean, std)
            pi = len(self._patches)
            self._patches.append(padded)
            self.patch_ids.append(pid)

            rows, cols = np.nonzero(label > 0)  # labelled pixels only
            for r, c in zip(rows, cols):
                index.append((pi, int(r) + self.half, int(c) + self.half))
            targets.append((label[rows, cols] == DEBRIS_CLASS).astype(np.uint8))

        if missing:
            print(f"Warning: {missing}/{len(ids)} patches not found under {patches_root}")
        if not index:
            raise RuntimeError(f"No labelled pixels found for split {split_file}")

        self.index = np.asarray(index, dtype=np.int32)
        self.targets = np.concatenate(targets).astype(np.float32)
        self.seed = seed
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.index)

    def patch_index(self) -> np.ndarray:
        """
        The source patch index of every window, shape (N,), as int32.

        Lets the patch-level bootstrap resample *patches* rather than pixels,
        the same unit ``src/validation/bootstrap.py`` already uses, so the CNN
        can join the existing paired comparison.
        """
        return self.index[:, 0].copy()

    def _rng_for_worker(self) -> np.random.Generator:
        """
        Augmentation RNG, created lazily so each DataLoader worker gets its own.

        A ``Generator`` built in ``__init__`` would be *copied* into every worker
        process, so all of them would replay an identical augmentation stream and
        the effective augmentation diversity would drop by a factor of
        ``num_workers``. ``torch.initial_seed()`` is already distinct per worker
        (PyTorch derives it as base seed + worker id), so seeding from it gives
        each worker an independent stream while staying reproducible.
        """
        if self._rng is None:
            self._rng = np.random.default_rng(
                (int(torch.initial_seed()) + self.seed) % (2 ** 32)
            )
        return self._rng

    def __getitem__(self, idx: int):
        pi, r, c = self.index[idx]
        padded = self._patches[pi]

        if self.augment:
            rng = self._rng_for_worker()
            crop = _augment_and_crop(
                padded, int(r), int(c), self.window,
                int(rng.integers(4)), bool(rng.random() < 0.5), bool(rng.random() < 0.5),
            )
        else:
            crop = crop_window(padded, int(r), int(c), self.window)

        return (
            torch.from_numpy(np.ascontiguousarray(crop)),  # (C, W, W)
            torch.tensor(self.targets[idx]),               # scalar
        )

    def compute_pos_weight(self, cap: float | None = 20.0) -> float:
        """
        BCE ``pos_weight`` = (# non-debris windows) / (# debris windows).

        Capped at ``cap`` by default, exactly as ``train_unet`` does: the raw
        ratio is ~220 on the train split, which destabilizes training. Pass
        ``cap=None`` for the uncapped value.
        """
        n_pos = float(self.targets.sum())
        if n_pos == 0:
            return 1.0
        raw = (len(self.targets) - n_pos) / n_pos
        return raw if cap is None else min(raw, cap)


if __name__ == "__main__":
    # Smoke check against the real split: shapes, label alignment, augmentation.
    root, split = "data/raw/patches", "data/raw/splits/val_X.txt"
    ds = MaridaWindowDataset(root, split, window=16)
    print(f"windows: {len(ds):,} | bands: {ds.bands} | "
          f"debris: {int(ds.targets.sum()):,} | pos_weight: {ds.compute_pos_weight():.2f}")
    x, y = ds[0]
    print(f"window {tuple(x.shape)} {x.dtype} | target {y.item()}")
    assert x.shape == (7, 16, 16)

    # The un-augmented crop must carry the target pixel at its centre offset.
    pi, r, c = ds.index[5]
    assert ds[5][0][0, 8, 8].item() == ds._patches[pi][0, r, c]

    # Augmented crops must keep that property for every transform.
    aug = MaridaWindowDataset(root, split, window=16, augment=True, seed=1)
    for i in (0, 1, 2, 50, 500):
        pi, r, c = aug.index[i]
        got = aug[i][0][0, 8, 8].item()
        assert got == aug._patches[pi][0, r, c], (i, got)
    print("centre-pixel alignment holds with and without augmentation -> OK")
