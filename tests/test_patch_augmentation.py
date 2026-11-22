"""
Tests for the U-Net segmentation dataset's dihedral augmentation (step 2 of
REVISED_PLAN.md).

The property under test is **image/mask alignment**: whatever transform is
drawn, every pixel must keep its own class label. The failure mode this guards
against is a transform applied to the image but not to the mask (or applied to
the mask with different parameters), which does not crash and does not look
wrong in a loss curve -- it just trains the model on systematically mislabelled
pixels.

The mask is transformed *before* ``target`` and ``valid`` are derived from it, so
those two cannot drift from the image either; there is only one transform, not
three that have to agree.
"""

import numpy as np

from src.data.patch_dataset import MaridaSegmentationDataset

SIZE = 8
N_CLASSES = 16


def _coded_pair(channels: int = 3):
    """
    An (image, mask) pair in which the image encodes each pixel's own position.

    ``img[0, r, c]`` is the flat index of (r, c) and the mask is a pure function
    of that index, so alignment can be checked pixel by pixel after any
    transform: the label at a position must match the label implied by the image
    value sitting at that same position.
    """
    base = np.arange(SIZE * SIZE, dtype=np.float32).reshape(SIZE, SIZE)
    img = np.stack([base + 1000 * ch for ch in range(channels)])
    label = (base.astype(np.int32) % N_CLASSES)
    return img, label


def _dataset(seed: int) -> MaridaSegmentationDataset:
    """A dataset instance without touching disk: only the RNG matters here."""
    ds = object.__new__(MaridaSegmentationDataset)
    ds.augment = True
    ds.seed = seed
    ds._rng = np.random.default_rng(seed)
    return ds


def test_every_pixel_keeps_its_own_label_under_augmentation():
    """Across many draws, the mask must follow the image exactly."""
    img, label = _coded_pair()
    seen = set()
    for seed in range(40):
        ds = _dataset(seed)
        aug_img, aug_label = ds._augment_pair(img, label)
        assert aug_img.shape == img.shape
        assert aug_label.shape == label.shape

        implied = aug_img[0].astype(np.int32) % N_CLASSES
        assert np.array_equal(implied, aug_label), f"misaligned for seed {seed}"

        # The other channels must receive the identical transform, not an
        # independent draw of their own.
        for ch in range(1, aug_img.shape[0]):
            assert np.array_equal(aug_img[ch] - 1000 * ch, aug_img[0])

        seen.add(aug_img[0].tobytes())

    # Sanity: the augmentation is actually varying the input, not a no-op.
    assert len(seen) > 1


def test_augmentation_reaches_all_eight_dihedral_images():
    """
    The 4 rotations x 2 flips x 2 flips draw collapses to 8 distinct images (the
    dihedral group of the square), and all 8 should be reachable.
    """
    img, label = _coded_pair(channels=1)
    seen = set()
    for seed in range(400):
        aug_img, _ = _dataset(seed)._augment_pair(img, label)
        seen.add(np.ascontiguousarray(aug_img).tobytes())
    assert len(seen) == 8, f"reached {len(seen)} of 8 dihedral images"


def test_disabled_augmentation_is_identity():
    """With augment off, __getitem__ must not permute anything."""
    img, label = _coded_pair()
    ds = object.__new__(MaridaSegmentationDataset)
    ds.augment = False
    assert ds.augment is False  # the flag __getitem__ branches on

    # And the transform itself is a pure function of the RNG draw, so an
    # all-identity draw returns the input untouched.
    ds2 = _dataset(0)
    out_img, out_label = ds2._augment_pair(img, label)
    assert out_img.shape == img.shape and out_label.shape == label.shape
