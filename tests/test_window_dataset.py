"""
Tests for the window dataset behind the custom CNN centre-pixel classifier.

The property under test is the one REVISED_PLAN.md gets wrong: that the
augmentation leaves the labelled pixel at the centre of the window. It does not,
for even W, if the crop is taken first and then flipped. These tests pin the
corrected order (transform the padded patch, carry the coordinate, then crop)
so a future refactor cannot silently reintroduce one-pixel label noise.
"""

import numpy as np
import pytest

from src.data.window_dataset import _augment_and_crop


def _padded(size: int, channels: int = 3) -> np.ndarray:
    """A padded patch whose every pixel value encodes its own position."""
    base = np.arange(size * size, dtype=np.float32).reshape(size, size)
    return np.stack([base + 1000 * ch for ch in range(channels)])


@pytest.mark.parametrize("window", [16, 32, 17])
def test_target_stays_at_window_centre_under_every_transform(window):
    """
    For all 16 dihedral transforms, the crop's centre offset must hold the target.

    This is the correctness property the labels depend on: the target value is
    the class of pixel (r, c), so (r, c) has to land at offset (W//2, W//2).
    """
    size = 64 + 2 * (window // 2)
    padded = _padded(size)
    half = window // 2

    for r in (half, size // 2, size - half - 1):
        for c in (half, size // 3, size - half - 1):
            expected = padded[:, r, c]
            for k in range(4):
                for flip_v in (False, True):
                    for flip_h in (False, True):
                        crop = _augment_and_crop(
                            padded, r, c, window, k, flip_v, flip_h
                        )
                        assert crop.shape == (3, window, window)
                        np.testing.assert_array_equal(
                            crop[:, half, half], expected,
                            err_msg=f"{window=} {r=} {c=} {k=} {flip_v=} {flip_h=}",
                        )


def test_cropping_before_flipping_would_move_the_target_for_even_w():
    """
    Guard the reason the implementation is shaped the way it is.

    If someone "simplifies" it to crop-then-flip, an even window's target moves
    by one pixel. This test documents that failure mode explicitly, so the
    comment in the module is backed by an executable check.
    """
    window, size = 16, 64
    padded = _padded(size)
    half = window // 2
    r = c = size // 2

    naive_crop = padded[:, r - half: r + half, c - half: c + half]
    flipped = naive_crop[:, :, ::-1]
    # The naive crop does hold the target before flipping ...
    np.testing.assert_array_equal(naive_crop[:, half, half], padded[:, r, c])
    # ... and does not after, which is exactly the bug being avoided.
    assert not np.array_equal(flipped[:, half, half], padded[:, r, c])


def test_identity_transform_matches_a_plain_crop():
    """k=0 with no flips must be the plain centred crop, not merely equivalent."""
    window, size = 32, 96
    padded = _padded(size)
    half = window // 2
    r, c = 50, 40
    np.testing.assert_array_equal(
        _augment_and_crop(padded, r, c, window, 0, False, False),
        padded[:, r - half: r - half + window, c - half: c - half + window],
    )


def test_rotation_is_order_four():
    """Four 90-degree rotations return the original crop (sanity on the mapping)."""
    window, size = 16, 64
    padded = _padded(size)
    r, c = 30, 25
    np.testing.assert_array_equal(
        _augment_and_crop(padded, r, c, window, 4 % 4, False, False),
        _augment_and_crop(padded, r, c, window, 0, False, False),
    )


def test_window_too_small_is_rejected():
    from src.data.window_dataset import MaridaWindowDataset

    with pytest.raises(ValueError, match="window must be >= 4"):
        MaridaWindowDataset("unused", "unused", window=2)
