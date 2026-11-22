"""
Tests for the U-Net checkpoint's band bookkeeping (step 2 of REVISED_PLAN.md).

Once the U-Net can be trained on either 6 or 7 bands, every consumer
(evaluation, GeoTIFF export, the bootstrap, the figures) has to learn the band
list from the checkpoint instead of assuming the 6-band default. The failure
these tests guard against is silent: feeding a model the wrong spectra produces
output of exactly the right shape, and no metric looks wrong.
"""

import numpy as np
import pytest

from src.data.dataset_loader import DEFAULT_BANDS, RF_BANDS
from src.models.unet import (
    build_unet,
    load_checkpoint_bands,
    resolve_bands,
    save_checkpoint,
)


def _save(tmp_path, n_channels, bands):
    """A minimal checkpoint on disk; base_channels is tiny to keep this fast."""
    model = build_unet(in_channels=n_channels, n_classes=1, base_channels=4)
    path = tmp_path / f"unet_{n_channels}ch.pt"
    save_checkpoint(
        model, str(path),
        mean=np.zeros(n_channels, dtype=np.float32),
        std=np.ones(n_channels, dtype=np.float32),
        bands=bands,
    )
    return model, str(path)


def test_recorded_bands_round_trip(tmp_path):
    model, path = _save(tmp_path, len(RF_BANDS), RF_BANDS)
    assert load_checkpoint_bands(path) == list(RF_BANDS)
    assert resolve_bands(path, model) == list(RF_BANDS)


def test_old_checkpoint_without_bands_falls_back_to_six(tmp_path):
    """Checkpoints predating the key are all 6-band, so the fallback is right."""
    model, path = _save(tmp_path, len(DEFAULT_BANDS), None)
    assert load_checkpoint_bands(path) is None
    assert resolve_bands(path, model) == list(DEFAULT_BANDS)


def test_band_count_disagreeing_with_the_model_is_rejected(tmp_path):
    """
    A 7-channel model with no recorded bands must not quietly take the 6-band
    fallback -- that is the mismatch that yields plausible nonsense.
    """
    model, path = _save(tmp_path, len(RF_BANDS), None)
    with pytest.raises(ValueError, match="7 input channels"):
        resolve_bands(path, model)


def test_wrong_recorded_bands_are_rejected(tmp_path):
    """The check also catches a checkpoint whose own band list is inconsistent."""
    model = build_unet(in_channels=len(RF_BANDS), n_classes=1, base_channels=4)
    path = tmp_path / "inconsistent.pt"
    save_checkpoint(model, str(path),
                    mean=np.zeros(7, dtype=np.float32),
                    std=np.ones(7, dtype=np.float32),
                    bands=DEFAULT_BANDS)          # 6 names, 7-channel model
    with pytest.raises(ValueError, match="band list"):
        resolve_bands(str(path), model)
