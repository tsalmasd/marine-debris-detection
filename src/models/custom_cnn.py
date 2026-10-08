"""
Custom CNN patch classifier for marine debris detection.

The third model of the thesis. Unlike the Random Forest (per-pixel) and the
U-Net (per-pixel dense segmentation), this one answers a *patch-level* question:

    "does this 256x256 Sentinel-2 patch contain any marine debris?"

    - Input:  (N, C, H, W) Sentinel-2 patch, default 6 bands
              ([B02, B03, B04, B08, B11, B12]), same subset as the U-Net
    - Output: (N, 1) single logit per patch (default). Pair with
              BCEWithLogitsLoss and sigmoid -> threshold for the decision.
    - n_classes > 1 switches to one logit per class, for CrossEntropyLoss.

Note there is no Softmax layer: ``CrossEntropyLoss`` and ``BCEWithLogitsLoss``
both apply their own normalization internally, so a Softmax here would be
applied twice. The model always returns raw logits, matching ``unet.UNet``.

Architecture
------------
Three convolutional stages widening 32 -> 64 -> 128, then a two-layer dense
head, as originally sketched::

    Input (6, 256, 256)
        -> Conv(32) + BN + ReLU -> MaxPool   (32, 128, 128)
        -> Conv(64) + BN + ReLU -> MaxPool   (64,  64,  64)
        -> Conv(128) + BN + ReLU             (128, 64,  64)
        -> global max pool                   (128,)
        -> Dropout -> Dense(128) + ReLU -> Dense(n_classes)

Two deliberate departures from that sketch, both forced by the data:

**Global max pooling instead of Flatten.** Flattening (128, 64, 64) into
Dense(128) is 67M parameters on a 694-patch training split — roughly 97k
parameters per training example, which cannot be fit honestly. Global *max*
pooling over the final feature map costs nothing and gives ~110k parameters
total. It is also the right pooling for this label: a positive patch holds a
median of 4 debris pixels out of 65,536, so the patch score should be driven by
the single most debris-like location, not by an average that dilutes 4 pixels
into 65,536. That makes the model a multiple-instance classifier, which is what
"contains any debris" actually means.

**BatchNorm after each convolution**, matching ``unet.DoubleConv``. The sketch
omitted it; it is standard, and it keeps the two deep models comparable in
everything except the architecture under test.

Why a single logit by default
-----------------------------
A single sigmoid score per patch keeps the scoring path identical to the
U-Net's: threshold sweeps, PR curves and average precision all work unchanged
(``src/validation/threshold.py``, ``src/validation/metrics.py``). Two softmax
logits would carry the same information in a shape nothing else in the repo
consumes.

Class imbalance
---------------
Counted in *patches*, not pixels. The pixel-level 197:1 figure quoted for the
U-Net does not apply here, and neither do Dice or Tversky — an overlap loss on
a single scalar label is a category error.

    split   patches   contain debris        clean   neg:pos
    train       694      190 (27.4%)          504    2.65:1
    val         328       82 (25.0%)          246    3.00:1
    test        359      101 (28.1%)          258    2.55:1

2.65:1 is mild. Weighted cross-entropy with ``pos_weight = 2.65`` covers it;
focal loss has nothing to do here and is not worth a pre-registered variant.
The hard part of this task is not the class balance, it is that the positive
evidence is a handful of pixels per patch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Default input band count, aligned with dataset_loader.DEFAULT_BANDS
# ([B02, B03, B04, B08, B11, B12]).
DEFAULT_IN_CHANNELS = 6

# Measured neg:pos patch ratio on the train split (190 of 694 patches contain
# at least one debris pixel). The default BCE ``pos_weight``; recompute with
# MaridaPatchClassificationDataset.compute_pos_weight() if the split changes.
TRAIN_POS_WEIGHT = 2.65


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU, optionally followed by a 2x max-pool."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Patch-level debris classifier.

    Args:
        in_channels: number of input bands (default 6, the U-Net subset).
        n_classes: output logits. 1 (default) for a single debris score, to be
            used with BCEWithLogitsLoss; >1 for one logit per class with
            CrossEntropyLoss.
        base_channels: width of the first stage; the three stages are
            ``base_channels`` x1, x2, x4 (default 32 -> 64 -> 128).
        hidden_dim: width of the dense layer before the output.
        dropout: dropout probability applied to the pooled features. The model
            has few parameters but very little training data, so this is on by
            default.

    Forward input:  (N, in_channels, H, W) — any H, W; the global pool makes
        the model resolution-independent.
    Forward output: (N, n_classes) raw logits.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        n_classes: int = 1,
        base_channels: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_channels = base_channels
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout

        c = base_channels
        self.features = nn.Sequential(
            ConvBlock(in_channels, c, pool=True),       # -> (c,   H/2, W/2)
            ConvBlock(c, c * 2, pool=True),             # -> (2c,  H/4, W/4)
            ConvBlock(c * 2, c * 4, pool=False),        # -> (4c,  H/4, W/4)
        )
        # Max, not average: a positive patch may hold only a handful of debris
        # pixels, and averaging would dilute them away. See the module docstring.
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)  # (N, 4c)
        return self.classifier(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Debris probability per patch.

        Binary mode (n_classes == 1): sigmoid of the logit -> (N,) in [0, 1].
        Multi-class mode: softmax over the class axis -> (N, n_classes).
        """
        self.eval()
        logits = self.forward(x)
        if self.n_classes == 1:
            return torch.sigmoid(logits.squeeze(1))
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Hard patch label: 1 if the patch is predicted to contain debris.

        Binary mode thresholds the sigmoid score; multi-class mode takes the
        argmax and ``threshold`` is ignored. Returns an integer tensor of
        shape (N,).
        """
        probs = self.predict_proba(x)
        if self.n_classes == 1:
            return (probs >= threshold).long()
        return probs.argmax(dim=1)


def build_custom_cnn(
    in_channels: int = DEFAULT_IN_CHANNELS,
    n_classes: int = 1,
    base_channels: int = 32,
    hidden_dim: int = 128,
    dropout: float = 0.3,
) -> CustomCNN:
    """Factory for a :class:`CustomCNN` with the project's default configuration."""
    return CustomCNN(
        in_channels=in_channels,
        n_classes=n_classes,
        base_channels=base_channels,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def save_checkpoint(
    model: CustomCNN,
    path: str,
    mean,
    std,
    extra: dict | None = None,
) -> None:
    """
    Persist a trained classifier together with everything inference needs.

    Mirrors :func:`src.models.unet.save_checkpoint`: the architecture config so
    the model can be rebuilt without knowing the hyperparameters, and the
    per-channel normalization stats so inputs are scaled at inference exactly
    as during training. ``extra`` records provenance such as the best epoch,
    the selection metric and the chosen threshold.
    """
    import numpy as np

    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "in_channels": model.in_channels,
            "n_classes": model.n_classes,
            "base_channels": model.base_channels,
            "hidden_dim": model.hidden_dim,
            "dropout": model.dropout_p,
        },
        "norm_mean": np.asarray(mean, dtype="float32"),
        "norm_std": np.asarray(std, dtype="float32"),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: str, device: str = "cpu"):
    """
    Rebuild a classifier from a checkpoint written by :func:`save_checkpoint`.

    Returns ``(model_in_eval_mode, mean, std)`` where mean/std are the (C,)
    normalization arrays to apply to inputs before a forward pass.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_custom_cnn(
        in_channels=cfg["in_channels"],
        n_classes=cfg["n_classes"],
        base_channels=cfg["base_channels"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["norm_mean"], ckpt["norm_std"]


if __name__ == "__main__":
    # Shape sanity check: a batch of 2 six-band 256x256 patches should map to a
    # (2, 1) logit vector in binary mode, and to (2,) scores / labels.
    model = build_custom_cnn()
    n_params = sum(p.numel() for p in model.parameters())
    dummy = torch.randn(2, DEFAULT_IN_CHANNELS, 256, 256)
    out = model(dummy)
    probs = model.predict_proba(dummy)
    labels = model.predict(dummy)
    print(f"CustomCNN parameters: {n_params:,}")
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Logits: {tuple(out.shape)}")
    print(f"Probs:  {tuple(probs.shape)} (range {probs.min():.3f}-{probs.max():.3f})")
    print(f"Labels: {tuple(labels.shape)} (unique values: {labels.unique().tolist()})")
    assert out.shape == (2, 1)
    assert probs.shape == (2,)
    assert labels.shape == (2,)

    # Resolution independence: the global pool means a 128x128 patch works too.
    assert model(torch.randn(1, DEFAULT_IN_CHANNELS, 128, 128)).shape == (1, 1)

    # Multi-class mode: 2 logits for CrossEntropyLoss.
    mc = build_custom_cnn(n_classes=2)
    assert mc(dummy).shape == (2, 2)
    assert mc.predict_proba(dummy).shape == (2, 2)
    assert mc.predict(dummy).shape == (2,)
    print("OK")
