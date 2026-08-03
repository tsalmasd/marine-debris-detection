"""
U-Net segmentation model for marine debris detection.

A self-contained PyTorch implementation of the classic U-Net encoder-decoder
(Ronneberger et al., 2015) adapted for Sentinel-2 patches:

    - Input:  C-band Sentinel-2 patch (default 6 bands, [B02, B03, B04, B08, B11, B12])
    - Output: per-pixel segmentation logits, same H x W as the input
    - Binary mode (n_classes == 1): a single logit per pixel; use with
      BCEWithLogitsLoss and threshold at 0 (== probability 0.5) for the mask.
    - Multi-class mode (n_classes > 1): one logit per class; use with
      CrossEntropyLoss and argmax over the channel axis for the mask.

The default configuration produces a binary debris / non-debris mask, matching
the semantics of the Random Forest baseline so the two models are comparable on
the same task.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default input band count, aligned with dataset_loader.DEFAULT_BANDS
# ([B02, B03, B04, B08, B11, B12]).
DEFAULT_IN_CHANNELS = 6


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x 2, the basic U-Net building block."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscaling step: max-pool by 2 then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """
    Upscaling step: upsample, concatenate the encoder skip connection, DoubleConv.

    When ``bilinear`` is True, upsampling is a parameter-free bilinear interpolation
    (lighter, fewer artifacts); otherwise a learned transposed convolution is used.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # in_channels = upsampled channels + skip channels (both == in_channels // 2)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Pad if the spatial dims drifted (e.g. odd input sizes) so the skip
        # connection concatenates cleanly.
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )

        x = torch.cat([skip, x], dim=1)  # concat along the channel axis
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution mapping to the number of output classes."""

    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for semantic segmentation of Sentinel-2 patches.

    Args:
        in_channels: number of input bands (default 6, the RF-baseline subset).
        n_classes: number of output classes. Use 1 for binary debris / non-debris
            segmentation (single logit, pair with BCEWithLogitsLoss); use >1 for
            multi-class segmentation (pair with CrossEntropyLoss).
        base_channels: channel width of the first encoder stage. The four
            downsampling stages scale it up by 2x each (default 64 -> 128 -> 256
            -> 512 -> 1024).
        bilinear: if True, upsample via bilinear interpolation; otherwise use
            learned transposed convolutions.

    Forward input:  (N, in_channels, H, W)
    Forward output: (N, n_classes, H, W) of raw logits.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        n_classes: int = 1,
        base_channels: int = 64,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        c = base_channels
        factor = 2 if bilinear else 1  # bottleneck is halved when bilinear upsampling

        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16 // factor)

        self.up1 = Up(c * 16, c * 8 // factor, bilinear)
        self.up2 = Up(c * 8, c * 4 // factor, bilinear)
        self.up3 = Up(c * 4, c * 2 // factor, bilinear)
        self.up4 = Up(c * 2, c, bilinear)
        self.outc = OutConv(c, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Convenience inference helper: logits -> integer class mask.

        Binary mode (n_classes == 1): sigmoid, then threshold -> {0, 1}.
        Multi-class mode: argmax over the channel axis -> {0 .. n_classes-1}.

        Args:
            x: input batch of shape (N, in_channels, H, W).
            threshold: probability cut-off for the binary case.

        Returns:
            Integer mask of shape (N, H, W).
        """
        self.eval()
        logits = self.forward(x)
        if self.n_classes == 1:
            probs = torch.sigmoid(logits.squeeze(1))
            return (probs >= threshold).long()
        return logits.argmax(dim=1)


def build_unet(
    in_channels: int = DEFAULT_IN_CHANNELS,
    n_classes: int = 1,
    base_channels: int = 64,
    bilinear: bool = True,
) -> UNet:
    """Factory for a :class:`UNet` with the project's default configuration."""
    return UNet(
        in_channels=in_channels,
        n_classes=n_classes,
        base_channels=base_channels,
        bilinear=bilinear,
    )


if __name__ == "__main__":
    # Shape sanity check: a batch of 2 six-band 256x256 patches should map to a
    # (2, 1, 256, 256) logit map in binary mode.
    model = build_unet()
    n_params = sum(p.numel() for p in model.parameters())
    dummy = torch.randn(2, DEFAULT_IN_CHANNELS, 256, 256)
    out = model(dummy)
    mask = model.predict_mask(dummy)
    print(f"UNet parameters: {n_params:,}")
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Logits: {tuple(out.shape)}")
    print(f"Mask:   {tuple(mask.shape)} (unique values: {mask.unique().tolist()})")
    assert out.shape == (2, 1, 256, 256)
    assert mask.shape == (2, 256, 256)
    print("OK")
