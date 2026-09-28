"""Anisotropic 3D U-Net for Baseline v1.

Plain encoder-decoder U-Net. Deliberately minimal -- no attention, no residual
connections, no transformer, no deep supervision, no pretraining. Those are all
later experiments; this is the reference baseline they will be compared against.

Why "anisotropic"
-----------------
The voxels are strongly anisotropic: 2.0 mm along z versus 0.667 mm in-plane, a
3:1 ratio. The tensor layout is ``(C, D, H, W)`` with ``D = Z`` (see
``data.crop_spec``), so the *depth* axis is the thick-slice direction.

Blindly applying 3x3x3 kernels everywhere would mix 6 mm of real anatomy along z
with 2 mm in-plane, which is a poor match to the underlying resolution. Instead
the two finest levels convolve and downsample only in-plane:

    level 0,1 : kernel (1,3,3), stride/pool (1,2,2)
    level 2,3 : kernel (3,3,3), stride/pool (2,2,2)
    bottleneck: kernel (3,3,3)

By level 2 the in-plane resolution has already been reduced 4x, so the effective
sampling there is ~2.7 mm in-plane vs 2.0 mm through-plane -- roughly isotropic,
which is why full 3x3x3 is appropriate from that point on.

Shape trace for the frozen crop (D,H,W) = (32, 96, 96):

    enc0 (32, 96, 96) -> pool(1,2,2)   ->       (32, 48, 48)
    enc1 (32, 48, 48) -> pool(1,2,2)   ->       (32, 24, 24)
    enc2 (32, 24, 24) -> pool(2,2,2)   ->       (16, 12, 12)
    enc3 (16, 12, 12) -> pool(2,2,2)   ->       ( 8,  6,  6)
    bottleneck        ( 8,  6,  6)
    dec3              (16, 12, 12)  dec2 (32, 24, 24)
    dec1              (32, 48, 48)  dec0 (32, 96, 96)  -> head -> (4, 32, 96, 96)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import crop_spec as cs


@dataclass
class UNet3DConfig:
    """Architecture hyper-parameters for :class:`AnisotropicUNet3D`."""

    in_channels: int = cs.IN_CHANNELS
    num_classes: int = cs.NUM_CLASSES
    base_channels: int = 16
    #: channel width at each scale; len() defines the number of scales
    channel_multipliers: Sequence[int] = field(default_factory=lambda: (1, 2, 4, 8, 16))
    #: how many of the finest encoder levels use in-plane-only kernels/pooling
    anisotropic_levels: int = 2
    #: how many encoder levels perform downsampling (the last scale is the bottleneck)
    n_downsample: int = 4
    negative_slope: float = 0.01

    @property
    def channels(self) -> list[int]:
        return [self.base_channels * m for m in self.channel_multipliers]


class DoubleConv3d(nn.Module):
    """(conv3d -> InstanceNorm3d -> LeakyReLU) x2 with a configurable kernel.

    InstanceNorm rather than BatchNorm: training uses a batch size of 1-2 because
    of the 8 GB VRAM budget, and BatchNorm statistics are too noisy at that size.
    ``affine=True`` keeps a learnable scale/shift per channel.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: tuple[int, int, int], negative_slope: float) -> None:
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AnisotropicUNet3D(nn.Module):
    """Encoder-decoder U-Net with anisotropic kernels on the finest levels."""

    def __init__(self, config: UNet3DConfig | None = None) -> None:
        super().__init__()
        self.config = config or UNet3DConfig()
        cfg = self.config
        channels = cfg.channels

        if len(channels) != cfg.n_downsample + 1:
            raise ValueError(
                f"channel_multipliers has {len(channels)} entries but "
                f"n_downsample={cfg.n_downsample} implies {cfg.n_downsample + 1}"
            )

        # ---- encoder ------------------------------------------------------ #
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = cfg.in_channels
        for level in range(cfg.n_downsample):
            out = channels[level]
            kernel = (1, 3, 3) if level < cfg.anisotropic_levels else (3, 3, 3)
            self.encoder_blocks.append(
                DoubleConv3d(prev, out, kernel, cfg.negative_slope)
            )
            # Downsample in-plane only on the anisotropic levels: the z axis is
            # already coarse (2 mm), so pooling it early would discard anatomy.
            self.pools.append(
                nn.MaxPool3d(kernel_size=(1, 2, 2) if level < cfg.anisotropic_levels
                             else (2, 2, 2))
            )
            prev = out

        # ---- bottleneck ---------------------------------------------------- #
        self.bottleneck = DoubleConv3d(prev, channels[-1], (3, 3, 3), cfg.negative_slope)

        # ---- decoder ------------------------------------------------------- #
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for level in range(cfg.n_downsample - 1, -1, -1):
            skip = channels[level]
            deeper = channels[level + 1]
            # 1x1x1 channel reduction on the upsampled features before the concat
            self.upconvs.append(nn.Conv3d(deeper, skip, kernel_size=1))
            kernel = (1, 3, 3) if level < cfg.anisotropic_levels else (3, 3, 3)
            self.decoder_blocks.append(
                DoubleConv3d(skip * 2, skip, kernel, cfg.negative_slope)
            )

        self.head = nn.Conv3d(channels[0], cfg.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"expected (N,C,D,H,W), got shape {tuple(x.shape)}")

        skips: list[torch.Tensor] = []
        for block, pool in zip(self.encoder_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, block, skip in zip(self.upconvs, self.decoder_blocks, reversed(skips)):
            x = upconv(x)
            # Interpolate to the skip's exact spatial size rather than assuming a
            # clean doubling -- this keeps the decoder mirroring the encoder even
            # for input shapes where a dimension is odd.
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear",
                              align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return self.head(x)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def build_baseline_model(base_channels: int = 16) -> AnisotropicUNet3D:
    """The exact Baseline v1 configuration."""
    return AnisotropicUNet3D(UNet3DConfig(base_channels=base_channels))
