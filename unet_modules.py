"""Shared building blocks for the conditional MGM U-Nets."""

import math

import torch
from torch import nn
import torch.nn.functional as F

__all__ = ["make_group_norm", "TimeEmbedding", "ResidualBlock", "Downsample", "Upsample"]


def make_group_norm(channels, max_groups=8):
    if channels < 1 or max_groups < 1:
        raise ValueError("channels and max_groups must be positive.")
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        if embedding_dim < 4 or embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be an even integer >= 4.")
        half_dim = embedding_dim // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half_dim, dtype=torch.float32)
            / (half_dim - 1)
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, t):
        t = t.reshape(-1).float()
        angles = (1000.0 * t[:, None]) * self.frequencies[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels=None):
        super().__init__()
        self.norm1 = make_group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = make_group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )
        self.time_projection = (
            nn.Linear(time_channels, 2 * out_channels)
            if time_channels is not None else None
        )

    def forward(self, x, t_emb=None):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h)
        if self.time_projection is not None:
            if t_emb is None:
                raise ValueError("This block requires a time embedding.")
            scale, shift = self.time_projection(F.silu(t_emb)).chunk(2, dim=1)
            h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return (self.skip(x) + h) / math.sqrt(2.0)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, spatial_size):
        x = F.interpolate(x, size=spatial_size, mode="nearest")
        return self.conv(x)
