"""
3D Rotary Position Embedding (RoPE) for volumetric data.

Extends DINOv3's 2D RoPE to 3 spatial dimensions.

DINOv3 original (rope_position_encoding.py):
    - Takes H, W as input
    - Creates 2D coordinate grid [H, W, 2]
    - Splits embedding dims: embed_dim // (4 * num_heads) frequencies per axis
    - Each axis uses half the frequency dimensions

Our 3D version:
    - Takes D, H, W as input
    - Creates 3D coordinate grid [D, H, W, 3]
    - Splits embedding dims: embed_dim // (6 * num_heads) frequencies per axis
    - Each axis uses one-third of the frequency dimensions
    - Supports physical spacing for anisotropic medical volumes
"""

import math

import numpy as np
import torch
from torch import Tensor, nn


class RopePositionEmbedding3D(nn.Module):
    """
    3D Rotary Position Embedding.

    Key difference from DINOv3's 2D version:
    - 2D: embed_dim must be divisible by (4 * num_heads), each of the 2 axes gets D_head//4 frequencies
    - 3D: embed_dim must be divisible by (6 * num_heads), each of the 3 axes gets D_head//6 frequencies

    Why 6? Because RoPE operates on pairs of dims (sin, cos), and we have 3 axes:
    3 axes * 2 (sin/cos) = 6 groups of frequency dimensions.

    Physical spacing support: medical images have non-uniform voxel spacing
    (e.g., CT: 0.7mm in-plane, 5mm slice thickness). We scale coordinates by
    physical spacing so the model understands real-world distances.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float = 100.0,
        normalize_coords: str = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        # 3D requires divisible by 6 (3 axes * 2 for sin/cos pairs)
        assert embed_dim % (6 * num_heads) == 0, (
            f"embed_dim ({embed_dim}) must be divisible by 6 * num_heads ({6 * num_heads}). "
            f"This is because 3D RoPE splits dimensions across 3 spatial axes."
        )

        D_head = embed_dim // num_heads
        self.base = base
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # D_head // 6 frequencies per axis (3 axes, each with sin+cos pairs)
        self.dtype = dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 6, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(
        self,
        *,
        D: int,
        H: int,
        W: int,
        spacing: tuple[float, float, float] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            D, H, W: spatial dimensions of the patch grid
            spacing: (spacing_d, spacing_h, spacing_w) physical voxel spacing in mm.
                     If provided, coordinates are scaled by physical distance.

        Returns:
            (sin, cos): each of shape [D*H*W, D_head]
        """
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Build coordinates — exactly matching DINOv3's logic per axis
        # DINOv3: coords_h = arange(0.5, H) / H  ->  range [0, 1)  ->  *2 - 1  ->  [-1, +1]
        if self.normalize_coords == "separate":
            coords_d = torch.arange(0.5, D, **dd) / D  # [D], range [0, 1)
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        elif self.normalize_coords == "max":
            max_DHW = max(D, H, W)
            coords_d = torch.arange(0.5, D, **dd) / max_DHW
            coords_h = torch.arange(0.5, H, **dd) / max_DHW
            coords_w = torch.arange(0.5, W, **dd) / max_DHW
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        # Apply physical spacing scaling if provided
        if spacing is not None:
            s_d, s_h, s_w = spacing
            coords_d = coords_d * s_d
            coords_h = coords_h * s_h
            coords_w = coords_w * s_w

        # Create 3D meshgrid: [D, H, W, 3]
        grid_d, grid_h, grid_w = torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij")
        coords = torch.stack([grid_d, grid_h, grid_w], dim=-1)  # [D, H, W, 3]
        coords = coords.flatten(0, 2)  # [D*H*W, 3]
        coords = 2.0 * coords - 1.0  # shift range [0,1) to [-1, +1)

        # Training-time augmentations (same logic as DINOv3 but for 3 dims)
        if self.training and self.shift_coords is not None:
            shift = torch.empty(3, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift[None, :]

        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter = torch.empty(3, **dd).uniform_(-jitter_max, jitter_max).exp()
            coords *= jitter[None, :]

        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale = torch.empty(1, **dd).uniform_(-rescale_max, rescale_max).exp()
            coords *= rescale

        # Compute angles: [D*H*W, 3, n_freqs] where n_freqs = D_head // 6
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2)  # [D*H*W, 3 * n_freqs] = [D*H*W, D_head//2]
        angles = angles.tile(2)  # [D*H*W, D_head]

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return (sin, cos)

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        n_freqs = self.D_head // 6
        periods = self.base ** (
            2 * torch.arange(n_freqs, device=device, dtype=dtype) / (n_freqs * 2)
        )
        self.periods.data = periods
