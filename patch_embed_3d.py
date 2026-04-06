"""
3D Patch Embedding for Medical Volumes.

Replaces DINOv3's 2D PatchEmbed (Conv2d-based) with a Conv3d-based version.

DINOv3 original (patch_embed.py):
    - Input:  (B, C, H, W)        e.g. (B, 3, 224, 224)
    - Conv2d: kernel=16x16, stride=16x16
    - Output: (B, H//16, W//16, D) e.g. (B, 14, 14, 768) = 196 tokens

Our 3D version:
    - Input:  (B, C, D, H, W)       e.g. (B, 3, 96, 96, 96)
    - Conv3d: kernel=16x16x16, stride=16x16x16
    - Output: (B, D//16, H//16, W//16, embed_dim) e.g. (B, 6, 6, 6, 768) = 216 tokens
"""

import math
from typing import Tuple, Union

from torch import Tensor, nn


def make_3tuple(x) -> Tuple[int, int, int]:
    if isinstance(x, tuple):
        assert len(x) == 3
        return x
    assert isinstance(x, int)
    return (x, x, x)


class PatchEmbed3D(nn.Module):
    """
    3D volume to patch embedding: (B, C, D, H, W) -> (B, D', H', W', embed_dim)

    This is the 3D counterpart of DINOv3's PatchEmbed.
    The key change: Conv2d -> Conv3d, and all spatial logic handles 3 dimensions.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]] = 96,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer=None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_DHW = make_3tuple(img_size)
        patch_DHW = make_3tuple(patch_size)
        patch_grid_size = (
            image_DHW[0] // patch_DHW[0],
            image_DHW[1] // patch_DHW[1],
            image_DHW[2] // patch_DHW[2],
        )

        self.img_size = image_DHW
        self.patch_size = patch_DHW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1] * patch_grid_size[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        # Core change: Conv2d -> Conv3d
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=patch_DHW,
            stride=patch_DHW,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, D, H, W)
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        D, H, W = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)  # (B, D'*H'*W', embed_dim)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, D, H, W, self.embed_dim)  # (B, D', H', W', embed_dim)
        return x

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] * self.patch_size[1] * self.patch_size[2]))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))
