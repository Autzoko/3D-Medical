"""
3D Vision Transformer for Medical DINO.

Adapts DINOv3's DinoVisionTransformer to handle 3D volumetric inputs.

DINOv3 original (vision_transformer.py):
    - PatchEmbed:  Conv2d, output (B, H, W, D)
    - RoPE:        2D coords (H, W)
    - Tokens:      [CLS] + [storage_tokens] + [H*W patch tokens]
    - Shape logic: B, H, W, _ = x.shape; x.flatten(1, 2)

Our 3D version:
    - PatchEmbed3D:  Conv3d, output (B, D', H', W', embed_dim)
    - RoPE3D:        3D coords (D', H', W')
    - Tokens:        [CLS] + [storage_tokens] + [D'*H'*W' patch tokens]
    - Shape logic:   B, D', H', W', _ = x.shape; x.flatten(1, 3)

The attention blocks (SelfAttentionBlock), DINOHead, and all loss functions
are UNCHANGED because they operate on token sequences, not spatial grids.
"""

import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.init
from torch import Tensor, nn

from .patch_embed_3d import PatchEmbed3D
from .rope_3d import RopePositionEmbedding3D

logger = logging.getLogger("medical_dino3d")


# ---------------------------------------------------------------------------
# Reuse DINOv3 components that are dimension-agnostic
# ---------------------------------------------------------------------------

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, device=None):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_values, device=device))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma

    def reset_parameters(self):
        nn.init.ones_(self.gamma)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.0, bias=True, device=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias, device=device)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

    def forward_list(self, x_list):
        return [self.forward(x) for x in x_list]


class SelfAttention(nn.Module):
    """Standard multi-head self-attention with optional RoPE. Dimension-agnostic."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True,
                 attn_drop=0.0, proj_drop=0.0, mask_k_bias=False, device=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, device=device)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)

    def forward(self, x: Tensor, rope=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, heads, N, head_dim)

        # Apply RoPE if provided
        if rope is not None:
            sin, cos = rope
            # sin, cos: [N_patches, D_head] — only for patch tokens, not CLS/register
            # We need to figure out how many prefix tokens there are
            n_prefix = N - sin.shape[0]
            if n_prefix > 0:
                q_prefix, q_patch = q[:, :, :n_prefix], q[:, :, n_prefix:]
                k_prefix, k_patch = k[:, :, :n_prefix], k[:, :, n_prefix:]

                q_patch = self._apply_rope(q_patch, sin, cos)
                k_patch = self._apply_rope(k_patch, sin, cos)

                q = torch.cat([q_prefix, q_patch], dim=2)
                k = torch.cat([k_prefix, k_patch], dim=2)
            else:
                q = self._apply_rope(q, sin, cos)
                k = self._apply_rope(k, sin, cos)

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = attn.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

    def _apply_rope(self, x, sin, cos):
        """Apply rotary position embedding."""
        # x: (B, heads, N, head_dim)
        # sin, cos: (N, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, N, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        sin_half = sin[..., 0::2]
        cos_half = cos[..., 0::2]

        return torch.cat([
            x1 * cos_half - x2 * sin_half,
            x2 * cos_half + x1 * sin_half,
        ], dim=-1)

    def forward_list(self, x_list, rope_list=None):
        if rope_list is None:
            rope_list = [None] * len(x_list)
        return [self.forward(x, rope=rope) for x, rope in zip(x_list, rope_list)]


class SelfAttentionBlock(nn.Module):
    """Transformer block. Identical to DINOv3's — dimension-agnostic."""

    def __init__(self, dim, num_heads, ffn_ratio=4.0, qkv_bias=False, proj_bias=True,
                 ffn_bias=True, drop_path=0.0, init_values=None,
                 norm_layer=None, act_layer=nn.GELU, ffn_layer=None,
                 mask_k_bias=False, device=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        ffn_layer = ffn_layer or Mlp

        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                             bias=ffn_bias, device=device)
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()
        self.drop_path_rate = drop_path

    def forward(self, x_or_x_list, rope_or_rope_list=None):
        if isinstance(x_or_x_list, Tensor):
            return self._forward_single(x_or_x_list, rope_or_rope_list)
        else:
            if rope_or_rope_list is None:
                rope_or_rope_list = [None] * len(x_or_x_list)
            return [self._forward_single(x, rope) for x, rope in zip(x_or_x_list, rope_or_rope_list)]

    def _forward_single(self, x, rope=None):
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# DINOHead — completely unchanged from DINOv3
# ---------------------------------------------------------------------------

class DINOHead(nn.Module):
    """Projection head. Identical to DINOv3's — operates on token embeddings."""

    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256, nlayers=3):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(nlayers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


# ---------------------------------------------------------------------------
# 3D Vision Transformer
# ---------------------------------------------------------------------------

def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed3D):
        module.reset_parameters()


class DinoVisionTransformer3D(nn.Module):
    """
    3D Vision Transformer for Medical DINO.

    This is the direct 3D adaptation of DINOv3's DinoVisionTransformer.
    Changes from the 2D version:
        1. PatchEmbed  -> PatchEmbed3D  (Conv2d -> Conv3d)
        2. RoPE 2D    -> RoPE 3D       (H,W coords -> D,H,W coords)
        3. Shape handling: (B, H, W, D) -> (B, D', H', W', embed_dim)

    Everything else is identical: CLS token, storage tokens, mask token,
    attention blocks, normalization, output format.
    """

    def __init__(
        self,
        *,
        img_size: int = 96,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        n_storage_tokens: int = 4,
        # RoPE 3D params
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_normalize_coords: str = "separate",
        device: Any | None = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        # 3D Patch Embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        # Special tokens (same as DINOv3)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

        # 3D RoPE
        self.rope_embed = RopePositionEmbedding3D(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            normalize_coords=pos_embed_rope_normalize_coords,
            dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device=device,
        )

        # Transformer blocks (completely unchanged from DINOv3)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                drop_path=drop_path_rate,
                init_values=layerscale_init,
                norm_layer=norm_layer,
                device=device,
            )
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

    def init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        self.rope_embed._init_weights()
        for name, module in self.named_modules():
            init_weights_vit(module, name)

    def prepare_tokens_with_masks(self, x: Tensor, masks=None):
        """
        Patch-embed, apply masking, prepend CLS + storage tokens.

        DINOv3:  B, H, W, _ = x.shape; x = x.flatten(1, 2)
        Ours:    B, D, H, W, _ = x.shape; x = x.flatten(1, 3)
        """
        x = self.patch_embed(x)  # (B, D', H', W', embed_dim) if flatten=False
        B, D, H, W, _ = x.shape
        x = x.flatten(1, 3)  # (B, D'*H'*W', embed_dim)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token

        storage_tokens = (
            self.storage_tokens if self.n_storage_tokens > 0
            else torch.empty(1, 0, cls_token.shape[-1], dtype=cls_token.dtype, device=cls_token.device)
        )

        x = torch.cat([
            cls_token.expand(B, -1, -1),
            storage_tokens.expand(B, -1, -1),
            x,
        ], dim=1)

        return x, (D, H, W)

    def forward_features_list(
        self,
        x_list: List[Tensor],
        masks_list: List[Tensor],
        spacing: tuple[float, float, float] | None = None,
    ) -> List[Dict[str, Tensor]]:
        """
        Process multiple crops jointly through the transformer.

        This is the core forward method, identical in structure to DINOv3's
        forward_features_list, but using 3D RoPE.
        """
        x = []
        rope_params = []  # (D, H, W) tuples for RoPE
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, dhw = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope_params.append(dhw)

        # Run through transformer blocks
        for blk in self.blocks:
            rope_sincos = [
                self.rope_embed(D=D, H=H, W=W, spacing=spacing) for D, H, W in rope_params
            ]
            x = blk(x, rope_sincos)

        # Apply final norm and separate output tokens
        output = []
        for idx, (xi, masks) in enumerate(zip(x, masks_list)):
            x_norm = self.norm(xi)
            x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
            x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append({
                "x_norm_clstoken": x_norm_cls_reg[:, 0],          # (B, embed_dim)
                "x_storage_tokens": x_norm_cls_reg[:, 1:],         # (B, n_reg, embed_dim)
                "x_norm_patchtokens": x_norm_patch,                # (B, n_patches, embed_dim)
                "x_prenorm": xi,                                    # (B, 1+n_reg+n_patches, embed_dim)
                "masks": masks,
            })
        return output

    def forward_features(self, x, masks=None, spacing=None):
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks], spacing=spacing)[0]
        else:
            return self.forward_features_list(x, masks, spacing=spacing)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


# ---------------------------------------------------------------------------
# Model factory functions (mirroring DINOv3's vit_small, vit_base, etc.)
# ---------------------------------------------------------------------------

def vit3d_small(patch_size=16, **kwargs):
    """~21M params, 216 tokens for 96^3 input. Good for debugging."""
    # embed_dim must be divisible by 6 * num_heads = 6 * 6 = 36
    # 384 / 36 = 10.67 -> not divisible! Use 360 or 432
    # Alternative: use num_heads=4, then 6*4=24, 384/24=16 -> OK
    return DinoVisionTransformer3D(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=4,
        ffn_ratio=4, **kwargs,
    )


def vit3d_base(patch_size=16, **kwargs):
    """~86M params. Best cost/performance ratio for medical imaging."""
    # embed_dim=768, num_heads=12 -> 6*12=72, 768/72=10.67 -> not OK
    # Use num_heads=8: 6*8=48, 768/48=16 -> OK
    return DinoVisionTransformer3D(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=8,
        ffn_ratio=4, **kwargs,
    )


def vit3d_large(patch_size=16, **kwargs):
    """~300M params. Use only with sufficient data (>10K volumes)."""
    # embed_dim=1024, num_heads=16 -> 6*16=96, 1024/96=10.67 -> not OK
    # Use embed_dim=1056 or num_heads=8: 6*8=48, 1024/48=21.33 -> not OK
    # Use embed_dim=960, num_heads=10: 6*10=60, 960/60=16 -> OK
    return DinoVisionTransformer3D(
        patch_size=patch_size, embed_dim=960, depth=24, num_heads=10,
        ffn_ratio=4, **kwargs,
    )
