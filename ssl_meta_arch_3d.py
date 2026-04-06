"""
3D Medical DINO: Self-Supervised Learning Meta-Architecture.

Adapts DINOv3's SSLMetaArch to 3D medical volumes.

What stays the SAME (directly from DINOv3):
    - Student-Teacher EMA self-distillation framework
    - DINOLoss with Sinkhorn-Knopp centering
    - iBOTPatchLoss for masked patch prediction
    - KoLeoLoss for feature uniformity
    - DINOHead projection heads
    - EMA update logic

What CHANGES:
    - Backbone: DinoVisionTransformer3D (3D patches, 3D RoPE)
    - Data: 3D medical volumes with medical augmentations
    - Gram loss: removed (too memory-intensive for 3D)
    - Multi-crop: 2 global + 4 local (reduced from 2+8)
"""

import logging
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from .vision_transformer_3d import (
    DinoVisionTransformer3D,
    DINOHead,
    vit3d_small,
    vit3d_base,
)

logger = logging.getLogger("medical_dino3d")


# ---------------------------------------------------------------------------
# Loss functions — copied from DINOv3 WITHOUT modification
# They operate on token embeddings, not spatial grids.
# ---------------------------------------------------------------------------

class DINOLoss(nn.Module):
    """Identical to DINOv3's DINOLoss."""

    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        Q = torch.exp(teacher_output / teacher_temp).t()
        B = Q.shape[1]
        K = Q.shape[0]

        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for _ in range(n_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B
        return Q.t()

    def forward(self, student_logits, teacher_probs, ignore_diagonal=False):
        student_crops, B, K = student_logits.shape
        teacher_crops, _, _ = teacher_probs.shape
        student_logits = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)

        if not ignore_diagonal:
            loss = -torch.einsum("s b k, t b k -> ", student_logits, teacher_probs)
            return loss / (B * student_crops * teacher_crops)
        else:
            loss = -torch.einsum("s b k, t b k -> s t", student_logits, teacher_probs)
            min_st = min(student_crops, teacher_crops)
            loss = torch.diagonal_scatter(loss, loss.new_zeros(min_st))
            return loss.sum() / (B * student_crops * teacher_crops - B * min_st)


class iBOTPatchLoss(nn.Module):
    """Identical to DINOv3's iBOTPatchLoss."""

    def __init__(self, patch_out_dim, student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_masked_patches_tensor=None, n_iterations=3):
        teacher_output = teacher_output.float()
        Q = torch.exp(teacher_output / teacher_temp).t()
        B = n_masked_patches_tensor if n_masked_patches_tensor is not None else Q.shape[1]
        K = Q.shape[0]

        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for _ in range(n_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B
        return Q.t()

    def forward_masked(self, student_masked, teacher_masked, student_masks_flat,
                        n_masked_patches=None, masks_weight=None):
        loss = torch.sum(
            teacher_masked.float() * F.log_softmax(student_masked.float() / self.student_temp, dim=-1),
            dim=-1,
        )
        if masks_weight is None:
            masks_weight = (
                (1.0 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1).expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight
        return -loss.sum() / student_masks_flat.shape[0]


class KoLeoLoss(nn.Module):
    """Identical to DINOv3's KoLeoLoss."""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def forward(self, student_output, eps=1e-8):
        with torch.autocast("cuda", enabled=False):
            student_output = F.normalize(student_output.float(), eps=eps, p=2, dim=-1)
            dots = torch.mm(student_output, student_output.t())
            n = student_output.shape[0]
            dots.view(-1)[:: (n + 1)].fill_(-1)
            _, indices = torch.max(dots, dim=1)
            distances = self.pdist(student_output, student_output[indices])
            loss = -torch.log(distances + eps).mean()
        return loss


# ---------------------------------------------------------------------------
# 3D Medical DINO Meta-Architecture
# ---------------------------------------------------------------------------

class MedicalDINO3D(nn.Module):
    """
    3D Medical DINO self-supervised learning framework.

    Architecture:
        Student: backbone (ViT3D) + dino_head + ibot_head
        Teacher: backbone (ViT3D) + dino_head + ibot_head  (EMA of student)

    Training objective:
        Total loss = DINO_loss (CLS self-distillation)
                   + iBOT_loss (masked patch prediction)
                   + KoLeo_loss (feature uniformity)
    """

    def __init__(
        self,
        # Model
        arch: str = "vit3d_base",
        img_size: int = 96,
        patch_size: int = 16,
        n_storage_tokens: int = 4,
        # DINO head
        dino_head_n_prototypes: int = 65536,
        dino_head_hidden_dim: int = 2048,
        dino_head_bottleneck_dim: int = 256,
        dino_head_nlayers: int = 3,
        # iBOT head
        ibot_head_n_prototypes: int = 8192,
        ibot_head_hidden_dim: int = 2048,
        ibot_head_bottleneck_dim: int = 256,
        ibot_head_nlayers: int = 3,
        # Loss weights
        dino_loss_weight: float = 1.0,
        ibot_loss_weight: float = 1.0,
        koleo_loss_weight: float = 0.1,
        # Multi-crop
        n_local_crops: int = 4,
        ignore_diagonal: bool = True,
    ):
        super().__init__()

        # Build backbone
        arch_fn = {"vit3d_small": vit3d_small, "vit3d_base": vit3d_base}[arch]
        student_backbone = arch_fn(img_size=img_size, patch_size=patch_size,
                                    n_storage_tokens=n_storage_tokens)
        teacher_backbone = arch_fn(img_size=img_size, patch_size=patch_size,
                                    n_storage_tokens=n_storage_tokens)
        embed_dim = student_backbone.embed_dim

        # Build heads
        student_dino_head = DINOHead(embed_dim, dino_head_n_prototypes,
                                      dino_head_hidden_dim, dino_head_bottleneck_dim,
                                      dino_head_nlayers)
        teacher_dino_head = DINOHead(embed_dim, dino_head_n_prototypes,
                                      dino_head_hidden_dim, dino_head_bottleneck_dim,
                                      dino_head_nlayers)
        student_ibot_head = DINOHead(embed_dim, ibot_head_n_prototypes,
                                      ibot_head_hidden_dim, ibot_head_bottleneck_dim,
                                      ibot_head_nlayers)
        teacher_ibot_head = DINOHead(embed_dim, ibot_head_n_prototypes,
                                      ibot_head_hidden_dim, ibot_head_bottleneck_dim,
                                      ibot_head_nlayers)

        self.student = nn.ModuleDict({
            "backbone": student_backbone,
            "dino_head": student_dino_head,
            "ibot_head": student_ibot_head,
        })
        self.teacher = nn.ModuleDict({
            "backbone": teacher_backbone,
            "dino_head": teacher_dino_head,
            "ibot_head": teacher_ibot_head,
        })
        self.teacher.requires_grad_(False)

        # Losses
        self.dino_loss = DINOLoss(dino_head_n_prototypes)
        self.ibot_patch_loss = iBOTPatchLoss(ibot_head_n_prototypes)
        self.koleo_loss = KoLeoLoss()

        # Config
        self.n_local_crops = n_local_crops
        self.dino_loss_weight = dino_loss_weight
        self.ibot_loss_weight = ibot_loss_weight
        self.koleo_loss_weight = koleo_loss_weight
        self.ignore_diagonal = ignore_diagonal
        self.embed_dim = embed_dim

        # EMA param cache
        self._ema_params = None

    def init_weights(self):
        self.student.backbone.init_weights()
        self.student.dino_head.init_weights()
        self.student.ibot_head.init_weights()
        # Initialize teacher as copy of student
        self.teacher.load_state_dict(self.student.state_dict())

    @torch.no_grad()
    def get_teacher_output(self, global_crops, *, teacher_temp, mask_indices_list,
                            n_masked_patches_tensor):
        """Teacher forward on global crops. No gradients."""
        n_crops, B = global_crops.shape[:2]
        images = global_crops.flatten(0, 1)  # (n_crops*B, C, D, H, W)

        out = self.teacher.backbone(images, is_training=True)
        cls = out["x_norm_clstoken"]           # (n_crops*B, embed_dim)
        patches = out["x_norm_patchtokens"]    # (n_crops*B, n_tokens, embed_dim)

        # iBOT head only on masked patches
        masked_patches = torch.index_select(patches.flatten(0, 1), 0, mask_indices_list)
        masked_after_head = self.teacher.ibot_head(masked_patches)

        # DINO head on CLS
        cls_after_head = self.teacher.dino_head(cls)

        # Sinkhorn-Knopp centering
        cls_centered = self.dino_loss.sinkhorn_knopp_teacher(cls_after_head, teacher_temp)
        cls_centered = cls_centered.unflatten(0, (n_crops, B))

        masked_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
            masked_after_head, teacher_temp, n_masked_patches_tensor,
        )

        return {
            "cls_after_head": cls_after_head.unflatten(0, (n_crops, B)),
            "cls_centered": cls_centered,
            "masked_patch_centered": masked_centered,
            "patch_pre_head": patches.unflatten(0, (n_crops, B)),
        }

    def get_student_output(self, *, global_crops, local_crops, masks, mask_indices_list):
        """Student forward on all crops. With gradients."""
        n_global, B = global_crops.shape[:2]
        n_local = local_crops.shape[0]

        # Forward both global and local through backbone together
        global_out, local_out = self.student.backbone(
            [global_crops.flatten(0, 1), local_crops.flatten(0, 1)],
            masks=[masks, None],
            is_training=True,
        )

        g_cls = global_out["x_norm_clstoken"]
        g_patch = global_out["x_norm_patchtokens"]
        l_cls = local_out["x_norm_clstoken"]

        # iBOT head on masked patches only
        masked_patches = torch.index_select(g_patch.flatten(0, 1), 0, mask_indices_list)
        masked_after_head = self.student.ibot_head(masked_patches)

        # DINO head on all CLS tokens
        all_cls = torch.cat([g_cls, l_cls], dim=0)
        all_cls_after_head = self.student.dino_head(all_cls)
        g_cls_head, l_cls_head = all_cls_after_head.split([n_global * B, n_local * B], dim=0)

        return {
            "global": {
                "cls_after_head": g_cls_head.unflatten(0, (n_global, B)),
                "cls_pre_head": g_cls.unflatten(0, (n_global, B)),
                "masked_patch_after_head": masked_after_head,
            },
            "local": {
                "cls_after_head": l_cls_head.unflatten(0, (n_local, B)),
            },
        }

    def compute_losses(self, teacher_out, student_out, masks, mask_indices_list,
                        masks_weight, n_global_crops=2):
        """Compute all losses. Identical structure to DINOv3."""
        n_local = student_out["local"]["cls_after_head"].shape[0]
        loss_dict = {}
        loss_total = 0.0

        # Scaling factors (same as DINOv3)
        dino_global_terms = n_global_crops * (n_global_crops - 1) if self.ignore_diagonal else n_global_crops ** 2
        dino_local_terms = n_global_crops * n_local
        dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
        dino_local_scale = dino_local_terms / (dino_global_terms + dino_local_terms)

        # DINO local loss: student(local CLS) vs teacher(global CLS)
        dino_local = self.dino_loss(
            student_out["local"]["cls_after_head"],
            teacher_out["cls_centered"],
        )
        loss_dict["dino_local"] = dino_local.item()
        loss_total += self.dino_loss_weight * dino_local_scale * dino_local

        # DINO global loss: student(global CLS) vs teacher(global CLS)
        dino_global = self.dino_loss(
            student_out["global"]["cls_after_head"],
            teacher_out["cls_centered"],
            ignore_diagonal=self.ignore_diagonal,
        )
        loss_dict["dino_global"] = dino_global.item()
        loss_total += self.dino_loss_weight * dino_global_scale * dino_global

        # KoLeo: regularize student global CLS pre-head
        koleo = sum(
            self.koleo_loss(x) for x in student_out["global"]["cls_pre_head"]
        ) / n_global_crops
        loss_dict["koleo"] = koleo.item()
        loss_total += self.koleo_loss_weight * n_global_crops * koleo

        # iBOT: masked patch prediction
        ibot = self.ibot_patch_loss.forward_masked(
            student_out["global"]["masked_patch_after_head"],
            teacher_out["masked_patch_centered"],
            student_masks_flat=masks,
            n_masked_patches=mask_indices_list.shape[0],
            masks_weight=masks_weight,
        )
        loss_dict["ibot"] = ibot.item()
        loss_total += self.ibot_loss_weight * ibot

        loss_dict["total"] = loss_total.item()
        return loss_total, loss_dict

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """EMA update: teacher = m * teacher + (1-m) * student."""
        if self._ema_params is None:
            s_params, t_params = [], []
            for k in self.student.keys():
                for ps, pt in zip(self.student[k].parameters(), self.teacher[k].parameters()):
                    s_params.append(ps)
                    t_params.append(pt)
            self._ema_params = (s_params, t_params)

        s_params, t_params = self._ema_params
        torch._foreach_mul_(t_params, momentum)
        torch._foreach_add_(t_params, s_params, alpha=1 - momentum)

    def train(self):
        super().train()
        self.teacher.eval()

    def forward(self, inputs):
        raise NotImplementedError("Use forward_backward() for training")

    def forward_backward(self, data, *, teacher_temp, iteration=0):
        """Full forward-backward pass. Main training entry point."""
        n_global_crops = 2
        B = data["collated_local_crops"].shape[0] // self.n_local_crops

        device = next(self.parameters()).device
        global_crops = data["collated_global_crops"].to(device)
        local_crops = data["collated_local_crops"].to(device)
        masks = data["collated_masks"].to(device)
        mask_indices = data["mask_indices_list"].to(device)
        masks_weight = data["masks_weight"].to(device)
        n_masked = data["n_masked_patches"].to(device)

        # Teacher forward (no grad)
        teacher_out = self.get_teacher_output(
            global_crops.unflatten(0, (n_global_crops, B)),
            teacher_temp=teacher_temp,
            mask_indices_list=mask_indices,
            n_masked_patches_tensor=n_masked,
        )

        # Student forward (with grad)
        student_out = self.get_student_output(
            global_crops=global_crops.unflatten(0, (n_global_crops, B)),
            local_crops=local_crops.unflatten(0, (self.n_local_crops, B)),
            masks=masks,
            mask_indices_list=mask_indices,
        )

        # Losses
        loss, loss_dict = self.compute_losses(
            teacher_out, student_out, masks, mask_indices, masks_weight,
            n_global_crops=n_global_crops,
        )

        # Backward
        loss.backward()

        return loss, loss_dict
