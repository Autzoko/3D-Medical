"""
3D Medical Image Augmentations for DINO training.

Replaces DINOv3's 2D augmentations (DataAugmentationDINO in augmentations.py)
with medically-appropriate 3D transforms.

DINOv3 original augmentations (all 2D, for natural images):
    - RandomResizedCrop, RandomHorizontalFlip
    - ColorJitter, RandomGrayscale (meaningless for CT/MRI)
    - GaussianBlur, RandomSolarize

Our 3D medical augmentations:
    - 3D RandomResizedCrop (volumetric cropping)
    - 3D RandomFlip (all 3 axes independently)
    - RandomGamma, RandomNoise, RandomBlur3D (intensity augmentations)
    - Multi-channel normalization (CT windows / MRI quantiles)

Note: We use pure PyTorch operations to avoid extra dependencies.
For production, consider using MONAI or TorchIO for more robust transforms.
"""

import random
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Multi-channel normalization (converting single-channel medical volumes to 3ch)
# ---------------------------------------------------------------------------

def ct_multi_channel_normalize(volume: Tensor) -> Tensor:
    """
    Convert a single-channel CT volume to 3 channels using different HU windows.
    This is the approach used by MASS (Section 6.1 of the paper).

    Args:
        volume: (1, D, H, W) tensor in Hounsfield Units

    Returns:
        (3, D, H, W) tensor with each channel showing different tissue contrasts:
        - Channel 0: Soft tissue window (center=60 HU, width=350 HU)
        - Channel 1: Contrast window (center=15 HU, width=250 HU)
        - Channel 2: Bone window (center=150 HU, width=1200 HU)
    """
    windows = [
        (60, 350),    # soft tissue
        (15, 250),    # contrast
        (150, 1200),  # bone
    ]
    channels = []
    for center, width in windows:
        lo = center - width / 2
        hi = center + width / 2
        ch = (volume - lo) / (hi - lo)
        ch = ch.clamp(0.0, 1.0)
        channels.append(ch)
    return torch.cat(channels, dim=0)  # (3, D, H, W)


def mri_multi_channel_normalize(volume: Tensor) -> Tensor:
    """
    Convert a single-channel MRI/PET volume to 3 channels using quantile normalization.
    Different quantile ranges highlight different tissue characteristics.

    Args:
        volume: (1, D, H, W) tensor

    Returns:
        (3, D, H, W) tensor
    """
    quantile_ranges = [
        (0.05, 0.95),
        (0.15, 0.85),
        (0.01, 0.99),
    ]
    channels = []
    flat = volume.flatten()
    for lo_q, hi_q in quantile_ranges:
        lo = torch.quantile(flat.float(), lo_q)
        hi = torch.quantile(flat.float(), hi_q)
        ch = (volume - lo) / (hi - lo + 1e-8)
        ch = ch.clamp(0.0, 1.0)
        channels.append(ch)
    return torch.cat(channels, dim=0)  # (3, D, H, W)


# ---------------------------------------------------------------------------
# 3D spatial transforms
# ---------------------------------------------------------------------------

def random_resized_crop_3d(
    volume: Tensor,
    output_size: int | Tuple[int, int, int],
    scale: Tuple[float, float] = (0.5, 1.0),
) -> Tensor:
    """
    3D version of RandomResizedCrop.

    Randomly crops a sub-volume and resizes it to output_size.

    Args:
        volume: (C, D, H, W) tensor
        output_size: target spatial size
        scale: (min_scale, max_scale) fraction of the original volume

    Returns:
        (C, out_D, out_H, out_W) tensor
    """
    if isinstance(output_size, int):
        output_size = (output_size,) * 3

    C, D, H, W = volume.shape
    total_voxels = D * H * W
    target_fraction = random.uniform(scale[0], scale[1])
    target_voxels = int(total_voxels * target_fraction)

    # Compute crop dimensions (roughly cubic, with some random aspect ratio)
    aspect = random.uniform(0.75, 1.33)
    crop_d = int(round((target_voxels * aspect) ** (1.0 / 3.0)))
    crop_h = int(round((target_voxels / max(crop_d, 1)) ** 0.5))
    crop_w = int(round(target_voxels / max(crop_d * crop_h, 1)))

    crop_d = max(1, min(crop_d, D))
    crop_h = max(1, min(crop_h, H))
    crop_w = max(1, min(crop_w, W))

    # Random position
    z = random.randint(0, D - crop_d)
    y = random.randint(0, H - crop_h)
    x = random.randint(0, W - crop_w)

    cropped = volume[:, z : z + crop_d, y : y + crop_h, x : x + crop_w]

    # Resize to output_size using trilinear interpolation
    cropped = cropped.unsqueeze(0).float()  # (1, C, d, h, w)
    resized = F.interpolate(cropped, size=output_size, mode="trilinear", align_corners=False)
    return resized.squeeze(0)  # (C, out_D, out_H, out_W)


def random_flip_3d(volume: Tensor, p: float = 0.5) -> Tensor:
    """Randomly flip along each of the 3 spatial axes independently."""
    # volume: (C, D, H, W)
    if random.random() < p:
        volume = volume.flip(1)  # flip depth
    if random.random() < p:
        volume = volume.flip(2)  # flip height
    if random.random() < p:
        volume = volume.flip(3)  # flip width
    return volume


# ---------------------------------------------------------------------------
# 3D intensity transforms (applied per-crop independently)
# ---------------------------------------------------------------------------

def random_gamma(volume: Tensor, gamma_range: Tuple[float, float] = (0.7, 1.5), p: float = 0.3) -> Tensor:
    """Random gamma correction to simulate contrast variations."""
    if random.random() > p:
        return volume
    gamma = random.uniform(*gamma_range)
    # Only apply to positive values, preserve zeros
    return volume.clamp(min=0).pow(gamma)


def random_noise(volume: Tensor, std_range: Tuple[float, float] = (0.0, 0.05), p: float = 0.2) -> Tensor:
    """Add random Gaussian noise."""
    if random.random() > p:
        return volume
    std = random.uniform(*std_range)
    noise = torch.randn_like(volume) * std
    return (volume + noise).clamp(0.0, 1.0)


def random_blur_3d(volume: Tensor, p: float = 0.15) -> Tensor:
    """Simple 3D blur using average pooling + upsample."""
    if random.random() > p:
        return volume
    C, D, H, W = volume.shape
    k = random.choice([3, 5])
    pad = k // 2
    blurred = F.avg_pool3d(
        volume.unsqueeze(0), kernel_size=k, stride=1, padding=pad
    )
    return blurred.squeeze(0)


def random_brightness(volume: Tensor, factor_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.3) -> Tensor:
    """Random multiplicative brightness adjustment."""
    if random.random() > p:
        return volume
    factor = random.uniform(*factor_range)
    return (volume * factor).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Main augmentation class (replaces DINOv3's DataAugmentationDINO)
# ---------------------------------------------------------------------------

class DataAugmentationMedical3D:
    """
    3D medical volume augmentation pipeline for DINO training.

    Replaces DINOv3's DataAugmentationDINO which uses 2D torchvision transforms.

    Multi-crop strategy (adapted for 3D memory constraints):
        - 2 global crops: large sub-volumes covering 50-100% of the input
        - 4 local crops: small sub-volumes covering 10-50% of the input
        (DINOv3 uses 2 global + 8 local; we reduce local count for 3D memory)
    """

    def __init__(
        self,
        global_crops_scale: Tuple[float, float] = (0.5, 1.0),
        local_crops_scale: Tuple[float, float] = (0.1, 0.5),
        n_local_crops: int = 4,
        global_crops_size: int = 96,
        local_crops_size: int = 64,
        modality: str = "ct",  # "ct", "mri", or "pet"
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.n_local_crops = n_local_crops
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.modality = modality

    def _apply_intensity_augmentation(self, crop: Tensor) -> Tensor:
        """Apply random intensity transforms to a single crop."""
        crop = random_gamma(crop, p=0.3)
        crop = random_noise(crop, p=0.2)
        crop = random_blur_3d(crop, p=0.15)
        crop = random_brightness(crop, p=0.3)
        return crop

    def __call__(self, volume: Tensor) -> dict:
        """
        Args:
            volume: (1, D, H, W) single-channel medical volume
                    (already preprocessed: resampled to uniform spacing, intensity normalized)

        Returns:
            dict with:
                "global_crops": list of 2 tensors, each (3, gs, gs, gs)
                "local_crops": list of n_local_crops tensors, each (3, ls, ls, ls)
                "global_crops_teacher": list of 2 tensors (same spatial region, less augmentation)
        """
        # Step 1: Multi-channel normalization (1ch -> 3ch)
        if self.modality == "ct":
            volume_3ch = ct_multi_channel_normalize(volume)
        else:
            volume_3ch = mri_multi_channel_normalize(volume)

        # Step 2: Random spatial augmentation (shared across all crops)
        volume_3ch = random_flip_3d(volume_3ch, p=0.5)

        # Step 3: Generate global crops
        global_crops = []
        global_crops_teacher = []
        for _ in range(2):
            crop = random_resized_crop_3d(
                volume_3ch, self.global_crops_size, self.global_crops_scale
            )
            # Teacher version: same crop, no intensity augmentation
            global_crops_teacher.append(crop.clone())
            # Student version: with intensity augmentation
            crop = self._apply_intensity_augmentation(crop)
            global_crops.append(crop)

        # Step 4: Generate local crops
        local_crops = []
        for _ in range(self.n_local_crops):
            crop = random_resized_crop_3d(
                volume_3ch, self.local_crops_size, self.local_crops_scale
            )
            crop = self._apply_intensity_augmentation(crop)
            local_crops.append(crop)

        return {
            "global_crops": global_crops,
            "local_crops": local_crops,
            "global_crops_teacher": global_crops_teacher,
        }
