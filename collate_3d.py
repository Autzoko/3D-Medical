"""
3D Collation and Masking for Medical DINO.

Adapts DINOv3's collate_data_and_cast to handle 3D volumes.

DINOv3 original (collate.py):
    - Stacks 2D crops: (n_crops * B, C, H, W)
    - Generates 2D masks: (2*B, H_patches, W_patches)
    - Uses torch.roll with 2D shifts: (shift_x, shift_y)

Our 3D version:
    - Stacks 3D crops: (n_crops * B, C, D, H, W)
    - Generates 3D masks: (2*B, D_patches, H_patches, W_patches)
    - Uses torch.roll with 3D shifts: (shift_d, shift_h, shift_w)
"""

import random
from typing import Tuple

import torch
from torch import Tensor

from masking_3d import MaskingGenerator3D


def collate_medical_3d(
    samples_list: list,
    mask_ratio_tuple: Tuple[float, float] = (0.1, 0.5),
    mask_probability: float = 0.5,
    n_tokens: int = 216,
    mask_generator: MaskingGenerator3D = None,
    patch_grid_size: Tuple[int, int, int] = (6, 6, 6),
) -> dict:
    """
    Collate function for 3D medical DINO training.

    Args:
        samples_list: list of (augmented_dict, target) tuples from the dataset
        mask_ratio_tuple: (min_ratio, max_ratio) for iBOT masking
        mask_probability: fraction of batch to apply masking to
        n_tokens: number of patch tokens per volume
        mask_generator: MaskingGenerator3D instance
        patch_grid_size: (D', H', W') patch grid dimensions

    Returns:
        dict with collated crops, masks, and mask metadata
    """
    n_global_crops = 2
    B = len(samples_list)

    # Separate crops from augmentation output
    global_crops = [[] for _ in range(n_global_crops)]
    global_crops_teacher = [[] for _ in range(n_global_crops)]
    local_crops_lists = None

    for sample, _target in samples_list:
        for i in range(n_global_crops):
            global_crops[i].append(sample["global_crops"][i])
            global_crops_teacher[i].append(sample["global_crops_teacher"][i])
        if local_crops_lists is None:
            n_local = len(sample["local_crops"])
            local_crops_lists = [[] for _ in range(n_local)]
        for i, lc in enumerate(sample["local_crops"]):
            local_crops_lists[i].append(lc)

    # Stack into batched tensors
    # Shape: (n_crops * B, C, D, H, W) — interleaved by crop index
    collated_global = torch.stack(
        [crop for crop_list in global_crops for crop in crop_list]
    )  # (n_global * B, C, D, H, W)
    collated_global_teacher = torch.stack(
        [crop for crop_list in global_crops_teacher for crop in crop_list]
    )
    collated_local = torch.stack(
        [crop for crop_list in local_crops_lists for crop in crop_list]
    )  # (n_local * B, C, d, h, w)

    # Generate masks for iBOT (only for global crops)
    if mask_generator is None:
        mask_generator = MaskingGenerator3D(patch_grid_size)

    n_samples_masked = int(B * mask_probability)
    masks_list = []
    probs = torch.linspace(mask_ratio_tuple[0], mask_ratio_tuple[1], n_samples_masked + 1)

    for i in range(n_samples_masked):
        prob_max = probs[i + 1].item()
        num_patches_to_mask = int(n_tokens * prob_max)
        mask = mask_generator(num_patches_to_mask)
        masks_list.append(torch.from_numpy(mask))

    # Non-masked samples get empty masks
    for _ in range(n_samples_masked, B):
        masks_list.append(torch.from_numpy(mask_generator(0)))

    random.shuffle(masks_list)

    # Duplicate masks for each global crop: (n_global * B, D_p, H_p, W_p)
    collated_masks_list = []
    for _ in range(n_global_crops):
        collated_masks_list.extend(masks_list)
    collated_masks = torch.stack(collated_masks_list)  # (n_global * B, D_p, H_p, W_p)

    # Flatten masks for indexing
    masks_flat = collated_masks.flatten(1)  # (n_global * B, n_tokens)

    # Compute mask indices and weights (same logic as DINOv3)
    mask_indices_list = masks_flat.flatten().nonzero(as_tuple=False).squeeze(-1)
    n_masked_per_sample = masks_flat.sum(dim=-1).clamp(min=1.0)

    # Weights: inverse of per-sample mask count (normalizes loss contribution)
    masks_weight = (1.0 / n_masked_per_sample).unsqueeze(-1).expand_as(masks_flat)[masks_flat]

    return {
        "collated_global_crops": collated_global,
        "collated_global_crops_teacher": collated_global_teacher,
        "collated_local_crops": collated_local,
        "collated_masks": masks_flat.bool(),           # (n_global*B, n_tokens)
        "mask_indices_list": mask_indices_list.long(),
        "masks_weight": masks_weight.float(),
        "n_masked_patches": torch.tensor(mask_indices_list.shape[0]),
        "upperbound": int(masks_flat.sum().item()),
    }
