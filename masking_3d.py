"""
3D Masking Generator for iBOT masked image modeling.

Extends DINOv3's 2D MaskingGenerator to 3D volumes.

DINOv3 original (masking.py):
    - Works on 2D patch grid (H, W)
    - Generates 2D rectangular mask regions
    - mask[top:top+h, left:left+w] = True

Our 3D version:
    - Works on 3D patch grid (D, H, W)
    - Generates 3D cuboid mask regions
    - mask[z:z+d, y:y+h, x:x+w] = True
"""

import math
import random

import numpy as np


class MaskingGenerator3D:
    """
    Generate random 3D cuboid masks on a (D, H, W) patch grid.

    This mirrors DINOv3's MaskingGenerator but adds a depth dimension.
    The algorithm:
    1. Repeatedly sample random cuboid regions until we reach the target count
    2. Each cuboid has random volume and aspect ratios
    3. Fill remaining patches randomly if cuboids don't reach the target
    """

    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 3
        assert len(input_size) == 3

        self.depth, self.height, self.width = input_size
        self.num_patches = self.depth * self.height * self.width

        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches or num_masking_patches
        self.log_aspect_ratio_range = (math.log(min_aspect), math.log(1 / min_aspect))

    def get_shape(self):
        return (self.depth, self.height, self.width)

    def _mask(self, mask, max_mask_patches):
        """Try to place one random 3D cuboid region on the mask."""
        delta = 0
        for _ in range(10):  # 10 attempts to find a valid region
            target_volume = random.uniform(self.min_num_patches, max_mask_patches)

            # Random aspect ratios for the 3D cuboid
            aspect_hw = math.exp(random.uniform(*self.log_aspect_ratio_range))
            aspect_d = math.exp(random.uniform(*self.log_aspect_ratio_range))

            # Compute cuboid dimensions
            d = int(round((target_volume * aspect_d) ** (1.0 / 3.0)))
            h = int(round((target_volume / (aspect_d * aspect_hw)) ** (1.0 / 3.0)))
            w = int(round(target_volume / max(d * h, 1)))

            # Clamp to grid bounds
            d = min(d, self.depth)
            h = min(h, self.height)
            w = min(w, self.width)

            if d >= 1 and h >= 1 and w >= 1:
                z = random.randint(0, self.depth - d)
                y = random.randint(0, self.height - h)
                x = random.randint(0, self.width - w)

                region = mask[z : z + d, y : y + h, x : x + w]
                num_new = int((~region).sum())
                if 0 < num_new <= max_mask_patches:
                    mask[z : z + d, y : y + h, x : x + w] = True
                    delta += num_new

                if delta > 0:
                    break

        return delta

    def __call__(self, num_masking_patches=None):
        """Generate a 3D mask with approximately num_masking_patches masked."""
        if num_masking_patches is None:
            num_masking_patches = self.num_masking_patches

        mask = np.zeros(self.get_shape(), dtype=bool)

        if num_masking_patches == 0:
            return mask

        # Phase 1: Place random cuboid regions
        mask_count = 0
        while mask_count < num_masking_patches:
            max_patches = num_masking_patches - mask_count
            max_patches = min(max_patches, self.max_num_patches)

            delta = self._mask(mask, max_patches)
            if delta == 0:
                break
            mask_count += delta

        # Phase 2: Fill remaining randomly (same as DINOv3)
        if mask_count < num_masking_patches:
            flat_mask = mask.flatten()
            unmasked_indices = np.where(~flat_mask)[0]
            remaining = min(num_masking_patches - mask_count, len(unmasked_indices))
            if remaining > 0:
                to_add = np.random.choice(unmasked_indices, size=remaining, replace=False)
                flat_mask[to_add] = True
                mask = flat_mask.reshape(self.get_shape())

        return mask
