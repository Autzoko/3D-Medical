"""
Medical 3D Volume Dataset for DINO training.

Supports loading NIfTI (.nii.gz) volumes from multiple sources.
Handles CT, MRI, and PET with appropriate preprocessing.
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

from augmentations_3d import DataAugmentationMedical3D


class Medical3DDataset(Dataset):
    """
    Dataset for loading 3D medical volumes.

    Expected directory structure:
        root/
            volume_001.nii.gz
            volume_002.nii.gz
            ...
        OR
        root/
            dataset_a/
                volume_001.nii.gz
                ...
            dataset_b/
                volume_001.nii.gz
                ...

    The dataset loads volumes, resamples them to a target spacing,
    and applies 3D augmentations.
    """

    def __init__(
        self,
        root: str,
        modality: str = "ct",
        target_size: Tuple[int, int, int] = (128, 128, 128),
        augmentation: Optional[DataAugmentationMedical3D] = None,
    ):
        self.root = Path(root)
        self.modality = modality
        self.target_size = target_size
        self.augmentation = augmentation or DataAugmentationMedical3D(modality=modality)

        # Find all NIfTI files
        self.file_paths = sorted(
            list(self.root.rglob("*.nii.gz")) + list(self.root.rglob("*.nii"))
        )
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No NIfTI files found in {root}")

        print(f"Found {len(self.file_paths)} volumes in {root} (modality={modality})")

    def __len__(self):
        return len(self.file_paths)

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        """Load a NIfTI volume and preprocess it."""
        if not HAS_NIBABEL:
            raise ImportError("nibabel is required to load NIfTI files. pip install nibabel")

        nii = nib.load(str(path))
        volume = nii.get_fdata().astype("float32")
        volume = torch.from_numpy(volume)

        # Add channel dim if needed: (D, H, W) -> (1, D, H, W)
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        elif volume.ndim == 4:
            # Multi-channel volume (e.g., multi-sequence MRI): take first channel
            volume = volume[..., 0].unsqueeze(0)

        # Basic intensity preprocessing
        if self.modality == "ct":
            # Clip to reasonable HU range
            volume = volume.clamp(-1024, 3071)
        else:
            # Percentile normalization for MRI/PET
            p1 = torch.quantile(volume.float(), 0.01)
            p99 = torch.quantile(volume.float(), 0.99)
            volume = (volume - p1) / (p99 - p1 + 1e-8)
            volume = volume.clamp(0, 1)

        # Resize to target size using trilinear interpolation
        volume = torch.nn.functional.interpolate(
            volume.unsqueeze(0).float(),  # (1, 1, D, H, W)
            size=self.target_size,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)  # (1, D, H, W)

        return volume

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        volume = self._load_and_preprocess(path)
        augmented = self.augmentation(volume)
        return augmented, 0  # target=0 (unused in SSL)


class DummyMedical3DDataset(Dataset):
    """
    Dummy dataset for testing without real data.
    Generates random 3D volumes.
    """

    def __init__(self, n_samples=100, modality="ct", **kwargs):
        self.n_samples = n_samples
        self.augmentation = DataAugmentationMedical3D(modality=modality, **kwargs)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Random volume simulating a CT scan
        volume = torch.randn(1, 128, 128, 128) * 200 + 40  # HU-like values
        augmented = self.augmentation(volume)
        return augmented, 0
