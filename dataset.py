"""
Medical 3D Volume Dataset for DINO training.

Supports loading NIfTI (.nii.gz) volumes from multiple sources.
Handles CT, MRI, and PET with appropriate preprocessing.
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import ConcatDataset, Dataset

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

from augmentations_3d import DataAugmentationMedical3D


# Known dataset modality mapping
DATASET_MODALITY = {
    "acdc": "mri",
    "amos": "ct",       # AMOS has both CT and MRI, but majority is CT
    "kits": "ct",
    "kits23": "ct",
    "brats": "mri",
    "totalsegmentator": "ct",
    "totalseg": "ct",
    "flare": "ct",
    "flare22": "ct",
    "lits": "ct",
    "autopet": "ct",
    "bcv": "ct",
    "mms": "mri",
    "ct_org": "ct",
    "word": "ct",
    "abdomenatlas": "ct",
    "structseg": "ct",
    "pelvic": "ct",
}


def guess_modality(dir_name: str) -> str:
    """Guess modality from directory name."""
    dir_lower = dir_name.lower()
    for key, modality in DATASET_MODALITY.items():
        if key in dir_lower:
            return modality
    # Default to CT
    return "ct"


class Medical3DDataset(Dataset):
    """
    Dataset for loading 3D medical volumes from a single directory.
    """

    def __init__(
        self,
        root: str,
        modality: str = "auto",
        target_size: Tuple[int, int, int] = (128, 128, 128),
        augmentation: Optional[DataAugmentationMedical3D] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        self.root = Path(root)
        self.target_size = target_size

        # Auto-detect modality from directory name
        if modality == "auto":
            self.modality = guess_modality(self.root.name)
        else:
            self.modality = modality

        self.augmentation = augmentation or DataAugmentationMedical3D(modality=self.modality)

        # Patterns to exclude (labels, segmentation masks, etc.)
        self.exclude_patterns = exclude_patterns or [
            "label", "seg", "mask", "gt", "truth", "annotation"
        ]

        # Find all NIfTI files, filtering out labels
        all_files = sorted(
            list(self.root.rglob("*.nii.gz")) + list(self.root.rglob("*.nii"))
        )
        self.file_paths = [f for f in all_files if not self._is_label(f)]

        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No NIfTI image files found in {root}")

        print(f"  {self.root.name}: {len(self.file_paths)} volumes "
              f"(modality={self.modality}, filtered from {len(all_files)} total files)")

    def _is_label(self, path: Path) -> bool:
        """Check if a file is a label/segmentation file by its name."""
        name_lower = str(path).lower()
        return any(p in name_lower for p in self.exclude_patterns)

    def __len__(self):
        return len(self.file_paths)

    def _load_and_preprocess(self, path: Path) -> torch.Tensor:
        """Load a NIfTI volume and preprocess it."""
        if not HAS_NIBABEL:
            raise ImportError("nibabel is required. pip install nibabel")

        nii = nib.load(str(path))
        volume = nii.get_fdata().astype("float32")
        volume = torch.from_numpy(volume)

        # Handle dimensions
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)  # (D, H, W) -> (1, D, H, W)
        elif volume.ndim == 4:
            # Multi-channel: take first channel
            if volume.shape[-1] <= volume.shape[0]:
                # (D, H, W, C) format
                volume = volume[..., 0].unsqueeze(0)
            else:
                # (C, D, H, W) format
                volume = volume[0:1]

        # Intensity preprocessing based on modality
        if self.modality == "ct":
            volume = volume.clamp(-1024, 3071)
        else:
            # MRI/PET: percentile normalization
            p1 = torch.quantile(volume.float(), 0.01)
            p99 = torch.quantile(volume.float(), 0.99)
            volume = (volume - p1) / (p99 - p1 + 1e-8)
            volume = volume.clamp(0, 1)

        # Resize to target size
        volume = torch.nn.functional.interpolate(
            volume.unsqueeze(0).float(),
            size=self.target_size,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        return volume

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            volume = self._load_and_preprocess(path)
            augmented = self.augmentation(volume)
            return augmented, 0
        except Exception as e:
            # If a file fails to load, return a random other sample
            print(f"  [WARN] Failed to load {path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))


def build_mixed_dataset(
    data_dirs: List[str],
    augmentation: DataAugmentationMedical3D,
    target_size: Tuple[int, int, int] = (128, 128, 128),
) -> Dataset:
    """
    Build a combined dataset from multiple directories.
    Auto-detects modality per directory.

    Args:
        data_dirs: list of paths, e.g. ["/data/amos", "/data/kits", "/data/brats"]
        augmentation: shared augmentation pipeline
        target_size: volume size after resampling

    Returns:
        ConcatDataset combining all directories
    """
    datasets = []
    total = 0
    print(f"Building mixed dataset from {len(data_dirs)} sources:")
    for d in data_dirs:
        d = str(d).strip()
        if not d or not Path(d).exists():
            print(f"  [SKIP] {d} does not exist")
            continue
        try:
            ds = Medical3DDataset(
                root=d,
                modality="auto",
                target_size=target_size,
                augmentation=augmentation,
            )
            datasets.append(ds)
            total += len(ds)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")

    if len(datasets) == 0:
        raise FileNotFoundError("No valid datasets found in any of the provided directories")

    combined = ConcatDataset(datasets)
    print(f"Total: {total} volumes from {len(datasets)} datasets")
    return combined


class DummyMedical3DDataset(Dataset):
    """Dummy dataset for testing without real data."""

    def __init__(self, n_samples=100, modality="ct", **kwargs):
        self.n_samples = n_samples
        self.augmentation = DataAugmentationMedical3D(modality=modality, **kwargs)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        volume = torch.randn(1, 128, 128, 128) * 200 + 40
        augmented = self.augmentation(volume)
        return augmented, 0
