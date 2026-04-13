"""
Evaluation tools for 3D Medical DINO.

Three evaluation protocols to verify pretrained features are useful:

1. k-NN evaluation:    Freeze backbone, extract features, nearest-neighbor classify
2. Linear probing:     Freeze backbone, train a single linear layer
3. Segmentation head:  Freeze backbone, train a lightweight 3D segmentation decoder

All three compare: pretrained vs random init (from scratch)
If pretrained beats random init, self-supervised pretraining is working.

Usage:
    # k-NN evaluation (fastest, no training)
    python evaluate.py --mode knn \
        --checkpoint checkpoints/checkpoint_0099.pt \
        --data_dir /data/medical3d/acdc \
        --label_dir /data/medical3d/acdc

    # Linear probing
    python evaluate.py --mode linear \
        --checkpoint checkpoints/checkpoint_0099.pt \
        --data_dir /data/medical3d/bcv

    # Compare pretrained vs random init
    python evaluate.py --mode linear \
        --checkpoint checkpoints/checkpoint_0099.pt \
        --compare_random \
        --data_dir /data/medical3d/bcv
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval")

try:
    import nibabel as nib
except ImportError:
    nib = None


# ============================================================================
# 1. Feature extraction (shared by all evaluation methods)
# ============================================================================

def load_backbone(checkpoint_path: str, arch: str = "vit3d_small", device: str = "cuda"):
    """Load pretrained backbone from a training checkpoint."""
    from vision_transformer_3d import vit3d_small, vit3d_base

    arch_fn = {"vit3d_small": vit3d_small, "vit3d_base": vit3d_base}[arch]
    model = arch_fn(img_size=96, patch_size=16, n_storage_tokens=4)

    if checkpoint_path and checkpoint_path != "random":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("student", ckpt)
        # Extract backbone weights (strip "backbone." prefix)
        backbone_state = {}
        for k, v in state.items():
            if k.startswith("backbone."):
                backbone_state[k[len("backbone."):]] = v
        if backbone_state:
            model.load_state_dict(backbone_state, strict=False)
            logger.info(f"Loaded pretrained backbone from {checkpoint_path} ({len(backbone_state)} params)")
        else:
            logger.warning(f"No backbone weights found in {checkpoint_path}, using random init")
    else:
        model.init_weights()
        logger.info("Using randomly initialized backbone (no pretraining)")

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, dataloader, device="cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract CLS token features and labels from a dataloader."""
    all_features = []
    all_labels = []
    for batch_idx, (volumes, labels) in enumerate(dataloader):
        volumes = volumes.to(device)
        out = model(volumes, is_training=True)
        cls_features = out["x_norm_clstoken"]  # (B, embed_dim)
        all_features.append(cls_features.cpu())
        all_labels.append(labels)
        if (batch_idx + 1) % 20 == 0:
            logger.info(f"  Extracted {(batch_idx+1) * volumes.shape[0]} samples...")
    return torch.cat(all_features), torch.cat(all_labels)


# ============================================================================
# 2. Labeled dataset for evaluation
# ============================================================================

class LabeledMedical3DDataset(Dataset):
    """
    Dataset that loads volumes WITH labels for evaluation.

    For classification: each volume gets a single class label
    For segmentation: each volume has a corresponding label volume

    Supports common dataset layouts:
        - ACDC: patient001/patient001_frame01.nii.gz + patient001_frame01_gt.nii.gz
        - BCV:  img0001.nii.gz + label0001.nii.gz
        - General: imageXXX.nii.gz + labelXXX.nii.gz (matched by number)
    """

    def __init__(
        self,
        data_dir: str,
        target_size: Tuple[int, int, int] = (96, 96, 96),
        task: str = "classification",  # "classification" or "segmentation"
        modality: str = "auto",
    ):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.task = task

        # Auto-detect modality
        from dataset import guess_modality
        self.modality = guess_modality(self.data_dir.name) if modality == "auto" else modality

        # Find image-label pairs
        self.pairs = self._find_pairs()
        if len(self.pairs) == 0:
            raise FileNotFoundError(f"No image-label pairs found in {data_dir}")
        logger.info(f"Found {len(self.pairs)} labeled samples in {data_dir} (task={task})")

    def _find_pairs(self) -> List[Tuple[Path, int]]:
        """Find image files and assign labels."""
        all_nifti = sorted(
            list(self.data_dir.rglob("*.nii.gz")) + list(self.data_dir.rglob("*.nii"))
        )

        label_keywords = ["label", "seg", "mask", "gt", "truth"]

        if self.task == "classification":
            # For classification: use parent directory name as class label
            # e.g., acdc/patient001/ -> group by patient, label by condition
            images = [f for f in all_nifti
                      if not any(kw in f.name.lower() for kw in label_keywords)]

            # Group by parent directory (each directory = one class or patient)
            dir_to_label = {}
            pairs = []
            for img in images:
                parent = img.parent.name
                if parent not in dir_to_label:
                    dir_to_label[parent] = len(dir_to_label)
                pairs.append((img, dir_to_label[parent]))

            n_classes = len(dir_to_label)
            logger.info(f"  Classification: {n_classes} classes from directory structure")
            return pairs

        else:  # segmentation
            # Match images with label files
            images = [f for f in all_nifti
                      if not any(kw in f.name.lower() for kw in label_keywords)]
            label_files = [f for f in all_nifti
                           if any(kw in f.name.lower() for kw in label_keywords)]

            # Try to match by filename similarity
            pairs = []
            for img in images:
                # Look for a label with similar name
                img_id = ''.join(filter(str.isdigit, img.stem.replace('.nii', '')))
                for lbl in label_files:
                    lbl_id = ''.join(filter(str.isdigit, lbl.stem.replace('.nii', '')))
                    if img_id and img_id == lbl_id and img.parent == lbl.parent:
                        pairs.append((img, lbl))
                        break

            logger.info(f"  Segmentation: {len(pairs)} image-label pairs matched")
            # Return (image_path, label_path) — label is a path, not int
            return pairs

    def _load_volume(self, path, is_label=False):
        nii_data = nib.load(str(path))
        vol = torch.from_numpy(nii_data.get_fdata().astype("float32"))
        if vol.ndim == 3:
            vol = vol.unsqueeze(0)
        elif vol.ndim == 4:
            vol = vol[..., 0].unsqueeze(0)

        if is_label:
            # Labels: nearest neighbor interpolation to preserve class indices
            vol = F.interpolate(
                vol.unsqueeze(0), size=self.target_size, mode="nearest"
            ).squeeze(0).long()
        else:
            # Images: intensity preprocessing + trilinear
            if self.modality == "ct":
                vol = vol.clamp(-1024, 3071)
            else:
                p1 = torch.quantile(vol.float(), 0.01)
                p99 = torch.quantile(vol.float(), 0.99)
                vol = (vol - p1) / (p99 - p1 + 1e-8)
                vol = vol.clamp(0, 1)

            vol = F.interpolate(
                vol.unsqueeze(0).float(), size=self.target_size,
                mode="trilinear", align_corners=False
            ).squeeze(0)

            # Multi-channel normalization
            from augmentations_3d import ct_multi_channel_normalize, mri_multi_channel_normalize
            if self.modality == "ct":
                vol = ct_multi_channel_normalize(vol)
            else:
                vol = mri_multi_channel_normalize(vol)

        return vol

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.task == "classification":
            img_path, label = self.pairs[idx]
            volume = self._load_volume(img_path)
            return volume, label
        else:
            img_path, lbl_path = self.pairs[idx]
            volume = self._load_volume(img_path)
            label = self._load_volume(lbl_path, is_label=True)
            return volume, label.squeeze(0)  # (D, H, W) long tensor


# ============================================================================
# 3. k-NN Evaluation (simplest, no training needed)
# ============================================================================

def evaluate_knn(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    k_values: List[int] = [5, 10, 20],
    device: str = "cuda",
) -> Dict[str, float]:
    """
    k-NN evaluation on frozen features.
    Same protocol as DINOv3's eval/knn.py.
    """
    logger.info("Extracting train features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    logger.info("Extracting test features...")
    test_features, test_labels = extract_features(model, test_loader, device)

    # L2 normalize
    train_features = F.normalize(train_features, dim=-1)
    test_features = F.normalize(test_features, dim=-1)

    results = {}
    for k in k_values:
        # Cosine similarity -> find k nearest neighbors
        sim = test_features @ train_features.t()  # (N_test, N_train)
        topk_sim, topk_idx = sim.topk(k, dim=1)

        # Weighted vote
        topk_labels = train_labels[topk_idx]  # (N_test, k)
        n_classes = train_labels.max().item() + 1

        # Temperature-weighted voting (same as DINOv3, T=0.07)
        weights = (topk_sim / 0.07).exp()  # (N_test, k)
        votes = torch.zeros(test_features.shape[0], n_classes)
        for c in range(n_classes):
            mask = (topk_labels == c).float()
            votes[:, c] = (weights * mask).sum(dim=1)

        preds = votes.argmax(dim=1)
        acc = (preds == test_labels).float().mean().item() * 100
        results[f"knn_k{k}_acc"] = acc
        logger.info(f"  k-NN (k={k}): {acc:.1f}%")

    return results


# ============================================================================
# 4. Linear Probing (train one linear layer on frozen features)
# ============================================================================

def evaluate_linear_probe(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_classes: int,
    epochs: int = 50,
    lr: float = 0.01,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Linear probing: freeze backbone, train a linear classifier.
    Same protocol as DINOv3's eval/linear.py.
    """
    # Extract features once (frozen backbone)
    logger.info("Extracting train features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    logger.info("Extracting test features...")
    test_features, test_labels = extract_features(model, test_loader, device)

    embed_dim = train_features.shape[1]
    classifier = nn.Linear(embed_dim, n_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Train
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    batch_size = 256

    best_acc = 0
    for epoch in range(epochs):
        classifier.train()
        perm = torch.randperm(len(train_features))
        epoch_loss = 0
        n_batches = 0
        for i in range(0, len(train_features), batch_size):
            idx = perm[i:i+batch_size]
            logits = classifier(train_features[idx])
            loss = criterion(logits, train_labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Evaluate
        classifier.eval()
        with torch.no_grad():
            test_logits = classifier(test_features.to(device))
            test_preds = test_logits.argmax(dim=1)
            acc = (test_preds == test_labels.to(device)).float().mean().item() * 100
            best_acc = max(best_acc, acc)

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/n_batches:.4f} | Acc: {acc:.1f}%")

    logger.info(f"  Linear probe best accuracy: {best_acc:.1f}%")
    return {"linear_probe_acc": best_acc}


# ============================================================================
# 5. Segmentation Head (lightweight decoder on frozen backbone)
# ============================================================================

class SegmentationHead3D(nn.Module):
    """
    Lightweight 3D segmentation head on frozen backbone features.
    Takes patch tokens, reshapes to 3D, upsamples to original resolution.
    """

    def __init__(self, embed_dim: int, n_classes: int, patch_grid: int = 6, output_size: int = 96):
        super().__init__()
        self.patch_grid = patch_grid
        self.output_size = output_size

        self.head = nn.Sequential(
            nn.Conv3d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, n_classes, kernel_size=1),
        )

    def forward(self, patch_tokens):
        # patch_tokens: (B, N, D) where N = patch_grid^3
        B, N, D = patch_tokens.shape
        g = self.patch_grid
        x = patch_tokens.transpose(1, 2).reshape(B, D, g, g, g)  # (B, D, g, g, g)
        x = F.interpolate(x, size=self.output_size, mode="trilinear", align_corners=False)
        x = self.head(x)  # (B, n_classes, out, out, out)
        return x


def compute_dice(pred: torch.Tensor, target: torch.Tensor, n_classes: int) -> Dict[str, float]:
    """Compute per-class Dice score."""
    dice_scores = {}
    pred_classes = pred.argmax(dim=1)  # (B, D, H, W)
    for c in range(1, n_classes):  # skip background
        pred_c = (pred_classes == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2 * intersection / (union + 1e-8)).item()
        dice_scores[f"dice_class_{c}"] = dice
    dice_scores["dice_mean"] = sum(dice_scores.values()) / max(len(dice_scores), 1)
    return dice_scores


def evaluate_segmentation(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_classes: int,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
    freeze_backbone: bool = True,
) -> Dict[str, float]:
    """Train a segmentation head on frozen (or fine-tuned) backbone."""
    embed_dim = model.embed_dim
    seg_head = SegmentationHead3D(embed_dim, n_classes).to(device)

    if freeze_backbone:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        params = seg_head.parameters()
        logger.info("Segmentation: backbone FROZEN, training head only")
    else:
        model.train()
        params = list(model.parameters()) + list(seg_head.parameters())
        logger.info("Segmentation: FINE-TUNING backbone + head")

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_dice = 0
    for epoch in range(epochs):
        seg_head.train()
        if not freeze_backbone:
            model.train()
        epoch_loss = 0
        n_batches = 0

        for volumes, labels in train_loader:
            volumes = volumes.to(device)
            labels = labels.to(device).long()

            with torch.set_grad_enabled(not freeze_backbone):
                out = model(volumes, is_training=True)
            patch_tokens = out["x_norm_patchtokens"]

            if freeze_backbone:
                patch_tokens = patch_tokens.detach()

            logits = seg_head(patch_tokens)

            # Resize labels to match output if needed
            if logits.shape[2:] != labels.shape[1:]:
                labels = F.interpolate(
                    labels.unsqueeze(1).float(), size=logits.shape[2:], mode="nearest"
                ).squeeze(1).long()

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Evaluate
        seg_head.eval()
        model.eval()
        all_dice = []
        with torch.no_grad():
            for volumes, labels in test_loader:
                volumes = volumes.to(device)
                labels = labels.to(device).long()
                out = model(volumes, is_training=True)
                logits = seg_head(out["x_norm_patchtokens"])
                if logits.shape[2:] != labels.shape[1:]:
                    labels = F.interpolate(
                        labels.unsqueeze(1).float(), size=logits.shape[2:], mode="nearest"
                    ).squeeze(1).long()
                dice = compute_dice(logits, labels, n_classes)
                all_dice.append(dice["dice_mean"])

        mean_dice = sum(all_dice) / max(len(all_dice), 1)
        best_dice = max(best_dice, mean_dice)

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/max(n_batches,1):.4f} | "
                f"Dice: {mean_dice:.4f}"
            )

    logger.info(f"  Best mean Dice: {best_dice:.4f}")
    return {"seg_dice_mean": best_dice}


# ============================================================================
# 6. Full comparison: pretrained vs random init
# ============================================================================

def compare_pretrained_vs_random(args):
    """Run the same evaluation on pretrained and random init, print comparison."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build dataset
    dataset = LabeledMedical3DDataset(
        data_dir=args.data_dir,
        target_size=(96, 96, 96),
        task=args.task,
    )

    # Split train/test
    n_test = max(int(len(dataset) * 0.2), 1)
    n_train = len(dataset) - n_test
    train_set, test_set = random_split(dataset, [n_train, n_test],
                                        generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    results = {}

    # --- Pretrained ---
    logger.info("=" * 60)
    logger.info("Evaluating PRETRAINED model")
    logger.info("=" * 60)
    pretrained_model = load_backbone(args.checkpoint, args.arch, device)

    if args.mode == "knn":
        results["pretrained"] = evaluate_knn(pretrained_model, train_loader, test_loader, device=device)
    elif args.mode == "linear":
        n_classes = max(dataset.pairs, key=lambda x: x[1])[1] + 1 if args.task == "classification" else 2
        results["pretrained"] = evaluate_linear_probe(
            pretrained_model, train_loader, test_loader, n_classes, device=device
        )
    elif args.mode == "segmentation":
        n_classes = args.n_classes
        results["pretrained"] = evaluate_segmentation(
            pretrained_model, train_loader, test_loader, n_classes, device=device
        )

    # --- Random init ---
    if args.compare_random:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Evaluating RANDOM INIT model (no pretraining)")
        logger.info("=" * 60)
        random_model = load_backbone("random", args.arch, device)

        if args.mode == "knn":
            results["random"] = evaluate_knn(random_model, train_loader, test_loader, device=device)
        elif args.mode == "linear":
            n_classes = max(dataset.pairs, key=lambda x: x[1])[1] + 1 if args.task == "classification" else 2
            results["random"] = evaluate_linear_probe(
                random_model, train_loader, test_loader, n_classes, device=device
            )
        elif args.mode == "segmentation":
            n_classes = args.n_classes
            results["random"] = evaluate_segmentation(
                random_model, train_loader, test_loader, n_classes, device=device
            )

    # --- Print comparison ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, r in results.items():
        print(f"\n  {name.upper()}:")
        for metric, value in r.items():
            print(f"    {metric}: {value:.2f}")

    if "pretrained" in results and "random" in results:
        print("\n  IMPROVEMENT (pretrained - random):")
        for metric in results["pretrained"]:
            if metric in results["random"]:
                diff = results["pretrained"][metric] - results["random"][metric]
                symbol = "+" if diff > 0 else ""
                print(f"    {metric}: {symbol}{diff:.2f}")
    print()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser("3D Medical DINO Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to training checkpoint, or 'random' for no pretraining")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with labeled medical data")
    parser.add_argument("--mode", type=str, default="knn",
                        choices=["knn", "linear", "segmentation"])
    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification", "segmentation"])
    parser.add_argument("--arch", type=str, default="vit3d_small")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--compare_random", action="store_true",
                        help="Also evaluate a randomly initialized model for comparison")
    parser.add_argument("--n_classes", type=int, default=4,
                        help="Number of segmentation classes (for segmentation mode)")

    args = parser.parse_args()
    compare_pretrained_vs_random(args)


if __name__ == "__main__":
    main()
