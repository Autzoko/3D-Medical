"""
Training script for 3D Medical DINO.

Adapts DINOv3's training loop (train.py) for 3D medical volumes.
Demonstrates the complete training pipeline including:
    - Model initialization
    - Data loading with 3D augmentations
    - Cosine learning rate schedule with warmup
    - Teacher EMA update with momentum schedule
    - Teacher temperature warmup
    - Gradient clipping
    - Checkpointing

Usage:
    python -m medical_dino3d.train --data_dir /path/to/nifti/volumes
    python -m medical_dino3d.train --dummy  # test with random data
"""

import argparse
import logging
import math
import os
import time
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from augmentations_3d import DataAugmentationMedical3D
from collate_3d import collate_medical_3d
from dataset import DummyMedical3DDataset, Medical3DDataset
from masking_3d import MaskingGenerator3D
from ssl_meta_arch_3d import MedicalDINO3D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical_dino3d")


# ---------------------------------------------------------------------------
# Schedulers (same as DINOv3's cosine_lr_scheduler.py)
# ---------------------------------------------------------------------------

def cosine_schedule(base_value, final_value, total_iters, warmup_iters=0, warmup_start_value=0):
    """Cosine schedule with linear warmup. Identical logic to DINOv3."""
    schedule = []
    # Warmup
    for i in range(warmup_iters):
        schedule.append(warmup_start_value + i * (base_value - warmup_start_value) / max(warmup_iters, 1))
    # Cosine decay
    for i in range(total_iters - warmup_iters):
        progress = i / max(total_iters - warmup_iters - 1, 1)
        schedule.append(final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * progress)))
    return schedule


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    # ---- Configuration ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # ---- Model ----
    model = MedicalDINO3D(
        arch=args.arch,
        img_size=args.global_crop_size,
        patch_size=args.patch_size,
        n_local_crops=args.n_local_crops,
        dino_head_n_prototypes=args.n_prototypes,
        ibot_head_n_prototypes=args.ibot_n_prototypes,
        dino_loss_weight=args.dino_loss_weight,
        ibot_loss_weight=args.ibot_loss_weight,
        koleo_loss_weight=args.koleo_loss_weight,
    ).to(device)

    model.init_weights()
    logger.info(f"Model: {args.arch}, params: {sum(p.numel() for p in model.student.parameters()) / 1e6:.1f}M")

    # ---- Data ----
    augmentation = DataAugmentationMedical3D(
        global_crops_size=args.global_crop_size,
        local_crops_size=args.local_crop_size,
        n_local_crops=args.n_local_crops,
        modality=args.modality,
    )

    if args.dummy:
        dataset = DummyMedical3DDataset(
            n_samples=args.n_dummy_samples,
            modality=args.modality,
            global_crops_size=args.global_crop_size,
            local_crops_size=args.local_crop_size,
            n_local_crops=args.n_local_crops,
        )
    else:
        dataset = Medical3DDataset(
            root=args.data_dir,
            modality=args.modality,
            augmentation=augmentation,
        )

    # Patch grid size for masking
    gs = args.global_crop_size // args.patch_size
    patch_grid_size = (gs, gs, gs)
    n_tokens = gs ** 3
    mask_generator = MaskingGenerator3D(patch_grid_size, num_masking_patches=int(n_tokens * 0.3))

    collate_fn = partial(
        collate_medical_3d,
        mask_ratio_tuple=(args.mask_ratio_min, args.mask_ratio_max),
        mask_probability=args.mask_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        patch_grid_size=patch_grid_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.student.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    # ---- Schedules (same as DINOv3) ----
    iters_per_epoch = len(dataloader)
    total_iters = args.epochs * iters_per_epoch

    lr_schedule = cosine_schedule(
        args.lr, args.min_lr, total_iters,
        warmup_iters=args.warmup_epochs * iters_per_epoch,
    )
    momentum_schedule = cosine_schedule(
        args.momentum_teacher, 1.0, total_iters,
    )
    teacher_temp_schedule = cosine_schedule(
        args.teacher_temp, args.teacher_temp, total_iters,
        warmup_iters=args.warmup_teacher_temp_epochs * iters_per_epoch,
        warmup_start_value=args.warmup_teacher_temp,
    )

    # ---- Training loop ----
    logger.info(f"Starting training: {args.epochs} epochs, {iters_per_epoch} iters/epoch")
    logger.info(f"Patch grid: {patch_grid_size}, tokens per volume: {n_tokens}")
    logger.info(f"Global crops: 2 x {args.global_crop_size}^3, Local crops: {args.n_local_crops} x {args.local_crop_size}^3")

    iteration = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t_epoch = time.time()

        for batch_idx, batch_data in enumerate(dataloader):
            # Update LR
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[iteration]

            # Add global batch size info
            batch_data["global_batch_size"] = args.batch_size

            # Forward-backward
            optimizer.zero_grad()
            teacher_temp = teacher_temp_schedule[iteration]

            loss, loss_dict = model.forward_backward(
                batch_data,
                teacher_temp=teacher_temp,
                iteration=iteration,
            )

            # Gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(model.student.parameters(), args.clip_grad)

            # Optimizer step
            optimizer.step()

            # EMA teacher update
            momentum = momentum_schedule[iteration]
            model.update_teacher(momentum)

            epoch_loss += loss.item()
            iteration += 1

            # Logging
            if batch_idx % args.log_freq == 0:
                lr = lr_schedule[iteration - 1]
                logger.info(
                    f"Epoch {epoch}/{args.epochs} | Iter {batch_idx}/{iters_per_epoch} | "
                    f"Loss: {loss.item():.4f} | "
                    f"DINO_g: {loss_dict.get('dino_global', 0):.4f} | "
                    f"DINO_l: {loss_dict.get('dino_local', 0):.4f} | "
                    f"iBOT: {loss_dict.get('ibot', 0):.4f} | "
                    f"KoLeo: {loss_dict.get('koleo', 0):.4f} | "
                    f"LR: {lr:.6f} | Momentum: {momentum:.4f} | "
                    f"Grad: {grad_norm:.2f}"
                )

        epoch_time = time.time() - t_epoch
        avg_loss = epoch_loss / iters_per_epoch
        logger.info(f"Epoch {epoch} done in {epoch_time:.1f}s | Avg loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "iteration": iteration,
                "student": model.student.state_dict(),
                "teacher": model.teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser("3D Medical DINO Training")

    # Data
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--dummy", action="store_true", help="Use random dummy data for testing")
    parser.add_argument("--n_dummy_samples", type=int, default=64)
    parser.add_argument("--modality", type=str, default="ct", choices=["ct", "mri", "pet"])
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument("--arch", type=str, default="vit3d_small", choices=["vit3d_small", "vit3d_base"])
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--global_crop_size", type=int, default=96)
    parser.add_argument("--local_crop_size", type=int, default=64)
    parser.add_argument("--n_local_crops", type=int, default=4)

    # Loss
    parser.add_argument("--n_prototypes", type=int, default=65536)
    parser.add_argument("--ibot_n_prototypes", type=int, default=8192)
    parser.add_argument("--dino_loss_weight", type=float, default=1.0)
    parser.add_argument("--ibot_loss_weight", type=float, default=1.0)
    parser.add_argument("--koleo_loss_weight", type=float, default=0.1)

    # Masking
    parser.add_argument("--mask_ratio_min", type=float, default=0.1)
    parser.add_argument("--mask_ratio_max", type=float, default=0.5)
    parser.add_argument("--mask_probability", type=float, default=0.5)

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.04)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--clip_grad", type=float, default=3.0)

    # Teacher
    parser.add_argument("--momentum_teacher", type=float, default=0.996)
    parser.add_argument("--teacher_temp", type=float, default=0.07)
    parser.add_argument("--warmup_teacher_temp", type=float, default=0.04)
    parser.add_argument("--warmup_teacher_temp_epochs", type=int, default=30)

    # Output
    parser.add_argument("--output_dir", type=str, default="./checkpoints_medical_dino3d")
    parser.add_argument("--save_freq", type=int, default=20)
    parser.add_argument("--log_freq", type=int, default=10)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
