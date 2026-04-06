"""
Smoke test: verify all 3D Medical DINO components work together.
Run with: python -m medical_dino3d.test_smoke
"""
import torch
import sys


def test_patch_embed_3d():
    from medical_dino3d.patch_embed_3d import PatchEmbed3D
    pe = PatchEmbed3D(img_size=96, patch_size=16, in_chans=3, embed_dim=384, flatten_embedding=False)
    x = torch.randn(2, 3, 96, 96, 96)
    out = pe(x)
    assert out.shape == (2, 6, 6, 6, 384), f"Expected (2,6,6,6,384), got {out.shape}"
    print(f"  PatchEmbed3D: input {x.shape} -> output {out.shape}  (216 tokens)")


def test_rope_3d():
    from medical_dino3d.rope_3d import RopePositionEmbedding3D
    # embed_dim=384, num_heads=4 -> D_head=96, 96/6=16 freqs per axis
    rope = RopePositionEmbedding3D(embed_dim=384, num_heads=4, dtype=torch.float32)
    sin, cos = rope(D=6, H=6, W=6)
    assert sin.shape == (216, 96), f"Expected (216, 96), got {sin.shape}"
    print(f"  RoPE3D: grid (6,6,6) -> sin/cos {sin.shape}  (216 positions x 96 dims)")


def test_masking_3d():
    from medical_dino3d.masking_3d import MaskingGenerator3D
    mg = MaskingGenerator3D(input_size=(6, 6, 6), num_masking_patches=80)
    mask = mg(80)
    assert mask.shape == (6, 6, 6)
    n_masked = mask.sum()
    print(f"  Masking3D: grid (6,6,6), target=80, actual={n_masked} masked patches")


def test_augmentation_3d():
    from medical_dino3d.augmentations_3d import DataAugmentationMedical3D
    aug = DataAugmentationMedical3D(
        global_crops_size=96, local_crops_size=64, n_local_crops=4, modality="ct"
    )
    volume = torch.randn(1, 128, 128, 128) * 200 + 40
    out = aug(volume)
    g = out["global_crops"]
    l = out["local_crops"]
    print(f"  Augmentation3D: {len(g)} global crops {g[0].shape}, {len(l)} local crops {l[0].shape}")
    assert g[0].shape == (3, 96, 96, 96)
    assert l[0].shape == (3, 64, 64, 64)


def test_vision_transformer_3d():
    from medical_dino3d.vision_transformer_3d import vit3d_small
    model = vit3d_small(img_size=96, patch_size=16, n_storage_tokens=4)
    model.init_weights()
    x = torch.randn(2, 3, 96, 96, 96)
    out = model(x, is_training=True)
    cls = out["x_norm_clstoken"]
    patch = out["x_norm_patchtokens"]
    reg = out["x_storage_tokens"]
    print(f"  ViT3D-Small: CLS {cls.shape}, patches {patch.shape}, registers {reg.shape}")
    assert cls.shape == (2, 384)
    assert patch.shape == (2, 216, 384)
    assert reg.shape == (2, 4, 384)


def test_vit3d_multi_crop():
    from medical_dino3d.vision_transformer_3d import vit3d_small
    model = vit3d_small(img_size=96, patch_size=16, n_storage_tokens=4)
    model.init_weights()
    # Simulate multi-crop: 2 global (96^3) + 4 local (64^3)
    global_crop = torch.randn(4, 3, 96, 96, 96)   # 2 crops * batch 2
    local_crop = torch.randn(8, 3, 64, 64, 64)     # 4 crops * batch 2
    out = model.forward_features_list(
        [global_crop, local_crop],
        [None, None],
    )
    g_cls = out[0]["x_norm_clstoken"]
    l_cls = out[1]["x_norm_clstoken"]
    g_patch = out[0]["x_norm_patchtokens"]
    l_patch = out[1]["x_norm_patchtokens"]
    print(f"  Multi-crop: global CLS {g_cls.shape}, patches {g_patch.shape}")
    print(f"              local  CLS {l_cls.shape}, patches {l_patch.shape}")
    # Global: 96/16=6 -> 6^3=216 tokens
    assert g_patch.shape == (4, 216, 384)
    # Local: 64/16=4 -> 4^3=64 tokens
    assert l_patch.shape == (8, 64, 384)


def test_model_init():
    from medical_dino3d.ssl_meta_arch_3d import MedicalDINO3D
    model = MedicalDINO3D(
        arch="vit3d_small",
        img_size=96,
        patch_size=16,
        n_local_crops=4,
        dino_head_n_prototypes=4096,  # smaller for testing
        ibot_head_n_prototypes=2048,
    )
    model.init_weights()
    n_student = sum(p.numel() for p in model.student.parameters()) / 1e6
    n_teacher = sum(p.numel() for p in model.teacher.parameters()) / 1e6
    print(f"  MedicalDINO3D: student {n_student:.1f}M, teacher {n_teacher:.1f}M params")


def main():
    print("=" * 60)
    print("3D Medical DINO - Smoke Test")
    print("=" * 60)

    tests = [
        ("PatchEmbed3D", test_patch_embed_3d),
        ("RoPE3D", test_rope_3d),
        ("Masking3D", test_masking_3d),
        ("Augmentation3D", test_augmentation_3d),
        ("ViT3D", test_vision_transformer_3d),
        ("ViT3D Multi-Crop", test_vit3d_multi_crop),
        ("MedicalDINO3D Init", test_model_init),
    ]

    passed = 0
    for name, test_fn in tests:
        try:
            print(f"\n[{name}]")
            test_fn()
            print(f"  PASSED")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    print(f"{'=' * 60}")

    if passed < len(tests):
        sys.exit(1)


if __name__ == "__main__":
    main()
