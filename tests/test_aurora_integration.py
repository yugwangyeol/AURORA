"""
AURORA integration tests with a tiny Qwen backbone.
Run with:
  PYTHONPATH=/home/jovyan/Scale-RAE /home/jovyan/.conda/envs/scale_rae/bin/python tests/test_aurora_integration.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
os.environ["PYTHONNOUSERSITE"] = "1"

import torch


def build_tiny_aurora_model():
    from scale_rae.model.language_model.scale_rae_qwen2 import (
        ScaleRAEQwenConfig,
        ScaleRAEQwenForCausalLM,
    )

    cfg = ScaleRAEQwenConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
        max_position_embeddings=2048,
        vision_loss="diffusion-loss",
        vision_loss_mode="query",
        vision_tower_aux_token_len_list=[256],
        diffusion_model_hidden_size=64,
        diffusion_model_channels=32,
        diffusion_model_depth=2,
        diffusion_model_heads=4,
        diffusion_model_z_channels=0,
        dit_cls="DiT",
        use_aurora=True,
        aurora_max_slots=4,
        aurora_n_register=4,
        aurora_cmd_length=8,
        aurora_dino_dim=24,
    )
    model = ScaleRAEQwenForCausalLM(cfg)
    return model, cfg


def attach_mock_encoders(model, cfg, batch_size: int):
    D = cfg.hidden_size
    feat_dim = cfg.diffusion_model_channels
    dino_dim = cfg.aurora_dino_dim
    N_img = 256

    def mock_encode(images):
        raw = torch.randn(batch_size, N_img, feat_dim, device=images.device)
        img = torch.randn(batch_size, N_img, D, device=images.device)
        return raw, img, raw.clone()

    def mock_dino(pixel_values_dino=None, dino_features=None):
        if dino_features is not None:
            return dino_features.to(dtype=torch.float32)
        return torch.randn(batch_size, N_img, dino_dim, device=pixel_values_dino.device)

    model._encode_images_aurora = mock_encode
    model._encode_dino_aurora = mock_dino


def build_batch(batch_size: int, cfg):
    images = torch.randn(batch_size, 3, 384, 384)
    target_images = torch.randn(batch_size, 3, 384, 384)
    pixel_values_dino = torch.randn(batch_size, 3, 224, 224)
    n_objects = torch.tensor([1, 3][:batch_size], dtype=torch.long)
    gt_masks = torch.rand(batch_size, cfg.aurora_max_slots - 1, 256)
    inpaint_mask = torch.rand(batch_size, 256)
    return images, target_images, pixel_values_dino, n_objects, gt_masks, inpaint_mask


def test_init_aurora():
    model, cfg = build_tiny_aurora_model()
    D = cfg.hidden_size
    assert model.aurora_cmd_query.shape == (cfg.aurora_cmd_length, D)
    assert model.aurora_op_init.shape == (1, D)
    assert model.aurora_register_init.shape == (cfg.aurora_n_register, D)
    assert model.aurora_null_slot.shape == (1, 1, D)
    assert not model.get_model().latent_queries.requires_grad
    print("init OK")


def test_forward_aurora():
    model, cfg = build_tiny_aurora_model()
    model.train()
    B = 2
    attach_mock_encoders(model, cfg, B)
    images, target_images, pixel_values_dino, n_objects, gt_masks, inpaint_mask = build_batch(B, cfg)

    loss, info = model._forward_aurora(
        images=images,
        n_objects=n_objects,
        gt_masks_patches=gt_masks,
        target_images=target_images,
        inpaint_mask_patches=inpaint_mask,
        pixel_values_dino=pixel_values_dino,
    )
    assert loss.dim() == 0
    assert loss.requires_grad
    for key in ["loss_full", "loss_slot_feat", "loss_slot_attn", "loss_stop", "loss_div", "loss_inpaint"]:
        assert key in info
    print("forward OK")


def test_forward_aurora_with_cached_dino():
    model, cfg = build_tiny_aurora_model()
    model.train()
    B = 2
    attach_mock_encoders(model, cfg, B)
    images, target_images, _, n_objects, gt_masks, inpaint_mask = build_batch(B, cfg)
    dino_features = torch.randn(B, 256, cfg.aurora_dino_dim)

    loss, info = model._forward_aurora(
        images=images,
        n_objects=n_objects,
        gt_masks_patches=gt_masks,
        target_images=target_images,
        inpaint_mask_patches=inpaint_mask,
        dino_features=dino_features,
        pixel_values_dino=None,
    )
    assert loss.dim() == 0
    assert "loss_slot_feat" in info
    print("forward cached DINO OK")


def test_backward():
    model, cfg = build_tiny_aurora_model()
    model.train()
    B = 2
    attach_mock_encoders(model, cfg, B)
    images, target_images, pixel_values_dino, n_objects, gt_masks, inpaint_mask = build_batch(B, cfg)

    loss, _ = model._forward_aurora(
        images=images,
        n_objects=n_objects,
        gt_masks_patches=gt_masks,
        target_images=target_images,
        inpaint_mask_patches=inpaint_mask,
        pixel_values_dino=pixel_values_dino,
    )
    loss.backward()
    grads = {
        "aurora_cmd_query": model.aurora_cmd_query.grad,
        "aurora_op_init": model.aurora_op_init.grad,
        "slot_head_q": model.aurora_slot_head.q_proj.weight.grad,
        "slot_proj": model.aurora_supervision_proj.proj[0].weight.grad,
    }
    for name, grad in grads.items():
        assert grad is not None, f"{name} grad missing"
        assert not torch.isnan(grad).any(), f"{name} grad has NaN"
    print("backward OK")


def test_generate_and_edit():
    model, cfg = build_tiny_aurora_model()
    model.eval()
    B = 1
    attach_mock_encoders(model, cfg, B)
    images = torch.randn(B, 3, 384, 384)

    result = model.generate_aurora(images, max_slots=3)
    assert "generated" in result
    assert "slots" in result
    assert "attn_maps" in result
    assert "K_per_sample" in result
    assert result["generated"].shape[1] == 256
    assert result["K_per_sample"].shape == (B,)
    assert int(result["K_per_sample"][0].item()) <= 3

    edited = model.edit_aurora(images, remove_indices=[0], keep_only=[0])
    assert "generated" in edited
    assert "edited_slots" in edited
    print("generate/edit OK")


if __name__ == "__main__":
    test_init_aurora()
    test_forward_aurora()
    test_forward_aurora_with_cached_dino()
    test_backward()
    test_generate_and_edit()
    print("All AURORA integration tests passed.")
