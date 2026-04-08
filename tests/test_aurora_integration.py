"""
AURORA v2 integration tests with a tiny Qwen backbone.
Run with:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/jovyan/AURORA /home/jovyan/.conda/envs/scale_rae/bin/python tests/test_aurora_integration.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
os.environ["PYTHONNOUSERSITE"] = "1"

import torch


def build_tiny_aurora_model(stage=1):
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
        aurora_inpaint_weight=0.5,
        aurora_training_stage=stage,
    )
    model = ScaleRAEQwenForCausalLM(cfg)
    model.im_start_id = 1
    model.im_end_id = 2
    model.config.im_start_id = 1
    model.config.im_end_id = 2
    return model, cfg


def attach_mock_aurora_io(model, cfg, batch_size: int):
    D = cfg.hidden_size
    feat_dim = cfg.diffusion_model_channels
    n_img = 256

    def mock_encode(images):
        current_batch = images.shape[0]
        raw = torch.randn(current_batch, n_img, feat_dim, device=images.device)
        img = torch.randn(current_batch, n_img, D, device=images.device)
        return raw, img, raw.clone()

    def mock_diffusion_loss(hidden, target):
        return (hidden.float().mean() - target.float().mean()).pow(2)

    def mock_infer(z, guidance_level=1.0):
        return z.float()

    model._encode_images_aurora = mock_encode
    model._aurora_compute_diffusion_loss = mock_diffusion_loss
    model.diff_head.infer = mock_infer


def build_batch(batch_size: int, cfg):
    images = torch.randn(batch_size, 3, 384, 384)
    target_images = torch.randn(batch_size, 3, 384, 384)
    n_objects = torch.tensor([1, 3][:batch_size], dtype=torch.long)
    gt_masks = torch.rand(batch_size, cfg.aurora_max_slots, 256)
    inpaint_mask = torch.rand(batch_size, 256)
    has_inpaint = torch.tensor([True] * batch_size, dtype=torch.bool)
    return images, target_images, n_objects, gt_masks, inpaint_mask, has_inpaint


def test_init_aurora():
    model, cfg = build_tiny_aurora_model(stage=1)
    D = cfg.hidden_size
    assert model.aurora_cmd_embeddings.shape == (cfg.aurora_cmd_length, D)
    assert model.aurora_obj_embedding_pool.shape == (cfg.aurora_max_slots, D)
    assert model.aurora_reg_embeddings.shape == (cfg.aurora_n_register, D)
    assert not model.get_model().latent_queries.requires_grad
    print("init OK")


def test_init_aurora_stage2():
    model, cfg = build_tiny_aurora_model(stage=2)
    assert model.get_model().latent_queries.requires_grad
    print("init stage2 OK")


def test_forward_aurora_stage1():
    model, cfg = build_tiny_aurora_model(stage=1)
    model.train()
    B = 2
    attach_mock_aurora_io(model, cfg, B)
    images, target_images, n_objects, gt_masks, inpaint_mask, has_inpaint = build_batch(B, cfg)

    loss, info = model._forward_aurora(
        images=images,
        n_objects=n_objects,
        gt_masks_patches=gt_masks,
        target_images=target_images,
        inpaint_mask_patches=inpaint_mask,
        has_inpaint=has_inpaint,
        aurora_inpaint_weight_override=0.0,
    )
    assert loss.dim() == 0
    assert loss.requires_grad
    for key in ["loss_recon", "loss_mask", "loss_div", "loss_inpaint", "inpaint_weight", "K_sampled"]:
        assert key in info, f"missing key: {key}"
    assert info["inpaint_weight"].item() == 0.0
    print("forward stage1 OK")


def test_forward_aurora_stage2():
    model, cfg = build_tiny_aurora_model(stage=2)
    model.train()
    B = 2
    attach_mock_aurora_io(model, cfg, B)
    images, target_images, n_objects, gt_masks, inpaint_mask, has_inpaint = build_batch(B, cfg)

    loss, info = model._forward_aurora(
        images=images,
        n_objects=n_objects,
        gt_masks_patches=gt_masks,
        target_images=target_images,
        inpaint_mask_patches=inpaint_mask,
        has_inpaint=has_inpaint,
        aurora_inpaint_weight_override=0.5,
    )
    assert loss.dim() == 0
    assert info["inpaint_weight"].item() == 0.5
    assert info["loss_inpaint"].item() >= 0.0
    print("forward stage2 OK")


def test_backward():
    model, cfg = build_tiny_aurora_model(stage=2)
    model.train()
    B = 2
    attach_mock_aurora_io(model, cfg, B)
    images, target_images, n_objects, gt_masks, inpaint_mask, has_inpaint = build_batch(B, cfg)

    loss, _ = model._forward_aurora(
        images=images,
        n_objects=n_objects,
        gt_masks_patches=gt_masks,
        target_images=target_images,
        inpaint_mask_patches=inpaint_mask,
        has_inpaint=has_inpaint,
        aurora_inpaint_weight_override=0.5,
    )
    loss.backward()
    grads = {
        "aurora_cmd_embeddings": model.aurora_cmd_embeddings.grad,
        "aurora_obj_embedding_pool": model.aurora_obj_embedding_pool.grad,
        "aurora_reg_embeddings": model.aurora_reg_embeddings.grad,
    }
    for name, grad in grads.items():
        assert grad is not None, f"{name} grad missing"
        assert not torch.isnan(grad).any(), f"{name} grad has NaN"
    print("backward OK")


def test_generate_edit_and_transfer():
    model, cfg = build_tiny_aurora_model(stage=2)
    model.eval()
    B = 1
    attach_mock_aurora_io(model, cfg, B)
    source = torch.randn(B, 3, 384, 384)
    target = torch.randn(B, 3, 384, 384)
    removal_mask = torch.rand(B, 256)

    generated = model.generate_aurora(source, K=3)
    assert generated["generated"].shape[1] == model.get_model().latent_queries.shape[0]
    assert generated["attn_maps"].shape[:2] == (B, 3)

    edited = model.edit_aurora(source, removal_masks=removal_mask, K=3)
    assert edited["generated"].shape[1] == model.get_model().latent_queries.shape[0]
    assert edited["removed_indices"].shape == (B,)

    transferred = model.transfer_aurora(
        source_images=source,
        target_images=target,
        source_masks=removal_mask,
        target_slot_indices=0,
        K=3,
    )
    assert transferred["generated"].shape[1] == model.get_model().latent_queries.shape[0]
    assert transferred["source_indices"].shape == (B,)
    print("inference OK")


if __name__ == "__main__":
    test_init_aurora()
    test_init_aurora_stage2()
    test_forward_aurora_stage1()
    test_forward_aurora_stage2()
    test_backward()
    test_generate_edit_and_transfer()
    print("All AURORA integration tests passed.")
