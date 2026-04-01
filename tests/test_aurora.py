"""
AURORA module/unit tests.
Run with:
  PYTHONPATH=/home/jovyan/Scale-RAE /home/jovyan/.conda/envs/scale_rae/bin/python tests/test_aurora.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import torch

from scale_rae.train.aurora_trainer import AlternatingTaskBatchSampler, AURORAMixedDataset
from scale_rae.model.object_centric.slot_head import SlotHead, SlotSupervisionProjector
from scale_rae.model.object_centric.aurora_utils import (
    build_bidirectional_mask,
    build_full_attention_mask,
    build_phase3_cache_attention_bias,
    build_phase3_mask,
    check_stopping,
    compute_diversity_loss,
    match_slot_to_mask,
)


def test_slot_head():
    D = 1536
    B = 2
    N_img = 256
    head = SlotHead(D, n_heads=8)
    query = torch.randn(B, 1, D)
    key_value = torch.randn(B, N_img, D)
    bias = torch.zeros(B, 1, 1, N_img)

    slot, attn_map = head(query, key_value, attn_bias=bias)
    assert slot.shape == (B, 1, D)
    assert attn_map.shape == (B, N_img)
    print("SlotHead OK")


def test_slot_supervision_projector():
    proj = SlotSupervisionProjector(1536, 768)
    out = proj(torch.randn(2, 1, 1536))
    assert out.shape == (2, 768)
    print("SlotSupervisionProjector OK")


def test_attention_masks():
    device = torch.device("cpu")
    bidir = build_bidirectional_mask(264, device)
    assert bidir.shape == (1, 1, 264, 264)
    assert (bidir == 0).all()

    full = build_full_attention_mask(256, 8, 3, 8, 256, device)
    L = 256 + 8 + 3 + 8 + 256
    assert full.shape == (1, 1, L, L)
    slot0 = 256 + 8
    slot1 = slot0 + 1
    reg0 = 256 + 8 + 3
    rae0 = reg0 + 8
    assert full[0, 0, slot0, 0].item() == 0.0
    assert full[0, 0, slot1, slot0].item() == 0.0
    assert full[0, 0, slot0, slot1].item() == float("-inf")
    assert full[0, 0, rae0, 0].item() == 0.0

    p3 = build_phase3_mask(8, 256, device)
    assert p3.shape == (1, 1, 264, 264)
    assert p3[0, 0, 0, 8].item() == float("-inf")
    assert p3[0, 0, 8, 0].item() == 0.0

    p3_cache = build_phase3_cache_attention_bias(
        prefix_len=270,
        n_reg=8,
        n_rae=256,
        batch_size=2,
        device=device,
        dtype=torch.float32,
    )
    assert p3_cache.shape == (2, 1, 264, 534)
    assert p3_cache[0, 0, 10, 100].item() == 0.0
    assert p3_cache[0, 0, 0, 270 + 8].item() == float("-inf")
    print("Attention masks OK")


def test_stopping():
    B = 2
    N = 256
    uniform = torch.ones(B, N) / N
    slot = torch.randn(B, 1, 32)
    prev = slot.clone()
    stop = check_stopping(uniform, slot, prev, n_patches=N)
    assert stop.all()

    peaked = torch.zeros(B, N)
    peaked[:, 0] = 1.0
    new_slot = torch.randn(B, 1, 32)
    stop2 = check_stopping(peaked, new_slot, prev, n_patches=N)
    assert not stop2.all()
    print("Stopping OK")


def test_diversity_and_match():
    B = 2
    D = 16
    slot_a = torch.randn(B, 1, D)
    slot_b = torch.randn(B, 1, D)
    n_obj = torch.tensor([2, 2])
    div = compute_diversity_loss([slot_a, slot_b], n_obj)
    assert div.dim() == 0

    attn0 = torch.zeros(B, 256)
    attn1 = torch.zeros(B, 256)
    attn0[:, :128] = 1.0 / 128
    attn1[:, 128:] = 1.0 / 128
    mask = torch.zeros(B, 256)
    mask[:, 128:] = 1.0
    idx = match_slot_to_mask([attn0, attn1], mask, n_obj)
    assert (idx == 1).all()
    print("Diversity/match OK")


def test_alternating_task_batch_sampler():
    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self, length):
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return idx

    mixed = AURORAMixedDataset(
        datasets=[TinyDataset(8), TinyDataset(8)],
        task_names=["reconstruction", "inpainting"],
        data_args=None,
    )
    sampler = AlternatingTaskBatchSampler(
        dataset=mixed,
        batch_size=2,
        warmup_steps=2,
        num_processes=2,
        drop_last=False,
        seed=123,
        step_provider=lambda: 0,
    )
    batches = list(iter(sampler))
    assert len(batches) == len(sampler)
    assert all(all(idx < 8 for idx in batch) for batch in batches[:6])
    assert all(all(idx >= 8 for idx in batch) for batch in batches[6:8])
    print("AlternatingTaskBatchSampler OK")


if __name__ == "__main__":
    test_slot_head()
    test_slot_supervision_projector()
    test_attention_masks()
    test_stopping()
    test_diversity_and_match()
    test_alternating_task_batch_sampler()
    print("All AURORA unit tests passed.")
