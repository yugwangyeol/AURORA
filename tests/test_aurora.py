"""
AURORA v2 utility tests.
Run with:
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/jovyan/AURORA /home/jovyan/.conda/envs/scale_rae/bin/python tests/test_aurora.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import torch

from scale_rae.model.object_centric.aurora_utils import (
    build_aurora_v2_attention_mask,
    compute_diversity_loss,
    compute_mask_loss,
    extract_attention_logits,
    extract_attention_maps,
    hungarian_match,
    sample_k_for_batch,
)


def test_attention_mask():
    device = torch.device("cpu")
    mask = build_aurora_v2_attention_mask(256, 8, 3, 4, 256, device, n_rae_anchor=1)
    L = 256 + 8 + 3 + 4 + 1 + 256
    assert mask.shape == (1, 1, L, L)

    cmd0 = 0
    img0 = 8
    obj0 = 8 + 256
    obj1 = obj0 + 1
    reg0 = obj0 + 3
    anchor0 = reg0 + 4
    rae0 = anchor0 + 1

    assert mask[0, 0, obj0, cmd0].item() == 0.0
    assert mask[0, 0, obj0, img0].item() == 0.0
    assert mask[0, 0, obj1, obj0].item() == 0.0
    assert mask[0, 0, obj0, obj1].item() == float("-inf")
    assert mask[0, 0, anchor0, anchor0].item() == 0.0
    assert mask[0, 0, anchor0, cmd0].item() == float("-inf")
    assert mask[0, 0, rae0, cmd0].item() == float("-inf")
    assert mask[0, 0, rae0, img0].item() == float("-inf")
    assert mask[0, 0, rae0, anchor0].item() == 0.0
    assert mask[0, 0, rae0, obj0].item() == 0.0
    assert mask[0, 0, reg0, obj1].item() == 0.0
    print("attention mask OK")


def test_sampling_and_matching():
    for _ in range(8):
        k = sample_k_for_batch([1, 3, 4], 5)
        assert 1 <= k <= 1

    pred_logits = torch.tensor([
        [4.0] * 128 + [-4.0] * 128,
        [-4.0] * 128 + [4.0] * 128,
    ])
    gt = torch.tensor([
        [1.0] * 128 + [0.0] * 128,
        [0.0] * 128 + [1.0] * 128,
    ])
    matches = hungarian_match(pred_logits, gt)
    assert matches == [(0, 0), (1, 1)]
    print("sampling/matching OK")


def test_attention_map_and_losses():
    B, L, D = 2, 270, 16
    img_start, img_end = 0, 256
    obj_positions = [256, 257]

    lm_output = torch.randn(B, L, D)
    logits = extract_attention_logits(lm_output, obj_positions, img_start, img_end)
    maps = extract_attention_maps(lm_output, obj_positions, img_start, img_end)
    assert logits.shape == (B, 2, 256)
    assert maps.shape == (B, 2, 256)

    gt_masks = torch.rand(B, 2, 256)
    matching = [[(0, 0), (1, 1)] for _ in range(B)]
    mask_loss = compute_mask_loss(
        pred_logits=logits,
        gt_masks=gt_masks,
        all_matchings=matching,
    )
    div_loss = compute_diversity_loss(maps)
    assert mask_loss.dim() == 0
    assert div_loss.dim() == 0
    print("maps/losses OK")


if __name__ == "__main__":
    test_attention_mask()
    test_sampling_and_matching()
    test_attention_map_and_losses()
    print("All AURORA unit tests passed.")
