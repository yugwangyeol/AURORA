"""Segmentation & reconstruction metrics, ported from AURORA's
`eval_captionslot_checkpoint.py` so PGOT numbers are directly comparable to
CODA (FG-ARI / mBO / mIoU / rFID).
"""
from typing import Dict, List, Optional

import math
import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# fARI (Foreground Adjusted Rand Index) — CODA's main slot metric
# ----------------------------------------------------------------------
def _adjusted_rand_index(true_ids: torch.Tensor, pred_ids: torch.Tensor, ignore_background: bool = False) -> torch.Tensor:
    if len(true_ids.shape) == 3:
        true_ids = true_ids.unsqueeze(1)
    if len(pred_ids.shape) == 3:
        pred_ids = pred_ids.unsqueeze(1)

    true_oh = F.one_hot(true_ids).float()
    pred_oh = F.one_hot(pred_ids).float()
    if ignore_background:
        true_oh = true_oh[..., 1:]

    N = torch.einsum("bthwc,bthwk->bck", true_oh, pred_oh)
    A = torch.sum(N, dim=-1)
    B = torch.sum(N, dim=-2)
    num_points = torch.sum(A, dim=1)

    rindex = torch.sum(N * (N - 1), dim=[1, 2])
    aindex = torch.sum(A * (A - 1), dim=1)
    bindex = torch.sum(B * (B - 1), dim=1)
    expected_rindex = aindex * bindex / torch.clamp(num_points * (num_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator
    return torch.where(denominator != 0, ari, torch.tensor(1.0, device=ari.device, dtype=ari.dtype))


def fari_metric(gt_mask: torch.Tensor, pred_mask: torch.Tensor) -> float:
    """gt_mask, pred_mask: (B, H, W) integer maps. 0 = background."""
    assert "int" in str(gt_mask.dtype) and "int" in str(pred_mask.dtype)
    return _adjusted_rand_index(gt_mask, pred_mask, ignore_background=True).mean().item()


# ----------------------------------------------------------------------
# mBO (Mean Best Overlap, foreground only)
# ----------------------------------------------------------------------
def _mean_best_overlap_single(gt_mask: torch.Tensor, pred_mask: torch.Tensor) -> float:
    """gt_mask, pred_mask: (H*W,)"""
    if gt_mask.max().item() == 0:
        return float("nan")
    true_oh = F.one_hot(gt_mask).float()[..., 1:]   # drop bg
    pred_oh = F.one_hot(pred_mask).float()
    intersect = (true_oh[:, :, None] * pred_oh[:, None, :]).sum(0)
    union = true_oh.sum(0)[:, None] + pred_oh.sum(0)[None] - intersect
    iou = intersect / (union + 1e-8)
    if iou.numel() == 0:
        return float("nan")
    return float(iou.max(dim=1).values.mean().item())


def mbo_metric(gt_mask: torch.Tensor, pred_mask: torch.Tensor) -> float:
    gt_flat = gt_mask.flatten(1, 2)
    pred_flat = pred_mask.flatten(1, 2)
    scores = []
    for i in range(gt_flat.shape[0]):
        scores.append(_mean_best_overlap_single(gt_flat[i], pred_flat[i]))
    valid = [s for s in scores if not math.isnan(s)]
    return float(np.mean(valid)) if valid else float("nan")


# ----------------------------------------------------------------------
# mIoU (Hungarian-matched)
# ----------------------------------------------------------------------
def _max_assignment_sum(iou: torch.Tensor) -> torch.Tensor:
    """Bitmask DP max-weight bipartite matching. iou: (n_true, n_pred)."""
    n_rows, n_cols = int(iou.shape[0]), int(iou.shape[1])
    if n_rows == 0 or n_cols == 0:
        return iou.new_tensor(0.0)
    if n_cols > 18:  # safeguard — fall back to scipy for very large col counts
        from scipy.optimize import linear_sum_assignment
        # We want to MAXIMIZE iou. linear_sum_assignment minimizes cost.
        cost = -iou.detach().cpu().numpy()
        # Pad rows if n_rows > n_cols (rare); scipy handles unequal sizes.
        row_ind, col_ind = linear_sum_assignment(cost)
        sel = iou[row_ind, col_ind].sum()
        return sel
    num_states = 1 << n_cols
    state_ids = torch.arange(num_states, device=iou.device, dtype=torch.long)
    neg_inf = torch.tensor(float("-inf"), device=iou.device, dtype=iou.dtype)
    dp = torch.full((num_states,), neg_inf, device=iou.device, dtype=iou.dtype)
    dp[0] = 0.0
    for row_idx in range(n_rows):
        row_vals = iou[row_idx]
        next_dp = dp.clone()
        for col_idx in range(n_cols):
            bit = 1 << col_idx
            free_states = (state_ids & bit) == 0
            if not bool(free_states.any()):
                continue
            src_states = state_ids[free_states]
            dst_states = src_states | bit
            candidate_vals = dp[src_states] + row_vals[col_idx]
            next_dp.scatter_reduce_(0, dst_states, candidate_vals, reduce="amax", include_self=True)
        dp = next_dp
    return dp.max()


def _hungarian_miou(gt_mask: torch.Tensor, pred_mask: torch.Tensor, ignore_background: bool) -> float:
    if gt_mask.max().item() == 0 and ignore_background:
        return float("nan")
    true_oh = F.one_hot(gt_mask).float()
    if ignore_background:
        true_oh = true_oh[..., 1:]
    pred_oh = F.one_hot(pred_mask).float()
    n_true, n_pred = true_oh.shape[-1], pred_oh.shape[-1]
    if n_true == 0:
        return float("nan")
    if n_pred == 0:
        return 0.0
    intersect = (true_oh[:, :, None] * pred_oh[:, None, :]).sum(0)
    union = true_oh.sum(0)[:, None] + pred_oh.sum(0)[None] - intersect
    iou = intersect / (union + 1e-8)
    best_sum = _max_assignment_sum(iou)
    return float((best_sum / float(n_true)).item())


def miou_metric(gt_mask: torch.Tensor, pred_mask: torch.Tensor) -> float:
    gt_flat = gt_mask.flatten(1, 2)
    pred_flat = pred_mask.flatten(1, 2)
    scores = []
    for i in range(gt_flat.shape[0]):
        scores.append(_hungarian_miou(gt_flat[i], pred_flat[i], ignore_background=False))
    valid = [s for s in scores if not math.isnan(s)]
    return float(np.mean(valid)) if valid else float("nan")


# ----------------------------------------------------------------------
# rFID (reconstruction Fréchet Inception Distance) via torchmetrics
# ----------------------------------------------------------------------
class FIDAccumulator:
    """Streaming wrapper around torchmetrics' FrechetInceptionDistance."""
    def __init__(self, device="cuda", feature: int = 2048):
        from torchmetrics.image.fid import FrechetInceptionDistance
        self.metric = FrechetInceptionDistance(feature=feature, normalize=True).to(device)
        self.device = device

    @torch.no_grad()
    def add(self, real: torch.Tensor, fake: torch.Tensor):
        """real, fake: (B, 3, H, W) in [0, 1]."""
        if real.dtype != torch.uint8:
            real = (real.clamp(0, 1) * 255).to(torch.uint8) if not real.is_floating_point() else real.clamp(0, 1)
            fake = (fake.clamp(0, 1) * 255).to(torch.uint8) if not fake.is_floating_point() else fake.clamp(0, 1)
        # torchmetrics expects normalize=True with float [0,1] -> works directly
        self.metric.update(real.to(self.device).float() / (255.0 if real.dtype == torch.uint8 else 1.0), real=True)
        self.metric.update(fake.to(self.device).float() / (255.0 if fake.dtype == torch.uint8 else 1.0), real=False)

    def compute(self) -> float:
        return float(self.metric.compute().item())


# ----------------------------------------------------------------------
# Reconstruction metrics (PSNR/SSIM/MSE/MAE) — optional companion to rFID
# ----------------------------------------------------------------------
def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    g2d = g1d.unsqueeze(0) * g1d.unsqueeze(1)
    window = g2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, window_size, window_size).contiguous()
    return window


def compute_recon_metrics(target: torch.Tensor, pred: torch.Tensor) -> Dict[str, torch.Tensor]:
    """PSNR/SSIM/MSE/MAE on (B,C,H,W) tensors in [0,1] or [-1,1]."""
    target = target.float()
    pred = pred.float()
    diff = target - pred
    mse = (diff ** 2).mean(dim=(1, 2, 3))
    mae = diff.abs().mean(dim=(1, 2, 3))
    psnr = -10.0 * torch.log10(mse.clamp(min=1e-20))
    C = target.shape[1]
    window = _gaussian_window(11, 1.5, C, target.device, target.dtype)
    pad = 5
    mu1 = F.conv2d(target, window, padding=pad, groups=C)
    mu2 = F.conv2d(pred, window, padding=pad, groups=C)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    s1 = F.conv2d(target * target, window, padding=pad, groups=C) - mu1_sq
    s2 = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu2_sq
    s12 = F.conv2d(target * pred, window, padding=pad, groups=C) - mu1_mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / (((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2)).clamp(min=1e-12))
    ssim = ssim_map.mean(dim=(1, 2, 3))
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}


# ----------------------------------------------------------------------
# OVT attention map -> integer pred panoptic mask
# ----------------------------------------------------------------------
def build_pred_mask_readout(
    ovt_logits: torch.Tensor,        # (B, M, P) raw logits
    ovt_valid_mask: torch.Tensor,    # (B, M) bool
    target_size: int,
    n_ovt_per_object: int,
    patch_grid: int = 32,
    *,
    merge: str = "mean",             # "mean" | "max"  (over the n_ovt_per_object tokens)
    competition: str = "sigmoid",    # "sigmoid" (independent) | "softmax" (per-pixel over objects)
    temp: float = 1.0,
    bg_threshold: float = 0.05,
    use_bg_channel: bool = True,     # True: bg as a competing channel; False: post-hoc threshold
) -> torch.Tensor:
    """Flexible readout: per-object merge -> competition -> upsample -> argmax.

    Returns integer panoptic mask (B, H, W), 0 = background, 1..K = objects.
    """
    B, M, P = ovt_logits.shape
    assert P == patch_grid * patch_grid
    K = M // n_ovt_per_object
    logits = ovt_logits[:, : K * n_ovt_per_object].reshape(B, K, n_ovt_per_object, P).float()
    if merge == "max":
        obj_logits = logits.amax(dim=2)            # (B, K, P)
    else:
        obj_logits = logits.mean(dim=2)
    obj_valid = ovt_valid_mask.reshape(B, K, n_ovt_per_object).any(dim=2)  # (B, K)

    # spatial
    obj_2d = obj_logits.reshape(B, K, patch_grid, patch_grid)
    up = F.interpolate(obj_2d, size=(target_size, target_size), mode="bilinear", align_corners=False)
    # (B, K, H, W) raw logits upsampled

    temp = max(float(temp), 1e-6)
    if competition == "softmax":
        # invalid objects -> -inf before softmax so they get 0 probability
        neg = torch.finfo(up.dtype).min
        up_for_comp = up.masked_fill(~obj_valid.view(B, K, 1, 1), neg)
        scores = (up_for_comp / temp).softmax(dim=1)   # per-pixel competition over objects
    else:
        scores = torch.sigmoid(up / temp)
        scores = scores * obj_valid.view(B, K, 1, 1).to(scores.dtype)

    neg = torch.finfo(scores.dtype).min
    scores_masked = scores.masked_fill(~obj_valid.view(B, K, 1, 1), neg)

    if use_bg_channel:
        bg = torch.full((B, 1, target_size, target_size), float(bg_threshold),
                        device=scores.device, dtype=scores.dtype)
        stack = torch.cat([bg, scores_masked], dim=1)   # ch 0 = bg
        pred = stack.argmax(dim=1)                       # 0 = bg, 1..K = objects
    else:
        pred = scores_masked.argmax(dim=1) + 1
        max_score = scores.amax(dim=1)
        pred = torch.where(max_score < float(bg_threshold), torch.zeros_like(pred), pred)
    return pred.to(torch.int64)


def ovt_logits_to_pred_mask(
    ovt_logits: torch.Tensor,
    ovt_valid_mask: torch.Tensor,
    target_size: int,
    n_ovt_per_object: int,
    patch_grid: int = 16,
    bg_threshold: float = 0.05,
) -> torch.Tensor:
    """Convert per-OVT mask logits to an integer panoptic mask (B, H, W).

    Pipeline:
      1) sigmoid → per-OVT probability (B, M, P)
      2) merge n_ovt_per_object consecutive OVTs into one object map (mean)
      3) reshape to spatial (patch_grid × patch_grid), bilinear interp to target
      4) argmax over object channel → integer index (1..K)
      5) Background: where max prob < bg_threshold → 0
    """
    B, M, P = ovt_logits.shape
    assert P == patch_grid * patch_grid
    probs = torch.sigmoid(ovt_logits.float())  # (B, M, P)

    K_max = M // n_ovt_per_object
    if K_max * n_ovt_per_object < M:
        probs = probs[:, : K_max * n_ovt_per_object]
    object_maps = probs.reshape(B, K_max, n_ovt_per_object, P).mean(dim=2)  # (B, K, P)

    # Mask out objects with all-padded ovts
    valid = ovt_valid_mask.reshape(B, K_max, n_ovt_per_object).any(dim=2)  # (B, K)
    object_maps = object_maps * valid.unsqueeze(-1).to(object_maps.dtype)

    # Reshape and upsample
    object_maps_2d = object_maps.reshape(B, K_max, patch_grid, patch_grid)
    upsampled = F.interpolate(object_maps_2d, size=(target_size, target_size), mode="bilinear", align_corners=False)
    # upsampled: (B, K, H, W)

    # Background channel: complement
    max_prob, argmax_obj = upsampled.max(dim=1)  # (B, H, W)
    pred = argmax_obj + 1  # 1..K  (0 reserved for background)
    pred = torch.where(max_prob < bg_threshold, torch.zeros_like(pred), pred)
    return pred.to(torch.int64)
