"""RESULT: the RAE decoder is an EXACT INVERSE of the frozen SigLIP encoder, not a robust
decoder. It cannot be used to measure the information content of a compressed code.

This started as an attempt to measure the rFID ceiling of an object-centric code without
training: pool GT SigLIP features inside each GT object mask into n_q codes per object,
route them back onto the patch grid, decode with the frozen RAE decoder. It does not work,
and the reason is the point.

Measured on 32 val images (PSNR):
    o0, the exact GT feature map (256 patches)          22.47
    smooth_f2  — the map merely 2x average-pooled and   10.79   <-- the MILDEST possible
                 bilinearly upsampled (64 dof, smooth,          degradation already destroys
                 no flat regions)                               reconstruction
    n_q=16     — 71 piecewise-constant codes             9.96
    n_q=1      — 6.3 codes (~CODA's 7-slot budget)       7.96

Averaging two ADJACENT SigLIP patches is already enough to break it. Adjacent patch features
are near-independent in feature space, so any pooling / downsampling / regression throws the
map off the SigLIP manifold, and the decoder — trained only to invert exact encoder outputs —
emits garbage regardless of how much information the code actually retains.

CONSEQUENCES
  * There is NO training-free way to measure a code's rFID ceiling through this decoder.
    Measure information content in FEATURE space instead (see probe_ovt_appearance.py).
  * A generative model that lands on the SigLIP manifold is MANDATORY. That is precisely the
    DiT's job, and it explains V18: its regressed latents scored rFID 258 because they were
    off-manifold, not because the code was empty. It also explains why the diffusion path
    (rFID 30) beats the direct path (258) at identical PSNR — it produces on-manifold maps
    with the wrong content.
  * Therefore the DiT cannot be dropped, and the only remaining lever on the decoder side is
    HOW it is conditioned (AdaLN today; cross-attention in CODA).

Kept as the experiment that establishes decoder brittleness.
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, "/home/jovyan/PGOT")

from pgot.eval.eval_recon_oracles import build_loader, encode_gt_siglip, load_model_and_tokenizer
from pgot.eval.pgot_metrics import FIDAccumulator, compute_recon_metrics
from pgot.eval.run_eval import decode_to_image, denormalize_images, load_rae_decoder


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pgot.probe_object_code_ceiling")


# ----------------------------------------------------------------------
def median_split(idx: torch.Tensor, coords: torch.Tensor, n_groups: int) -> List[torch.Tensor]:
    """Recursively split a patch index set into n_groups by coordinate median.

    Deterministic (no k-means randomness) and derivable from the mask alone, so it smuggles
    no information the decoder would not already have from the segmentation.
    """
    if n_groups <= 1 or idx.numel() <= 1:
        return [idx]
    xy = coords[idx]                                     # (n, 2)
    spread = xy.max(dim=0).values - xy.min(dim=0).values
    axis = int(torch.argmax(spread).item())
    order = torch.argsort(xy[:, axis])
    idx = idx[order]
    half = idx.numel() // 2
    left, right = idx[:half], idx[half:]
    n_left = n_groups // 2
    n_right = n_groups - n_left
    out = []
    if left.numel() > 0:
        out += median_split(left, coords, n_left)
    if right.numel() > 0:
        out += median_split(right, coords, n_right)
    return [g for g in out if g.numel() > 0]


@torch.no_grad()
def routed_map(
    gt_siglip: torch.Tensor,     # (P, C) target-tower features on a side x side grid
    masks: torch.Tensor,         # (K, P) GT object masks, already on that grid
    coords: torch.Tensor,        # (P, 2) patch coordinates
    n_q: int,
) -> (torch.Tensor, int):
    """Pool GT features into n_q codes per object (+ background) and route them back."""
    P, C = gt_siglip.shape
    out = torch.zeros_like(gt_siglip)
    n_codes = 0

    # Hard ownership: a patch belongs to the object with the strongest mask, else background.
    if masks.numel() > 0:
        best, owner = masks.max(dim=0)                    # (P,), (P,)
        owner = torch.where(best > 0.5, owner, torch.full_like(owner, -1))
    else:
        owner = torch.full((P,), -1, dtype=torch.long, device=gt_siglip.device)

    regions = [(k, (owner == k).nonzero(as_tuple=False).flatten()) for k in range(masks.shape[0])]
    regions.append((-1, (owner == -1).nonzero(as_tuple=False).flatten()))   # background

    for _, idx in regions:
        if idx.numel() == 0:
            continue
        for group in median_split(idx, coords, n_q):
            out[group] = gt_siglip[group].mean(dim=0, keepdim=True)          # one code vector
            n_codes += 1
    return out, n_codes


# ----------------------------------------------------------------------
@torch.no_grad()
def run(args, model, loader, decoder, device) -> List[Dict[str, float]]:
    n_qs = [int(x) for x in str(args.codes_per_object).split(",") if x.strip()]
    # CONTROL: same information budget, but a SMOOTH map instead of a piecewise-constant one.
    # avgpool by f then bilinear-upsample -> (side/f)^2 degrees of freedom, no flat regions.
    # smooth_f2 (64 dof) vs n_q=16 (~71 codes) is the matched-budget comparison that separates
    # "the code lost information" from "the decoder cannot eat piecewise-constant input".
    fs = [int(x) for x in str(args.smooth_factors).split(",") if x.strip()]
    settings = (["o0_full_gt_map"] if args.include_o0 else []) \
        + [f"n_q={q}" for q in n_qs] + [f"smooth_f{f}" for f in fs]

    fid = {s: FIDAccumulator(device=device, feature=2048) for s in settings}
    mets = {s: {"psnr": [], "ssim": [], "mse": [], "mae": []} for s in settings}
    codes = {s: [] for s in settings}
    n_samples = 0

    vt_list = model.get_vision_tower_aux_list()
    tp = vt_list[1].image_processor if len(vt_list) > 1 else vt_list[0].image_processor
    t_mean, t_std = torch.tensor(tp.image_mean), torch.tensor(tp.image_std)

    n_ovt = int(args.n_ovt_per_object)
    coords_cache = {}

    for batch in tqdm(loader, desc="Ceiling sweep"):
        gt_siglip = encode_gt_siglip(model, batch, device)             # (B, P, C)
        B, P, _ = gt_siglip.shape
        side = int(round(P ** 0.5))
        if side not in coords_cache:
            yy, xx = torch.meshgrid(
                torch.arange(side, device=device), torch.arange(side, device=device), indexing="ij"
            )
            coords_cache[side] = torch.stack([yy.flatten(), xx.flatten()], dim=-1).float()
        coords = coords_cache[side]

        m_raw = batch["gt_masks_per_ovt"].to(device=device, dtype=torch.float32).clamp(0, 1)
        valid = batch["ovt_valid_mask"].to(device=device, dtype=torch.bool)
        K = m_raw.shape[1] // n_ovt
        # collapse the n_ovt tokens of an object (they share a mask) and resize to the target grid
        m_obj = m_raw[:, : K * n_ovt].reshape(B, K, n_ovt, -1).amax(dim=2)
        v_obj = valid[:, : K * n_ovt].reshape(B, K, n_ovt).any(dim=2)
        src_side = int(round(m_obj.shape[-1] ** 0.5))
        if src_side != side:
            m_obj = F.interpolate(
                m_obj.reshape(B * K, 1, src_side, src_side), size=(side, side),
                mode="bilinear", align_corners=False,
            ).reshape(B, K, P)
        m_obj = m_obj * v_obj.unsqueeze(-1).float()

        real = denormalize_images(batch["target_images"].to(device).float(), t_mean, t_std)

        for s in settings:
            gen = torch.empty_like(gt_siglip)
            if s == "o0_full_gt_map":
                gen = gt_siglip
                for b in range(B):
                    codes[s].append(P)
            elif s.startswith("smooth_f"):
                f = int(s.split("f")[1])
                grid = gt_siglip.reshape(B, side, side, -1).permute(0, 3, 1, 2)   # (B,C,s,s)
                small = F.avg_pool2d(grid, kernel_size=f)
                back = F.interpolate(small, size=(side, side), mode="bilinear", align_corners=False)
                gen = back.permute(0, 2, 3, 1).reshape(B, P, -1).contiguous()
                for b in range(B):
                    codes[s].append((side // f) ** 2)
            else:
                q = int(s.split("=")[1])
                for b in range(B):
                    keep = m_obj[b][v_obj[b]] if bool(v_obj[b].any()) else m_obj[b][:0]
                    gen[b], n_c = routed_map(gt_siglip[b], keep, coords, q)
                    codes[s].append(n_c)

            fake = decode_to_image(decoder, gen, device)
            r = real
            if r.shape[-2:] != fake.shape[-2:]:
                r = F.interpolate(real, size=fake.shape[-2:], mode="bilinear", align_corners=False)
            fid[s].add(r, fake)
            rec = compute_recon_metrics(r, fake)
            for key in mets[s]:
                mets[s][key].extend([float(x) for x in rec[key].detach().cpu()])
        n_samples += B

    rows = []
    for s in settings:
        row = {
            "setting": s,
            "num_samples": n_samples,
            "mean_codes_per_image": sum(codes[s]) / max(len(codes[s]), 1),
            "rFID": float(fid[s].compute()),
        }
        for key, vals in mets[s].items():
            row[f"recon_{key}"] = sum(vals) / max(len(vals), 1)
        rows.append(row)
        log.info("[%s] %s", s, json.dumps(row))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/home/jovyan/PGOT/checkpoints/pgot_main_v15_bce_bottleneck")
    parser.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--codes_per_object", default="1,2,4,8,16")
    parser.add_argument("--smooth_factors", default="2,4", help="control: avgpool by f then bilinear up")
    parser.add_argument("--include_o0", action="store_true", default=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--max_caption_tokens", type=int, default=2048)
    parser.add_argument("--n_ovt_per_object", type=int, default=2)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--image_preprocess_mode", choices=["default", "coda_center_crop"], default="coda_center_crop")
    parser.add_argument("--coda_crop_size", type=int, default=512)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--diffusion_inference_steps", type=int, default=25)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer, device, _ = load_model_and_tokenizer(args)
    loader = build_loader(args, tokenizer, model, args.val_jsonl, shuffle=False, max_samples=args.max_samples)
    decoder = load_rae_decoder(model, device=device, dtype=torch.float32)
    log.info("Decoder ready. No DiT, no projector — pooling + routing only.")

    rows = run(args, model, loader, decoder, device)

    with open(Path(args.output_dir) / "summary.json", "w") as f:
        json.dump({"settings": rows}, f, indent=2)
    fields = ["setting", "num_samples", "mean_codes_per_image", "rFID",
              "recon_psnr", "recon_ssim", "recon_mse", "recon_mae"]
    with open(Path(args.output_dir) / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    print("\n" + "=" * 82)
    print("OBJECT-CODE CEILING — rFID of a PERFECT object-centric code (GT features, no DiT)")
    print("=" * 82)
    print(f"{'setting':<18} {'codes/img':>10} {'rFID':>9} {'PSNR':>8} {'SSIM':>8}")
    for r in rows:
        print(f"{r['setting']:<18} {r['mean_codes_per_image']:>10.1f} {r['rFID']:>9.2f} "
              f"{r['recon_psnr']:>8.2f} {r['recon_ssim']:>8.4f}")
    print("=" * 82)
    print("CODA reference: 7 slots, rFID 10.65.  o0 (full map) is the decoder's own ceiling.")
    log.info("Wrote %s", Path(args.output_dir) / "summary.json")


if __name__ == "__main__":
    main()
