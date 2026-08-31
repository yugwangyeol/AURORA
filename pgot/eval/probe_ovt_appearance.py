"""Probe: does the OVT hidden state carry OBJECT APPEARANCE, or only semantics?

No training of the model. We cache frozen features on the val set and fit closed-form
ridge regressions, then compare against a no-information baseline.

Targets (per object k):
    a_k = mask-pooled GT SigLIP target feature (the space the RAE decoder consumes)
          -> the object's appearance, in decoder space. [C=1152]

Predictors:
    P0_mean      constant per-channel mean of a_k          -> the ZERO-INFORMATION floor
    P1_ovt       the object's OVT hidden states (TEXT stream, concat of n_ovt)  [n_ovt*D]
    P2_llm_img   mask-pooled LLM image hidden states (IMAGE stream)             [D]
    P3_word      the object's caption phrase, embedded with the FROZEN input embedding
                 table and mean-pooled. Sees the WORD but NEVER the image.      [D]
    P4_raw_siglip mask-pooled RAW SigLIP patch features (before projector+LLM)  [C_raw]
                 -> the ceiling: how much appearance a purely visual aggregate carries.

THE CONFOUND this version fixes: the target a_k is dominated by CATEGORY identity
("a zebra's mean SigLIP feature" is mostly determined by *being a zebra*). So a purely
SEMANTIC code scores well on it. P3 measures exactly that semantic component, and we then
re-run every probe on the INSTANCE RESIDUAL  Y_res = a_k - P3_prediction(a_k), i.e. on what
is left after the word is accounted for. R^2 on Y_res is appearance proper.

Reading the result (residual probes are the decisive ones):
    P1_res ~= 0                -> the OVT is a PURE SEMANTIC code; it knows the name, not the
                                  look. The encoder read path must be opened.
    P1_res >> 0, near P4_res   -> the OVT really does carry instance appearance; the
                                  bottleneck is downstream (the DiT conditioning interface).
    P4_res >> P2_res           -> the projector+LLM destroyed appearance on the way IN,
                                  independent of the OVT read-out.

Also reports the effective dimensionality (participation ratio, dims for 90/99% variance)
of each representation -> quantifies "semantic manifold collapse".
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
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "/home/jovyan/PGOT")

from pgot.eval.eval_recon_oracles import build_loader, encode_gt_siglip, load_model_and_tokenizer
from pgot.eval.pgot_inference import pgot_forward_eval
from pgot.model.pgot_utils import gather_ovt_hidden_states


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pgot.probe_ovt_appearance")


# ----------------------------------------------------------------------
# Feature caching
# ----------------------------------------------------------------------
def pool_with_mask(feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """feats (P, C), mask (P,) in [0,1] -> (C,) mask-weighted mean."""
    denom = mask.sum().clamp_min(1e-6)
    return (mask.unsqueeze(-1) * feats).sum(dim=0) / denom


def resize_mask(mask: torch.Tensor, out_side: int) -> torch.Tensor:
    """mask (P_src,) on a square grid -> (out_side**2,) bilinear."""
    src_side = int(round(mask.numel() ** 0.5))
    if src_side == out_side:
        return mask
    m = mask.reshape(1, 1, src_side, src_side).float()
    m = F.interpolate(m, size=(out_side, out_side), mode="bilinear", align_corners=False)
    return m.reshape(-1)


@torch.no_grad()
def raw_siglip_patches(model, images: torch.Tensor) -> torch.Tensor:
    """RAW mask-tower SigLIP patch features — before mm_projector, before the LLM."""
    vt = model.get_vision_tower_aux_list()[0]
    p = next(vt.parameters(), None)
    feats = vt(images.to(device=p.device, dtype=p.dtype))
    if not torch.is_tensor(feats):
        feats = getattr(feats, "last_hidden_state", None) or feats[0]
    return feats.float()                                             # (B, P_raw, C_raw)


@torch.no_grad()
def word_embeddings(model, caption_ids: torch.Tensor, ovt_pos: torch.Tensor,
                    k: int, n_ovt: int, b: int) -> torch.Tensor:
    """Mean-pooled FROZEN input embeddings of object k's caption phrase.

    The phrase is the token span between the previous object's last <ovt> and this
    object's first <ovt>. Uses only the embedding table -> ZERO image information.
    """
    emb = model.get_input_embeddings()
    first = int(ovt_pos[b, k * n_ovt].item())
    prev_last = int(ovt_pos[b, k * n_ovt - 1].item()) if k > 0 else -1
    start, end = prev_last + 1, first
    if end <= start:                       # degenerate span -> fall back to the token before
        start, end = max(first - 1, 0), max(first, 1)
    ids = caption_ids[b, start:end].to(emb.weight.device)
    if ids.numel() == 0:
        return torch.zeros(emb.weight.shape[1], dtype=torch.float32)
    return emb(ids).float().mean(dim=0).cpu()


@torch.no_grad()
def cache_features(args, model, loader, device) -> Dict[str, torch.Tensor]:
    """Returns per-object feature matrices (float32, CPU)."""
    X_ovt: List[torch.Tensor] = []      # (n_ovt * D,)  OVT hidden (text stream)
    X_ovt_slots: List[List[torch.Tensor]] = []
    ovt_pair_cosines: List[torch.Tensor] = []
    X_img: List[torch.Tensor] = []      # (D,)          mask-pooled LLM image stream
    X_word: List[torch.Tensor] = []     # (D,)          frozen phrase embedding (no image)
    X_raw: List[torch.Tensor] = []      # (C_raw,)      mask-pooled RAW SigLIP patches
    X_visual_memory: List[torch.Tensor] = []  # (J*D,) E8/E11 image-only object memory
    X_visual_memory_mean: List[torch.Tensor] = []
    X_visual_memory_slots: List[List[torch.Tensor]] = []
    Y_app: List[torch.Tensor] = []      # (C,)          mask-pooled GT SigLIP target
    img_ids: List[int] = []             # sample index -> for a leakage-free split

    n_ovt = int(args.n_ovt_per_object)
    X_ovt_slots = [[] for _ in range(n_ovt)]
    n_skipped = 0

    for sample_idx, batch in enumerate(tqdm(loader, desc="Caching")):
        out = pgot_forward_eval(
            model,
            images=batch["images"],
            target_images=batch["target_images"],
            caption_input_ids=batch["caption_input_ids"],
            caption_attention_mask=batch["caption_attention_mask"],
            ovt_positions_in_caption=batch["ovt_positions_in_caption"],
            ovt_valid_mask=batch["ovt_valid_mask"],
        )
        gt_siglip = encode_gt_siglip(model, batch, device)          # (B, P_t=256, C)
        ovt_hidden = gather_ovt_hidden_states(
            out["hidden"], out["ovt_abs_positions"], out["ovt_valid_mask"]
        ).float()                                                    # (B, M, D)
        img_hidden = out["img_hidden"].float()                       # (B, P_i=1024, D)
        raw_feats = raw_siglip_patches(model, batch["images"])       # (B, P_raw, C_raw)

        masks = batch["gt_masks_per_ovt"].to(device=device, dtype=torch.float32).clamp(0, 1)
        valid = batch["ovt_valid_mask"].to(device=device, dtype=torch.bool)
        caption_ids = batch["caption_input_ids"]
        ovt_pos = batch["ovt_positions_in_caption"]

        B = gt_siglip.shape[0]
        side_t = int(round(gt_siglip.shape[1] ** 0.5))
        side_i = int(round(img_hidden.shape[1] ** 0.5))
        side_r = int(round(raw_feats.shape[1] ** 0.5))
        K = masks.shape[1] // n_ovt
        visual_memory = out.get("visual_memory")
        if visual_memory is not None and not X_visual_memory_slots:
            memory_count = visual_memory.shape[2] if visual_memory.ndim == 4 else 1
            if visual_memory.ndim == 4:
                configured_object_count = int(
                    getattr(
                        model.config,
                        "pgot_e11_object_memories_per_owner",
                        memory_count,
                    )
                    or memory_count
                )
                memory_count = min(memory_count, configured_object_count)
            X_visual_memory_slots = [[] for _ in range(memory_count)]

        for b in range(B):
            for k in range(K):
                sl = slice(k * n_ovt, (k + 1) * n_ovt)
                if not bool(valid[b, sl].any()):
                    continue
                # both OVTs of an object share the GT mask; amax is a no-op safety net
                m_src = masks[b, sl].amax(dim=0)                      # (P_src=1024,)
                if float(m_src.sum()) < 1e-3:
                    n_skipped += 1
                    continue

                m_t = resize_mask(m_src, side_t)                      # target-tower grid
                m_i = resize_mask(m_src, side_i)                      # LLM image grid
                m_r = resize_mask(m_src, side_r)                      # raw SigLIP grid
                if min(float(m_t.sum()), float(m_i.sum()), float(m_r.sum())) < 1e-3:
                    n_skipped += 1
                    continue

                Y_app.append(pool_with_mask(gt_siglip[b], m_t).cpu())
                ovt_obj = ovt_hidden[b, sl]
                X_ovt.append(ovt_obj.reshape(-1).cpu())               # concat of n_ovt states
                for j in range(n_ovt):
                    X_ovt_slots[j].append(ovt_obj[j].cpu())           # each OVT slot alone
                if n_ovt >= 2 and bool(valid[b, sl].all()):
                    ovt_pair_cosines.append(
                        F.cosine_similarity(ovt_obj[0].float(), ovt_obj[1].float(), dim=0).cpu()
                    )
                X_img.append(pool_with_mask(img_hidden[b], m_i).cpu())
                X_raw.append(pool_with_mask(raw_feats[b].to(device), m_r).cpu())
                X_word.append(word_embeddings(model, caption_ids, ovt_pos, k, n_ovt, b))
                if visual_memory is not None:
                    # E8/E10: [D]. E11: [J,D].  Concatenation tests the full
                    # capacity increase; the mean and individual memories show
                    # whether the gain is distributed or collapses to one ID.
                    object_memory = visual_memory[b, k].float()
                    if object_memory.ndim == 1:
                        object_memory = object_memory.unsqueeze(0)
                    object_memory = object_memory[: len(X_visual_memory_slots)]
                    X_visual_memory.append(object_memory.flatten().cpu())
                    X_visual_memory_mean.append(object_memory.mean(dim=0).cpu())
                    for j in range(object_memory.shape[0]):
                        X_visual_memory_slots[j].append(object_memory[j].cpu())
                img_ids.append(sample_idx * B + b)

    if not Y_app:
        raise RuntimeError("No objects cached — check the dataset / masks.")

    log.info("Cached %d objects (%d skipped: empty mask).", len(Y_app), n_skipped)
    out = {
        "X_ovt": torch.stack(X_ovt).float(),
        "X_img": torch.stack(X_img).float(),
        "X_word": torch.stack(X_word).float(),
        "X_raw": torch.stack(X_raw).float(),
        "Y_app": torch.stack(Y_app).float(),
        "img_ids": torch.tensor(img_ids, dtype=torch.long),
    }
    for j, xs in enumerate(X_ovt_slots):
        if xs:
            out[f"X_ovt_slot{j}"] = torch.stack(xs).float()
    if X_visual_memory:
        if len(X_visual_memory) != len(Y_app):
            raise RuntimeError("E8 visual-memory/object target count mismatch")
        out["X_visual_memory"] = torch.stack(X_visual_memory).float()
        out["X_visual_memory_mean"] = torch.stack(X_visual_memory_mean).float()
        for j, xs in enumerate(X_visual_memory_slots):
            if xs:
                out[f"X_visual_memory_slot{j}"] = torch.stack(xs).float()
    if ovt_pair_cosines:
        out["ovt_pair_cosine"] = torch.stack(ovt_pair_cosines).float()
    return out


# ----------------------------------------------------------------------
# Closed-form ridge + metrics
# ----------------------------------------------------------------------
def ridge_probe(
    X: torch.Tensor,
    Y: torch.Tensor,
    is_fit: torch.Tensor,
    lambdas: List[float],
    device: torch.device,
    return_pred_all: bool = False,
):
    """Fit Y ~ X by ridge on the fit split; report held-out R^2 vs the train-mean predictor.

    If return_pred_all, also returns the best model's prediction for EVERY row of X
    (used to build the instance residual Y - Yhat_word).
    """
    X = X.to(device=device, dtype=torch.float64)
    Y = Y.to(device=device, dtype=torch.float64)
    X_tr, Y_tr = X[is_fit], Y[is_fit]
    X_te, Y_te = X[~is_fit], Y[~is_fit]

    xm = X_tr.mean(dim=0, keepdim=True)
    ym = Y_tr.mean(dim=0, keepdim=True)
    Xc, Yc = X_tr - xm, Y_tr - ym

    # Floor: always predict the train mean.
    sse_mean = ((Y_te - ym) ** 2).sum()

    best = {"r2": -float("inf")}
    best_W = None
    best_alpha = None
    d = Xc.shape[1]
    use_dual = d > Xc.shape[0]
    if use_dual:
        G = Xc @ Xc.T
        eye = torch.eye(Xc.shape[0], device=device, dtype=torch.float64)
    else:
        G = Xc.T @ Xc
        XtY = Xc.T @ Yc
        eye = torch.eye(d, device=device, dtype=torch.float64)
    for lam in lambdas:
        try:
            if use_dual:
                alpha = torch.linalg.solve(G + lam * eye, Yc)
                pred = ((X_te - xm) @ Xc.T) @ alpha + ym
            else:
                W = torch.linalg.solve(G + lam * eye, XtY)
                pred = (X_te - xm) @ W + ym
        except RuntimeError:
            continue
        sse = ((Y_te - pred) ** 2).sum()
        r2 = float(1.0 - sse / sse_mean)
        cos = float(F.cosine_similarity(pred, Y_te, dim=-1).mean())
        rel_l2 = float(((pred - Y_te).norm(dim=-1) / Y_te.norm(dim=-1).clamp_min(1e-8)).mean())
        if r2 > best["r2"]:
            best = {"r2": r2, "cosine": cos, "rel_l2": rel_l2, "lambda": float(lam)}
            if use_dual:
                best_alpha = alpha
            else:
                best_W = W
    if not return_pred_all:
        return best
    pred_all = (
        ((X - xm) @ Xc.T) @ best_alpha + ym
        if use_dual
        else (X - xm) @ best_W + ym
    )
    return best, pred_all.float().cpu()


def baseline_floor(Y_tr: torch.Tensor, Y_te: torch.Tensor, device: torch.device) -> Dict[str, float]:
    Y_tr = Y_tr.to(device=device, dtype=torch.float64)
    Y_te = Y_te.to(device=device, dtype=torch.float64)
    ym = Y_tr.mean(dim=0, keepdim=True)
    pred = ym.expand_as(Y_te)
    cos = float(F.cosine_similarity(pred, Y_te, dim=-1).mean())
    rel_l2 = float(((pred - Y_te).norm(dim=-1) / Y_te.norm(dim=-1).clamp_min(1e-8)).mean())
    return {"r2": 0.0, "cosine": cos, "rel_l2": rel_l2, "lambda": None}


# ----------------------------------------------------------------------
# Effective dimensionality
# ----------------------------------------------------------------------
def effective_dim(X: torch.Tensor, device: torch.device) -> Dict[str, float]:
    Xc = (X - X.mean(dim=0, keepdim=True)).to(device=device, dtype=torch.float64)
    s = torch.linalg.svdvals(Xc)
    ev = s ** 2
    ev = ev / ev.sum().clamp_min(1e-30)
    pr = float(1.0 / (ev ** 2).sum())            # participation ratio
    c = torch.cumsum(ev, dim=0)
    d90 = int((c < 0.90).sum().item()) + 1
    d99 = int((c < 0.99).sum().item()) + 1
    return {
        "nominal_dim": int(X.shape[1]),
        "participation_ratio": pr,
        "dims_for_90pct_var": d90,
        "dims_for_99pct_var": d99,
        "spectrum": ev[:256].cpu().tolist(),
    }


def summarize_1d(x: torch.Tensor) -> Dict[str, float]:
    x = x.float().cpu()
    qs = torch.quantile(x, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
    return {
        "count": int(x.numel()),
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "min": float(x.min()),
        "p10": float(qs[0]),
        "p25": float(qs[1]),
        "median": float(qs[2]),
        "p75": float(qs[3]),
        "p90": float(qs[4]),
        "max": float(x.max()),
    }


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/home/jovyan/PGOT/checkpoints/pgot_main_v15_bce_bottleneck")
    parser.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split_ratio", type=float, default=0.8, help="fraction of IMAGES used to fit")
    parser.add_argument("--lambdas", default="1e-1,1e0,1e1,1e2,1e3,1e4,1e5")
    # loader args (consumed by build_loader / load_model_and_tokenizer)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--max_caption_tokens", type=int, default=2048)
    parser.add_argument("--n_ovt_per_object", type=int, default=2)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--image_preprocess_mode", choices=["default", "coda_center_crop"], default="coda_center_crop")
    parser.add_argument("--coda_crop_size", type=int, default=512)
    # fp32: the V14/V15 router runs its LayerNorms in fp32, so bf16 hidden states blow up there.
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--diffusion_inference_steps", type=int, default=25)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lambdas = [float(x) for x in str(args.lambdas).split(",") if x.strip()]

    model, tokenizer, device, _ = load_model_and_tokenizer(args)
    loader = build_loader(args, tokenizer, model, args.val_jsonl, shuffle=False, max_samples=args.max_samples)
    log.info("Val set ready. Caching frozen features (no training).")

    cache = cache_features(args, model, loader, device)
    del model
    torch.cuda.empty_cache()

    # Leakage-free split: by image, not by object.
    ids = cache["img_ids"]
    uniq = torch.unique(ids)
    n_fit = max(int(len(uniq) * args.split_ratio), 1)
    fit_ids = set(uniq[:n_fit].tolist())
    is_fit = torch.tensor([int(i) in fit_ids for i in ids.tolist()], dtype=torch.bool)
    log.info(
        "Split: %d images (%d objects) fit / %d images (%d objects) test",
        n_fit, int(is_fit.sum()), len(uniq) - n_fit, int((~is_fit).sum()),
    )
    if int((~is_fit).sum()) < 2:
        raise RuntimeError("Test split has <2 objects — raise --max_samples.")

    Y = cache["Y_app"]
    probe_keys = [("P1_ovt_text_stream", "X_ovt")]
    for j in range(int(args.n_ovt_per_object)):
        key = f"X_ovt_slot{j}"
        if key in cache:
            probe_keys.append((f"P1_ovt_slot{j}", key))
    if "X_visual_memory" in cache:
        probe_keys.append(("P1b_e8_visual_memory", "X_visual_memory"))
        probe_keys.append(("P1c_e8_visual_memory_mean", "X_visual_memory_mean"))
        for j in range(len([k for k in cache if k.startswith("X_visual_memory_slot")])):
            key = f"X_visual_memory_slot{j}"
            if key in cache:
                probe_keys.append((f"P1d_e8_visual_memory_slot{j}", key))
    probe_keys.extend([
        ("P2_llm_image_stream", "X_img"),
        ("P3_word_only", "X_word"),
        ("P4_raw_siglip", "X_raw"),
    ])

    # ---- Probes on the RAW target (category-dominated) ----
    results = {"P0_mean_floor": baseline_floor(Y[is_fit], Y[~is_fit], device)}
    for name, key in probe_keys:
        results[name] = ridge_probe(cache[key], Y, is_fit, lambdas, device)
        log.info("[raw target | %s] %s", name, json.dumps(results[name]))

    # ---- Instance residual: strip what the WORD alone already explains ----
    _, pred_word = ridge_probe(cache["X_word"], Y, is_fit, lambdas, device, return_pred_all=True)
    Y_res = Y - pred_word
    frac = float((Y_res[~is_fit] ** 2).sum() / ((Y[~is_fit] - Y[is_fit].mean(0, keepdim=True)) ** 2).sum())
    log.info("Instance residual keeps %.1f%% of the held-out target variance.", 100 * frac)

    results_res = {"P0_mean_floor": baseline_floor(Y_res[is_fit], Y_res[~is_fit], device)}
    for name, key in probe_keys:
        if name == "P3_word_only":
            continue  # by construction ~0
        results_res[name] = ridge_probe(cache[key], Y_res, is_fit, lambdas, device)
        log.info("[INSTANCE RESIDUAL | %s] %s", name, json.dumps(results_res[name]))

    dims = {
        "OVT_text_stream": effective_dim(cache["X_ovt"], device),
        "LLM_image_stream": effective_dim(cache["X_img"], device),
        "raw_siglip_pooled": effective_dim(cache["X_raw"], device),
        "appearance_target": effective_dim(cache["Y_app"], device),
    }
    if "X_visual_memory" in cache:
        dims["E8_visual_memory"] = effective_dim(cache["X_visual_memory"], device)
        dims["E8_visual_memory_mean"] = effective_dim(
            cache["X_visual_memory_mean"], device
        )
        for j in range(len([k for k in cache if k.startswith("X_visual_memory_slot")])):
            key = f"X_visual_memory_slot{j}"
            if key in cache:
                dims[f"E8_visual_memory_slot{j}"] = effective_dim(cache[key], device)
    for j in range(int(args.n_ovt_per_object)):
        key = f"X_ovt_slot{j}"
        if key in cache:
            dims[f"OVT_slot{j}"] = effective_dim(cache[key], device)

    ovt_pair = summarize_1d(cache["ovt_pair_cosine"]) if "ovt_pair_cosine" in cache else {}

    summary = {
        "model_path": args.model_path,
        "num_images": int(len(uniq)),
        "num_objects": int(Y.shape[0]),
        "split_ratio": args.split_ratio,
        "probes_raw_target": results,
        "probes_instance_residual": results_res,
        "residual_variance_fraction": frac,
        "ovt_pair_cosine": ovt_pair,
        "effective_dim": {k: {kk: vv for kk, vv in v.items() if kk != "spectrum"} for k, v in dims.items()},
    }
    with open(Path(args.output_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(Path(args.output_dir) / "spectra.json", "w") as f:
        json.dump({k: v["spectrum"] for k, v in dims.items()}, f)
    with open(Path(args.output_dir) / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "probe", "r2", "cosine", "rel_l2", "lambda"])
        w.writeheader()
        for tgt, res in [("raw", results), ("instance_residual", results_res)]:
            for k, v in res.items():
                w.writerow({"target": tgt, "probe": k,
                            **{kk: v.get(kk) for kk in ["r2", "cosine", "rel_l2", "lambda"]}})

    print("\n" + "=" * 78)
    print("OVT APPEARANCE PROBE — can the OVT predict its object's appearance?")
    print("=" * 78)
    print(f"[A] RAW TARGET (category-dominated — a purely semantic code already scores here)")
    print(f"{'probe':<26} {'held-out R^2':>13} {'cosine':>9} {'rel_L2':>9}")
    for k, v in results.items():
        print(f"{k:<26} {v['r2']:>13.4f} {v['cosine']:>9.4f} {v['rel_l2']:>9.4f}")
    print()
    print(f"[B] INSTANCE RESIDUAL (word removed — THIS is appearance proper). "
          f"Residual keeps {100*frac:.1f}% of target variance.")
    print(f"{'probe':<26} {'held-out R^2':>13} {'cosine':>9} {'rel_L2':>9}")
    for k, v in results_res.items():
        print(f"{k:<26} {v['r2']:>13.4f} {v['cosine']:>9.4f} {v['rel_l2']:>9.4f}")
    if ovt_pair:
        print("-" * 78)
        print(
            "OVT pair cosine "
            f"(slot0 vs slot1): mean={ovt_pair['mean']:.4f}, "
            f"median={ovt_pair['median']:.4f}, p90={ovt_pair['p90']:.4f}, "
            f"count={ovt_pair['count']}"
        )
    print("-" * 78)
    print(f"{'representation':<26} {'nominal':>9} {'part.ratio':>11} {'d90':>6} {'d99':>6}")
    for k, v in dims.items():
        print(f"{k:<26} {v['nominal_dim']:>9} {v['participation_ratio']:>11.1f} "
              f"{v['dims_for_90pct_var']:>6} {v['dims_for_99pct_var']:>6}")
    print("=" * 78)
    log.info("Wrote %s", Path(args.output_dir) / "summary.json")


if __name__ == "__main__":
    main()
