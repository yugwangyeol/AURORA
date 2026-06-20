"""Encoder feature-ceiling comparison (NO PGOT model, NO training).

Question: at the SAME 32x32 patch grid, which frozen encoder's features can
separate COCO instances best (GT-supervised prototype oracle)?

Compares raw frozen patch features from:
  - google/siglip2-so400m-patch16-512   (current PGOT encoder; 512->32x32)
  - facebook/dinov2-base                 (448->32x32, interpolate_pos_encoding)
  - facebook/dinov2-with-registers-base  (448->32x32)

For each image: LN(patch features) -> per-GT-object prototype (mean) + bg
prototype -> assign each patch to nearest (cosine) -> upsample nearest to 256
-> fARI/mBO vs coco_instance GT (overlap excluded). This is the SAME oracle as
probe_feature_ceiling.R1 but on raw encoder features, so the numbers are
directly comparable to R1_feature_oracle (LLM-contextualised SigLIP = 0.417).

Usage:
  PYTHONPATH=/home/jovyan/PGOT python pgot/eval/probe_encoder_compare.py \
     --output_dir /home/jovyan/PGOT/outputs/probe_encoder_compare --max_samples 512
"""
import argparse, json, logging, os, sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/home/jovyan/PGOT")
from pgot.eval.pgot_metrics import fari_metric, mbo_metric
from pgot.eval.run_eval import CocoInstanceMaskCache
from transformers import AutoModel, AutoImageProcessor

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("enc")

ENCODERS = {
    "siglip2_512":      ("google/siglip2-so400m-patch16-512", 512, False),
    "dinov2_base":      ("facebook/dinov2-base",              518, True),
    "dinov2_reg_base":  ("facebook/dinov2-with-registers-base", 518, True),
}


def up_nearest(lab_g, g, size):
    t = torch.from_numpy(lab_g.reshape(g, g).astype(np.float32))[None, None]
    return F.interpolate(t, size=(size, size), mode="nearest")[0, 0].to(torch.int64)


def gt_down(gt256, g):
    return F.interpolate(gt256.float()[None, None], size=(g, g), mode="nearest")[0, 0].to(torch.int64)


@torch.no_grad()
def patch_feats(model, proc, name, img, size, is_dino, patch_size, device, target_g=32):
    """Return (target_g, target_g, D) patch features for one PIL image.

    DINO is run at its NATIVE pretrained resolution (no pos-emb interpolation),
    then its feature grid is bilinearly resampled to target_g so all encoders are
    compared at the same 32x32 grid.
    """
    if is_dino:
        # Force native square input (size = config image_size) so pretrained
        # position embeddings match exactly (this tf version has no
        # interpolate_pos_encoding kwarg).
        px = proc(images=img, return_tensors="pt",
                  size={"shortest_edge": size},
                  crop_size={"height": size, "width": size})["pixel_values"].to(device)
        out = model(pixel_values=px)
        h = out.last_hidden_state  # (1, 1+reg+P, D)
        g0 = px.shape[-1] // patch_size
        n_extra = h.shape[1] - g0 * g0
        h = h[:, n_extra:, :]      # drop CLS (+ register) tokens
        feats = h[0].reshape(g0, g0, -1)
    else:
        px = proc(images=img, return_tensors="pt")["pixel_values"].to(device)
        vm = getattr(model, "vision_model", model)   # Siglip2Model -> vision_model
        out = vm(pixel_values=px)
        h = out.last_hidden_state  # (1, P, D) siglip patch tokens, no CLS
        g0 = int(round(h.shape[1] ** 0.5))
        feats = h[0].reshape(g0, g0, -1)
    if g0 != target_g:
        feats = F.interpolate(feats.permute(2, 0, 1).unsqueeze(0).float(),
                              size=(target_g, target_g), mode="bilinear",
                              align_corners=False)[0].permute(1, 2, 0)
    return feats, target_g


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_samples", type=int, default=512)
    p.add_argument("--coco_mask_cache", default="/home/jovyan/PGOT/data/coco_inst_mask_cache_coda256")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    samples = [json.loads(l) for l in open(args.val_jsonl)][: args.max_samples]
    cache = CocoInstanceMaskCache(args.coco_mask_cache)
    eval_size = cache.size

    res = defaultdict(lambda: {"fari": [], "mbo": []})
    for name, (repo, size, is_dino) in ENCODERS.items():
        log.info(f"loading {name} ({repo}) ...")
        proc = AutoImageProcessor.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo, torch_dtype=torch.float32).to(device).eval()
        patch_size = int(getattr(model.config, "patch_size", 14))
        done = 0
        for s in samples:
            gt256 = cache.get(int(s["image_id"]))
            if gt256 is None or int(gt256.max()) == 0:
                continue
            ov = cache.get_overlap(int(s["image_id"]))
            ovb = (ov if ov is not None else torch.zeros_like(gt256, dtype=torch.uint8))[None].to(device)
            gtb = gt256[None].to(device)
            try:
                img = Image.open(s["image_path"]).convert("RGB")
            except Exception:
                continue
            feats2d, g = patch_feats(model, proc, name, img, size, is_dino, patch_size, device)  # (g,g,D)
            D = feats2d.shape[-1]
            feats = F.layer_norm(feats2d.reshape(g * g, D).float(), (D,))
            gtg = gt_down(gt256, g).cpu().numpy().reshape(-1)
            objs = [o for o in np.unique(gtg) if o != 0]
            if not objs:
                continue
            pl = [feats[torch.from_numpy(gtg == 0).to(device)].mean(0)] if (gtg == 0).any() \
                else [torch.zeros(D, device=device)]
            for o in objs:
                pl.append(feats[torch.from_numpy(gtg == o).to(device)].mean(0))
            protos = F.layer_norm(torch.stack(pl), (D,))
            lab = (feats @ protos.t()).argmax(1).cpu().numpy()
            pr = up_nearest(lab, g, eval_size)[None].to(device)
            fa = fari_metric(gtb, pr, ovb); mb = mbo_metric(gtb, pr, ovb)
            if not np.isnan(fa): res[name]["fari"].append(fa)
            if not np.isnan(mb): res[name]["mbo"].append(mb)
            done += 1
            if done % 100 == 0:
                log.info(f"  {name}: {done} done, grid={g}x{g}, D={D}, fARI={np.mean(res[name]['fari']):.3f}")
        del model
        torch.cuda.empty_cache()
        log.info(f"== {name}: fARI={np.mean(res[name]['fari']):.4f} mBO={np.mean(res[name]['mbo']):.4f} (n={len(res[name]['fari'])})")

    summary = {k: {"fARI": float(np.mean(v["fari"])), "mBO": float(np.mean(v["mbo"])), "n": len(v["fari"])}
               for k, v in res.items()}
    summary["_note"] = "GT-prototype oracle on RAW frozen patch features @32x32. Compare to PGOT R1_feature_oracle=0.417 (LLM-SigLIP). CODA fARI target=0.475."
    json.dump(summary, open(os.path.join(args.output_dir, "encoder_compare.json"), "w"), indent=2)
    log.info("=" * 60)
    for k in ENCODERS:
        if k in summary:
            log.info(f"{k:18s} fARI={summary[k]['fARI']:.4f}  mBO={summary[k]['mBO']:.4f}  (n={summary[k]['n']})")
    log.info(f"written -> {args.output_dir}/encoder_compare.json")


if __name__ == "__main__":
    main()
