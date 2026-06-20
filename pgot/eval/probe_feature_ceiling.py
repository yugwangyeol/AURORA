"""Feature-ceiling diagnostic probe (NO training).

Decomposes the FG-ARI gap of a trained PGOT checkpoint into:
  R0  current readout (threshold, thing-only)            -> what we have now
  R1  GT-supervised feature-prototype oracle on img_hidden -> frozen-feature partition ceiling
  R2  model OVT maps + per-image oracle background threshold -> ceiling if bg/competition were perfect
  R3  GT-supervised OVT-vector-as-prototype oracle assignment -> ceiling of the OVT vectors themselves

All assignments are done on the 32x32 patch grid, upsampled NEAREST to the GT
resolution, and scored with the SAME metric functions / overlap exclusion as
run_eval.py (coco_instance GT).

Usage:
  PYTHONPATH=/home/jovyan/PGOT python pgot/eval/probe_feature_ceiling.py \
     --model_path /home/jovyan/PGOT/checkpoints/pgot_main_v11_bal_bce_ce03_mean \
     --output_dir /home/jovyan/PGOT/outputs/probe_v11_feature_ceiling \
     --max_samples 512 --batch_size 8
"""
import argparse, glob, json, logging, os, sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/jovyan/PGOT")
from pgot.constants import OVT_TOKEN, SCENE_END_TOKEN, NEW_SPECIAL_TOKENS
from pgot.model.pgot_qwen2 import PGOTQwen2ForCausalLM
from pgot.train.pgot_dataset import Pix2CapPGOTDataset, PGOTDataCollator
from pgot.eval.pgot_metrics import fari_metric, mbo_metric, miou_metric
from pgot.eval.pgot_inference import pgot_forward_eval
from pgot.eval.run_eval import CocoInstanceMaskCache, load_thing_categories
from transformers import AutoTokenizer, AutoConfig

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("probe")


def build_model(model_path, device, dtype):
    config = AutoConfig.from_pretrained(model_path)
    model = PGOTQwen2ForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=dtype, ignore_mismatched_sizes=True)
    model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="right")
    # LoRA re-inject if present
    has_lora = False
    for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        import safetensors.torch as st
        with st.safe_open(shard, framework="pt", device="cpu") as f:
            if any("lora_" in k for k in f.keys()):
                has_lora = True
        if has_lora:
            break
    if has_lora:
        from peft import LoraConfig, inject_adapter_in_model
        import safetensors.torch as st
        lc = LoraConfig(r=int(getattr(config, "captionslot_lora_r", 16)),
                        lora_alpha=int(getattr(config, "captionslot_lora_alpha", 32)),
                        lora_dropout=0.0, bias="none",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                        task_type="CAUSAL_LM")
        inject_adapter_in_model(lc, model, adapter_name="default")
        sd = {}
        for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
            with st.safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
        model.load_state_dict(sd, strict=False)
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"; tok.pad_token_id = 151643
    if OVT_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": NEW_SPECIAL_TOKENS})
        model.resize_token_embeddings(len(tok))
    parsed_towers = getattr(config, "mm_vision_tower_aux_list", None) or json.loads(
        getattr(config, "vision_tower_aux_list", '["google/siglip2-so400m-patch14-224"]'))
    parsed_lens = getattr(config, "mm_vision_tower_aux_token_len_list", None) or [256]
    from types import SimpleNamespace
    vt_args = SimpleNamespace(
        vision_tower_aux_list=parsed_towers, vision_tower_aux_token_len_list=parsed_lens,
        mm_vision_select_layer=-1, mm_vision_select_feature="patch", mm_projector_type="mlp2x_gelu",
        mm_use_im_start_end=True, mm_use_im_patch_token=False, unfreeze_mm_vision_tower=False,
        vision_hidden_size=1024, connector_only=True, pretrain_mm_mlp_adapter=None,
        pretrain_adapter_and_vision_head=None,
        diffusion_norm_stats_path=getattr(config, "diffusion_norm_stats_path", None))
    model.get_model().initialize_vision_modules(model_args=vt_args, fsdp=None)
    model.load_vision_head(model_args=vt_args)
    for vt in model.get_vision_tower_aux_list():
        vt.to(dtype=dtype, device=device)
    model.pgot_ovt_token_id = tok.convert_tokens_to_ids(OVT_TOKEN)
    model.pgot_scene_end_token_id = tok.convert_tokens_to_ids(SCENE_END_TOKEN)
    blocks = {
        "pgot_system_prefix_ids": "<|im_start|>system\nYou are a vision assistant that describes scenes with grounded objects.",
        "pgot_system_suffix_ids": "<|im_end|>\n",
        "pgot_user_prefix_ids": "<|im_start|>user\n",
        "pgot_user_suffix_ids": "\nDescribe all objects and regions in this scene with grounded tokens.<|im_end|>\n",
        "pgot_assistant_prefix_ids": "<|im_start|>assistant\n",
        "pgot_assistant_suffix_ids": "<|im_end|>"}
    for a, t in blocks.items():
        setattr(model, a, tok.encode(t, add_special_tokens=False))
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, tok


def labelmap_up_nearest(lab32, size):
    t = torch.from_numpy(lab32.astype(np.float32))[None, None]
    up = F.interpolate(t, size=(size, size), mode="nearest")
    return up[0, 0].to(torch.int64)


def gt_down_nearest(gt256, g):
    t = gt256.float()[None, None]
    d = F.interpolate(t, size=(g, g), mode="nearest")
    return d[0, 0].to(torch.int64)  # (g,g)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_samples", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grid_size", type=int, default=32)
    p.add_argument("--n_ovt_per_object", type=int, default=2)
    p.add_argument("--max_objects", type=int, default=50)
    p.add_argument("--coco_mask_cache", default="/home/jovyan/PGOT/data/coco_inst_mask_cache_coda256")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"; dtype = torch.float32
    g = args.grid_size; P = g * g; n = args.n_ovt_per_object

    model, tok = build_model(args.model_path, device, dtype)
    log.info("model loaded")

    vt = model.get_vision_tower_aux_list()
    ip = vt[0].image_processor
    tp = vt[1].image_processor if len(vt) > 1 else ip
    ds = Pix2CapPGOTDataset(jsonl_path=args.val_jsonl, tokenizer=tok, image_processor=ip,
                            target_image_processor=tp, grid_size=g, n_ovt_per_object=n,
                            max_objects=args.max_objects,
                            panoptic_categories_json="/home/jovyan/data/coco/annotations/panoptic_val2017.json")
    samples = ds.samples
    if args.max_samples:
        ds = torch.utils.data.Subset(ds, list(range(min(args.max_samples, len(ds)))))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         collate_fn=PGOTDataCollator(pad_token_id=tok.pad_token_id),
                                         num_workers=4)
    cache = CocoInstanceMaskCache(args.coco_mask_cache)
    eval_size = cache.size

    res = defaultdict(lambda: {"fari": [], "mbo": []})
    bg_taus = np.linspace(0.05, 0.6, 6)

    for bi, batch in enumerate(loader):
        out = pgot_forward_eval(
            model, images=batch["images"], target_images=batch["target_images"],
            caption_input_ids=batch["caption_input_ids"],
            caption_attention_mask=batch["caption_attention_mask"],
            ovt_positions_in_caption=batch["ovt_positions_in_caption"],
            ovt_valid_mask=batch["ovt_valid_mask"])
        img_hidden = out["img_hidden"].float()          # (B,P,D)
        ovt_logits = out["ovt_logits"].float()          # (B,M,P)
        ovt_valid = out["ovt_valid_mask"].to(device)
        ovt_thing = batch["ovt_is_thing"].to(device).bool()
        B = img_hidden.shape[0]
        D = img_hidden.shape[-1]
        imgLN = F.layer_norm(img_hidden, (D,))          # (B,P,D)

        # object-level thing maps (merge n tokens, mean) -> prob via sigmoid
        M = ovt_logits.shape[1]; K = M // n
        obj_logits = ovt_logits[:, :K * n].reshape(B, K, n, P).mean(2)        # (B,K,P)
        obj_valid = ovt_valid[:, :K * n].reshape(B, K, n).any(2)             # (B,K)
        obj_thing = ovt_thing[:, :K * n].reshape(B, K, n).any(2)            # (B,K)
        obj_prob = torch.sigmoid(obj_logits)                                 # (B,K,P)

        for b in range(B):
            gidx = bi * args.batch_size + b
            gt256 = cache.get(int(samples[gidx]["image_id"]))
            if gt256 is None:
                continue
            ov = cache.get_overlap(int(samples[gidx]["image_id"]))
            overlap256 = (ov if ov is not None else torch.zeros_like(gt256, dtype=torch.uint8))
            if int(gt256.max()) == 0:
                continue
            gt256 = gt256.to(device); overlap256 = overlap256.to(device)
            gtb = gt256[None]; ovb = overlap256[None]
            gt32 = gt_down_nearest(gt256.cpu(), g).numpy().reshape(-1)        # (P,)
            feats = imgLN[b]                                                  # (P,D)

            def score(lab32, tag):
                pr = labelmap_up_nearest(lab32.reshape(g, g), eval_size)[None].to(device)
                fa = fari_metric(gtb, pr, ovb); mb = mbo_metric(gtb, pr, ovb)
                if not np.isnan(fa): res[tag]["fari"].append(fa)
                if not np.isnan(mb): res[tag]["mbo"].append(mb)

            # ---- R_res: resolution ceiling. Upper bound of ANY method that emits a
            # 32x32 partition: take the GT itself, downsampled to 32 and upsampled back.
            score(gt32.astype(np.int64), "Rres_resolution_ceiling")

            # ---- Multi-resolution: does raising the output grid (with a feature
            # decoder = bilinear-upsampled frozen features) lift the ceiling past CODA?
            feats_2d = feats.reshape(g, g, D).permute(2, 0, 1).unsqueeze(0)   # (1,D,g,g)
            for G in (64, 128):
                # resolution ceiling at G (GT only)
                gtG = gt_down_nearest(gt256.cpu(), G).numpy().reshape(-1)
                prG = labelmap_up_nearest(gtG.reshape(G, G), eval_size)[None].to(device)
                fa = fari_metric(gtb, prG, ovb)
                if not np.isnan(fa):
                    res[f"Rres_g{G}"]["fari"].append(fa)
                # frozen-feature oracle at G (bilinear-upsampled features + GT prototypes)
                fG = F.interpolate(feats_2d, size=(G, G), mode="bilinear", align_corners=False)
                fG = fG[0].permute(1, 2, 0).reshape(G * G, D)                  # (G*G,D)
                objsG = [o for o in np.unique(gtG) if o != 0]
                if objsG:
                    pl = [fG[torch.from_numpy(gtG == 0).to(device)].mean(0)] if (gtG == 0).any() \
                        else [torch.zeros(D, device=device)]
                    for o in objsG:
                        pl.append(fG[torch.from_numpy(gtG == o).to(device)].mean(0))
                    pr = F.layer_norm(torch.stack(pl), (D,))
                    labG = (fG @ pr.t()).argmax(1)
                    prG2 = labelmap_up_nearest(labG.cpu().numpy().reshape(G, G), eval_size)[None].to(device)
                    fa2 = fari_metric(gtb, prG2, ovb)
                    if not np.isnan(fa2):
                        res[f"R1_feature_oracle_g{G}"]["fari"].append(fa2)

            # ---- R0: current threshold readout (thing-only, sigmoid, bg<0.05) on 32 grid
            tk = obj_thing[b] & obj_valid[b]
            if tk.any():
                pm = obj_prob[b][tk]                                          # (Kt,P)
                mx, am = pm.max(0)
                lab = (am + 1).clone()
                lab[mx < 0.05] = 0
                score(lab.cpu().numpy(), "R0_current_threshold")
            else:
                score(np.zeros(P, dtype=np.int64), "R0_current_threshold")

            # ---- R1: GT feature-prototype oracle (frozen-feature ceiling)
            objs = [o for o in np.unique(gt32) if o != 0]
            if objs:
                protos = [feats[torch.from_numpy(gt32 == 0).to(device)].mean(0)] if (gt32 == 0).any() \
                    else [torch.zeros(D, device=device)]
                for o in objs:
                    protos.append(feats[torch.from_numpy(gt32 == o).to(device)].mean(0))
                protos = F.layer_norm(torch.stack(protos), (D,))             # (1+Ko, D)
                sim = feats @ protos.t()                                      # (P, 1+Ko)
                lab = sim.argmax(1)                                          # 0=bg else obj idx
                score(lab.cpu().numpy(), "R1_feature_oracle")

            # ---- R2: model OVT maps + per-image oracle bg threshold
            if tk.any():
                pm = obj_prob[b][tk]; mx, am = pm.max(0)
                best = None; best_fa = -2
                for tau in bg_taus:
                    lab = (am + 1).clone(); lab[mx < tau] = 0
                    pr = labelmap_up_nearest(lab.cpu().numpy().reshape(g, g), eval_size)[None].to(device)
                    fa = fari_metric(gtb, pr, ovb)
                    if not np.isnan(fa) and fa > best_fa:
                        best_fa = fa; best = lab.cpu().numpy()
                if best is not None:
                    score(best, "R2_ovtmap_oracle_bg")

            # ---- R3: GT-supervised OVT-vector-as-prototype (ceiling of OVT vectors)
            # use ovt_hidden? not returned; approximate via obj_logits argmax with GT bg threshold
            # (skip; R1+R2 are the decisive pair)

        if (bi + 1) % 10 == 0:
            log.info(f"batch {bi+1}: "
                     + " | ".join(f"{k}:fARI={np.mean(v['fari']):.3f}" for k, v in sorted(res.items())))

    summary = {"model": args.model_path, "n_eval": len(res["R0_current_threshold"]["fari"])}
    for k, v in sorted(res.items()):
        summary[k] = {"fARI": float(np.mean(v["fari"])) if v["fari"] else None,
                      "mBO": float(np.mean(v["mbo"])) if v["mbo"] else None,
                      "n": len(v["fari"])}
    with open(os.path.join(args.output_dir, "probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info("=" * 60)
    for k in ["R0_current_threshold", "R1_feature_oracle", "R2_ovtmap_oracle_bg",
              "Rres_resolution_ceiling", "R1_feature_oracle_g64", "Rres_g64",
              "R1_feature_oracle_g128", "Rres_g128"]:
        if k in summary:
            d = summary[k]
            mbo = f"{d['mBO']:.4f}" if d['mBO'] is not None else "  -  "
            log.info(f"{k:26s} fARI={d['fARI']:.4f}  mBO={mbo}  (n={d['n']})")
    log.info(f"written -> {args.output_dir}/probe_summary.json")


if __name__ == "__main__":
    main()
