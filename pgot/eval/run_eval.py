"""PGOT evaluation entry point.

Computes CODA-comparable metrics on Pix2Cap val:
    fARI / mBO / mIoU              (segmentation)
    rFID / PSNR / SSIM / MSE / MAE (reconstruction, optional)
    active_count                   (avg # of OVTs that contributed signal)
    per-object-count split         (1-3, 4-6, 7-10, 11+)

Usage:
    PYTHONPATH=/home/jovyan/PGOT python pgot/eval/run_eval.py \
        --model_path /home/jovyan/PGOT/checkpoints/pgot_main/checkpoint-10000 \
        --val_jsonl  /home/jovyan/PGOT/data/pgot_val.jsonl \
        --output_dir /home/jovyan/PGOT/outputs/eval_main_10k \
        --batch_size 4 --compute_rfid False
"""
import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project imports
sys.path.insert(0, "/home/jovyan/PGOT")
from pgot.constants import OVT_TOKEN, SCENE_END_TOKEN, NEW_SPECIAL_TOKENS
from pgot.model.pgot_qwen2 import PGOTQwen2ForCausalLM
from pgot.train.pgot_dataset import Pix2CapPGOTDataset, PGOTDataCollator
from pgot.eval.pgot_metrics import (
    fari_metric, mbo_metric, miou_metric,
    ovt_logits_to_pred_mask,
    build_pred_mask_spatial_readout,
    FIDAccumulator,
    compute_recon_metrics,
)
from pgot.model.pgot_utils import (
    build_pred_mask_competition_eval,
    build_pred_mask_llm_attention_eval,
    build_pred_mask_null_bg_eval,
    build_pred_mask_ovt_owner_eval,
)
from pgot.eval.pgot_inference import pgot_forward_eval, generate_siglip_latent

from transformers import AutoTokenizer, AutoConfig

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s :: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pgot.eval")


# ----------------------------------------------------------------------
# Build GT panoptic mask at target_size  (per-image; 0=bg, 1..N=segments in caption order)
# ----------------------------------------------------------------------
def load_gt_panoptic_mask(panoptic_path: str, segment_ids: list, target_size: int) -> torch.Tensor:
    rgb = np.array(Image.open(panoptic_path).convert("RGB"))
    seg_id = (rgb[..., 0].astype(np.int64) + rgb[..., 1].astype(np.int64) * 256 + rgb[..., 2].astype(np.int64) * 256 * 256)
    out = np.zeros((seg_id.shape[0], seg_id.shape[1]), dtype=np.int64)
    for k, sid in enumerate(segment_ids):
        out[seg_id == int(sid)] = k + 1  # 1..K (0=background)
    # Resize to target_size via nearest
    mask_t = torch.from_numpy(out).unsqueeze(0).unsqueeze(0).float()
    mask_t = F.interpolate(mask_t, size=(target_size, target_size), mode="nearest")
    return mask_t.squeeze(0).squeeze(0).to(torch.int64)


# ----------------------------------------------------------------------
# RAE decoder loader (nyu-visionx/siglip2_decoder via scale_rae's MultimodalDecoder)
# ----------------------------------------------------------------------
def load_rae_decoder(pgot_model, device, dtype=torch.float32):
    from huggingface_hub import hf_hub_download
    from scale_rae.model.multimodal_decoder import MultimodalDecoder

    repo_id = "nyu-visionx/siglip2_decoder"
    vision_towers = list(getattr(pgot_model.config, "mm_vision_tower_aux_list",
                                 ["google/siglip2-so400m-patch14-224"]))
    encoder_path = vision_towers[1] if len(vision_towers) > 1 else vision_towers[0]
    encoder_path = encoder_path.split("-interp")[0]
    num_patches = int(getattr(pgot_model.config, "diffusion_target_token_len", 256))
    log.info(f"Loading RAE decoder ({repo_id}) for encoder={encoder_path} P={num_patches}")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="model.pt")
    dec = MultimodalDecoder(
        pretrained_encoder_path=encoder_path,
        general_decoder_config=config_path,
        num_patches=num_patches,
        drop_cls_token=True,
        decoder_path=ckpt_path,
    )
    dec.eval()
    dec.to(device=device, dtype=dtype)
    return dec


@torch.no_grad()
def decode_to_image(decoder, generated: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Replicates AURORA's decode_generated_images."""
    if generated is None:
        return torch.empty(0, 3, 1, 1, device=device)
    decoder = decoder.to(device=device)
    decoder_dtype = next(decoder.parameters()).dtype
    if generated.dtype != decoder_dtype:
        generated = generated.to(dtype=decoder_dtype)
    if hasattr(decoder, "image_mean"):
        decoder.image_mean = decoder.image_mean.to(device=device, dtype=decoder_dtype)
        decoder.image_std = decoder.image_std.to(device=device, dtype=decoder_dtype)
    empty_cls = torch.zeros((generated.shape[0], 1, generated.shape[-1]), device=device, dtype=decoder_dtype)
    image_features = torch.cat([empty_cls, generated], dim=1)
    recon = decoder(image_features)
    recon = torch.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=0.0)
    return recon.clamp(0.0, 1.0).detach().float()  # (B, 3, H, W) in [0,1]


def denormalize_images(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.view(1, -1, 1, 1).to(device=images.device, dtype=images.dtype)
    std = std.view(1, -1, 1, 1).to(device=images.device, dtype=images.dtype)
    return (images * std + mean).clamp(0.0, 1.0)


# ----------------------------------------------------------------------
# CODA-style instance mask cache loader
# ----------------------------------------------------------------------
def load_thing_categories(panoptic_json_path: str) -> set:
    """COCO panoptic JSON contains a 'categories' field with isthing flags."""
    with open(panoptic_json_path) as f:
        d = json.load(f)
    return {c["name"] for c in d.get("categories", []) if int(c.get("isthing", 0)) == 1}


class CocoInstanceMaskCache:
    """Loads AURORA's coco_mask_cache (mode=coda, size=256). Look up by COCO image_id."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.masks = np.load(os.path.join(cache_dir, "masks.npy"), mmap_mode="r")
        overlap_path = os.path.join(cache_dir, "overlap_masks.npy")
        self.overlap_masks = (
            np.load(overlap_path, mmap_mode="r") if os.path.exists(overlap_path) else None
        )
        self.image_ids = np.load(os.path.join(cache_dir, "image_ids.npy")).tolist()
        self.id2idx = {int(iid): i for i, iid in enumerate(self.image_ids)}
        with open(os.path.join(cache_dir, "meta.json")) as f:
            self.meta = json.load(f)
        log.info(f"CocoInstanceMaskCache: {self.masks.shape}, size={self.meta.get('size')}, mode={self.meta.get('mode')}")
        if self.overlap_masks is None:
            log.warning(
                "overlap_masks.npy not found; overlap pixels will not be excluded. "
                "Run pgot.eval.build_coco_overlap_cache for CODA-matched metrics."
            )
        elif self.overlap_masks.shape != self.masks.shape:
            raise ValueError(
                f"Overlap cache shape {self.overlap_masks.shape} does not match "
                f"GT cache shape {self.masks.shape}."
            )
        else:
            log.info(f"Loaded CODA overlap masks: {self.overlap_masks.shape}")

    @property
    def size(self) -> int:
        return int(self.meta["size"])

    def get(self, image_id: int):
        idx = self.id2idx.get(int(image_id))
        if idx is None:
            return None
        return torch.from_numpy(np.asarray(self.masks[idx], dtype=np.int64))

    def get_overlap(self, image_id: int):
        if self.overlap_masks is None:
            return None
        idx = self.id2idx.get(int(image_id))
        if idx is None:
            return None
        return torch.from_numpy(np.asarray(self.overlap_masks[idx], dtype=np.uint8))


# ----------------------------------------------------------------------
# Main eval loop
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="PGOT trained checkpoint")
    p.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--grid_size", type=int, default=32)   # v3 default
    p.add_argument("--guidance_scale", type=float, default=2.5,
                   help="CFG guidance scale for rFID inference (1.0 = no guidance).")
    p.add_argument("--eval_size", type=int, default=224, help="GT/pred mask resolution")
    p.add_argument("--max_caption_tokens", type=int, default=2048)
    p.add_argument("--n_ovt_per_object", type=int, default=2)
    p.add_argument("--max_objects", type=int, default=50)
    p.add_argument("--bg_threshold", type=float, default=0.05)
    p.add_argument("--eval_merge", choices=["mean", "max"], default="max",
                   help="How to merge the n_ovt_per_object logits into one object score for competition.")
    p.add_argument(
        "--readout",
        choices=[
            "competition", "threshold", "nullbg", "spatial",
            "spatial_trainmatch", "llm_attention", "ovt_owner", "slot_owner",
        ],
        default="competition",
                   help="competition: argmax over {K objects, register-bg} (v5/v6 style). "
                        "threshold: filter stuff OVTs, sigmoid+bg_threshold on thing OVTs only (v3 style). "
                        "nullbg: argmax over {thing OVT objects, null-bg} (v7 style). "
                        "spatial: configurable per-OVT patch-axis softmax readout. "
                        "spatial_trainmatch: V8.2 training-matched patch-axis softmax, "
                        "checkpoint temperature, and mean merge. "
                        "llm_attention: V8.4/V8.5 internal-attention maps with void. "
                        "ovt_owner: V12 OVT-owner softmax maps with void. "
                        "slot_owner is accepted as a legacy alias.")
    p.add_argument("--spatial_temperature", type=float, default=1.0,
                   help="Patch-axis softmax temperature for --readout spatial.")
    p.add_argument("--compute_rfid", action="store_true", help="Decode + FID (slow)")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--gt_source", choices=["pix2cap_panoptic", "coco_instance"], default="pix2cap_panoptic")
    p.add_argument("--coco_mask_cache", default="/home/jovyan/PGOT/data/coco_inst_mask_cache_coda256",
                   help="CODA-style mask cache (built by AURORA's coco_mask_cache.py mode=coda)")
    p.add_argument(
        "--image_preprocess_mode",
        choices=["default", "coda_center_crop"],
        default="default",
        help="default: use the model's image processor on the raw image. "
             "coda_center_crop: first apply CODA's resize-min-side + center crop "
             "so predictions align with CODA-style COCO instance masks.",
    )
    p.add_argument(
        "--coda_crop_size",
        type=int,
        default=512,
        help="Square size used for the pre-SigLIP CODA center crop.",
    )
    p.add_argument("--diffusion_inference_steps", type=int, default=10,
                   help="Number of RF denoising steps for rFID (AURORA uses 10).")
    args = p.parse_args()
    if args.readout == "slot_owner":
        args.readout = "ovt_owner"

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    # ---- Load model
    log.info(f"Loading model from: {args.model_path}")
    raw_config_path = os.path.join(args.model_path, "config.json")
    raw_name_or_path = args.model_path
    if os.path.exists(raw_config_path):
        with open(raw_config_path, "r") as f:
            raw_cfg = json.load(f)
        raw_name_or_path = str(raw_cfg.get("_name_or_path", args.model_path))
    config = AutoConfig.from_pretrained(args.model_path)
    import glob
    has_lora_in_ckpt = False
    index_path = os.path.join(args.model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index_json = json.load(f)
        has_lora_in_ckpt = any(
            "lora_" in k or ".base_layer." in k
            for k in index_json.get("weight_map", {}).keys()
        )

    model_init_path = args.model_path
    if has_lora_in_ckpt:
        model_init_path = raw_name_or_path
        log.info(
            "[LoRA] adapter checkpoint detected; bootstrap model from base path: %s",
            model_init_path,
        )

    model = PGOTQwen2ForCausalLM.from_pretrained(
        model_init_path, config=config, torch_dtype=dtype, ignore_mismatched_sizes=True,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = 151643

    # If <ovt> not in vocab (checkpoint pre-resize), add them. Usually they ARE saved.
    if OVT_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": NEW_SPECIAL_TOKENS})
        model.resize_token_embeddings(len(tokenizer))

    # Vision tower init
    parsed_towers = getattr(config, "mm_vision_tower_aux_list", None) or json.loads(
        getattr(config, "vision_tower_aux_list", '["google/siglip2-so400m-patch14-224"]')
    )
    parsed_token_lens = (
        getattr(config, "mm_vision_tower_aux_token_len_list", None) or [256]
    )
    from types import SimpleNamespace
    vt_args = SimpleNamespace(
        vision_tower_aux_list=parsed_towers,
        vision_tower_aux_token_len_list=parsed_token_lens,
        mm_vision_select_layer=-1,
        mm_vision_select_feature="patch",
        mm_projector_type="mlp2x_gelu",
        mm_use_im_start_end=True,
        mm_use_im_patch_token=False,
        unfreeze_mm_vision_tower=False,
        vision_hidden_size=1024,
        connector_only=True,
        pretrain_mm_mlp_adapter=None,
        pretrain_adapter_and_vision_head=None,
        diffusion_norm_stats_path=getattr(config, "diffusion_norm_stats_path", None),
    )
    model.get_model().initialize_vision_modules(model_args=vt_args, fsdp=None)
    model.load_vision_head(model_args=vt_args)
    for vt in model.get_vision_tower_aux_list():
        vt.to(dtype=dtype, device=device)

    # Register ovt token ids on model
    model.pgot_ovt_token_id = tokenizer.convert_tokens_to_ids(OVT_TOKEN)
    model.pgot_scene_end_token_id = tokenizer.convert_tokens_to_ids(SCENE_END_TOKEN)

    # If checkpoint contains LoRA-wrapped weights, inject the adapters AFTER
    # all PGOT/vision modules exist, then load the checkpoint state once.
    if has_lora_in_ckpt:
        import safetensors.torch as safe_torch
        from peft import LoraConfig, inject_adapter_in_model

        lora_cfg = LoraConfig(
            r=int(getattr(config, "captionslot_lora_r", 16)) if hasattr(config, "captionslot_lora_r") else 16,
            lora_alpha=int(getattr(config, "captionslot_lora_alpha", 32)) if hasattr(config, "captionslot_lora_alpha") else 32,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        inject_adapter_in_model(lora_cfg, model, adapter_name="default")

        sd = {}
        for shard in sorted(glob.glob(os.path.join(args.model_path, "*.safetensors"))):
            with safe_torch.safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log.info(
            "[LoRA] re-loaded ckpt after adapter injection | missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )

    # Template ids
    blocks = {
        "pgot_system_prefix_ids": "<|im_start|>system\nYou are a vision assistant that describes scenes with grounded objects.",
        "pgot_system_suffix_ids": "<|im_end|>\n",
        "pgot_user_prefix_ids":   "<|im_start|>user\n",
        "pgot_user_suffix_ids":   "\nDescribe all objects and regions in this scene with grounded tokens.<|im_end|>\n",
        "pgot_assistant_prefix_ids": "<|im_start|>assistant\n",
        "pgot_assistant_suffix_ids": "<|im_end|>",
    }
    for attr, txt in blocks.items():
        setattr(model, attr, tokenizer.encode(txt, add_special_tokens=False))

    model.to(device=device, dtype=dtype)
    model.eval()
    log.info("Model loaded.")

    # ---- Dataset
    vt_list = model.get_vision_tower_aux_list()
    image_proc = vt_list[0].image_processor
    target_proc = vt_list[1].image_processor if len(vt_list) > 1 else image_proc
    val_dataset = Pix2CapPGOTDataset(
        jsonl_path=args.val_jsonl,
        tokenizer=tokenizer,
        image_processor=image_proc,
        target_image_processor=target_proc,
        grid_size=args.grid_size,
        max_caption_tokens=args.max_caption_tokens,
        n_ovt_per_object=args.n_ovt_per_object,
        max_objects=args.max_objects,
        panoptic_categories_json="/home/jovyan/data/coco/annotations/panoptic_val2017.json",
        image_preprocess_mode=args.image_preprocess_mode,
        coda_crop_size=args.coda_crop_size,
    )
    if args.max_samples is not None:
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(min(args.max_samples, len(val_dataset)))))
    log.info(f"Eval set: {len(val_dataset)} samples")

    collator = PGOTDataCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=args.num_workers, pin_memory=True,
    )

    # ---- Metric accumulators
    fari_scores, mbo_scores, miou_scores = [], [], []
    by_count = defaultdict(lambda: {"fari": [], "mbo": [], "miou": []})
    active_counts = []
    image_ids = []
    n_objects_list = []

    # CODA-style GT cache (optional)
    coco_cache = None
    thing_categories = None
    if args.gt_source == "coco_instance":
        coco_cache = CocoInstanceMaskCache(args.coco_mask_cache)
        args.eval_size = coco_cache.size   # honor cache resolution
        # When GT is thing-only, filter out stuff OVTs so they don't count as false positives.
        thing_categories = load_thing_categories("/home/jovyan/data/coco/annotations/panoptic_val2017.json")
        log.info(f"Loaded {len(thing_categories)} thing categories for OVT filtering.")

    # rFID accumulators
    fid_acc = None
    rae_decoder = None
    recon_psnr_list, recon_ssim_list, recon_mse_list, recon_mae_list = [], [], [], []

    if args.compute_rfid:
        rae_decoder = load_rae_decoder(model, device=device, dtype=dtype)
        # Patch inference steps for speed (10 vs default 50)
        if args.diffusion_inference_steps and args.diffusion_inference_steps != 50:
            try:
                from scale_rae.model.diffusion_loss.diffusion import create_diffusion
                inf = model.diff_head.inference_flow
                size_ratio = float(getattr(inf, "size_ratio", 1.0))
                d_steps = int(getattr(inf, "diffusion_steps", 1000))
                model.diff_head.inference_flow = create_diffusion(
                    str(args.diffusion_inference_steps),
                    noise_schedule="linear", use_kl=False, sigma_small=False,
                    predict_xstart=False, learn_sigma=False, rescale_learned_sigmas=False,
                    diffusion_steps=d_steps, input_base_dimension_ratio=size_ratio,
                    diffusion_type="rf", use_loss_weighting=False,
                )
                log.info(f"[rFID] Patched RF inference_flow to {args.diffusion_inference_steps} steps.")
            except Exception as e:
                log.warning(f"[rFID] Could not patch diffusion steps: {e}")
        try:
            fid_acc = FIDAccumulator(device=device, feature=2048)
        except (ImportError, ModuleNotFoundError):
            log.warning("torchmetrics not available — skipping rFID.")
            fid_acc = None

    # Image normalization stats (for un-normalizing model input → [0,1] real images)
    img_proc = vt_list[0].image_processor
    img_mean = torch.tensor(img_proc.image_mean).view(1, -1, 1, 1)
    img_std = torch.tensor(img_proc.image_std).view(1, -1, 1, 1)

    # ---- Loop
    samples_iter = val_dataset.dataset.samples if isinstance(val_dataset, torch.utils.data.Subset) else val_dataset.samples
    for batch_idx, batch in enumerate(tqdm(loader, desc="Eval")):
        out = pgot_forward_eval(
            model,
            images=batch["images"],
            target_images=batch["target_images"],
            caption_input_ids=batch["caption_input_ids"],
            caption_attention_mask=batch["caption_attention_mask"],
            ovt_positions_in_caption=batch["ovt_positions_in_caption"],
            ovt_valid_mask=batch["ovt_valid_mask"],
            return_llm_attention_maps=(args.readout == "llm_attention"),
        )

        # ── Readout: either v5/v6 competition (thing+stuff+register-bg argmax)
        # or v3-style threshold (stuff OVTs filtered out, sigmoid+bg_threshold
        # on thing OVTs only). The latter is the cleaner background mechanism
        # used in v3 and is useful for diagnosing whether v6's dual-background
        # (stuff OVT + register) is what's hurting fARI.
        valid_for_pred = out["ovt_valid_mask"].clone()
        ovt_is_thing = batch.get("ovt_is_thing")
        has_batch_thing = ovt_is_thing is not None
        if has_batch_thing:
            ovt_is_thing = ovt_is_thing.to(valid_for_pred.device, dtype=torch.bool)
        else:
            ovt_is_thing = torch.zeros_like(valid_for_pred, dtype=torch.bool)
        if thing_categories is not None:
            for b in range(valid_for_pred.shape[0]):
                global_idx = batch_idx * args.batch_size + b
                segs = samples_iter[global_idx]["segments"]
                for k, seg in enumerate(segs):
                    s = k * args.n_ovt_per_object
                    e = s + args.n_ovt_per_object
                    if e > valid_for_pred.shape[1]:
                        continue
                    if seg["category"] in thing_categories:
                        ovt_is_thing[b, s:e] = True
        elif not has_batch_thing:
            ovt_is_thing = valid_for_pred.clone()

        readout = getattr(args, "readout", "competition")
        if readout == "threshold":
            # v3-style: filter out stuff OVTs, then sigmoid+bg_threshold readout
            # on the surviving thing OVTs only. No register involvement.
            valid_thing_only = valid_for_pred & ovt_is_thing
            pred_mask = ovt_logits_to_pred_mask(
                ovt_logits=out["ovt_logits"],
                ovt_valid_mask=valid_thing_only,
                target_size=args.eval_size,
                n_ovt_per_object=args.n_ovt_per_object,
                patch_grid=args.grid_size,
                bg_threshold=args.bg_threshold,
            )  # (B, H, W)
        elif readout == "nullbg":
            if out.get("null_bg_logits") is None:
                raise ValueError("--readout nullbg requires a checkpoint with pgot_n_null_bg > 0.")
            pred_mask = build_pred_mask_null_bg_eval(
                ovt_logits=out["ovt_logits"],
                null_bg_logits=out["null_bg_logits"],
                ovt_valid_mask=valid_for_pred,
                ovt_is_thing=ovt_is_thing,
                target_size=args.eval_size,
                n_ovt_per_object=args.n_ovt_per_object,
                patch_grid=args.grid_size,
                merge=getattr(args, "eval_merge", "max"),
            )  # (B, H, W)
        elif readout in {"spatial", "spatial_trainmatch"}:
            if readout == "spatial_trainmatch":
                spatial_temp = float(
                    getattr(
                        model.config,
                        "pgot_mask_spatial_outside_log_temperature",
                        getattr(model.config, "pgot_mask_spatial_temperature", 1.0),
                    )
                )
                spatial_merge = "mean"
            else:
                spatial_temp = args.spatial_temperature
                spatial_merge = getattr(args, "eval_merge", "mean")
            pred_mask = build_pred_mask_spatial_readout(
                ovt_logits=out["ovt_logits"],
                ovt_valid_mask=valid_for_pred,
                target_size=args.eval_size,
                n_ovt_per_object=args.n_ovt_per_object,
                patch_grid=args.grid_size,
                merge=spatial_merge,
                temp=spatial_temp,
                ovt_is_thing=ovt_is_thing,
                map_stuff_to_bg=(args.gt_source == "coco_instance"),
            )
        elif readout == "llm_attention":
            if out.get("llm_attention_maps") is None:
                raise ValueError(
                    "--readout llm_attention requires a V8.4/V8.5 checkpoint "
                    "with internal LLM-attention supervision enabled."
                )
            pred_mask = build_pred_mask_llm_attention_eval(
                ovt_attention_maps=out["llm_attention_maps"],
                void_attention_maps=out["llm_attention_void_maps"],
                ovt_valid_mask=valid_for_pred,
                ovt_is_thing=ovt_is_thing,
                target_size=args.eval_size,
                n_ovt_per_object=args.n_ovt_per_object,
                patch_grid=args.grid_size,
                merge="mean",
                map_stuff_to_bg=(args.gt_source == "coco_instance"),
            )
        elif readout == "ovt_owner":
            if out.get("ovt_object_probs") is None:
                raise ValueError(
                    "--readout ovt_owner requires a V12/V14 checkpoint with OVT-owner outputs."
                )
            pred_mask = build_pred_mask_ovt_owner_eval(
                ovt_object_probs=out["ovt_object_probs"],
                ovt_void_probs=out["ovt_void_probs"],
                ovt_valid_mask=valid_for_pred,
                ovt_is_thing=ovt_is_thing,
                target_size=args.eval_size,
                n_ovt_per_object=args.n_ovt_per_object,
                patch_grid=args.grid_size,
                map_stuff_to_bg=(args.gt_source == "coco_instance"),
            )
        else:
            # v5/v6 competition readout (default).
            pred_mask = build_pred_mask_competition_eval(
                ovt_logits=out["ovt_logits"],
                reg_logits=out["reg_logits"],
                ovt_valid_mask=valid_for_pred,
                ovt_is_thing=ovt_is_thing,
                target_size=args.eval_size,
                n_ovt_per_object=args.n_ovt_per_object,
                patch_grid=args.grid_size,
                merge=getattr(args, "eval_merge", "max"),
            )  # (B, H, W)

        # GT mask
        B = pred_mask.shape[0]
        gt_masks = []
        overlap_masks = []
        for b in range(B):
            global_idx = batch_idx * args.batch_size + b
            samp = samples_iter[global_idx]
            if args.gt_source == "coco_instance":
                gt = coco_cache.get(int(samp["image_id"]))
                if gt is None:
                    # Image not in COCO val cache — fall back to panoptic-from-pix2cap
                    seg_ids = [int(s["segment_id"]) for s in samp["segments"]]
                    gt = load_gt_panoptic_mask(samp["panoptic_mask_path"], seg_ids, args.eval_size)
                gt_masks.append(gt)
                overlap = coco_cache.get_overlap(int(samp["image_id"]))
                if overlap is None:
                    overlap = torch.zeros_like(gt, dtype=torch.uint8)
                overlap_masks.append(overlap)
            else:
                seg_ids = [int(s["segment_id"]) for s in samp["segments"]]
                gt = load_gt_panoptic_mask(samp["panoptic_mask_path"], seg_ids, args.eval_size)
                gt_masks.append(gt)
        gt_mask = torch.stack(gt_masks).to(device=pred_mask.device)
        overlap_mask = (
            torch.stack(overlap_masks).to(device=pred_mask.device)
            if overlap_masks
            else None
        )

        # Per-sample metrics (loop to handle nan-safe averaging)
        for b in range(B):
            gt_b = gt_mask[b:b+1]  # (1, H, W)
            pr_b = pred_mask[b:b+1]
            overlap_b = overlap_mask[b:b+1] if overlap_mask is not None else None
            fa = fari_metric(gt_b, pr_b, overlap_b)
            mb = mbo_metric(gt_b, pr_b, overlap_b)
            mi = miou_metric(gt_b, pr_b, overlap_b)
            if not np.isnan(fa): fari_scores.append(fa)
            if not np.isnan(mb): mbo_scores.append(mb)
            if not np.isnan(mi): miou_scores.append(mi)
            # Per-count bucket
            n_obj = batch["n_objects_list"][b]
            bkt = "1-3" if n_obj <= 3 else ("4-6" if n_obj <= 6 else ("7-10" if n_obj <= 10 else "11+"))
            if not np.isnan(fa): by_count[bkt]["fari"].append(fa)
            if not np.isnan(mb): by_count[bkt]["mbo"].append(mb)
            if not np.isnan(mi): by_count[bkt]["miou"].append(mi)
            image_ids.append(batch["image_ids"][b])
            n_objects_list.append(int(n_obj))

        # Active count
        ovt_probs = torch.sigmoid(out["ovt_logits"].float())
        ovt_max = ovt_probs.amax(dim=-1)
        active = ((ovt_max > args.bg_threshold) & out["ovt_valid_mask"].to(ovt_max.device)).sum(dim=-1).float() / args.n_ovt_per_object
        active_counts.extend(active.cpu().tolist())

        # rFID
        if fid_acc is not None and rae_decoder is not None:
            try:
                generated_latent = generate_siglip_latent(model, out["rae_hidden"], guidance_level=float(args.guidance_scale))
                recon_images = decode_to_image(rae_decoder, generated_latent, device)
                # Real image: denormalize target_images (which was SigLIP-preprocessed)
                target_proc = vt_list[1].image_processor if len(vt_list) > 1 else vt_list[0].image_processor
                t_mean = torch.tensor(target_proc.image_mean).view(1, -1, 1, 1)
                t_std = torch.tensor(target_proc.image_std).view(1, -1, 1, 1)
                src = denormalize_images(batch["target_images"].to(device).float(), t_mean, t_std)
                # Match resolution to decoder output
                if src.shape[-2:] != recon_images.shape[-2:]:
                    src = F.interpolate(src, size=recon_images.shape[-2:], mode="bilinear", align_corners=False)
                fid_acc.add(src, recon_images)
                # Recon metrics
                rm = compute_recon_metrics(src, recon_images)
                recon_psnr_list.extend(rm["psnr"].cpu().tolist())
                recon_ssim_list.extend(rm["ssim"].cpu().tolist())
                recon_mse_list.extend(rm["mse"].cpu().tolist())
                recon_mae_list.extend(rm["mae"].cpu().tolist())
            except Exception as e:
                log.warning(f"rFID batch failed: {type(e).__name__}: {e}")

    # ---- Aggregate
    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    summary = {
        "ckpt": args.model_path,
        "readout": args.readout,
        "num_samples": len(image_ids),
        "fARI": _mean(fari_scores),
        "mBO": _mean(mbo_scores),
        "mIoU": _mean(miou_scores),
        "active_count_mean": _mean(active_counts),
        "by_object_count": {
            bkt: {
                "n": len(d["fari"]),
                "fARI": _mean(d["fari"]),
                "mBO": _mean(d["mbo"]),
                "mIoU": _mean(d["miou"]),
            } for bkt, d in by_count.items()
        },
    }
    summary["gt_source"] = args.gt_source
    summary["image_preprocess_mode"] = args.image_preprocess_mode
    summary["coda_crop_size"] = int(args.coda_crop_size)
    summary["coda_overlap_excluded"] = bool(
        coco_cache is not None and coco_cache.overlap_masks is not None
    )
    if args.readout == "spatial_trainmatch":
        summary["softmax_axis"] = "patch"
        summary["spatial_temperature"] = float(
            getattr(
                model.config,
                "pgot_mask_spatial_outside_log_temperature",
                getattr(model.config, "pgot_mask_spatial_temperature", 1.0),
            )
        )
        summary["eval_merge"] = "mean"
    if args.readout == "llm_attention":
        patch_attention = (
            float(
                getattr(
                    model.config, "pgot_mask_llm_patch_outside_weight", 0.0
                )
            ) > 0.0
            or float(
                getattr(model.config, "pgot_mask_llm_image_use_weight", 0.0)
            ) > 0.0
        )
        if patch_attention:
            summary["attention_source"] = (
                "exact_llm_post_rope_image_patch_softmax"
            )
            summary["attention_layers"] = str(
                getattr(
                    model.config,
                    "pgot_mask_llm_patch_outside_layers",
                    "last4",
                )
            )
            summary["attention_temperature"] = float(
                getattr(
                    model.config,
                    "pgot_mask_llm_patch_outside_temperature",
                    1.0,
                )
            )
            summary["image_use_weight"] = float(
                getattr(model.config, "pgot_mask_llm_image_use_weight", 0.0)
            )
            summary["image_use_margin"] = float(
                getattr(model.config, "pgot_mask_llm_image_use_margin", 0.05)
            )
        else:
            summary["attention_source"] = "exact_llm_post_rope_full_key_softmax"
            summary["attention_layers"] = str(
                getattr(
                    model.config,
                    "pgot_mask_llm_attention_outside_layers",
                    "last4",
                )
            )
        summary["eval_merge"] = "mean"
        summary["void_tokens"] = int(getattr(model.config, "pgot_n_null_bg", 0))
    if args.readout == "ovt_owner":
        if bool(getattr(model.config, "pgot_v14_enable", False)):
            summary["owner_source"] = "v14_ovt_bottleneck_route"
            summary["route_temperature"] = float(getattr(model.config, "pgot_v14_route_temperature", 1.0))
            summary["route_weight"] = float(getattr(model.config, "pgot_v14_route_weight", 1.0))
            summary["route_void_weight"] = float(getattr(model.config, "pgot_v14_void_weight", 0.5))
            summary["route_position_weight"] = float(getattr(model.config, "pgot_v14_position_weight", 1.0))
        else:
            summary["owner_source"] = "v12_ovt_owner_softmax"
            summary["ovt_layers"] = str(getattr(model.config, "pgot_v12_layers", ""))
            summary["ovt_temperature"] = float(
                getattr(
                    model.config,
                    "pgot_v12_ovt_temperature",
                    getattr(model.config, "pgot_v12_slot_temperature", 1.0),
                )
            )
            summary["owner_temperature"] = float(
                getattr(model.config, "pgot_v12_owner_temperature", 1.0)
            )
            summary["owner_weight"] = float(
                getattr(model.config, "pgot_v12_owner_weight", 1.0)
            )
        summary["void_tokens"] = int(getattr(model.config, "pgot_n_null_bg", 0))
    if fid_acc is not None:
        try:
            summary["rFID"] = fid_acc.compute()
            summary["recon_psnr"] = _mean(recon_psnr_list)
            summary["recon_ssim"] = _mean(recon_ssim_list)
            summary["recon_mse"] = _mean(recon_mse_list)
            summary["recon_mae"] = _mean(recon_mae_list)
        except Exception as e:
            summary["rFID_error"] = str(e)
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info("EVAL SUMMARY")
    log.info("=" * 60)
    log.info(f"  Samples:        {summary['num_samples']}")
    log.info(f"  fARI:           {summary['fARI']:.4f}")
    log.info(f"  mBO:            {summary['mBO']:.4f}")
    log.info(f"  mIoU:           {summary['mIoU']:.4f}")
    log.info(f"  Active count:   {summary['active_count_mean']:.2f}")
    log.info("  --- By object count ---")
    for bkt in ["1-3", "4-6", "7-10", "11+"]:
        if bkt in summary["by_object_count"]:
            d = summary["by_object_count"][bkt]
            log.info(f"    {bkt:>4}: n={d['n']:>4} fARI={d['fARI']:.4f} mBO={d['mBO']:.4f} mIoU={d['mIoU']:.4f}")
    if "rFID" in summary:
        log.info(f"  rFID:           {summary['rFID']:.4f}")
        log.info(f"  PSNR/SSIM:      {summary['recon_psnr']:.3f} / {summary['recon_ssim']:.4f}")
        log.info(f"  MSE/MAE:        {summary['recon_mse']:.4f} / {summary['recon_mae']:.4f}")
    log.info(f"  GT source:      {summary['gt_source']}")
    log.info(f"  Written to: {summary_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
