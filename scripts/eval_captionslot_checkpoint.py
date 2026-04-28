#!/usr/bin/env python
import argparse
import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# B200 has no pure fp32 tensor cores — TF32 path is required to get real throughput
# out of Qwen LLM / SigLIP / decoder fp32 GEMMs. Numerical impact on rFID/PSNR/SSIM is negligible.
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# einops.rearrange breaks torch.compile Dynamo tracing under dynamic shapes
# ("unhashable type: non-nested SymInt"). This helper registers the einops ops
# as compile-compatible so rotate_half / attention reshapes survive tracing.
try:
    from einops._torch_specific import allow_ops_in_compiled_graph as _einops_allow_compile
    _einops_allow_compile()
except Exception:
    pass


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
INFERENCE_ROOT = REPO_ROOT / "inference"
SCALE_RAE_ROOT = Path("/home/jovyan/Scale-RAE")
SCALE_RAE_INFERENCE_ROOT = SCALE_RAE_ROOT / "inference"
for path in (
    str(REPO_ROOT),
    str(INFERENCE_ROOT),
    str(SCALE_RAE_ROOT),
    str(SCALE_RAE_INFERENCE_ROOT),
):
    if path not in sys.path:
        sys.path.append(path)

if "IPython" not in sys.modules:
    ipython_stub = types.ModuleType("IPython")
    ipython_stub.get_ipython = lambda: None
    ipython_stub.version_info = (0, 0, 0, "")
    sys.modules["IPython"] = ipython_stub

from cli import ensure_output_dir  # type: ignore
from eval_caption_to_image_rfid import (  # type: ignore
    InceptionFeatureExtractor,
    build_abs_diff_image,
    compute_basic_metrics,
    compute_fid,
    gaussian_window,
    save_json,
    save_triptych,
    tensor_to_pil,
)
from scale_rae.train.captionslot_trainer import (  # type: ignore
    _register_captionslot_template_token_ids,
    _register_im_start_end_token_ids,
)
from scale_rae.utils import disable_torch_init  # type: ignore
from utils.load_model import load_scale_rae_model  # type: ignore


DEFAULT_MODEL_PATH = "/home/jovyan/AURORA/checkpoints/captionslot_firstslot_noprior_recon_stage1_fp32/checkpoint-18000"
DEFAULT_IMAGE_DIR = "/home/jovyan/data/coco/val2017"
DEFAULT_CAPTIONS_JSONL = "/home/jovyan/Scale-RAE/outputs/caption_to_image_recon_concise/captions.jsonl"
DEFAULT_OUTPUT_DIR = "/home/jovyan/AURORA/outputs/captionslot_eval_checkpoint"


def str2bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an AURORA CaptionSlot checkpoint on caption-conditioned reconstruction."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--captions-jsonl", default=DEFAULT_CAPTIONS_JSONL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--max-caption-tokens", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--guidance-level", type=float, default=1.0)
    parser.add_argument("--save-images", type=str2bool, default=True)
    parser.add_argument("--save-limit", type=int, default=50)
    parser.add_argument("--save-fixed-first-n", type=int, default=50)
    parser.add_argument("--report-losses", type=str2bool, default=True)
    # ── slot attention eval ──────────────────────────────────────────────────
    parser.add_argument(
        "--coco-instances",
        default="/home/jovyan/data/coco/annotations/instances_val2017.json",
        help="Path to COCO instances_val2017.json for GT mask-based slot eval.",
    )
    parser.add_argument("--eval-slot-attention", type=str2bool, default=True,
                        help="Run slot attention evaluation (MBO/IoU + visualizations).")
    parser.add_argument("--attn-threshold", type=float, default=0.5,
                        help="Sigmoid threshold for the primary binary slot attention mask.")
    parser.add_argument(
        "--attn-thresholds",
        default="",
        help="Comma-separated list of extra attn thresholds for slot_attention_metrics. "
             "If set, emits slot_attention_metrics_thr{T} per threshold in addition to the primary.",
    )
    parser.add_argument(
        "--segmentation-bg-threshold",
        type=float,
        default=0.5,
        help="Primary background score used for bg-aware dense segmentation eval.",
    )
    parser.add_argument(
        "--segmentation-bg-thresholds",
        default="",
        help="Comma-separated list of extra bg thresholds. "
             "If set, emits segmentation_metrics_bg_thr{T} per threshold in addition to the primary.",
    )
    parser.add_argument(
        "--attn-temperature",
        type=float,
        default=1.0,
        help="Temperature for sharpening attention maps before argmax. "
             "<1.0 sharpens (e.g. 0.3), >1.0 softens. Applied after MEAN merge.",
    )
    parser.add_argument("--save-attn-maps", type=str2bool, default=True,
                        help="Save attention map overlay images.")
    parser.add_argument("--save-attn-limit", type=int, default=100,
                        help="Max number of attn overlay images to save.")
    parser.add_argument(
        "--raw-slot-argmax",
        type=str2bool,
        default=False,
        help="Skip per-object MEAN merge; expose all raw slot maps for argmax (up to 192 channels). "
             "Closer to competitor unsupervised-slot protocol.",
    )
    parser.add_argument(
        "--gt-min-area-pct",
        type=float,
        default=0.0,
        help="Filter GT instances smaller than this fraction of image area (0=off). "
             "E.g. 0.005 removes instances covering <0.5%% of image. "
             "Recommended: 0.005 to 0.02 to match our 16x16 patch resolution.",
    )
    parser.add_argument("--torch-compile", type=str2bool, default=False,
                        help="Apply torch.compile to DiT (diff_head.model). "
                             "First batch is slow (compile), then ~20-40%% faster.")
    parser.add_argument("--diffusion-steps", type=int, default=10,
                        help="Diffusion inference steps (default 10; model default is 50). "
                             "RF models work well at 10-20 steps.")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def normalize_caption(text: str) -> str:
    return " ".join(str(text).strip().split())


def _parse_threshold_list(primary: float, extra: str) -> List[float]:
    """Return sorted unique list including the primary threshold plus extras parsed from a comma string."""
    values = {float(primary)}
    if extra:
        for tok in str(extra).split(","):
            tok = tok.strip()
            if tok:
                values.add(float(tok))
    return sorted(values)


def _thr_key(thr: float) -> str:
    """Stable short key for a threshold value: 0.3 -> '0.30'."""
    return f"{thr:.2f}"


def resolve_image_path(record: Dict[str, Any], image_dir: str) -> str:
    image = record.get("image") or record.get("file_name")
    if image is None:
        raise KeyError(f"Missing image/file_name in record: {record}")
    return image if os.path.isabs(image) else os.path.join(image_dir, image)


def load_caption_records(captions_jsonl: str, image_dir: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(captions_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            caption = normalize_caption(item.get("caption", ""))
            if not caption:
                continue
            image_path = resolve_image_path(item, image_dir)
            records.append(
                {
                    "image_id": Path(image_path).stem,
                    "image": image_path,
                    "file_name": Path(image_path).name,
                    "caption": caption,
                    "token_ids": item.get("token_ids"),
                    "noun_chunks": item.get("noun_chunks"),
                }
            )
            if max_samples is not None and len(records) >= max_samples:
                break
    records.sort(key=lambda x: x["file_name"])
    return records


def apply_round_robin_shard(samples: Sequence[Dict[str, Any]], num_shards: int, shard_index: int) -> List[Dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    return list(samples[shard_index::num_shards])


def select_allowed_save_stems(samples: Sequence[Dict[str, Any]], save_fixed_first_n: Optional[int]) -> Optional[set]:
    if save_fixed_first_n is None:
        return None
    if save_fixed_first_n < 0:
        raise ValueError("--save-fixed-first-n must be >= 0")
    return {Path(sample["image"]).stem for sample in samples[:save_fixed_first_n]}


def get_image_stats(image_processor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(getattr(image_processor, "image_mean", [0.5, 0.5, 0.5]), dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(getattr(image_processor, "image_std", [0.5, 0.5, 0.5]), dtype=torch.float32).view(3, 1, 1)
    return mean, std


def denormalize_images(
    images: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    to_cpu: bool = False,
) -> torch.Tensor:
    mean = mean.to(device=images.device, dtype=torch.float32)
    std = std.to(device=images.device, dtype=torch.float32)
    denorm = (images.float() * std + mean).clamp(0.0, 1.0)
    return denorm.cpu() if to_cpu else denorm


def load_eval_decoder(model):
    from huggingface_hub import hf_hub_download
    from scale_rae.model.multimodal_decoder import MultimodalDecoder  # type: ignore

    repo_id = "nyu-visionx/siglip2_decoder"
    vision_towers = list(getattr(model.config, "mm_vision_tower_aux_list", ["google/siglip2-so400m-patch14-224"]))
    encoder_path = vision_towers[1] if len(vision_towers) > 1 else vision_towers[0]
    encoder_path = encoder_path.split("-interp")[0]
    num_patches = int(
        getattr(
            model.config,
            "diffusion_target_token_len",
            getattr(model, "diffusion_target_token_len", 256),
        )
    )

    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="model.pt")
    return MultimodalDecoder(
        pretrained_encoder_path=encoder_path,
        general_decoder_config=config_path,
        num_patches=num_patches,
        drop_cls_token=True,
        decoder_path=ckpt_path,
    )


def patch_diffusion_steps(model, n_steps: int) -> None:
    """Replace diff_head.inference_flow with a faster n_steps version at runtime.

    RF (Rectified Flow) models work well at 10-20 steps vs the hardcoded 50.
    We rebuild inference_flow using the same size_ratio stored on the existing object.
    """
    import math
    try:
        from scale_rae.model.diffusion_loss.diffusion import create_diffusion  # type: ignore
        from scale_rae.model.diffusion_loss.diffusion.gaussian_diffusion import (  # type: ignore
            ScheduleType, ModelMeanType, LossType,
        )
    except ImportError:
        print(f"[WARN] Could not import diffusion helpers; keeping default inference_step.")
        return

    diff_head = getattr(model, "diff_head", None)
    if diff_head is None:
        return
    inf_flow = getattr(diff_head, "inference_flow", None)
    if inf_flow is None:
        return

    current_steps = len(getattr(inf_flow, "used_timesteps", [None] * 50))
    if current_steps == n_steps:
        print(f"[INFO] diffusion inference_flow already has {n_steps} steps, skipping patch.")
        return

    size_ratio = float(getattr(inf_flow, "size_ratio", 1.0))
    diffusion_steps = int(getattr(inf_flow, "diffusion_steps", 1000))

    new_flow = create_diffusion(
        str(n_steps),
        noise_schedule="linear",
        use_kl=False,
        sigma_small=False,
        predict_xstart=False,
        learn_sigma=False,
        rescale_learned_sigmas=False,
        diffusion_steps=diffusion_steps,
        input_base_dimension_ratio=size_ratio,
        diffusion_type="rf",
        use_loss_weighting=False,
        use_schedule_shift=True,
    )
    diff_head.inference_flow = new_flow
    print(f"[INFO] Patched diffusion inference_flow: {current_steps} steps → {n_steps} steps")


def decode_generated_images(decoder, generated: torch.Tensor, device: torch.device) -> torch.Tensor:
    if generated is None:
        return torch.empty(0, 3, 1, 1, device=device)
    decoder = decoder.to(device=device)
    decoder_dtype = next(decoder.parameters()).dtype
    if generated.dtype != decoder_dtype:
        generated = generated.to(dtype=decoder_dtype)
    if hasattr(decoder, "image_mean") and hasattr(decoder, "image_std"):
        decoder.image_mean = decoder.image_mean.to(device=device, dtype=decoder_dtype)
        decoder.image_std = decoder.image_std.to(device=device, dtype=decoder_dtype)
    empty_cls = torch.zeros((generated.shape[0], 1, generated.shape[-1]), device=device, dtype=decoder_dtype)
    image_features = torch.cat([empty_cls, generated], dim=1)
    recon = decoder(image_features)
    recon = torch.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=0.0)
    return recon.clamp(0.0, 1.0).detach().float()


class CaptionSlotEvalDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Dict[str, Any]],
        tokenizer,
        image_processor,
        target_image_processor,
        max_slots: int,
        slots_per_object: int,
        image_feature_token_len: int,
        max_caption_tokens: int,
    ) -> None:
        self.image_processor = image_processor
        self.target_image_processor = target_image_processor
        self.max_slots = max_slots
        self.slots_per_object = max(int(slots_per_object), 1)
        self.max_unique_objects = max(self.max_slots // self.slots_per_object, 1)
        self.image_feature_token_len = int(image_feature_token_len)
        self.max_caption_tokens = max_caption_tokens
        self.entries: List[Dict[str, Any]] = []

        missing_precomputed = 0
        for sample in samples:
            token_ids = sample.get("token_ids")
            noun_chunks = sample.get("noun_chunks")
            if token_ids is None or noun_chunks is None:
                missing_precomputed += 1
                continue
            token_ids = list(token_ids)[:max_caption_tokens]
            noun_chunks = list(noun_chunks)[: self.max_unique_objects]
            if not token_ids or not noun_chunks:
                continue
            self.entries.append(
                {
                    "image_id": sample["image_id"],
                    "image": sample["image"],
                    "file_name": sample["file_name"],
                    "caption": sample["caption"],
                    "token_ids": token_ids,
                    "noun_chunks": noun_chunks,
                }
            )

        if missing_precomputed:
            raise ValueError(
                f"{missing_precomputed} records lack precomputed token_ids/noun_chunks. "
                "Regenerate captions.jsonl with postprocess_stagea_object_captions.py "
                "(or an equivalent pipeline) so these fields are present."
            )
        if not self.entries:
            raise ValueError("No valid eval entries after filtering empty noun chunks.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        image = Image.open(entry["image"]).convert("RGB")
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        target_image_tensor = self.target_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        token_ids = torch.tensor(entry["token_ids"], dtype=torch.long)
        noun_chunk_spans = torch.full((self.max_slots, 2), -1, dtype=torch.long)
        n_unique = min(len(entry["noun_chunks"]), self.max_unique_objects)
        n_slots = min(n_unique * self.slots_per_object, self.max_slots)
        slot_idx = 0
        for chunk in entry["noun_chunks"][:n_unique]:
            for _ in range(self.slots_per_object):
                if slot_idx >= self.max_slots:
                    break
                noun_chunk_spans[slot_idx, 0] = int(chunk["token_start"])
                noun_chunk_spans[slot_idx, 1] = int(chunk["token_end"])
                slot_idx += 1

        head_prior_maps = torch.zeros((self.max_slots, self.image_feature_token_len), dtype=torch.float32)
        head_prior_valid_mask = torch.zeros(self.max_slots, dtype=torch.bool)

        return {
            "image": image_tensor,
            "target_image": target_image_tensor,
            "caption_input_ids": token_ids,
            "noun_chunk_spans": noun_chunk_spans,
            "n_slots": torch.tensor(n_slots, dtype=torch.long),
            "head_prior_maps": head_prior_maps,
            "head_prior_valid_mask": head_prior_valid_mask,
            "image_id": entry["image_id"],
            "image_path": entry["image"],
            "caption": entry["caption"],
            "file_name": entry["file_name"],
        }


class CaptionSlotEvalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        max_caption_len = max(inst["caption_input_ids"].shape[0] for inst in instances)
        batch_size = len(instances)

        caption_input_ids = torch.full((batch_size, max_caption_len), self.pad_token_id, dtype=torch.long)
        caption_attention_mask = torch.zeros((batch_size, max_caption_len), dtype=torch.bool)
        for idx, inst in enumerate(instances):
            length = inst["caption_input_ids"].shape[0]
            caption_input_ids[idx, :length] = inst["caption_input_ids"]
            caption_attention_mask[idx, :length] = True

        return {
            "images": torch.stack([inst["image"] for inst in instances]),
            "target_images": torch.stack([inst["target_image"] for inst in instances]),
            "caption_input_ids": caption_input_ids,
            "caption_attention_mask": caption_attention_mask,
            "noun_chunk_spans": torch.stack([inst["noun_chunk_spans"] for inst in instances]),
            "n_slots": torch.stack([inst["n_slots"] for inst in instances]),
            "head_prior_maps": torch.stack([inst["head_prior_maps"] for inst in instances]),
            "head_prior_valid_mask": torch.stack([inst["head_prior_valid_mask"] for inst in instances]),
            "image_ids": [inst["image_id"] for inst in instances],
            "image_path": [inst["image_path"] for inst in instances],
            "caption": [inst["caption"] for inst in instances],
            "file_name": [inst["file_name"] for inst in instances],
        }


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _preproc_masks_overlap(
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    inst_overlap_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inst_overlap_mask is None:
        return gt_mask, pred_mask
    gt_mask = gt_mask.clone()
    pred_mask = pred_mask.clone()
    gt_mask[inst_overlap_mask == 1] = 0
    pred_mask[inst_overlap_mask == 1] = pred_mask.max() + 1
    return gt_mask, pred_mask


def _adjusted_rand_index(
    true_ids: torch.Tensor,
    pred_ids: torch.Tensor,
    ignore_background: bool = False,
) -> torch.Tensor:
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


def _fari_metric(
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    inst_overlap_mask: Optional[torch.Tensor] = None,
) -> float:
    assert "int" in str(gt_mask.dtype)
    assert "int" in str(pred_mask.dtype)
    if inst_overlap_mask is not None:
        gt_mask = gt_mask.clone()
        pred_mask = pred_mask.clone()
        for idx in range(gt_mask.shape[0]):
            gt_mask[idx], pred_mask[idx] = _preproc_masks_overlap(
                gt_mask[idx], pred_mask[idx], inst_overlap_mask[idx]
            )
    return _adjusted_rand_index(gt_mask, pred_mask, ignore_background=True).mean().item()


def _max_assignment_sum(iou: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = int(iou.shape[0]), int(iou.shape[1])
    if n_rows == 0 or n_cols == 0:
        return iou.new_tensor(0.0)
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
        return np.nan

    true_oh = F.one_hot(gt_mask).float()
    if ignore_background:
        true_oh = true_oh[..., 1:]
    pred_oh = F.one_hot(pred_mask).float()
    n_true, n_pred = true_oh.shape[-1], pred_oh.shape[-1]
    if n_true == 0:
        return np.nan
    if n_pred == 0:
        return 0.0

    intersect = (true_oh[:, :, None] * pred_oh[:, None, :]).sum(0)
    union = true_oh.sum(0)[:, None] + pred_oh.sum(0)[None] - intersect
    iou = intersect / (union + 1e-8)
    best_sum = _max_assignment_sum(iou)
    return float((best_sum / float(n_true)).item())


def _mean_best_overlap(gt_mask: torch.Tensor, pred_mask: torch.Tensor) -> float:
    if gt_mask.max().item() == 0:
        return np.nan
    true_oh = F.one_hot(gt_mask).float()[..., 1:]
    pred_oh = F.one_hot(pred_mask).float()
    intersect = (true_oh[:, :, None] * pred_oh[:, None, :]).sum(0)
    union = true_oh.sum(0)[:, None] + pred_oh.sum(0)[None] - intersect
    iou = intersect / (union + 1e-8)
    if iou.numel() == 0:
        return np.nan
    return float(iou.max(dim=1).values.mean().item())


def _miou_metric(
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    inst_overlap_mask: Optional[torch.Tensor] = None,
) -> float:
    assert "int" in str(gt_mask.dtype)
    assert "int" in str(pred_mask.dtype)
    if inst_overlap_mask is None:
        overlap_masks = [None] * gt_mask.shape[0]
    else:
        overlap_masks = inst_overlap_mask.flatten(1, 2)
    gt_mask_flat = gt_mask.flatten(1, 2)
    pred_mask_flat = pred_mask.flatten(1, 2)
    scores: List[float] = []
    for idx in range(gt_mask_flat.shape[0]):
        gt_mask_i, pred_mask_i = _preproc_masks_overlap(gt_mask_flat[idx], pred_mask_flat[idx], overlap_masks[idx])
        scores.append(_hungarian_miou(gt_mask_i, pred_mask_i, ignore_background=False))
    valid_scores = [score for score in scores if not np.isnan(score)]
    return float(np.mean(valid_scores)) if valid_scores else float("nan")


def _mbo_metric(
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    inst_overlap_mask: Optional[torch.Tensor] = None,
) -> float:
    assert "int" in str(gt_mask.dtype)
    assert "int" in str(pred_mask.dtype)
    if inst_overlap_mask is None:
        overlap_masks = [None] * gt_mask.shape[0]
    else:
        overlap_masks = inst_overlap_mask.flatten(1, 2)
    gt_mask_flat = gt_mask.flatten(1, 2)
    pred_mask_flat = pred_mask.flatten(1, 2)
    scores: List[float] = []
    for idx in range(gt_mask_flat.shape[0]):
        gt_mask_i, pred_mask_i = _preproc_masks_overlap(gt_mask_flat[idx], pred_mask_flat[idx], overlap_masks[idx])
        scores.append(_mean_best_overlap(gt_mask_i, pred_mask_i))
    valid_scores = [score for score in scores if not np.isnan(score)]
    return float(np.mean(valid_scores)) if valid_scores else float("nan")


def _compute_recon_metrics_batched(
    target: torch.Tensor,
    pred: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Dict[str, torch.Tensor]:
    """Vectorised PSNR/SSIM/MSE/MAE across the batch; returns per-image GPU tensors of shape [B]."""
    target = target.float()
    pred = pred.float()
    diff = target - pred
    mse = (diff ** 2).mean(dim=(1, 2, 3))
    mae = diff.abs().mean(dim=(1, 2, 3))
    psnr = -10.0 * torch.log10(mse.clamp(min=1e-20))

    channels = target.shape[1]
    window = gaussian_window(window_size, sigma, channels, target.device, target.dtype)
    pad = window_size // 2
    mu1 = F.conv2d(target, window, padding=pad, groups=channels)
    mu2 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(target * pred, window, padding=pad, groups=channels) - mu1_mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)).clamp(min=1e-12)
    )
    ssim = ssim_map.mean(dim=(1, 2, 3))
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}


def _fari_metric_batched(
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    inst_overlap_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-sample fARI (no .mean() / no .item()); returns Tensor of shape [B]."""
    if inst_overlap_mask is not None:
        gt_mask = gt_mask.clone()
        pred_mask = pred_mask.clone()
        for idx in range(gt_mask.shape[0]):
            gt_mask[idx], pred_mask[idx] = _preproc_masks_overlap(
                gt_mask[idx], pred_mask[idx], inst_overlap_mask[idx]
            )
    return _adjusted_rand_index(gt_mask, pred_mask, ignore_background=True)


def _mbo_metric_batched(
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    inst_overlap_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-sample MBO. Keeps per-sample work on GPU; a single .tolist() defers sync to the caller."""
    B = gt_mask.shape[0]
    if inst_overlap_mask is None:
        overlap_iter: List[Optional[torch.Tensor]] = [None] * B
    else:
        overlap_iter = [inst_overlap_mask[idx].flatten() for idx in range(B)]
    gt_flat = gt_mask.flatten(1, 2)
    pred_flat = pred_mask.flatten(1, 2)
    scores: List[torch.Tensor] = []
    for idx in range(B):
        gt_i, pred_i = _preproc_masks_overlap(gt_flat[idx], pred_flat[idx], overlap_iter[idx])
        if int(gt_i.max().item()) == 0:
            scores.append(gt_i.new_tensor(float("nan"), dtype=torch.float32))
            continue
        true_oh = F.one_hot(gt_i).float()[..., 1:]
        pred_oh = F.one_hot(pred_i).float()
        intersect = (true_oh[:, :, None] * pred_oh[:, None, :]).sum(0)
        union = true_oh.sum(0)[:, None] + pred_oh.sum(0)[None] - intersect
        iou = intersect / (union + 1e-8)
        if iou.numel() == 0:
            scores.append(gt_i.new_tensor(float("nan"), dtype=torch.float32))
        else:
            scores.append(iou.max(dim=1).values.mean().float())
    return torch.stack(scores)


def _miou_metric_batched(
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    inst_overlap_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-sample mIoU via Hungarian assignment. Returns Tensor of shape [B] (GPU)."""
    B = gt_mask.shape[0]
    if inst_overlap_mask is None:
        overlap_iter: List[Optional[torch.Tensor]] = [None] * B
    else:
        overlap_iter = [inst_overlap_mask[idx].flatten() for idx in range(B)]
    gt_flat = gt_mask.flatten(1, 2)
    pred_flat = pred_mask.flatten(1, 2)
    scores: List[torch.Tensor] = []
    for idx in range(B):
        gt_i, pred_i = _preproc_masks_overlap(gt_flat[idx], pred_flat[idx], overlap_iter[idx])
        if int(gt_i.max().item()) == 0:
            scores.append(gt_i.new_tensor(float("nan"), dtype=torch.float32))
            continue
        true_oh = F.one_hot(gt_i).float()
        pred_oh = F.one_hot(pred_i).float()
        n_true, n_pred = true_oh.shape[-1], pred_oh.shape[-1]
        if n_true == 0:
            scores.append(gt_i.new_tensor(float("nan"), dtype=torch.float32))
            continue
        if n_pred == 0:
            scores.append(gt_i.new_tensor(0.0, dtype=torch.float32))
            continue
        intersect = (true_oh[:, :, None] * pred_oh[:, None, :]).sum(0)
        union = true_oh.sum(0)[:, None] + pred_oh.sum(0)[None] - intersect
        iou = intersect / (union + 1e-8)
        best_sum = _max_assignment_sum(iou)
        scores.append((best_sum / float(n_true)).float())
    return torch.stack(scores)


def _merge_object_maps(
    attn_maps: torch.Tensor,
    n_active_slots: int,
    slots_per_object: int,
) -> torch.Tensor:
    n_active = max(0, min(int(n_active_slots), int(attn_maps.shape[0])))
    if n_active == 0:
        return attn_maps.new_zeros((0, attn_maps.shape[-1]))
    active_maps = attn_maps[:n_active].float()
    if slots_per_object <= 1:
        return active_maps
    merged: List[torch.Tensor] = []
    for start_idx in range(0, n_active, slots_per_object):
        chunk = active_maps[start_idx : min(start_idx + slots_per_object, n_active)]
        merged.append(chunk.mean(dim=0))
    return torch.stack(merged, dim=0) if merged else attn_maps.new_zeros((0, attn_maps.shape[-1]))


def _upsample_slot_maps(
    slot_maps: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    if slot_maps.numel() == 0:
        return slot_maps.new_zeros((0, height, width))
    side = int(round(slot_maps.shape[-1] ** 0.5))
    slot_maps_2d = slot_maps.reshape(slot_maps.shape[0], 1, side, side)
    upsampled = F.interpolate(slot_maps_2d, size=(height, width), mode="bilinear", align_corners=False)
    return upsampled[:, 0].clamp(0.0, 1.0)


def _build_dense_pred_masks(
    object_maps: torch.Tensor,
    height: int,
    width: int,
    bg_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    object_maps_hw = _upsample_slot_maps(object_maps, height, width)
    if object_maps_hw.shape[0] == 0:
        empty_mask = torch.zeros((height, width), dtype=torch.long)
        return empty_mask, empty_mask.clone(), object_maps_hw

    strict_mask = object_maps_hw.argmax(dim=0).to(dtype=torch.long)
    bg_channel = torch.full(
        (1, height, width),
        fill_value=float(bg_threshold),
        dtype=object_maps_hw.dtype,
        device=object_maps_hw.device,
    )
    bg_mask = torch.cat([bg_channel, object_maps_hw], dim=0).argmax(dim=0).to(dtype=torch.long)
    return strict_mask, bg_mask, object_maps_hw


def _build_dense_pred_masks_batched(
    merged_maps_per_image: List[torch.Tensor],
    height: int,
    width: int,
    bg_thresholds: Sequence[float],
    device: torch.device,
) -> tuple[torch.Tensor, Dict[float, torch.Tensor]]:
    """Build strict/bg dense pred masks for the whole batch. Upsample is shared across all thresholds.

    merged_maps_per_image[i] has shape [Ni, P] (variable Ni, P=patch tokens).
    Returns (strict_mask [B,H,W], {thr: bg_mask [B,H,W]}).
    """
    thr_list = [float(t) for t in bg_thresholds]
    B = len(merged_maps_per_image)
    if B == 0:
        empty = torch.zeros((0, height, width), dtype=torch.long, device=device)
        return empty, {t: empty.clone() for t in thr_list}

    max_n = max((int(m.shape[0]) for m in merged_maps_per_image), default=0)
    P = int(merged_maps_per_image[0].shape[-1]) if merged_maps_per_image[0].ndim >= 1 else 0
    # Fallback P: any image with slots
    if P == 0:
        for m in merged_maps_per_image:
            if m.ndim >= 1 and m.shape[-1] > 0:
                P = int(m.shape[-1])
                break

    if max_n == 0 or P == 0:
        empty = torch.zeros((B, height, width), dtype=torch.long, device=device)
        return empty, {t: empty.clone() for t in thr_list}

    side = int(round(P ** 0.5))
    padded = torch.zeros((B, max_n, P), dtype=torch.float32, device=device)
    valid = torch.zeros((B, max_n), dtype=torch.bool, device=device)
    for i, m in enumerate(merged_maps_per_image):
        n = int(m.shape[0])
        if n == 0:
            continue
        padded[i, :n] = m.float()
        valid[i, :n] = True

    # Batched upsample: [B*max_n, 1, side, side] -> [B*max_n, H, W]
    x = padded.view(B * max_n, 1, side, side)
    up = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    up = up[:, 0].clamp(0.0, 1.0).view(B, max_n, height, width)

    # Mask out invalid slot channels so they can't win argmax.
    neg_inf = torch.finfo(up.dtype).min
    invalid = ~valid
    up_masked = up.masked_fill(invalid.view(B, max_n, 1, 1), neg_inf)

    has_any = valid.any(dim=1)  # [B]
    strict_mask = up_masked.argmax(dim=1).to(dtype=torch.long)
    if (~has_any).any():
        strict_mask[~has_any] = 0

    # bg variant per threshold: prepend constant bg channel, then argmax. Upsample is reused.
    bg_masks: Dict[float, torch.Tensor] = {}
    for thr in thr_list:
        bg_channel = torch.full(
            (B, 1, height, width),
            fill_value=float(thr),
            dtype=up.dtype,
            device=device,
        )
        bg_stack = torch.cat([bg_channel, up_masked], dim=1)
        bg_mask = bg_stack.argmax(dim=1).to(dtype=torch.long)
        if (~has_any).any():
            bg_mask[~has_any] = 0
        bg_masks[thr] = bg_mask
    return strict_mask, bg_masks


def _build_gt_cache_for_samples(
    samples: Sequence[Dict[str, Any]],
    coco,
    img_id_to_anns: Dict[int, List[Dict[str, Any]]],
    height: int,
    width: int,
    device: torch.device,
    patch_side: int,
    gt_min_area_pct: float = 0.0,
) -> Dict[int, Dict[str, torch.Tensor]]:
    gt_cache: Dict[int, Dict[str, torch.Tensor]] = {}
    if coco is None:
        return gt_cache

    ph = max(1, height // patch_side)
    pw = max(1, width // patch_side)
    unique_img_ids = []
    seen_ids = set()
    for sample in samples:
        image_id = sample.get("image_id")
        if image_id is None or not str(image_id).isdigit():
            continue
        img_int_id = int(image_id)
        if img_int_id in seen_ids:
            continue
        seen_ids.add(img_int_id)
        unique_img_ids.append(img_int_id)

    img_info_map: Dict[int, Dict] = {}
    if gt_min_area_pct > 0.0:
        for info in coco.loadImgs(unique_img_ids):
            img_info_map[info["id"]] = info

    for img_int_id in tqdm(unique_img_ids, desc="Preload COCO GT", unit="img", dynamic_ncols=True):
        anns = [a for a in img_id_to_anns.get(img_int_id, []) if not a.get("iscrowd", 0)]
        if gt_min_area_pct > 0.0 and img_int_id in img_info_map:
            info = img_info_map[img_int_id]
            img_area = info["width"] * info["height"]
            min_area = gt_min_area_pct * img_area
            anns = [a for a in anns if a.get("area", 0) >= min_area]
        instance_mask_np = np.zeros((height, width), dtype=np.int64)
        union_mask_np = np.zeros((height, width), dtype=bool)
        overlap_mask_np = np.zeros((height, width), dtype=bool)
        for inst_idx, ann in enumerate(anns, start=1):
            gt_mask = _decode_coco_mask(coco, ann, height, width).astype(bool)
            overlap_mask_np |= (union_mask_np & gt_mask)
            union_mask_np |= gt_mask
            instance_mask_np[gt_mask] = inst_idx

        instance_mask = torch.from_numpy(instance_mask_np).to(device=device, dtype=torch.long)
        union_mask = torch.from_numpy(union_mask_np).to(device=device, dtype=torch.bool)
        overlap_mask = torch.from_numpy(overlap_mask_np).to(device=device, dtype=torch.bool)
        union_patches = (
            union_mask.view(patch_side, ph, patch_side, pw)
            .permute(0, 2, 1, 3)
            .reshape(patch_side * patch_side, ph, pw)
            .any(dim=(1, 2))
        )
        gt_cache[img_int_id] = {
            "instance_mask": instance_mask,
            "union": union_mask,
            "overlap": overlap_mask,
            "union_patches": union_patches,
        }
    return gt_cache


def _compute_binary_mask_metrics(
    pred_bool: torch.Tensor,
    union_mask: torch.Tensor,
    union_patches: torch.Tensor,
    instance_mask: torch.Tensor,
    topk_patch_indices: torch.Tensor,
) -> tuple[Optional[float], Optional[float], float]:
    n_instances = int(instance_mask.max().item())
    mbo: Optional[float] = None
    miou: Optional[float] = None
    if n_instances > 0:
        gt_masks = F.one_hot(instance_mask, num_classes=n_instances + 1).permute(2, 0, 1)[1:].to(dtype=torch.bool)
        pred_mask = pred_bool.unsqueeze(0)
        inter = (gt_masks & pred_mask).flatten(1).sum(dim=1)
        union = (gt_masks | pred_mask).flatten(1).sum(dim=1)
        ious = torch.where(union > 0, inter.float() / union.float().clamp(min=1.0), torch.zeros_like(union, dtype=torch.float32))
        mbo = float(ious.max().item())
        miou = float(ious.mean().item())

    loc_acc = float(union_patches[topk_patch_indices].float().mean().item()) if union_mask.numel() > 0 else 0.0
    return mbo, miou, loc_acc


def evaluate_all(
    args: argparse.Namespace,
    model,
    dataloader: DataLoader,
    target_image_processor,
    decoder,
    device: torch.device,
    allowed_save_stems: Optional[set],
) -> Dict[str, Any]:
    """Single-pass eval: Loss + Recon + Slot Attention in one loop.

    Per batch:
      1. generate_captionslot() → generated (for recon) + attn_maps (for slot)  [1 LLM fwd + diffusion]
      2. model() for training loss                                                [1 LLM fwd, only if --report-losses]

    Previously these were 3 separate dataloader passes (3× LLM forward each).
    """
    import concurrent.futures as _cf

    mean, std = get_image_stats(target_image_processor)
    inception = InceptionFeatureExtractor().to(device)
    coco, img_id_to_anns = _load_coco_gt(args.coco_instances) if args.eval_slot_attention else (None, {})
    has_gt = coco is not None
    slots_per_object = int(
        getattr(
            model,
            "captionslot_slots_per_object",
            getattr(getattr(model, "config", None), "captionslot_slots_per_object", 1),
        )
    )
    patch_side = int(round(math.sqrt(int(getattr(model, "num_image_tokens", 256)))))

    # ── output file handles ───────────────────────────────────────────────────
    captions_path      = os.path.join(args.output_dir, "captions.jsonl")
    per_image_path     = os.path.join(args.output_dir, "per_image.jsonl")
    per_image_attn_path = os.path.join(args.output_dir, "per_image_attn.jsonl")
    features_path      = os.path.join(args.output_dir, "fid_features.npz")
    attn_dir           = os.path.join(args.output_dir, "attn_maps")
    if args.save_attn_maps and args.eval_slot_attention:
        os.makedirs(attn_dir, exist_ok=True)

    # Background thread pool for matplotlib overlay saves (prevents GPU stall).
    save_executor = _cf.ThreadPoolExecutor(max_workers=4) if (args.save_attn_maps and args.eval_slot_attention) else None
    pending_save_futures: List[Any] = []

    gt_cache: Dict[int, Dict[str, torch.Tensor]] = {}
    if has_gt:
        sample_hw = tuple(dataloader.dataset[0]["target_image"].shape[-2:])
        gt_cache = _build_gt_cache_for_samples(
            samples=getattr(dataloader.dataset, "entries", []),
            coco=coco,
            img_id_to_anns=img_id_to_anns,
            height=int(sample_hw[0]),
            width=int(sample_hw[1]),
            device=device,
            patch_side=patch_side,
            gt_min_area_pct=float(getattr(args, "gt_min_area_pct", 0.0)),
        )

    # ── threshold lists (primary + user-provided extras) ──────────────────────
    primary_bg_thr = float(args.segmentation_bg_threshold)
    primary_attn_thr = float(args.attn_threshold)
    bg_thr_list = _parse_threshold_list(primary_bg_thr, getattr(args, "segmentation_bg_thresholds", ""))
    attn_thr_list = _parse_threshold_list(primary_attn_thr, getattr(args, "attn_thresholds", ""))

    # ── accumulators ──────────────────────────────────────────────────────────
    real_features: List[np.ndarray] = []
    fake_features: List[np.ndarray] = []
    metrics_list:  List[Dict[str, float]] = []
    loc_acc_list:  List[float] = []  # threshold-independent (top16)
    strict_fari_list: List[float] = []
    strict_mbo_list: List[float] = []
    strict_miou_list: List[float] = []
    # Per-threshold accumulators
    bg_metrics_by_thr: Dict[float, Dict[str, List[float]]] = {
        thr: {"fari": [], "mbo": [], "miou": []} for thr in bg_thr_list
    }
    bin_metrics_by_thr: Dict[float, Dict[str, List[float]]] = {
        thr: {"mbo": [], "miou": []} for thr in attn_thr_list
    }
    total_loss = 0.0
    total_recon_loss = 0.0
    loss_count = 0
    generated_count = failed_count = saved_count = saved_attn = 0

    model.eval()
    with (
        open(captions_path,       "w", encoding="utf-8") as cap_f,
        open(per_image_path,      "w", encoding="utf-8") as img_f,
        open(per_image_attn_path, "w", encoding="utf-8") as attn_f,
        tqdm(total=len(dataloader.dataset), desc="Eval", unit="img", dynamic_ncols=True) as pbar,
    ):
        with torch.no_grad():
            for batch in dataloader:
                batch = move_batch_to_device(batch, device)
                B = batch["images"].shape[0]
                batch_n_slots = batch["n_slots"].detach().cpu().tolist()

                # ── 1. generate: LLM forward + diffusion sampling ─────────────
                output = model.generate_captionslot(
                    images=batch["images"],
                    caption_input_ids=batch["caption_input_ids"],
                    caption_attention_mask=batch["caption_attention_mask"],
                    noun_chunk_spans=batch["noun_chunk_spans"],
                    n_slots=batch["n_slots"],
                    head_prior_maps=batch["head_prior_maps"],
                    head_prior_valid_mask=batch["head_prior_valid_mask"],
                    guidance_level=args.guidance_level,
                    return_generated=True,
                )
                attn_maps    = output.get("attn_maps")    # [B, K, 256] or None
                recon_images = decode_generated_images(decoder, output.get("generated"), device=device)
                source_images = denormalize_images(batch["target_images"], mean, std, to_cpu=False)

                # ── 2. training loss (optional, 1 extra LLM fwd) ──────────────
                if args.report_losses:
                    loss_out = model(
                        input_ids=torch.zeros(B, 1, dtype=torch.long, device=device),
                        images=batch["images"],
                        target_images=batch["target_images"],
                        caption_input_ids=batch["caption_input_ids"],
                        caption_attention_mask=batch["caption_attention_mask"],
                        noun_chunk_spans=batch["noun_chunk_spans"],
                        n_slots=batch["n_slots"],
                        head_prior_maps=batch["head_prior_maps"],
                        head_prior_valid_mask=batch["head_prior_valid_mask"],
                        return_dict=True,
                    )
                    total_loss       += float(loss_out.loss.detach().float().item())
                    total_recon_loss += float(getattr(model, "captionslot_loss_recon", loss_out.loss).detach().float().item())
                    loss_count += 1

                # ── 3. Inception batch ────────────────────────────────────────
                try:
                    batch_real_feat = inception(source_images).detach().cpu().numpy()
                    batch_fake_feat = inception(recon_images).detach().cpu().numpy()
                except Exception:
                    batch_real_feat = batch_fake_feat = None

                # ── 4a. batched recon metrics ─────────────────────────────────
                try:
                    recon_t = _compute_recon_metrics_batched(source_images, recon_images)
                    recon_mse  = recon_t["mse"].detach().float().cpu().tolist()
                    recon_mae  = recon_t["mae"].detach().float().cpu().tolist()
                    recon_psnr = recon_t["psnr"].detach().float().cpu().tolist()
                    recon_ssim = recon_t["ssim"].detach().float().cpu().tolist()
                    recon_err: Optional[str] = None
                except Exception as exc:
                    recon_mse = recon_mae = recon_psnr = recon_ssim = [None] * B
                    recon_err = repr(exc)

                # ── 4b. batched slot-attention prep (ragged merge → batched) ──
                H, W = int(source_images.shape[-2]), int(source_images.shape[-1])
                merged_maps_list: List[torch.Tensor] = []
                overlay_maps_list: List[torch.Tensor] = []
                attn_soft_batch: Optional[torch.Tensor] = None
                # primary binary mask kept for backward-compat overlay save
                binary_mask_primary_batch: Optional[torch.Tensor] = None
                top16_indices_batch: Optional[torch.Tensor] = None
                attn_mean_list: List[float] = []
                attn_max_list: List[float] = []
                pred_fg_ratio_list: List[float] = []
                n_merged_list: List[int] = []
                strict_fari_vals = [float("nan")] * B
                strict_mbo_vals  = [float("nan")] * B
                strict_miou_vals = [float("nan")] * B
                # Per-threshold per-image results for this batch
                bg_fari_vals_by_thr: Dict[float, List[float]] = {thr: [float("nan")] * B for thr in bg_thr_list}
                bg_mbo_vals_by_thr:  Dict[float, List[float]] = {thr: [float("nan")] * B for thr in bg_thr_list}
                bg_miou_vals_by_thr: Dict[float, List[float]] = {thr: [float("nan")] * B for thr in bg_thr_list}
                bin_mbo_vals_by_thr:  Dict[float, List[Optional[float]]] = {thr: [None] * B for thr in attn_thr_list}
                bin_miou_vals_by_thr: Dict[float, List[Optional[float]]] = {thr: [None] * B for thr in attn_thr_list}
                bin_loc_acc_vals: List[float] = [0.0] * B  # threshold-independent
                cached_list: List[Optional[Dict[str, torch.Tensor]]] = [None] * B

                if args.eval_slot_attention and attn_maps is not None:
                    for idx in range(B):
                        n_active = int(batch_n_slots[idx])
                        if getattr(args, "raw_slot_argmax", False):
                            # Skip per-object merge: expose each slot as its own channel
                            n_active_clamped = max(0, min(n_active, int(attn_maps[idx].shape[0])))
                            m = attn_maps[idx][:n_active_clamped].float() if n_active_clamped > 0 else attn_maps[idx].new_zeros((0, attn_maps[idx].shape[-1]))
                        else:
                            m = _merge_object_maps(attn_maps[idx], n_active, slots_per_object)
                        # Temperature sharpening: divide by T then renormalize to [0,1]
                        _attn_temp = float(getattr(args, "attn_temperature", 1.0))
                        if _attn_temp != 1.0 and m.shape[0] > 0:
                            m = (m / _attn_temp).softmax(dim=0)
                        merged_maps_list.append(m)
                        if m.shape[0] > 0:
                            overlay_maps_list.append(m.max(dim=0).values)
                        else:
                            overlay_maps_list.append(
                                torch.zeros((patch_side * patch_side,), dtype=torch.float32, device=device)
                            )
                        n_merged_list.append(int(m.shape[0]))

                    overlay_stack = torch.stack(overlay_maps_list, dim=0)
                    attn_soft_batch = _upsample_slot_maps(overlay_stack, H, W)
                    binary_mask_primary_batch = attn_soft_batch > primary_attn_thr
                    top16_indices_batch = overlay_stack.topk(16, dim=-1).indices

                    attn_mean_list = attn_soft_batch.mean(dim=(1, 2)).detach().float().cpu().tolist()
                    attn_max_list  = attn_soft_batch.amax(dim=(1, 2)).detach().float().cpu().tolist()
                    pred_fg_ratio_list = binary_mask_primary_batch.float().mean(dim=(1, 2)).detach().cpu().tolist()

                    if has_gt:
                        for idx in range(B):
                            image_id_str = batch["image_ids"][idx]
                            try:
                                img_int_id = int(image_id_str)
                            except ValueError:
                                img_int_id = None
                            cached_list[idx] = gt_cache.get(img_int_id) if img_int_id is not None else None

                        # strict + bg @ each threshold (upsample shared)
                        strict_batch, bg_batch_by_thr = _build_dense_pred_masks_batched(
                            merged_maps_list, H, W, bg_thr_list, device
                        )

                        valid_idx: List[int] = [
                            i for i, c in enumerate(cached_list)
                            if c is not None and c.get("instance_mask") is not None
                        ]
                        if valid_idx:
                            gt_stack = torch.stack([cached_list[i]["instance_mask"] for i in valid_idx], dim=0).long()
                            overlap_stack = torch.stack([cached_list[i]["overlap"] for i in valid_idx], dim=0).long()

                            # strict metrics (threshold-independent)
                            strict_sub = strict_batch[valid_idx].long()
                            s_fari_l = _fari_metric_batched(gt_stack, strict_sub, overlap_stack).detach().float().cpu().tolist()
                            s_mbo_l  = _mbo_metric_batched(gt_stack, strict_sub, overlap_stack).detach().float().cpu().tolist()
                            s_miou_l = _miou_metric_batched(gt_stack, strict_sub, overlap_stack).detach().float().cpu().tolist()
                            for pos, idx in enumerate(valid_idx):
                                strict_fari_vals[idx] = s_fari_l[pos]
                                strict_mbo_vals[idx]  = s_mbo_l[pos]
                                strict_miou_vals[idx] = s_miou_l[pos]

                            # bg metrics per threshold
                            for thr in bg_thr_list:
                                bg_sub = bg_batch_by_thr[thr][valid_idx].long()
                                b_fari_l = _fari_metric_batched(gt_stack, bg_sub, overlap_stack).detach().float().cpu().tolist()
                                b_mbo_l  = _mbo_metric_batched(gt_stack, bg_sub, overlap_stack).detach().float().cpu().tolist()
                                b_miou_l = _miou_metric_batched(gt_stack, bg_sub, overlap_stack).detach().float().cpu().tolist()
                                for pos, idx in enumerate(valid_idx):
                                    bg_fari_vals_by_thr[thr][idx] = b_fari_l[pos]
                                    bg_mbo_vals_by_thr[thr][idx]  = b_mbo_l[pos]
                                    bg_miou_vals_by_thr[thr][idx] = b_miou_l[pos]

                        # Per-image binary-mask metrics (per attn threshold; loc_acc shared)
                        for idx in range(B):
                            cached = cached_list[idx]
                            if cached is None:
                                continue
                            gt_inst = cached.get("instance_mask")
                            gt_union = cached.get("union")
                            gt_union_patches = cached.get("union_patches")
                            if gt_inst is None or gt_union is None or gt_union_patches is None:
                                continue
                            loc_acc_this: Optional[float] = None
                            for thr in attn_thr_list:
                                pred_bool = attn_soft_batch[idx] > thr
                                mbo, miou, loc_acc = _compute_binary_mask_metrics(
                                    pred_bool=pred_bool,
                                    union_mask=gt_union,
                                    union_patches=gt_union_patches,
                                    instance_mask=gt_inst,
                                    topk_patch_indices=top16_indices_batch[idx],
                                )
                                bin_mbo_vals_by_thr[thr][idx] = mbo
                                bin_miou_vals_by_thr[thr][idx] = miou
                                if loc_acc_this is None:
                                    loc_acc_this = loc_acc  # loc_acc is threshold-independent
                            if loc_acc_this is not None:
                                bin_loc_acc_vals[idx] = float(loc_acc_this)

                # ── 4c. per-image record write + sample/overlay saves ─────────
                for idx in range(B):
                    image_path    = batch["image_path"][idx]
                    stem          = Path(image_path).stem
                    image_id_str  = batch["image_ids"][idx]
                    caption       = batch["caption"][idx]

                    record: Dict[str, Any] = {
                        "image": os.path.abspath(image_path),
                        "file_name": batch["file_name"][idx],
                        "image_id": image_id_str,
                        "caption": caption,
                        "generated_image": False,
                    }
                    if recon_err is None and recon_mse[idx] is not None:
                        metrics = {
                            "mse":  float(recon_mse[idx]),
                            "mae":  float(recon_mae[idx]),
                            "psnr": float(recon_psnr[idx]),
                            "ssim": float(recon_ssim[idx]),
                        }
                        metrics_list.append(metrics)
                        record["generated_image"] = True
                        record["metrics"] = metrics
                        generated_count += 1
                        if batch_real_feat is not None:
                            real_features.append(batch_real_feat[idx:idx+1])
                            fake_features.append(batch_fake_feat[idx:idx+1])
                        should_save = (
                            args.save_images
                            and saved_count < args.save_limit
                            and (allowed_save_stems is None or stem in allowed_save_stems)
                        )
                        if should_save:
                            sample_dir = os.path.join(args.output_dir, "samples", stem)
                            os.makedirs(sample_dir, exist_ok=True)
                            input_img  = tensor_to_pil(source_images[idx:idx+1])
                            recon_img  = tensor_to_pil(recon_images[idx:idx+1])
                            diff_img   = build_abs_diff_image(source_images[idx:idx+1], recon_images[idx:idx+1])
                            input_img.save(os.path.join(sample_dir, "input_processed.png"))
                            recon_img.save(os.path.join(sample_dir, "generated.png"))
                            diff_img.save(os.path.join(sample_dir, "abs_diff.png"))
                            save_triptych(input_img, recon_img, diff_img, os.path.join(sample_dir, "comparison_triptych.png"))
                            saved_count += 1
                    else:
                        failed_count += 1
                        record["error"] = recon_err if recon_err is not None else "metric_compute_failed"

                    cap_f.write(json.dumps({"image": os.path.abspath(image_path), "file_name": batch["file_name"][idx], "caption": caption}, ensure_ascii=False) + "\n")
                    img_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    if not args.eval_slot_attention or attn_maps is None:
                        continue
                    n_active_slots = int(batch_n_slots[idx])
                    attn_record: Dict[str, Any] = {
                        "image_id": image_id_str, "caption": caption,
                        "attn_mean": float(attn_mean_list[idx]),
                        "attn_max": float(attn_max_list[idx]),
                        "pred_fg_ratio": float(pred_fg_ratio_list[idx]),
                        "n_active_slots": n_active_slots,
                        "n_merged_objects": int(n_merged_list[idx]),
                        "slots_per_object": slots_per_object,
                    }
                    if has_gt:
                        cached = cached_list[idx]
                        if cached is not None and cached.get("instance_mask") is not None:
                            # Binary slot_attention metrics: primary (back-compat) + per-threshold
                            primary_mbo = bin_mbo_vals_by_thr[primary_attn_thr][idx]
                            primary_miou = bin_miou_vals_by_thr[primary_attn_thr][idx]
                            attn_record["MBO"] = primary_mbo
                            attn_record["mIoU"] = primary_miou
                            loc_acc_list.append(float(bin_loc_acc_vals[idx]))
                            attn_record["loc_acc_top16"] = float(bin_loc_acc_vals[idx])
                            for thr in attn_thr_list:
                                mbo_t = bin_mbo_vals_by_thr[thr][idx]
                                miou_t = bin_miou_vals_by_thr[thr][idx]
                                if mbo_t is not None:
                                    bin_metrics_by_thr[thr]["mbo"].append(float(mbo_t))
                                if miou_t is not None:
                                    bin_metrics_by_thr[thr]["miou"].append(float(miou_t))
                                if len(attn_thr_list) > 1:
                                    key = _thr_key(thr)
                                    attn_record[f"attn_thr{key}_MBO"]  = None if mbo_t is None else float(mbo_t)
                                    attn_record[f"attn_thr{key}_mIoU"] = None if miou_t is None else float(miou_t)

                            # Strict seg (threshold-independent)
                            sf = strict_fari_vals[idx]
                            sm = strict_mbo_vals[idx]
                            si = strict_miou_vals[idx]
                            strict_fari_list.append(float(sf))
                            if not math.isnan(sm):
                                strict_mbo_list.append(float(sm))
                            if not math.isnan(si):
                                strict_miou_list.append(float(si))
                            attn_record["strict_fARI"] = float(sf)
                            attn_record["strict_MBO"] = None if math.isnan(sm) else float(sm)
                            attn_record["strict_mIoU"] = None if math.isnan(si) else float(si)

                            # BG seg: primary (back-compat fields) + per-threshold
                            bf = bg_fari_vals_by_thr[primary_bg_thr][idx]
                            bm = bg_mbo_vals_by_thr[primary_bg_thr][idx]
                            bi = bg_miou_vals_by_thr[primary_bg_thr][idx]
                            attn_record["bg_fARI"] = float(bf)
                            attn_record["bg_MBO"] = None if math.isnan(bm) else float(bm)
                            attn_record["bg_mIoU"] = None if math.isnan(bi) else float(bi)
                            attn_record["segmentation_bg_threshold"] = primary_bg_thr
                            for thr in bg_thr_list:
                                bf_t = bg_fari_vals_by_thr[thr][idx]
                                bm_t = bg_mbo_vals_by_thr[thr][idx]
                                bi_t = bg_miou_vals_by_thr[thr][idx]
                                bg_metrics_by_thr[thr]["fari"].append(float(bf_t))
                                if not math.isnan(bm_t):
                                    bg_metrics_by_thr[thr]["mbo"].append(float(bm_t))
                                if not math.isnan(bi_t):
                                    bg_metrics_by_thr[thr]["miou"].append(float(bi_t))
                                if len(bg_thr_list) > 1:
                                    key = _thr_key(thr)
                                    attn_record[f"bg_thr{key}_fARI"] = float(bf_t)
                                    attn_record[f"bg_thr{key}_MBO"]  = None if math.isnan(bm_t) else float(bm_t)
                                    attn_record[f"bg_thr{key}_mIoU"] = None if math.isnan(bi_t) else float(bi_t)

                    if args.save_attn_maps and saved_attn < args.save_attn_limit and (allowed_save_stems is None or stem in allowed_save_stems):
                        src_np = (source_images[idx].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        overlay_path = os.path.join(attn_dir, f"{stem}_attn.png")
                        attn_cpu = attn_soft_batch[idx].detach().float().cpu().numpy().copy()
                        pending_save_futures.append(
                            save_executor.submit(
                                _save_attn_overlay,
                                src_np.copy(),
                                attn_cpu,
                                overlay_path,
                            )
                        )
                        attn_record["attn_overlay"] = os.path.abspath(overlay_path)
                        saved_attn += 1
                    attn_f.write(json.dumps(attn_record, ensure_ascii=False) + "\n")

                cap_f.flush()
                img_f.flush()
                attn_f.flush()
                pbar.update(B)
                pbar.set_postfix(gen=generated_count, fail=failed_count)

    # Drain any pending overlay saves before finalising.
    if save_executor is not None:
        for fut in pending_save_futures:
            try:
                fut.result(timeout=60)
            except Exception as exc:  # best-effort — don't fail eval on overlay errors
                print(f"[WARN] overlay save failed: {exc!r}")
        save_executor.shutdown(wait=True)

    # ── aggregate ─────────────────────────────────────────────────────────────
    real_arr = np.concatenate(real_features) if real_features else np.empty((0, 2048), dtype=np.float32)
    fake_arr = np.concatenate(fake_features) if fake_features else np.empty((0, 2048), dtype=np.float32)
    np.savez_compressed(features_path, real=real_arr, fake=fake_arr)

    result: Dict[str, Any] = {}

    if loss_count:
        result["loss_metrics"] = {
            "eval/loss":       total_loss / loss_count,
            "eval/loss_recon": total_recon_loss / loss_count,
        }

    recon: Dict[str, Any] = {
        "requested_samples": len(dataloader.dataset),
        "generated_count": generated_count, "failed_count": failed_count,
        "success_rate": generated_count / max(len(dataloader.dataset), 1),
    }
    if metrics_list:
        recon["PSNR"] = float(np.mean([m["psnr"] for m in metrics_list]))
        recon["SSIM"] = float(np.mean([m["ssim"] for m in metrics_list]))
        recon["MSE"]  = float(np.mean([m["mse"]  for m in metrics_list]))
        recon["MAE"]  = float(np.mean([m["mae"]  for m in metrics_list]))
    recon["rFID"] = compute_fid(real_arr, fake_arr) if real_arr.shape[0] >= 2 and fake_arr.shape[0] >= 2 else None
    result["reconstruction_metrics"] = recon

    if args.eval_slot_attention:
        # Primary slot_attention_metrics (uses primary attn threshold)
        prim_attn = bin_metrics_by_thr[primary_attn_thr]
        attn_summary: Dict[str, Any] = {
            "attn_threshold": primary_attn_thr,
            "per_image_attn_jsonl": os.path.abspath(per_image_attn_path),
            "evaluated_images": len(strict_fari_list),
        }
        if prim_attn["mbo"]:
            attn_summary["MBO"]           = float(np.mean(prim_attn["mbo"]))
            attn_summary["mIoU"]          = float(np.mean(prim_attn["miou"]))
            attn_summary["loc_acc_top16"] = float(np.mean(loc_acc_list))
        result["slot_attention_metrics"] = attn_summary

        # Per-threshold slot_attention_metrics_thr{T} (only if multiple thresholds)
        if len(attn_thr_list) > 1:
            for thr in attn_thr_list:
                entries = bin_metrics_by_thr[thr]
                block: Dict[str, Any] = {
                    "attn_threshold": thr,
                    "evaluated_images": len(entries["mbo"]),
                }
                if entries["mbo"]:
                    block["MBO"]           = float(np.mean(entries["mbo"]))
                    block["mIoU"]          = float(np.mean(entries["miou"]))
                    block["loc_acc_top16"] = float(np.mean(loc_acc_list))
                result[f"slot_attention_metrics_thr{_thr_key(thr)}"] = block

        result["segmentation_metrics_strict"] = {
            "variant": "pair_max_argmax_no_background",
            "evaluated_images": len(strict_fari_list),
            "fARI": float(np.mean(strict_fari_list)) if strict_fari_list else None,
            "MBO": float(np.mean(strict_mbo_list)) if strict_mbo_list else None,
            "mIoU": float(np.mean(strict_miou_list)) if strict_miou_list else None,
        }

        # Primary segmentation_metrics_bg (uses primary bg threshold)
        prim_bg = bg_metrics_by_thr[primary_bg_thr]
        result["segmentation_metrics_bg"] = {
            "variant": "pair_max_argmax_with_background",
            "evaluated_images": len(prim_bg["fari"]),
            "background_threshold": primary_bg_thr,
            "fARI": float(np.mean(prim_bg["fari"])) if prim_bg["fari"] else None,
            "MBO": float(np.mean(prim_bg["mbo"])) if prim_bg["mbo"] else None,
            "mIoU": float(np.mean(prim_bg["miou"])) if prim_bg["miou"] else None,
        }

        # Per-threshold segmentation_metrics_bg_thr{T} (only if multiple thresholds)
        if len(bg_thr_list) > 1:
            for thr in bg_thr_list:
                entries = bg_metrics_by_thr[thr]
                result[f"segmentation_metrics_bg_thr{_thr_key(thr)}"] = {
                    "variant": "pair_max_argmax_with_background",
                    "evaluated_images": len(entries["fari"]),
                    "background_threshold": thr,
                    "fARI": float(np.mean(entries["fari"])) if entries["fari"] else None,
                    "MBO": float(np.mean(entries["mbo"])) if entries["mbo"] else None,
                    "mIoU": float(np.mean(entries["miou"])) if entries["miou"] else None,
                }

    return result


def _load_coco_gt(coco_instances_path: str):
    """Load COCO GT annotation. Returns (coco_api, img_id_to_anns dict)."""
    try:
        from pycocotools.coco import COCO
    except ImportError:
        return None, {}
    coco = COCO(coco_instances_path)
    img_id_to_anns: Dict[int, List[Dict]] = {}
    for ann in coco.dataset.get("annotations", []):
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)
    return coco, img_id_to_anns


def _decode_coco_mask(coco, ann: Dict, height: int, width: int) -> np.ndarray:
    """Decode a COCO annotation segmentation to a binary (H, W) uint8 mask."""
    from pycocotools import mask as maskUtils
    rle = coco.annToRLE(ann)
    mask = maskUtils.decode(rle)          # (H, W) uint8
    if mask.shape[0] != height or mask.shape[1] != width:
        pil = Image.fromarray(mask * 255).resize((width, height), Image.NEAREST)
        mask = (np.array(pil) > 0).astype(np.uint8)
    return mask


def _compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = float((pred & gt).sum())
    union = float((pred | gt).sum())
    return intersection / union if union > 0 else 0.0


def _attn_map_to_image_mask(
    attn_map: torch.Tensor,     # [256] or [P]
    height: int,
    width: int,
    threshold: float,
) -> np.ndarray:
    """Upsample 1-D attn_map (256 patches → 16×16) to (H, W) binary mask."""
    side = int(round(attn_map.numel() ** 0.5))
    attn_2d = attn_map.float().reshape(1, 1, side, side)
    upsampled = F.interpolate(attn_2d, size=(height, width), mode="bilinear", align_corners=False)
    upsampled = upsampled[0, 0].cpu().numpy()            # (H, W) float
    binary = (upsampled > threshold).astype(np.uint8)
    return binary, upsampled


_JET_LUT: Optional[np.ndarray] = None


def _get_jet_lut() -> np.ndarray:
    """Build a 256-entry 'jet' colourmap lookup table once (PIL-based overlay)."""
    global _JET_LUT
    if _JET_LUT is not None:
        return _JET_LUT
    # simple jet approximation — ramps over 4 segments
    x = np.linspace(0, 1, 256)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    lut = (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)
    _JET_LUT = lut
    return lut


def _save_attn_overlay(
    source_image: np.ndarray,   # (H, W, 3) uint8
    attn_soft: np.ndarray,      # (H, W) float [0,1]
    save_path: str,
) -> None:
    """Save input | overlay | heatmap side-by-side using pure PIL/NumPy.

    Avoids matplotlib entirely so threaded saves don't contend on GIL or
    matplotlib's global state.
    """
    H, W = source_image.shape[:2]
    lut = _get_jet_lut()

    # Soft → 8-bit index → jet RGB
    idx = np.clip((attn_soft * 255).astype(np.int32), 0, 255)
    heatmap_rgb = lut[idx]                                   # (H, W, 3) uint8

    # Alpha-blend 50% onto source
    alpha = 0.5
    overlay_rgb = (
        alpha * heatmap_rgb + (1 - alpha) * source_image.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    # Side-by-side canvas: input | overlay | heatmap
    canvas = np.concatenate([source_image, overlay_rgb, heatmap_rgb], axis=1)
    Image.fromarray(canvas).save(save_path, optimize=False)



def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)
    disable_torch_init()

    global_samples = load_caption_records(args.captions_jsonl, args.image_dir, args.max_samples)
    if not global_samples:
        raise SystemExit("No evaluation samples found.")
    allowed_save_stems = select_allowed_save_stems(global_samples, args.save_fixed_first_n)
    samples = apply_round_robin_shard(global_samples, args.num_shards, args.shard_index)
    if not samples:
        raise SystemExit("This shard received zero samples.")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = resolve_dtype(args.dtype)
    tokenizer, model, image_processors, _ = load_scale_rae_model(
        model_path=args.model_path,
        device=str(device),
        dtype=dtype,
    )
    image_processor = image_processors[0]
    target_image_processor = image_processors[1] if len(image_processors) > 1 else image_processor
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = 151643
    _register_im_start_end_token_ids(model, tokenizer)
    if not hasattr(model, "captionslot_system_prefix_ids"):
        _register_captionslot_template_token_ids(model, tokenizer)

    # Detect LoRA by inspecting checkpoint keys directly (captionslot_lora_enable may not be
    # persisted in config.json, so we check for lora_ keys in the safetensors shards).
    import glob as _glob
    from safetensors import safe_open as _safe_open
    _shard_files = sorted(_glob.glob(os.path.join(args.model_path, "*.safetensors")))
    _has_lora = False
    _lora_keys_sample = []
    for _f in _shard_files:
        with _safe_open(_f, framework="pt", device="cpu") as _sf:
            for _k in _sf.keys():
                if "lora_" in _k:
                    _has_lora = True
                    _lora_keys_sample.append(_k)
                    break
        if _has_lora:
            break

    if _has_lora:
        from peft import LoraConfig, inject_adapter_in_model

        # Infer LoRA config from checkpoint keys (target modules = parent of lora_A/lora_B)
        _all_lora_parents = set()
        for _f in _shard_files:
            with _safe_open(_f, framework="pt", device="cpu") as _sf:
                for _k in _sf.keys():
                    if ".lora_A." in _k:
                        # e.g. model.layers.0.self_attn.q_proj.lora_A.default.weight → q_proj
                        _all_lora_parents.add(_k.split(".lora_A.")[0].split(".")[-1])
        lora_targets = sorted(_all_lora_parents)
        lora_config = LoraConfig(
            r=int(getattr(model.config, "captionslot_lora_r", 16)),
            lora_alpha=int(getattr(model.config, "captionslot_lora_alpha", 32)),
            lora_dropout=float(getattr(model.config, "captionslot_lora_dropout", 0.05)),
            target_modules=lora_targets,
            task_type="CAUSAL_LM",
        )
        inject_adapter_in_model(lora_config, model, adapter_name="default")
        print(f"[INFO] LoRA detected and injected: r={lora_config.r}, alpha={lora_config.lora_alpha}, targets={lora_targets}")

        # Reload base_layer + lora_ weights now that the model structure matches the checkpoint.
        reload_sd = {}
        for _f in _shard_files:
            with _safe_open(_f, framework="pt", device="cpu") as _sf:
                for _k in _sf.keys():
                    if "base_layer" in _k or "lora_" in _k:
                        reload_sd[_k] = _sf.get_tensor(_k)
        missing_keys, unexpected_keys = model.load_state_dict(reload_sd, strict=False)
        lora_missing = [k for k in missing_keys if "lora_" in k or "base_layer" in k]
        print(f"[INFO] LoRA weight reload: {len(reload_sd)} keys loaded, missing_lora={len(lora_missing)}, unexpected={len(unexpected_keys)}")
    else:
        print("[INFO] No LoRA weights found in checkpoint — loading as base model.")

    model = model.to(device)
    model.eval()

    patch_diffusion_steps(model, args.diffusion_steps)

    if getattr(args, "torch_compile", False):
        if hasattr(model, "diff_head") and hasattr(model.diff_head, "model"):
            print("[INFO] Applying torch.compile to diff_head.model ...")
            model.diff_head.model = torch.compile(model.diff_head.model)
            print("[INFO] torch.compile done. First batch will be slow (compile), then fast.")
        else:
            print("[WARN] torch.compile requested but diff_head.model not found, skipping.")

    max_slots = int(getattr(model.config, "captionslot_max_slots", 10))
    slots_per_object = int(
        getattr(
            model,
            "captionslot_slots_per_object",
            getattr(getattr(model, "config", None), "captionslot_slots_per_object", 1),
        )
    )
    image_feature_token_len = int(
        getattr(
            model.config,
            "image_feature_token_len",
            getattr(model, "num_image_tokens", 256),
        )
    )
    dataset = CaptionSlotEvalDataset(
        samples=samples,
        tokenizer=tokenizer,
        image_processor=image_processor,
        target_image_processor=target_image_processor,
        max_slots=max_slots,
        slots_per_object=slots_per_object,
        image_feature_token_len=image_feature_token_len,
        max_caption_tokens=args.max_caption_tokens,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
        prefetch_factor=4 if args.dataloader_num_workers > 0 else None,
        collate_fn=CaptionSlotEvalCollator(pad_token_id=tokenizer.pad_token_id),
        drop_last=False,
    )
    decoder = load_eval_decoder(model)

    summary: Dict[str, Any] = {
        "model_path": os.path.abspath(args.model_path),
        "image_dir": os.path.abspath(args.image_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "dtype": args.dtype,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "sample_count": len(dataset),
        "global_sample_count": len(global_samples),
        "captions_jsonl": os.path.abspath(args.captions_jsonl),
        "guidance_level": float(args.guidance_level),
        "max_slots": max_slots,
        "slots_per_object": slots_per_object,
        "image_feature_token_len": image_feature_token_len,
        "diffusion_target_token_len": int(
            getattr(
                model.config,
                "diffusion_target_token_len",
                getattr(model, "diffusion_target_token_len", 256),
            )
        ),
        "segmentation_bg_threshold": float(args.segmentation_bg_threshold),
    }
    results = evaluate_all(
        args=args,
        model=model,
        dataloader=dataloader,
        target_image_processor=target_image_processor,
        decoder=decoder,
        device=device,
        allowed_save_stems=allowed_save_stems,
    )
    summary.update(results)
    summary["notes"] = [
        "CaptionSlot reconstruction eval uses generate_captionslot() followed by the SigLIP decoder.",
        "PSNR, SSIM, MAE, and MSE are computed against the model-preprocessed input image.",
        "rFID is computed with a torchvision Inception-V3 backbone, so values may differ from clean-fid/pytorch-fid.",
        "When captions are provided via captions.jsonl, precomputed noun_chunks/token_ids from the jsonl are used directly (no spaCy) and no head prior maps are applied.",
        "Slot attention eval merges per-object slot pairs with max pooling before segmentation metrics are computed.",
        "segmentation_metrics_strict uses pair-merged object maps and argmax with no explicit background channel.",
        "segmentation_metrics_bg adds a constant background channel with score --segmentation-bg-threshold before argmax.",
        "slot_attention_metrics remains a localization-style summary over the merged max attention map.",
    ]
    save_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
