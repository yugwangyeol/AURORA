"""PGOT training entry point.

Usage:
    PYTHONPATH=/home/jovyan/PGOT python train.py \
        --model_name_or_path <path-to-scale-rae-qwen2-ckpt> \
        --train_jsonl /home/jovyan/PGOT/data/pgot_train.jsonl \
        --val_jsonl   /home/jovyan/PGOT/data/pgot_val.jsonl \
        --output_dir  /home/jovyan/PGOT/outputs/pgot_pilot \
        ...
"""

import json
import logging
import os
import sys

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

# Local imports
from pgot.constants import (
    OVT_TOKEN,
    SCENE_END_TOKEN,
    THING_TOKEN,
    STUFF_TOKEN,
    NEW_SPECIAL_TOKENS,
    PGOT_SYSTEM_PROMPT,
    PGOT_USER_INSTRUCTION,
    get_pgot_prompts,
)
from pgot.model.pgot_qwen2 import PGOTQwen2ForCausalLM, PGOTQwen2Config
from pgot.train.pgot_dataset import Pix2CapPGOTDataset, PGOTDataCollator
from pgot.train.pgot_trainer import (
    PGOTModelArguments,
    PGOTDataArguments,
    PGOTTrainingArguments,
    PGOTTrainer,
    freeze_for_pgot,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pgot.train")


def _register_template_token_ids(model, tokenizer, dataset_format: str = "pix2cap"):
    """Pre-tokenize ChatML template blocks ONCE and cache on the model."""
    system_prompt, user_instruction = get_pgot_prompts(dataset_format)
    blocks = {
        "pgot_system_prefix_ids": f"<|im_start|>system\n{system_prompt}",
        "pgot_system_suffix_ids": "<|im_end|>\n",
        "pgot_user_prefix_ids":   "<|im_start|>user\n<image>",
        "pgot_user_suffix_ids":   f"{user_instruction}<|im_end|>\n",
        "pgot_assistant_prefix_ids": "<|im_start|>assistant\n",
        "pgot_assistant_suffix_ids": "<|im_end|>",
    }
    for attr, txt in blocks.items():
        ids = tokenizer.encode(txt, add_special_tokens=False)
        # Strip the <image> placeholder if present — we don't want it in the
        # token stream; the actual image features are concatenated separately.
        if "<image>" in txt:
            try:
                placeholder_id = tokenizer.convert_tokens_to_ids("<image>")
                ids = [i for i in ids if i != placeholder_id]
            except Exception:
                pass
        setattr(model, attr, ids)


def _register_ovt_token_ids(model, tokenizer):
    """Register core object-format token ids on the model."""
    model.pgot_ovt_token_id = tokenizer.convert_tokens_to_ids(OVT_TOKEN)
    model.pgot_scene_end_token_id = tokenizer.convert_tokens_to_ids(SCENE_END_TOKEN)
    model.pgot_thing_token_id = tokenizer.convert_tokens_to_ids(THING_TOKEN)
    model.pgot_stuff_token_id = tokenizer.convert_tokens_to_ids(STUFF_TOKEN)
    logger.info(
        f"[PGOT] token ids: <ovt>={model.pgot_ovt_token_id}, "
        f"<scene_end>={model.pgot_scene_end_token_id}, "
        f"<thing>={model.pgot_thing_token_id}, <stuff>={model.pgot_stuff_token_id}"
    )


def train():
    parser = transformers.HfArgumentParser(
        (PGOTModelArguments, PGOTDataArguments, PGOTTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.info("=" * 70)
    logger.info("[PGOT] train() start")
    logger.info(f"  model_name_or_path: {model_args.model_name_or_path}")
    logger.info(f"  output_dir: {training_args.output_dir}")
    logger.info(f"  train_jsonl: {data_args.train_jsonl}")
    logger.info(f"  val_jsonl:   {data_args.val_jsonl}")
    logger.info("=" * 70)

    # ---- precision: fp32 per user decision
    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # ---- Build config (use ScaleRAE underneath)
    parsed_towers = json.loads(model_args.vision_tower_aux_list) if model_args.vision_tower_aux_list else None
    parsed_token_lens = (
        json.loads(model_args.vision_tower_aux_token_len_list)
        if model_args.vision_tower_aux_token_len_list else None
    )

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    # ScaleRAE base fields
    config.vision_loss = model_args.vision_loss
    config.vision_loss_mode = model_args.vision_loss_mode
    config.vision_coef = model_args.vision_coef
    config.diffusion_model_hidden_size = model_args.diffusion_model_hidden_size
    config.diffusion_model_channels = model_args.diffusion_model_channels
    config.diffusion_model_depth = model_args.diffusion_model_depth
    config.diffusion_model_heads = model_args.diffusion_model_heads
    config.diffusion_model_z_channels = model_args.diffusion_model_z_channels
    config.dit_cls = model_args.dit_cls
    config.use_aurora = False
    config.use_captionslot = False
    config.use_pgot = True
    config.pgot_dataset_format = str(data_args.dataset_format)

    if parsed_towers:
        config.mm_vision_tower_aux_list = parsed_towers
        config.mm_vision_tower_aux_token_len_list = parsed_token_lens
        config.vision_tower_aux_token_len_list = parsed_token_lens
        config.image_feature_token_len = int(
            model_args.image_feature_token_len if model_args.image_feature_token_len is not None
            else parsed_token_lens[0]
        )
        config.diffusion_target_token_len = int(
            model_args.diffusion_target_token_len if model_args.diffusion_target_token_len is not None
            else parsed_token_lens[-1]
        )
    else:
        config.image_feature_token_len = int(getattr(config, "image_feature_token_len", 256))
        config.diffusion_target_token_len = int(getattr(config, "diffusion_target_token_len", 256))

    # PGOT fields
    config.pgot_n_register = model_args.pgot_n_register
    config.pgot_n_null_bg = model_args.pgot_n_null_bg
    config.pgot_n_ovt_per_object = model_args.pgot_n_ovt_per_object
    config.pgot_max_objects = model_args.pgot_max_objects
    config.pgot_use_null_bg_competition = bool(model_args.pgot_use_null_bg_competition)
    config.pgot_mask_ce_weight = float(model_args.pgot_mask_ce_weight)
    config.pgot_mask_ce_temperature = float(model_args.pgot_mask_ce_temperature)
    config.pgot_mask_ce_merge = str(model_args.pgot_mask_ce_merge)
    config.pgot_mask_fg_weight = float(model_args.pgot_mask_fg_weight)
    config.pgot_mask_outside_weight = float(model_args.pgot_mask_outside_weight)
    config.pgot_mask_object_balanced_bce_weight = float(model_args.pgot_mask_object_balanced_bce_weight)
    config.pgot_mask_spatial_outside_weight = float(model_args.pgot_mask_spatial_outside_weight)
    config.pgot_mask_spatial_temperature = float(model_args.pgot_mask_spatial_temperature)
    config.pgot_mask_spatial_outside_log_weight = float(model_args.pgot_mask_spatial_outside_log_weight)
    config.pgot_mask_spatial_outside_log_temperature = float(model_args.pgot_mask_spatial_outside_log_temperature)
    config.pgot_mask_llm_qk_outside_weight = float(model_args.pgot_mask_llm_qk_outside_weight)
    config.pgot_mask_llm_qk_outside_temperature = float(model_args.pgot_mask_llm_qk_outside_temperature)
    config.pgot_mask_llm_qk_outside_layers = str(model_args.pgot_mask_llm_qk_outside_layers)
    config.pgot_mask_llm_attention_outside_weight = float(
        model_args.pgot_mask_llm_attention_outside_weight
    )
    config.pgot_mask_llm_attention_outside_layers = str(
        model_args.pgot_mask_llm_attention_outside_layers
    )
    config.pgot_mask_llm_attention_void_weight = float(
        model_args.pgot_mask_llm_attention_void_weight
    )
    config.pgot_mask_llm_patch_outside_weight = float(
        model_args.pgot_mask_llm_patch_outside_weight
    )
    config.pgot_mask_llm_patch_outside_layers = str(
        model_args.pgot_mask_llm_patch_outside_layers
    )
    config.pgot_mask_llm_patch_outside_temperature = float(
        model_args.pgot_mask_llm_patch_outside_temperature
    )
    config.pgot_mask_llm_patch_void_weight = float(
        model_args.pgot_mask_llm_patch_void_weight
    )
    config.pgot_mask_llm_image_use_weight = float(
        model_args.pgot_mask_llm_image_use_weight
    )
    config.pgot_mask_llm_image_use_margin = float(
        model_args.pgot_mask_llm_image_use_margin
    )
    config.pgot_core_outside_weight = float(model_args.pgot_core_outside_weight)
    config.pgot_core_outside_layers = str(model_args.pgot_core_outside_layers)
    config.pgot_core_outside_temperature = float(model_args.pgot_core_outside_temperature)
    config.pgot_core_void_weight = float(model_args.pgot_core_void_weight)
    config.pgot_core_tail_weight = float(model_args.pgot_core_tail_weight)
    config.pgot_core_tail_fraction = float(model_args.pgot_core_tail_fraction)
    config.pgot_core_register_outside_weight = float(
        model_args.pgot_core_register_outside_weight
    )
    config.pgot_e3_attention_competition_weight = float(
        model_args.pgot_e3_attention_competition_weight
    )
    config.pgot_e3_attention_competition_layers = str(
        model_args.pgot_e3_attention_competition_layers
    )
    config.pgot_e3_attention_competition_temperature = float(
        model_args.pgot_e3_attention_competition_temperature
    )
    config.pgot_e3_attention_competition_bg_weight = float(
        model_args.pgot_e3_attention_competition_bg_weight
    )
    config.pgot_e4_rae_isolated = bool(model_args.pgot_e4_rae_isolated)
    config.pgot_e4_full_inside_weight = float(
        model_args.pgot_e4_full_inside_weight
    )
    config.pgot_e4_full_inside_target = float(
        model_args.pgot_e4_full_inside_target
    )
    config.pgot_e4_rae_bind_weight = float(model_args.pgot_e4_rae_bind_weight)
    config.pgot_e4_rae_bind_layers = str(model_args.pgot_e4_rae_bind_layers)
    config.pgot_register_hard_gt_mask = bool(model_args.pgot_register_hard_gt_mask)
    config.pgot_register_hard_gt_mask_eval = bool(
        model_args.pgot_register_hard_gt_mask_eval
    )
    config.pgot_register_hard_gt_mask_threshold = float(
        model_args.pgot_register_hard_gt_mask_threshold
    )
    config.pgot_register_attends_caption = bool(model_args.pgot_register_attends_caption)
    config.pgot_ovt_caption_init = bool(model_args.pgot_ovt_caption_init)
    config.pgot_ovt_caption_init_scale = float(model_args.pgot_ovt_caption_init_scale)
    config.pgot_ovt_isolated_attention = bool(
        model_args.pgot_ovt_isolated_attention
    )
    config.pgot_v12_enable = bool(model_args.pgot_v12_enable)
    config.pgot_v12_layers = str(model_args.pgot_v12_layers)
    v12_ovt_temp = float(
        model_args.pgot_v12_ovt_temperature
        if model_args.pgot_v12_ovt_temperature is not None
        else model_args.pgot_v12_slot_temperature
    )
    config.pgot_v12_ovt_temperature = v12_ovt_temp
    config.pgot_v12_slot_temperature = v12_ovt_temp
    config.pgot_v12_owner_temperature = float(model_args.pgot_v12_owner_temperature)
    config.pgot_v12_owner_weight = float(model_args.pgot_v12_owner_weight)
    config.pgot_v14_enable = bool(model_args.pgot_v14_enable)
    config.pgot_v14_route_temperature = float(model_args.pgot_v14_route_temperature)
    config.pgot_v14_route_weight = float(model_args.pgot_v14_route_weight)
    config.pgot_v14_void_weight = float(model_args.pgot_v14_void_weight)
    config.pgot_v14_position_weight = float(model_args.pgot_v14_position_weight)
    config.pgot_v14_router_depth = int(model_args.pgot_v14_router_depth)
    config.pgot_v14_router_mlp_ratio = int(model_args.pgot_v14_router_mlp_ratio)
    config.pgot_dit_ovt_cross_attn_enable = bool(model_args.pgot_dit_ovt_cross_attn_enable)
    config.pgot_dit_ovt_cross_attn_start_block = int(model_args.pgot_dit_ovt_cross_attn_start_block)
    config.pgot_dit_ovt_cross_attn_every_n_blocks = int(model_args.pgot_dit_ovt_cross_attn_every_n_blocks)
    config.pgot_v17_enable = bool(model_args.pgot_v17_enable)
    config.pgot_v17_ownership_weight = float(model_args.pgot_v17_ownership_weight)
    config.pgot_v17_ownership_layers = str(model_args.pgot_v17_ownership_layers)
    config.pgot_v21_enable = bool(model_args.pgot_v21_enable)
    config.pgot_v21_ground_weight = float(model_args.pgot_v21_ground_weight)
    config.pgot_v21_ground_final_weight = float(model_args.pgot_v21_ground_final_weight)
    config.pgot_v21_ground_anneal_steps = int(model_args.pgot_v21_ground_anneal_steps)
    config.pgot_v21_temperature = float(model_args.pgot_v21_temperature)
    config.pgot_v21_position_weight = float(model_args.pgot_v21_position_weight)
    config.pgot_v21_code_dim = int(model_args.pgot_v21_code_dim)
    config.pgot_v22_attention_competition_weight = float(
        model_args.pgot_v22_attention_competition_weight
    )
    config.pgot_v22_attention_competition_layers = str(
        model_args.pgot_v22_attention_competition_layers
    )
    config.pgot_v22_attention_competition_temperature = float(
        model_args.pgot_v22_attention_competition_temperature
    )
    config.pgot_v22_attention_competition_include_void = bool(
        model_args.pgot_v22_attention_competition_include_void
    )
    config.pgot_v22_attention_competition_bg_weight = float(
        model_args.pgot_v22_attention_competition_bg_weight
    )
    config.pgot_latent_distill_enable = bool(model_args.pgot_latent_distill_enable)
    config.pgot_latent_distill_weight = float(model_args.pgot_latent_distill_weight)
    config.pgot_latent_distill_mse_weight = float(model_args.pgot_latent_distill_mse_weight)
    config.pgot_latent_distill_cos_weight = float(model_args.pgot_latent_distill_cos_weight)
    config.pgot_latent_distill_l1_weight = float(model_args.pgot_latent_distill_l1_weight)
    config.pgot_rae_bidirectional = model_args.pgot_rae_bidirectional
    config.pgot_attention_use_layer_norm = model_args.pgot_attention_use_layer_norm
    config.pgot_attention_temperature = model_args.pgot_attention_temperature
    config.pgot_lm_loss_weight = model_args.pgot_lm_loss_weight
    config.pgot_mask_loss_weight = model_args.pgot_mask_loss_weight
    config.pgot_recon_loss_weight = model_args.pgot_recon_loss_weight
    config.pgot_contrastive_loss_weight = model_args.pgot_contrastive_loss_weight  # start at 0
    config.pgot_contrastive_sampling_rate = model_args.pgot_contrastive_sampling_rate
    config.pgot_unfreeze_mm_projector = bool(model_args.pgot_unfreeze_mm_projector)
    config.image_preprocess_mode = str(data_args.image_preprocess_mode)
    config.coda_crop_size = int(data_args.coda_crop_size)
    if model_args.diffusion_norm_stats_path:
        config.diffusion_norm_stats_path = model_args.diffusion_norm_stats_path

    # ---- Load model
    model = PGOTQwen2ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
        ignore_mismatched_sizes=True,
    )
    model.config.use_cache = False

    # Copy mask/CFG hyperparameters from ModelArguments onto the model config so
    # _forward_pgot (which reads self.config via getattr) uses the CLI-provided values
    # rather than the getattr defaults.
    model.config.pgot_mask_ce_weight = float(model_args.pgot_mask_ce_weight)
    model.config.pgot_mask_ce_temperature = float(model_args.pgot_mask_ce_temperature)
    model.config.pgot_mask_ce_merge = str(model_args.pgot_mask_ce_merge)
    model.config.pgot_use_null_bg_competition = bool(model_args.pgot_use_null_bg_competition)
    model.config.pgot_n_null_bg = int(model_args.pgot_n_null_bg)
    model.config.pgot_mask_fg_weight = float(model_args.pgot_mask_fg_weight)
    model.config.pgot_mask_outside_weight = float(model_args.pgot_mask_outside_weight)
    model.config.pgot_mask_aux_competition_weight = float(model_args.pgot_mask_aux_competition_weight)
    model.config.pgot_mask_bce_weight = float(model_args.pgot_mask_bce_weight)
    model.config.pgot_mask_sigmoid_outside_weight = float(
        model_args.pgot_mask_sigmoid_outside_weight
    )
    model.config.pgot_register_foreground_suppression_weight = float(
        model_args.pgot_register_foreground_suppression_weight
    )
    model.config.pgot_mask_object_balanced_bce_weight = float(model_args.pgot_mask_object_balanced_bce_weight)
    model.config.pgot_mask_tversky_weight = float(model_args.pgot_mask_tversky_weight)
    model.config.pgot_mask_tversky_alpha = float(model_args.pgot_mask_tversky_alpha)
    model.config.pgot_mask_tversky_beta = float(model_args.pgot_mask_tversky_beta)
    model.config.pgot_mask_spatial_outside_weight = float(model_args.pgot_mask_spatial_outside_weight)
    model.config.pgot_mask_spatial_temperature = float(model_args.pgot_mask_spatial_temperature)
    model.config.pgot_mask_spatial_outside_log_weight = float(model_args.pgot_mask_spatial_outside_log_weight)
    model.config.pgot_mask_spatial_outside_log_temperature = float(model_args.pgot_mask_spatial_outside_log_temperature)
    model.config.pgot_mask_llm_qk_outside_weight = float(model_args.pgot_mask_llm_qk_outside_weight)
    model.config.pgot_mask_llm_qk_outside_temperature = float(model_args.pgot_mask_llm_qk_outside_temperature)
    model.config.pgot_mask_llm_qk_outside_layers = str(model_args.pgot_mask_llm_qk_outside_layers)
    model.config.pgot_mask_llm_attention_outside_weight = float(
        model_args.pgot_mask_llm_attention_outside_weight
    )
    model.config.pgot_mask_llm_attention_outside_layers = str(
        model_args.pgot_mask_llm_attention_outside_layers
    )
    model.config.pgot_mask_llm_attention_void_weight = float(
        model_args.pgot_mask_llm_attention_void_weight
    )
    model.config.pgot_mask_llm_patch_outside_weight = float(
        model_args.pgot_mask_llm_patch_outside_weight
    )
    model.config.pgot_mask_llm_patch_outside_layers = str(
        model_args.pgot_mask_llm_patch_outside_layers
    )
    model.config.pgot_mask_llm_patch_outside_temperature = float(
        model_args.pgot_mask_llm_patch_outside_temperature
    )
    model.config.pgot_mask_llm_patch_void_weight = float(
        model_args.pgot_mask_llm_patch_void_weight
    )
    model.config.pgot_mask_llm_image_use_weight = float(
        model_args.pgot_mask_llm_image_use_weight
    )
    model.config.pgot_mask_llm_image_use_margin = float(
        model_args.pgot_mask_llm_image_use_margin
    )
    model.config.pgot_core_outside_weight = float(model_args.pgot_core_outside_weight)
    model.config.pgot_core_outside_layers = str(model_args.pgot_core_outside_layers)
    model.config.pgot_core_outside_temperature = float(model_args.pgot_core_outside_temperature)
    model.config.pgot_core_void_weight = float(model_args.pgot_core_void_weight)
    model.config.pgot_core_tail_weight = float(model_args.pgot_core_tail_weight)
    model.config.pgot_core_tail_fraction = float(model_args.pgot_core_tail_fraction)
    model.config.pgot_core_register_outside_weight = float(
        model_args.pgot_core_register_outside_weight
    )
    model.config.pgot_e3_attention_competition_weight = float(
        model_args.pgot_e3_attention_competition_weight
    )
    model.config.pgot_e3_attention_competition_layers = str(
        model_args.pgot_e3_attention_competition_layers
    )
    model.config.pgot_e3_attention_competition_temperature = float(
        model_args.pgot_e3_attention_competition_temperature
    )
    model.config.pgot_e3_attention_competition_bg_weight = float(
        model_args.pgot_e3_attention_competition_bg_weight
    )
    model.config.pgot_e4_rae_isolated = bool(model_args.pgot_e4_rae_isolated)
    model.config.pgot_e4_full_inside_weight = float(
        model_args.pgot_e4_full_inside_weight
    )
    model.config.pgot_e4_full_inside_target = float(
        model_args.pgot_e4_full_inside_target
    )
    model.config.pgot_e4_rae_bind_weight = float(
        model_args.pgot_e4_rae_bind_weight
    )
    model.config.pgot_e4_rae_bind_layers = str(
        model_args.pgot_e4_rae_bind_layers
    )
    model.config.pgot_register_hard_gt_mask = bool(
        model_args.pgot_register_hard_gt_mask
    )
    model.config.pgot_register_hard_gt_mask_eval = bool(
        model_args.pgot_register_hard_gt_mask_eval
    )
    model.config.pgot_register_hard_gt_mask_threshold = float(
        model_args.pgot_register_hard_gt_mask_threshold
    )
    model.config.pgot_register_attends_caption = bool(
        model_args.pgot_register_attends_caption
    )
    model.config.pgot_ovt_caption_init = bool(model_args.pgot_ovt_caption_init)
    model.config.pgot_ovt_caption_init_scale = float(
        model_args.pgot_ovt_caption_init_scale
    )
    model.config.pgot_ovt_isolated_attention = bool(
        model_args.pgot_ovt_isolated_attention
    )
    model.config.pgot_dataset_format = str(data_args.dataset_format)
    model.config.pgot_v12_enable = bool(model_args.pgot_v12_enable)
    model.config.pgot_v12_layers = str(model_args.pgot_v12_layers)
    model.config.pgot_v12_ovt_temperature = v12_ovt_temp
    model.config.pgot_v12_slot_temperature = v12_ovt_temp
    model.config.pgot_v12_owner_temperature = float(model_args.pgot_v12_owner_temperature)
    model.config.pgot_v12_owner_weight = float(model_args.pgot_v12_owner_weight)
    model.config.pgot_v14_enable = bool(model_args.pgot_v14_enable)
    model.config.pgot_v14_route_temperature = float(model_args.pgot_v14_route_temperature)
    model.config.pgot_v14_route_weight = float(model_args.pgot_v14_route_weight)
    model.config.pgot_v14_void_weight = float(model_args.pgot_v14_void_weight)
    model.config.pgot_v14_position_weight = float(model_args.pgot_v14_position_weight)
    model.config.pgot_v14_router_depth = int(model_args.pgot_v14_router_depth)
    model.config.pgot_v14_router_mlp_ratio = int(model_args.pgot_v14_router_mlp_ratio)
    model.config.pgot_dit_ovt_cross_attn_enable = bool(model_args.pgot_dit_ovt_cross_attn_enable)
    model.config.pgot_dit_ovt_cross_attn_start_block = int(model_args.pgot_dit_ovt_cross_attn_start_block)
    model.config.pgot_dit_ovt_cross_attn_every_n_blocks = int(model_args.pgot_dit_ovt_cross_attn_every_n_blocks)
    model.config.pgot_v17_enable = bool(model_args.pgot_v17_enable)
    model.config.pgot_v17_ownership_weight = float(model_args.pgot_v17_ownership_weight)
    model.config.pgot_v17_ownership_layers = str(model_args.pgot_v17_ownership_layers)
    model.config.pgot_v21_enable = bool(model_args.pgot_v21_enable)
    model.config.pgot_v21_ground_weight = float(model_args.pgot_v21_ground_weight)
    model.config.pgot_v21_ground_final_weight = float(model_args.pgot_v21_ground_final_weight)
    model.config.pgot_v21_ground_anneal_steps = int(model_args.pgot_v21_ground_anneal_steps)
    model.config.pgot_v21_temperature = float(model_args.pgot_v21_temperature)
    model.config.pgot_v21_position_weight = float(model_args.pgot_v21_position_weight)
    model.config.pgot_v21_code_dim = int(model_args.pgot_v21_code_dim)
    model.config.pgot_v22_attention_competition_weight = float(
        model_args.pgot_v22_attention_competition_weight
    )
    model.config.pgot_v22_attention_competition_layers = str(
        model_args.pgot_v22_attention_competition_layers
    )
    model.config.pgot_v22_attention_competition_temperature = float(
        model_args.pgot_v22_attention_competition_temperature
    )
    model.config.pgot_v22_attention_competition_include_void = bool(
        model_args.pgot_v22_attention_competition_include_void
    )
    model.config.pgot_v22_attention_competition_bg_weight = float(
        model_args.pgot_v22_attention_competition_bg_weight
    )
    model.config.pgot_latent_distill_enable = bool(model_args.pgot_latent_distill_enable)
    model.config.pgot_latent_distill_weight = float(model_args.pgot_latent_distill_weight)
    model.config.pgot_latent_distill_mse_weight = float(model_args.pgot_latent_distill_mse_weight)
    model.config.pgot_latent_distill_cos_weight = float(model_args.pgot_latent_distill_cos_weight)
    model.config.pgot_latent_distill_l1_weight = float(model_args.pgot_latent_distill_l1_weight)
    model.config.pgot_cfg_drop_rate = float(model_args.pgot_cfg_drop_rate)
    model.config.pgot_rae_attends_caption = bool(model_args.pgot_rae_attends_caption)
    model.config.pgot_unfreeze_mm_projector = bool(model_args.pgot_unfreeze_mm_projector)
    model.config.image_preprocess_mode = str(data_args.image_preprocess_mode)
    model.config.coda_crop_size = int(data_args.coda_crop_size)
    logger.info(
        f"[PGOT] mask loss weights -> ce={model.config.pgot_mask_ce_weight} "
        f"fg={model.config.pgot_mask_fg_weight} outside={model.config.pgot_mask_outside_weight} "
        f"ce_aux={model.config.pgot_mask_aux_competition_weight} "
        f"bce={model.config.pgot_mask_bce_weight} "
        f"sigmoid_out={model.config.pgot_mask_sigmoid_outside_weight} "
        f"reg_fg_suppress={model.config.pgot_register_foreground_suppression_weight} "
        f"obj_bal_bce={model.config.pgot_mask_object_balanced_bce_weight} "
        f"tversky={model.config.pgot_mask_tversky_weight} "
        f"spatial_out={model.config.pgot_mask_spatial_outside_weight} "
        f"(spatial_temp={model.config.pgot_mask_spatial_temperature}) "
        f"spatial_out_log={model.config.pgot_mask_spatial_outside_log_weight} "
        f"(spatial_log_temp={model.config.pgot_mask_spatial_outside_log_temperature}) "
        f"llm_qk_out={model.config.pgot_mask_llm_qk_outside_weight} "
        f"(llm_qk_temp={model.config.pgot_mask_llm_qk_outside_temperature}, "
        f"llm_qk_layers={model.config.pgot_mask_llm_qk_outside_layers}) "
        f"llm_attn_out={model.config.pgot_mask_llm_attention_outside_weight} "
        f"(llm_attn_layers={model.config.pgot_mask_llm_attention_outside_layers}, "
        f"void_w={model.config.pgot_mask_llm_attention_void_weight}) "
        f"llm_patch_out={model.config.pgot_mask_llm_patch_outside_weight} "
        f"(llm_patch_layers={model.config.pgot_mask_llm_patch_outside_layers}, "
        f"temp={model.config.pgot_mask_llm_patch_outside_temperature}, "
        f"void_w={model.config.pgot_mask_llm_patch_void_weight}, "
        f"image_use_w={model.config.pgot_mask_llm_image_use_weight}, "
        f"image_use_margin={model.config.pgot_mask_llm_image_use_margin}) "
        f"core_out={model.config.pgot_core_outside_weight} "
        f"(layers={model.config.pgot_core_outside_layers}, "
        f"temp={model.config.pgot_core_outside_temperature}, "
        f"void_w={model.config.pgot_core_void_weight}, "
        f"register_w={model.config.pgot_core_register_outside_weight}, "
        f"register_hard_gt={model.config.pgot_register_hard_gt_mask}, "
        f"register_hard_gt_eval={model.config.pgot_register_hard_gt_mask_eval}, "
        f"register_hard_threshold={model.config.pgot_register_hard_gt_mask_threshold}, "
        f"tail_w={model.config.pgot_core_tail_weight}, "
        f"tail_frac={model.config.pgot_core_tail_fraction}) "
        f"ovt=(caption_init={model.config.pgot_ovt_caption_init}, "
        f"isolated_attention={model.config.pgot_ovt_isolated_attention}) "
        f"e3_comp=(w={model.config.pgot_e3_attention_competition_weight}, "
        f"layers={model.config.pgot_e3_attention_competition_layers}, "
        f"temp={model.config.pgot_e3_attention_competition_temperature}, "
        f"bg_w={model.config.pgot_e3_attention_competition_bg_weight}) "
        f"e4=(rae_isolated={model.config.pgot_e4_rae_isolated}, "
        f"full_inside_w={model.config.pgot_e4_full_inside_weight}, "
        f"full_inside_target={model.config.pgot_e4_full_inside_target}, "
        f"rae_bind_w={model.config.pgot_e4_rae_bind_weight}, "
        f"rae_bind_layers={model.config.pgot_e4_rae_bind_layers}) "
        f"v12={bool(getattr(model.config, 'pgot_v12_enable', False))} "
        f"(layers={getattr(model.config, 'pgot_v12_layers', '12,16,20,24')}, "
        f"ovt_temp={getattr(model.config, 'pgot_v12_ovt_temperature', getattr(model.config, 'pgot_v12_slot_temperature', 1.0))}, "
        f"owner_temp={getattr(model.config, 'pgot_v12_owner_temperature', 1.0)}, "
        f"owner_w={getattr(model.config, 'pgot_v12_owner_weight', 1.0)}) "
        f"v14={bool(getattr(model.config, 'pgot_v14_enable', False))} "
        f"(route_temp={getattr(model.config, 'pgot_v14_route_temperature', 1.0)}, "
        f"route_w={getattr(model.config, 'pgot_v14_route_weight', 1.0)}, "
        f"void_w={getattr(model.config, 'pgot_v14_void_weight', 0.5)}, "
        f"pos_w={getattr(model.config, 'pgot_v14_position_weight', 1.0)}, "
        f"router_depth={getattr(model.config, 'pgot_v14_router_depth', 1)}, "
        f"router_mlp={getattr(model.config, 'pgot_v14_router_mlp_ratio', 4)}, "
        f"dit_ovt_xattn={getattr(model.config, 'pgot_dit_ovt_cross_attn_enable', False)}, "
        f"xattn_start={getattr(model.config, 'pgot_dit_ovt_cross_attn_start_block', 25)}, "
        f"xattn_every={getattr(model.config, 'pgot_dit_ovt_cross_attn_every_n_blocks', 1)}) "
        f"v17={bool(getattr(model.config, 'pgot_v17_enable', False))} "
        f"(ownership_w={getattr(model.config, 'pgot_v17_ownership_weight', 0.0)}, "
        f"ownership_layers={getattr(model.config, 'pgot_v17_ownership_layers', 'last4')}) "
        f"v21={bool(getattr(model.config, 'pgot_v21_enable', False))} "
        f"(ground_w={getattr(model.config, 'pgot_v21_ground_weight', 0.0)}, "
        f"ground_final={getattr(model.config, 'pgot_v21_ground_final_weight', -1.0)}, "
        f"anneal={getattr(model.config, 'pgot_v21_ground_anneal_steps', 0)}, "
        f"temp={getattr(model.config, 'pgot_v21_temperature', 1.0)}, "
        f"pos_w={getattr(model.config, 'pgot_v21_position_weight', 1.0)}, "
        f"code_dim={getattr(model.config, 'pgot_v21_code_dim', 0)}) "
        f"v22a=(attn_comp_w={getattr(model.config, 'pgot_v22_attention_competition_weight', 0.0)}, "
        f"attn_comp_layers={getattr(model.config, 'pgot_v22_attention_competition_layers', '26,27')}, "
        f"attn_comp_temp={getattr(model.config, 'pgot_v22_attention_competition_temperature', 1.0)}, "
        f"attn_comp_void={getattr(model.config, 'pgot_v22_attention_competition_include_void', False)}) "
        f"(ce_temp={model.config.pgot_mask_ce_temperature}, ce_merge={model.config.pgot_mask_ce_merge}); "
        f"null_bg={model.config.pgot_use_null_bg_competition}; cfg_drop={model.config.pgot_cfg_drop_rate}"
    )
    logger.info("[PGOT] model checkpoint loaded")

    # ---- Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = 151643

    # ---- Add new special tokens
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": NEW_SPECIAL_TOKENS})
    if num_added > 0:
        # Resize embedding/lm_head to accommodate new tokens
        old_vocab = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(len(tokenizer))
        new_vocab = model.get_input_embeddings().weight.shape[0]
        # Initialize new rows
        with torch.no_grad():
            init_std = 0.02
            for tok in NEW_SPECIAL_TOKENS:
                tid = tokenizer.convert_tokens_to_ids(tok)
                model.get_input_embeddings().weight[tid].normal_(mean=0.0, std=init_std)
                if hasattr(model, "lm_head") and model.lm_head is not None:
                    model.lm_head.weight[tid].normal_(mean=0.0, std=init_std)
        logger.info(f"[PGOT] vocab resized {old_vocab} -> {new_vocab} ({num_added} new specials)")

    _register_ovt_token_ids(model, tokenizer)
    _register_template_token_ids(model, tokenizer, data_args.dataset_format)

    # ---- Initialize vision tower + diffusion head
    if parsed_towers is not None:
        model_args.vision_tower_aux_list = parsed_towers
        model_args.vision_tower_aux_token_len_list = parsed_token_lens
        model_args.unfreeze_mm_vision_tower = training_args.pgot_lora_enable and False  # keep frozen
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
        model.load_vision_head(model_args=model_args)
        logger.info("[PGOT] vision tower + diffusion head initialized")

        vt_list = model.get_vision_tower_aux_list()
        for vt in vt_list:
            vt.to(dtype=compute_dtype, device=training_args.device)

        data_args.image_processor_aux_list = [vt.image_processor for vt in vt_list]
        data_args.is_multimodal = True
        data_args.vision_tower_aux_token_len_list = parsed_token_lens
        model.config.image_aspect_ratio = model_args.image_aspect_ratio
        model.config.vision_tower_aux_token_len_list = parsed_token_lens
        model.config.image_feature_token_len = config.image_feature_token_len
        model.config.diffusion_target_token_len = config.diffusion_target_token_len
        model.config.si_token_len = model_args.si_token_len
        model.config.miv_token_len = model_args.miv_token_len

    # ---- Inject LoRA
    if bool(training_args.pgot_lora_enable):
        from peft import LoraConfig, inject_adapter_in_model
        lora_targets = [t.strip() for t in training_args.pgot_lora_target_modules.split(",") if t.strip()]
        lora_config = LoraConfig(
            r=int(training_args.pgot_lora_r),
            lora_alpha=int(training_args.pgot_lora_alpha),
            lora_dropout=float(training_args.pgot_lora_dropout),
            bias="none",
            target_modules=lora_targets,
            task_type="CAUSAL_LM",
        )
        inject_adapter_in_model(lora_config, model, adapter_name="default")
        n_lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
        logger.info(
            f"[PGOT/LoRA] injected r={training_args.pgot_lora_r} α={training_args.pgot_lora_alpha} "
            f"targets={lora_targets} | lora_params={n_lora_params:,}"
        )
        # from_pretrained() cannot match PEFT's base_layer/lora_ keys before
        # adapters exist. Reload those keys after injection so structural
        # variants such as V17 can warm-start from a prior PGOT LoRA checkpoint
        # with a fresh optimizer.
        try:
            import glob
            from safetensors import safe_open

            lora_reload_sd = {}
            shard_files = sorted(glob.glob(os.path.join(model_args.model_name_or_path, "*.safetensors")))
            for shard_file in shard_files:
                with safe_open(shard_file, framework="pt", device="cpu") as sf:
                    for key in sf.keys():
                        if "base_layer" in key or "lora_" in key:
                            lora_reload_sd[key] = sf.get_tensor(key)
            if lora_reload_sd:
                missing_keys, unexpected_keys = model.load_state_dict(lora_reload_sd, strict=False)
                lora_missing = [k for k in missing_keys if "base_layer" in k or "lora_" in k]
                logger.info(
                    "[PGOT/LoRA] reloaded checkpoint adapter weights | keys=%d missing_lora=%d unexpected=%d",
                    len(lora_reload_sd),
                    len(lora_missing),
                    len(unexpected_keys),
                )
        except Exception as exc:
            logger.warning("[PGOT/LoRA] checkpoint adapter reload skipped: %s", exc)

    # ---- Apply freeze policy
    model = freeze_for_pgot(
        model,
        freeze_dit_body=bool(model_args.freeze_dit_body),
        freeze_vision_tower=bool(model_args.freeze_vision_tower),
        unfreeze_mm_projector=bool(model_args.pgot_unfreeze_mm_projector),
        dit_unfreeze_last_n_blocks=int(getattr(model_args, "pgot_dit_unfreeze_last_n_blocks", 0)),
    )

    # ---- Move to device (vision tower already moved above)
    model.to(training_args.device)
    if parsed_towers is not None:
        for vt in model.get_vision_tower_aux_list():
            vt.to(dtype=compute_dtype, device=training_args.device)

    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # ---- Datasets
    rae_token_len = int(getattr(model.config, "diffusion_target_token_len", 256))
    rae_grid_size = int(round(rae_token_len ** 0.5))
    if rae_grid_size * rae_grid_size != rae_token_len:
        raise ValueError(
            "E4 RAE ownership requires a square diffusion target grid, got "
            f"diffusion_target_token_len={rae_token_len}"
        )
    train_dataset = Pix2CapPGOTDataset(
        jsonl_path=data_args.train_jsonl,
        tokenizer=tokenizer,
        image_processor=data_args.image_processor_aux_list[0],
        target_image_processor=(
            data_args.image_processor_aux_list[1]
            if len(data_args.image_processor_aux_list) > 1
            else data_args.image_processor_aux_list[0]
        ),
        grid_size=data_args.grid_size,
        rae_grid_size=rae_grid_size,
        max_caption_tokens=data_args.max_caption_tokens,
        n_ovt_per_object=model_args.pgot_n_ovt_per_object,
        max_objects=model_args.pgot_max_objects,
        panoptic_categories_json="/home/jovyan/data/coco/annotations/panoptic_train2017.json",
        image_preprocess_mode=data_args.image_preprocess_mode,
        coda_crop_size=data_args.coda_crop_size,
    )
    val_dataset = Pix2CapPGOTDataset(
        jsonl_path=data_args.val_jsonl,
        tokenizer=tokenizer,
        image_processor=data_args.image_processor_aux_list[0],
        target_image_processor=(
            data_args.image_processor_aux_list[1]
            if len(data_args.image_processor_aux_list) > 1
            else data_args.image_processor_aux_list[0]
        ),
        grid_size=data_args.grid_size,
        rae_grid_size=rae_grid_size,
        max_caption_tokens=data_args.max_caption_tokens,
        n_ovt_per_object=model_args.pgot_n_ovt_per_object,
        max_objects=model_args.pgot_max_objects,
        panoptic_categories_json="/home/jovyan/data/coco/annotations/panoptic_val2017.json",
        image_preprocess_mode=data_args.image_preprocess_mode,
        coda_crop_size=data_args.coda_crop_size,
    )
    # Cap eval size
    if len(val_dataset) > data_args.eval_num_images:
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(data_args.eval_num_images)))

    collator = PGOTDataCollator(pad_token_id=tokenizer.pad_token_id)

    logger.info(f"[PGOT] datasets ready | train={len(train_dataset)} | val={len(val_dataset)}")

    # ---- Optional: batch-size sweep mode
    if os.environ.get("PGOT_TUNE_BATCH"):
        import gc, traceback
        candidates = [int(x) for x in os.environ["PGOT_TUNE_BATCH"].split(",")]
        print(f"\n{'='*60}\n[PGOT/Tune] Trying per_device_train_batch_size = {candidates}\n{'='*60}\n")
        model.train()
        device = training_args.device
        results = []
        for bs in candidates:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            try:
                samples = [train_dataset[i] for i in range(bs)]
                batch = collator(samples)
                outputs = model(
                    images=batch["images"].to(device),
                    target_images=batch["target_images"].to(device),
                    caption_input_ids=batch["caption_input_ids"].to(device),
                    caption_attention_mask=batch["caption_attention_mask"].to(device),
                    caption_labels=batch["caption_labels"].to(device),
                    ovt_positions_in_caption=batch["ovt_positions_in_caption"].to(device),
                    ovt_valid_mask=batch["ovt_valid_mask"].to(device),
                    ovt_is_thing=batch["ovt_is_thing"].to(device),
                    gt_masks_per_ovt=batch["gt_masks_per_ovt"].to(device),
                    gt_rae_masks_per_ovt=batch["gt_rae_masks_per_ovt"].to(device),
                    pgot_contrastive_weight=0.0,
                )
                outputs.loss.backward()
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"  BS={bs:3d}  OK   peak_VRAM={peak:6.2f} GB  loss={float(outputs.loss):.3f}")
                results.append((bs, "OK", peak))
                model.zero_grad(set_to_none=True)
                del outputs, batch, samples
            except torch.cuda.OutOfMemoryError:
                print(f"  BS={bs:3d}  OOM")
                results.append((bs, "OOM", None))
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                gc.collect()
                break
            except Exception as e:
                traceback.print_exc()
                print(f"  BS={bs:3d}  ERROR  {type(e).__name__}: {e}")
                results.append((bs, "ERROR", None))
                break
        print("\n" + "=" * 60)
        print("[PGOT/Tune] Summary")
        print("=" * 60)
        last_ok_bs = None
        last_ok_peak = None
        for bs, status, peak in results:
            if status == "OK":
                last_ok_bs = bs
                last_ok_peak = peak
                print(f"  BS={bs:3d}  OK    peak={peak:.2f} GB")
            else:
                print(f"  BS={bs:3d}  {status}")
        if last_ok_bs is not None:
            print(f"\n  ★ Recommended per_device_train_batch_size = {last_ok_bs}  (~{last_ok_peak:.1f} GB / GPU)")
        return

    # ---- Trainer
    trainer = PGOTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # ---- Device sanity check log
    def _first_param_device(m):
        try:
            return str(next(m.parameters()).device)
        except StopIteration:
            return "none"
    inner = model
    logger.info("[PGOT/device-check]")
    logger.info(f"  backbone: {_first_param_device(inner.model)}")
    logger.info(f"  lm_head:  {_first_param_device(inner.lm_head)}")
    logger.info(f"  diff_head: {_first_param_device(inner.diff_head)}")
    logger.info(f"  diff_head_projector: {_first_param_device(inner.diff_head_projector)}")
    logger.info(f"  pgot_register: {inner.pgot_register_embeddings.device}")
    logger.info(f"  latent_queries: {inner.get_model().latent_queries.device}")
    if parsed_towers is not None:
        for i, vt in enumerate(model.get_vision_tower_aux_list()):
            logger.info(f"  vision_tower[{i}]: {_first_param_device(vt)}")

    # ---- Train
    resume = training_args.resume_from_checkpoint or None
    trainer.train(resume_from_checkpoint=resume)

    if os.environ.get("PGOT_SKIP_FINAL_SAVE", "").strip().lower() not in {"1", "true", "yes"}:
        trainer.save_state()
        trainer.save_model(training_args.output_dir)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()
