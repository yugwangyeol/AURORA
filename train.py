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
import math
import os
import sys
from functools import partial

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
    PGOTParameterUpdateAuditCallback,
    PGOTTrainer,
    freeze_for_pgot,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pgot.train")


def _reset_one_shot_scratch_modules(model) -> None:
    """Deterministically initialize PGOT modules absent from Scale-RAE base."""
    hidden_size = int(model.config.hidden_size)

    def _normal(parameter, seed, std=1.0):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        value = torch.randn(
            tuple(parameter.shape), generator=generator, dtype=torch.float32
        ) * float(std)
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    def _xavier(parameter, seed):
        fan_out, fan_in = parameter.shape
        bound = math.sqrt(6.0 / float(fan_in + fan_out))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        value = torch.empty(tuple(parameter.shape), dtype=torch.float32).uniform_(
            -bound, bound, generator=generator
        )
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    def _identity(linear):
        linear.weight.zero_()
        diagonal = min(linear.weight.shape)
        indices = torch.arange(diagonal, device=linear.weight.device)
        linear.weight[indices, indices] = 1
        if linear.bias is not None:
            linear.bias.zero_()

    def _reset_norm(norm):
        if getattr(norm, "weight", None) is not None:
            norm.weight.fill_(1.0)
        if getattr(norm, "bias", None) is not None:
            norm.bias.zero_()

    writer = model.pgot_e8_writer
    reader = model.pgot_e8_reader
    with torch.no_grad():
        _normal(model.pgot_register_embeddings, 7001, 1.0 / math.sqrt(hidden_size))
        if getattr(model, "pgot_ovt_caption_norm", None) is not None:
            _reset_norm(model.pgot_ovt_caption_norm)
            _identity(model.pgot_ovt_caption_projector)

        _reset_norm(writer.semantic_norm)
        _reset_norm(writer.image_norm)
        _reset_norm(writer.raw_value_norm)
        _identity(writer.query)
        _identity(writer.key)
        _xavier(writer.raw_value.weight, 7002)
        _normal(writer.memory_id_embeddings, 7003)

        for norm in (
            reader.query_norm,
            reader.semantic_norm,
            reader.memory_norm,
            reader.output_norm,
        ):
            _reset_norm(norm)
        for linear in (reader.query, reader.key, reader.value, reader.output):
            _identity(linear)
        if hasattr(reader, "content_key"):
            _identity(reader.content_key)
        if hasattr(reader, "memory_key_embeddings"):
            _normal(reader.memory_key_embeddings, 7004)

    logger.info(
        "[PGOT/Memory] deterministic scratch initialization complete "
        "(register, caption projector, Writer, Reader)"
    )


def _manual_fsdp_wrap(model, training_args):
    """Wrap before optimizer creation to preserve PGOT's per-group learning rates.

    Accelerate 0.23 rebuilds an optimizer with a single parameter group when it
    performs FSDP wrapping itself.  ``use_orig_params=True`` lets us wrap first,
    then let PGOTTrainer construct the grouped optimizer from original names.
    """
    if not training_args.fsdp:
        return model
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        checkpoint_wrapper,
    )
    config = dict(training_args.fsdp_config or {})
    class_names = set(
        config.get(
            "transformer_layer_cls_to_wrap",
            ["Qwen2DecoderLayer", "LightningDiTBlock"],
        )
    )
    layer_classes = {
        type(module) for module in model.modules()
        if module.__class__.__name__ in class_names
    }
    missing_classes = class_names - {cls.__name__ for cls in layer_classes}
    if missing_classes:
        raise ValueError(
            "FSDP transformer classes not found: " + ", ".join(sorted(missing_classes))
        )
    if training_args.bf16:
        dtype = torch.bfloat16
    elif training_args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    mixed_precision = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )

    def _as_bool(value, default):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes"}

    backward_prefetch_name = str(
        config.get(
            "backward_prefetch",
            config.get("fsdp_backward_prefetch", "backward_post"),
        )
    ).strip().lower()
    backward_prefetch = {
        "backward_pre": BackwardPrefetch.BACKWARD_PRE,
        "backward_post": BackwardPrefetch.BACKWARD_POST,
        "none": None,
        "null": None,
    }.get(backward_prefetch_name)
    if backward_prefetch_name not in {
        "backward_pre", "backward_post", "none", "null"
    }:
        raise ValueError(
            f"Unsupported fsdp_backward_prefetch={backward_prefetch_name!r}"
        )

    activation_checkpointing = bool(training_args.gradient_checkpointing)

    def _checkpoint(module):
        if not activation_checkpointing:
            return module
        return checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )

    fsdp_common = {
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "backward_prefetch": backward_prefetch,
        "forward_prefetch": _as_bool(config.get("forward_prefetch"), False),
        "use_orig_params": _as_bool(config.get("use_orig_params"), True),
        "sync_module_states": _as_bool(config.get("sync_module_states"), True),
        "limit_all_gathers": _as_bool(config.get("limit_all_gathers"), True),
        "device_id": training_args.device,
    }

    split_dit_fp32 = bool(
        getattr(model.config, "pgot_diffusion_force_fp32", True)
    )
    if split_dit_fp32:
        dit_model = getattr(getattr(model, "diff_head", None), "model", None)
        if dit_model is None or not hasattr(dit_model, "dit_blocks"):
            raise ValueError("FP32 DiT/FSDP split requested but DiT blocks are missing")
        # Promote the complete DiT before any child is wrapped.  The patch/time/
        # condition embedders and final layer participate in the same forward as
        # the transformer blocks, so leaving those modules in the root BF16 FSDP
        # unit creates float-input/BF16-bias mismatches at the DiT boundary.
        dit_model.float()
        fp32_policy = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        for index, block in enumerate(dit_model.dit_blocks):
            block.float()
            dit_model.dit_blocks[index] = FSDP(
                _checkpoint(block),
                mixed_precision=fp32_policy,
                **fsdp_common,
            )
        # A second, hierarchical FSDP unit owns the non-block DiT parameters.
        # Its already wrapped blocks remain independent shards, keeping the
        # 9.8B DiT within the two 90-GB GPUs' memory budget.
        model.diff_head.model = FSDP(
            dit_model,
            mixed_precision=fp32_policy,
            **fsdp_common,
        )
        # Prevent the legacy first-forward conversion from mutating FSDP flat
        # parameters. The blocks have already been promoted before wrapping.
        model.set_to_fp32 = True
        layer_classes = {
            cls for cls in layer_classes if cls.__name__ != "LightningDiTBlock"
        }
        logger.info(
            "[PGOT/FSDP] complete DiT uses hierarchical FP32 master/compute shards"
        )

    qwen_model = model.get_model()
    if not hasattr(qwen_model, "layers"):
        raise ValueError("FSDP expected Qwen decoder layers")
    for index, layer in enumerate(qwen_model.layers):
        qwen_model.layers[index] = FSDP(
            _checkpoint(layer),
            mixed_precision=mixed_precision,
            **fsdp_common,
        )

    if not split_dit_fp32:
        dit_model = getattr(getattr(model, "diff_head", None), "model", None)
        if dit_model is None or not hasattr(dit_model, "dit_blocks"):
            raise ValueError("FSDP expected DiT blocks")
        for index, block in enumerate(dit_model.dit_blocks):
            dit_model.dit_blocks[index] = FSDP(
                _checkpoint(block),
                mixed_precision=mixed_precision,
                **fsdp_common,
            )

    if activation_checkpointing:
        # Qwen and LightningDiT normally checkpoint *around* an already FSDP-
        # wrapped layer. That retains unsharded parameters during recompute and
        # exhausts a 90-GB MIG slice in full FP32. The wrappers above place
        # non-reentrant checkpointing inside each FSDP unit instead.
        qwen_model.gradient_checkpointing = False
        if dit_model is not None:
            dit_model.gradient_checkpointing = False
        training_args.gradient_checkpointing = False

    wrapped = FSDP(
        model,
        mixed_precision=mixed_precision,
        **fsdp_common,
    )
    logger.info(
        "[PGOT/FSDP] manually wrapped full-shard model before optimizer "
        "(manual Qwen=%d, DiT=%d, split_dit_fp32=%s, activation_checkpointing=%s, "
        "backward_prefetch=%s, use_orig_params=True)",
        len(qwen_model.layers),
        len(dit_model.dit_blocks),
        split_dit_fp32,
        activation_checkpointing,
        backward_prefetch_name,
    )
    return wrapped


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
    if (
        training_args.gradient_checkpointing
        and training_args.fsdp
        and training_args.gradient_checkpointing_kwargs is None
    ):
        # Transformers 4.37 cannot parse Optional[dict] from the CLI. Set the
        # FSDP-safe implementation directly instead of falling back to the
        # legacy reentrant checkpoint path.
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        logger.info(
            "[PGOT/FSDP] activation checkpointing uses use_reentrant=False"
        )
    one_shot_memory = bool(model_args.pgot_one_shot_reader_enable) and (
        model_args.pgot_one_shot_readout_mode in {"memory_content", "memory_id"}
    )
    if one_shot_memory and training_args.pgot_log_metric_profile == "all":
        training_args.pgot_log_metric_profile = "one_shot_memory"

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
    # Keep the source-checkpoint architecture before CLI overrides.  HF's
    # missing-key initializer zeros standalone Parameters that do not exist in
    # E10-R, so the first E11 run must explicitly restore its deterministic
    # memory-ID symmetry breakers after from_pretrained().
    source_checkpoint_is_e11 = bool(
        getattr(config, "pgot_e11_dual_m4_enable", False)
    )
    source_one_shot_memory = bool(getattr(config, "pgot_one_shot_reader_enable", False)) and (
        getattr(config, "pgot_one_shot_readout_mode", "") in {"memory_content", "memory_id"}
    )
    source_one_shot_mode = getattr(config, "pgot_one_shot_readout_mode", "")
    source_rae_query_adapter = bool(
        getattr(config, "pgot_one_shot_rae_query_adapter_enable", False)
    )
    source_rae_query_adapter_bottleneck = int(
        getattr(config, "pgot_one_shot_rae_query_adapter_bottleneck", 0)
    )
    source_memory_value_source = str(
        getattr(config, "pgot_one_shot_memory_value_source", "siglip")
    ).strip().lower()
    scratch_one_shot = bool(model_args.pgot_one_shot_scratch_init)
    if scratch_one_shot and (source_checkpoint_is_e11 or source_one_shot_memory):
        raise ValueError(
            "pgot_one_shot_scratch_init is only valid for a vanilla Scale-RAE source checkpoint"
        )
    if one_shot_memory and not (
        source_checkpoint_is_e11 or source_one_shot_memory or scratch_one_shot
    ):
        raise ValueError(
            "one-shot memory must initialize from E11/one-shot, or explicitly set "
            "pgot_one_shot_scratch_init=True for a cold-start experiment"
        )
    if one_shot_memory and source_one_shot_memory and source_one_shot_mode != model_args.pgot_one_shot_readout_mode:
        raise ValueError("For an ID/content comparison initialize both from E11; continue one-shot checkpoints in their original mode")
    source_legacy_memories = int(
        getattr(config, "pgot_e11_memories_per_owner", 1)
    )
    source_object_memories = int(
        getattr(
            config,
            "pgot_e11_object_memories_per_owner",
            source_legacy_memories,
        )
    )
    source_register_memories = int(
        getattr(
            config,
            "pgot_e11_register_memories_per_owner",
            source_legacy_memories,
        )
    )
    source_checkpoint_is_e12 = bool(
        getattr(config, "pgot_e12_centroid_reader_enable", False)
    )
    source_reader_num_layers = int(
        getattr(config, "pgot_e8_reader_num_layers", 1)
    )
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
    config.pgot_diffusion_force_fp32 = bool(model_args.pgot_diffusion_force_fp32)
    config.use_aurora = False
    config.use_captionslot = False
    config.use_pgot = True
    config.pgot_dataset_format = str(data_args.dataset_format)
    config.pgot_one_shot_scratch_init = scratch_one_shot

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
    config.pgot_ovt_attends_own_caption = bool(
        model_args.pgot_ovt_attends_own_caption
    )
    config.pgot_e5_forcing_probability = float(
        model_args.pgot_e5_forcing_probability
    )
    config.pgot_fvw_enable = bool(model_args.pgot_fvw_enable)
    config.pgot_fvw_layers = str(model_args.pgot_fvw_layers)
    config.pgot_fvw_num_heads = int(model_args.pgot_fvw_num_heads)
    config.pgot_fvw_temperature = float(model_args.pgot_fvw_temperature)
    config.pgot_fvw_write_outside_weight = float(
        model_args.pgot_fvw_write_outside_weight
    )
    config.pgot_fvw_full_probability = float(
        model_args.pgot_fvw_full_probability
    )
    config.pgot_fvw_ovt_only_probability = float(
        model_args.pgot_fvw_ovt_only_probability
    )
    config.pgot_e6_enable = bool(model_args.pgot_e6_enable)
    config.pgot_e6_model_id = str(model_args.pgot_e6_model_id)
    config.pgot_e6_coda_unet_checkpoint = str(
        model_args.pgot_e6_coda_unet_checkpoint
    )
    config.pgot_e6_context_dim = int(model_args.pgot_e6_context_dim)
    config.pgot_e6_cfg_drop_rate = float(model_args.pgot_e6_cfg_drop_rate)
    config.pgot_e6_attention_grid_size = int(
        model_args.pgot_e6_attention_grid_size
    )
    config.pgot_e7_enable = bool(model_args.pgot_e7_enable)
    config.pgot_e7_model_id = str(model_args.pgot_e7_model_id)
    config.pgot_e7_context_dim = int(model_args.pgot_e7_context_dim)
    config.pgot_e7_router_dim = int(model_args.pgot_e7_router_dim)
    config.pgot_e7_owner_num_heads = int(model_args.pgot_e7_owner_num_heads)
    config.pgot_e7_owner_temperature = float(model_args.pgot_e7_owner_temperature)
    config.pgot_e7_owner_weight = float(model_args.pgot_e7_owner_weight)
    config.pgot_e7_owner_bg_weight = float(model_args.pgot_e7_owner_bg_weight)
    config.pgot_e7_cfg_drop_rate = float(model_args.pgot_e7_cfg_drop_rate)
    config.pgot_e7_unet_lora_rank = int(model_args.pgot_e7_unet_lora_rank)
    config.pgot_e7_unet_lora_alpha = float(model_args.pgot_e7_unet_lora_alpha)
    config.pgot_e7_causal_weight = float(model_args.pgot_e7_causal_weight)
    config.pgot_e7_invariance_weight = float(model_args.pgot_e7_invariance_weight)
    config.pgot_e7_causal_min_timestep = int(
        model_args.pgot_e7_causal_min_timestep
    )
    config.pgot_e7_causal_margin = float(model_args.pgot_e7_causal_margin)
    config.pgot_e7_causal_temperature = float(
        model_args.pgot_e7_causal_temperature
    )
    config.pgot_e8_visual_memory_enable = bool(
        model_args.pgot_e8_visual_memory_enable
    )
    config.pgot_one_shot_reader_enable = bool(
        model_args.pgot_one_shot_reader_enable
    )
    config.pgot_one_shot_readout_mode = str(
        model_args.pgot_one_shot_readout_mode
    )
    config.pgot_one_shot_detach_owner_routing = bool(
        model_args.pgot_one_shot_detach_owner_routing
    )
    config.pgot_one_shot_owner_gradient_ramp_steps = int(
        model_args.pgot_one_shot_owner_gradient_ramp_steps
    )
    config.pgot_one_shot_writer_softmax_axis = str(
        model_args.pgot_one_shot_writer_softmax_axis
    )
    config.pgot_one_shot_writer_owner_prior = bool(
        model_args.pgot_one_shot_writer_owner_prior
    )
    config.pgot_one_shot_direct_rae_query = bool(
        model_args.pgot_one_shot_direct_rae_query
    )
    config.pgot_one_shot_rae_query_adapter_enable = bool(
        model_args.pgot_one_shot_rae_query_adapter_enable
    )
    config.pgot_one_shot_rae_query_adapter_bottleneck = int(
        model_args.pgot_one_shot_rae_query_adapter_bottleneck
    )
    config.pgot_one_shot_memory_contrastive_enable = bool(
        model_args.pgot_one_shot_memory_contrastive_enable
    )
    config.pgot_one_shot_memory_value_source = str(
        model_args.pgot_one_shot_memory_value_source
    ).strip().lower()
    config.pgot_one_shot_memory_value_dim = int(
        model_args.pgot_one_shot_memory_value_dim
    )
    config.pgot_one_shot_memory_contrastive_target_weight = float(
        training_args.pgot_contrastive_loss_target_weight
    )
    config.pgot_one_shot_memory_contrastive_warmup_steps = int(
        training_args.pgot_contrastive_warmup_steps
    )
    config.pgot_e8_layers = str(model_args.pgot_e8_layers)
    config.pgot_e8_owner_temperature = float(
        model_args.pgot_e8_owner_temperature
    )
    config.pgot_e8_owner_weight = float(model_args.pgot_e8_owner_weight)
    config.pgot_e8_owner_bg_weight = float(model_args.pgot_e8_owner_bg_weight)
    config.pgot_e8_reader_num_heads = int(model_args.pgot_e8_reader_num_heads)
    config.pgot_e8_reader_num_layers = int(model_args.pgot_e8_reader_num_layers)
    config.pgot_e8_reader_temperature = float(
        model_args.pgot_e8_reader_temperature
    )
    config.pgot_e8_clean_refinement = bool(model_args.pgot_e8_clean_refinement)
    config.pgot_e8_inject_memory = bool(model_args.pgot_e8_inject_memory)
    config.pgot_e8_update_mode = str(model_args.pgot_e8_update_mode)
    config.pgot_e10_raw_value_enable = bool(
        model_args.pgot_e10_raw_value_enable
    )
    config.pgot_e11_dual_m4_enable = bool(
        model_args.pgot_e11_dual_m4_enable
    )
    config.pgot_e11_memories_per_owner = int(
        model_args.pgot_e11_memories_per_owner
    )
    config.pgot_e11_object_memories_per_owner = int(
        model_args.pgot_e11_object_memories_per_owner
        if model_args.pgot_e11_object_memories_per_owner > 0
        else model_args.pgot_e11_memories_per_owner
    )
    config.pgot_e11_register_memories_per_owner = int(
        model_args.pgot_e11_register_memories_per_owner
        if model_args.pgot_e11_register_memories_per_owner > 0
        else model_args.pgot_e11_memories_per_owner
    )
    config.pgot_e11_query_separation_enable = bool(
        model_args.pgot_e11_query_separation_enable
    )
    config.pgot_e12_centroid_reader_enable = bool(
        model_args.pgot_e12_centroid_reader_enable
    )
    config.pgot_e12_centroid_gate_init = float(
        model_args.pgot_e12_centroid_gate_init
    )
    config.pgot_e9_update_dim = int(model_args.pgot_e9_update_dim)
    config.pgot_e9_mlp_ratio = float(model_args.pgot_e9_mlp_ratio)
    config.pgot_e8_reader_supervision_mode = str(
        model_args.pgot_e8_reader_supervision_mode
    )
    config.pgot_e8_reader_object_weight = float(
        model_args.pgot_e8_reader_object_weight
    )
    config.pgot_e8_reader_background_weight = float(
        model_args.pgot_e8_reader_background_weight
    )
    config.pgot_e8_causal_enable = bool(model_args.pgot_e8_causal_enable)
    config.pgot_e8_causal_margin = float(model_args.pgot_e8_causal_margin)
    config.pgot_e8_register_margin = float(model_args.pgot_e8_register_margin)
    config.pgot_e8_need_weight = float(model_args.pgot_e8_need_weight)
    config.pgot_e8_local_weight = float(model_args.pgot_e8_local_weight)
    config.pgot_e8_register_bg_weight = float(model_args.pgot_e8_register_bg_weight)
    config.pgot_e8_register_fg_weight = float(model_args.pgot_e8_register_fg_weight)
    config.pgot_e8_causal_batch_probability = float(
        model_args.pgot_e8_causal_batch_probability
    )
    config.pgot_e8_causal_ramp_steps = int(model_args.pgot_e8_causal_ramp_steps)
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
    config.pgot_dit_soft_routing_enable = bool(model_args.pgot_dit_soft_routing_enable)
    config.pgot_dit_soft_routing_scale = float(model_args.pgot_dit_soft_routing_scale)
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
    config.pgot_latent_distill_ramp_steps = int(model_args.pgot_latent_distill_ramp_steps)
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
    if one_shot_memory and scratch_one_shot:
        _reset_one_shot_scratch_modules(model)
    if bool(model_args.pgot_one_shot_rae_query_adapter_enable) and (
        not source_rae_query_adapter
        or source_rae_query_adapter_bottleneck
        != int(model_args.pgot_one_shot_rae_query_adapter_bottleneck)
    ):
        model.pgot_rae_query_adapter.reset_as_identity()
        logger.info(
            "[PGOT/RAE] initialized direct-query RMSNorm residual MLP "
            "as identity (bottleneck=%d)",
            int(model_args.pgot_one_shot_rae_query_adapter_bottleneck),
        )
    if one_shot_memory and model_args.pgot_one_shot_readout_mode == "memory_content":
        if not source_one_shot_memory:
            # Newly introduced content keys start from the trained E11 key
            # projection; resuming a memory_content checkpoint preserves them.
            with torch.no_grad():
                model.pgot_e8_reader.content_key.weight.copy_(model.pgot_e8_reader.key.weight)
            logger.info("[PGOT/Memory] initialized content-key projection from source Reader key")
    if (
        one_shot_memory
        and str(model_args.pgot_one_shot_memory_value_source).strip().lower()
        != source_memory_value_source
    ):
        # The E11 source contains a trained 1152->D SigLIP value projection.
        # DINOv2-base has 768 channels, so initialize its replacement explicitly
        # and deterministically instead of relying on HF mismatched-size init.
        writer = model.pgot_e8_writer
        with torch.no_grad():
            writer.raw_value_norm.weight.fill_(1.0)
            writer.raw_value_norm.bias.zero_()
            fan_out, fan_in = writer.raw_value.weight.shape
            bound = math.sqrt(6.0 / float(fan_in + fan_out))
            generator = torch.Generator(device="cpu")
            generator.manual_seed(1301)
            initialized = torch.empty(
                tuple(writer.raw_value.weight.shape),
                dtype=torch.float32,
                device="cpu",
            ).uniform_(-bound, bound, generator=generator)
            writer.raw_value.weight.copy_(
                initialized.to(
                    device=writer.raw_value.weight.device,
                    dtype=writer.raw_value.weight.dtype,
                )
            )
        logger.info(
            "[PGOT/Memory] deterministically initialized %s value projector (%d->%d)",
            str(model_args.pgot_one_shot_memory_value_source).strip().lower(),
            int(model_args.pgot_one_shot_memory_value_dim),
            int(model.config.hidden_size),
        )
    target_reader_num_layers = int(config.pgot_e8_reader_num_layers)
    if target_reader_num_layers > source_reader_num_layers:
        refinement_layers = model.pgot_e8_reader.refinement_layers
        for refinement in refinement_layers[source_reader_num_layers - 1 :]:
            refinement.reset_as_identity()
        logger.info(
            "[PGOT/Reader] deterministically initialized zero-output "
            "refinement layers source=%d target=%d",
            source_reader_num_layers,
            target_reader_num_layers,
        )
    target_max_memories = max(
        int(config.pgot_e11_object_memories_per_owner),
        int(config.pgot_e11_register_memories_per_owner),
    )
    source_max_memories = max(
        source_object_memories,
        source_register_memories,
    )
    if (bool(model_args.pgot_e11_dual_m4_enable) or one_shot_memory) and (
        not (source_checkpoint_is_e11 or source_one_shot_memory)
        or target_max_memories != source_max_memories
    ):
        with torch.no_grad():
            named_parameters = [
                (
                    1104,
                    "pgot_e8_writer.memory_id_embeddings",
                    model.pgot_e8_writer.memory_id_embeddings,
                ),
            ]
            if hasattr(model.pgot_e8_reader, "memory_key_embeddings"):
                named_parameters.append((
                    1105,
                    "pgot_e8_reader.memory_key_embeddings",
                    model.pgot_e8_reader.memory_key_embeddings,
                ))
            for seed, parameter_name, parameter in named_parameters:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed)
                initialized = torch.randn(
                    tuple(parameter.shape),
                    generator=generator,
                    dtype=torch.float32,
                    device="cpu",
                )
                parameter.copy_(
                    initialized.to(device=parameter.device, dtype=parameter.dtype)
                )
                if source_checkpoint_is_e11 or source_one_shot_memory:
                    try:
                        import glob
                        from safetensors import safe_open

                        source_tensor = None
                        for shard_file in sorted(
                            glob.glob(
                                os.path.join(
                                    model_args.model_name_or_path,
                                    "*.safetensors",
                                )
                            )
                        ):
                            with safe_open(
                                shard_file, framework="pt", device="cpu"
                            ) as sf:
                                if parameter_name in sf.keys():
                                    source_tensor = sf.get_tensor(parameter_name)
                                    break
                        if source_tensor is None:
                            raise KeyError(parameter_name)
                        rows = min(source_tensor.shape[0], parameter.shape[0])
                        parameter[:rows].copy_(
                            source_tensor[:rows].to(
                                device=parameter.device,
                                dtype=parameter.dtype,
                            )
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "Could not expand E11 memory IDs from source "
                            f"checkpoint for {parameter_name}"
                        ) from exc
        logger.info(
            "[PGOT/E11] initialized expanded Writer/Reader memory IDs "
            "source_max=%d target_max=%d",
            source_max_memories,
            target_max_memories,
        )
    if bool(model_args.pgot_e12_centroid_reader_enable) and not source_checkpoint_is_e12:
        # The E11 source checkpoint does not contain E12's positional path.
        # Initialize it explicitly so separate launches start from the same
        # gate-zero E11 function and the same Fourier-feature projector.
        reader = model.pgot_e8_reader
        with torch.no_grad():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(1201)
            fan_out, fan_in = reader.centroid_position_projector.weight.shape
            bound = math.sqrt(6.0 / float(fan_in + fan_out))
            initialized = torch.empty(
                tuple(reader.centroid_position_projector.weight.shape),
                dtype=torch.float32,
                device="cpu",
            ).uniform_(-bound, bound, generator=generator)
            reader.centroid_position_projector.weight.copy_(
                initialized.to(
                    device=reader.centroid_position_projector.weight.device,
                    dtype=reader.centroid_position_projector.weight.dtype,
                )
            )
            reader.centroid_position_norm.weight.fill_(1.0)
            reader.centroid_position_norm.bias.zero_()
            reader.centroid_object_gate.fill_(
                float(model_args.pgot_e12_centroid_gate_init)
            )
            reader.centroid_register_gate.fill_(
                float(model_args.pgot_e12_centroid_gate_init)
            )
        logger.info(
            "[PGOT/E12] deterministically initialized centroid Reader at gate zero"
        )

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
    model.config.pgot_ovt_attends_own_caption = bool(
        model_args.pgot_ovt_attends_own_caption
    )
    model.config.pgot_e5_forcing_probability = float(
        model_args.pgot_e5_forcing_probability
    )
    model.config.pgot_fvw_enable = bool(model_args.pgot_fvw_enable)
    model.config.pgot_fvw_layers = str(model_args.pgot_fvw_layers)
    model.config.pgot_fvw_num_heads = int(model_args.pgot_fvw_num_heads)
    model.config.pgot_fvw_temperature = float(model_args.pgot_fvw_temperature)
    model.config.pgot_fvw_write_outside_weight = float(
        model_args.pgot_fvw_write_outside_weight
    )
    model.config.pgot_fvw_full_probability = float(
        model_args.pgot_fvw_full_probability
    )
    model.config.pgot_fvw_ovt_only_probability = float(
        model_args.pgot_fvw_ovt_only_probability
    )
    model.config.pgot_e6_enable = bool(model_args.pgot_e6_enable)
    model.config.pgot_e6_model_id = str(model_args.pgot_e6_model_id)
    model.config.pgot_e6_coda_unet_checkpoint = str(
        model_args.pgot_e6_coda_unet_checkpoint
    )
    model.config.pgot_e6_context_dim = int(model_args.pgot_e6_context_dim)
    model.config.pgot_e6_cfg_drop_rate = float(model_args.pgot_e6_cfg_drop_rate)
    model.config.pgot_e6_attention_grid_size = int(
        model_args.pgot_e6_attention_grid_size
    )
    model.config.pgot_e7_enable = bool(model_args.pgot_e7_enable)
    model.config.pgot_e7_model_id = str(model_args.pgot_e7_model_id)
    model.config.pgot_e7_context_dim = int(model_args.pgot_e7_context_dim)
    model.config.pgot_e7_router_dim = int(model_args.pgot_e7_router_dim)
    model.config.pgot_e7_owner_num_heads = int(model_args.pgot_e7_owner_num_heads)
    model.config.pgot_e7_owner_temperature = float(
        model_args.pgot_e7_owner_temperature
    )
    model.config.pgot_e7_owner_weight = float(model_args.pgot_e7_owner_weight)
    model.config.pgot_e7_owner_bg_weight = float(model_args.pgot_e7_owner_bg_weight)
    model.config.pgot_e7_cfg_drop_rate = float(model_args.pgot_e7_cfg_drop_rate)
    model.config.pgot_e7_unet_lora_rank = int(model_args.pgot_e7_unet_lora_rank)
    model.config.pgot_e7_unet_lora_alpha = float(model_args.pgot_e7_unet_lora_alpha)
    model.config.pgot_e7_causal_weight = float(model_args.pgot_e7_causal_weight)
    model.config.pgot_e7_invariance_weight = float(
        model_args.pgot_e7_invariance_weight
    )
    model.config.pgot_e7_causal_min_timestep = int(
        model_args.pgot_e7_causal_min_timestep
    )
    model.config.pgot_e7_causal_margin = float(model_args.pgot_e7_causal_margin)
    model.config.pgot_e7_causal_temperature = float(
        model_args.pgot_e7_causal_temperature
    )
    model.config.pgot_e8_visual_memory_enable = bool(
        model_args.pgot_e8_visual_memory_enable
    )
    model.config.pgot_one_shot_reader_enable = bool(
        model_args.pgot_one_shot_reader_enable
    )
    model.config.pgot_one_shot_readout_mode = str(
        model_args.pgot_one_shot_readout_mode
    )
    model.config.pgot_one_shot_detach_owner_routing = bool(
        model_args.pgot_one_shot_detach_owner_routing
    )
    model.config.pgot_one_shot_owner_gradient_ramp_steps = int(
        model_args.pgot_one_shot_owner_gradient_ramp_steps
    )
    model.config.pgot_one_shot_writer_softmax_axis = str(
        model_args.pgot_one_shot_writer_softmax_axis
    )
    model.config.pgot_one_shot_writer_owner_prior = bool(
        model_args.pgot_one_shot_writer_owner_prior
    )
    model.config.pgot_one_shot_direct_rae_query = bool(
        model_args.pgot_one_shot_direct_rae_query
    )
    model.config.pgot_one_shot_rae_query_adapter_enable = bool(
        model_args.pgot_one_shot_rae_query_adapter_enable
    )
    model.config.pgot_one_shot_rae_query_adapter_bottleneck = int(
        model_args.pgot_one_shot_rae_query_adapter_bottleneck
    )
    model.config.pgot_one_shot_memory_contrastive_enable = bool(
        model_args.pgot_one_shot_memory_contrastive_enable
    )
    model.config.pgot_one_shot_memory_value_source = str(
        model_args.pgot_one_shot_memory_value_source
    ).strip().lower()
    model.config.pgot_one_shot_memory_value_dim = int(
        model_args.pgot_one_shot_memory_value_dim
    )
    model.config.pgot_one_shot_memory_contrastive_target_weight = float(
        training_args.pgot_contrastive_loss_target_weight
    )
    model.config.pgot_one_shot_memory_contrastive_warmup_steps = int(
        training_args.pgot_contrastive_warmup_steps
    )
    model.config.pgot_e8_layers = str(model_args.pgot_e8_layers)
    model.config.pgot_e8_owner_temperature = float(
        model_args.pgot_e8_owner_temperature
    )
    model.config.pgot_e8_owner_weight = float(model_args.pgot_e8_owner_weight)
    model.config.pgot_e8_owner_bg_weight = float(model_args.pgot_e8_owner_bg_weight)
    model.config.pgot_e8_reader_num_heads = int(
        model_args.pgot_e8_reader_num_heads
    )
    model.config.pgot_e8_reader_num_layers = int(
        model_args.pgot_e8_reader_num_layers
    )
    model.config.pgot_e8_reader_temperature = float(
        model_args.pgot_e8_reader_temperature
    )
    model.config.pgot_e8_clean_refinement = bool(
        model_args.pgot_e8_clean_refinement
    )
    model.config.pgot_e8_inject_memory = bool(model_args.pgot_e8_inject_memory)
    model.config.pgot_e10_raw_value_enable = bool(
        model_args.pgot_e10_raw_value_enable
    )
    model.config.pgot_e11_dual_m4_enable = bool(
        model_args.pgot_e11_dual_m4_enable
    )
    model.config.pgot_e11_memories_per_owner = int(
        model_args.pgot_e11_memories_per_owner
    )
    model.config.pgot_e11_object_memories_per_owner = int(
        config.pgot_e11_object_memories_per_owner
    )
    model.config.pgot_e11_register_memories_per_owner = int(
        config.pgot_e11_register_memories_per_owner
    )
    model.config.pgot_e11_query_separation_enable = bool(
        model_args.pgot_e11_query_separation_enable
    )
    model.config.pgot_e12_centroid_reader_enable = bool(
        model_args.pgot_e12_centroid_reader_enable
    )
    model.config.pgot_e12_centroid_gate_init = float(
        model_args.pgot_e12_centroid_gate_init
    )
    model.config.pgot_e8_reader_supervision_mode = str(
        model_args.pgot_e8_reader_supervision_mode
    )
    model.config.pgot_e8_reader_object_weight = float(
        model_args.pgot_e8_reader_object_weight
    )
    model.config.pgot_e8_reader_background_weight = float(
        model_args.pgot_e8_reader_background_weight
    )
    model.config.pgot_e8_causal_enable = bool(model_args.pgot_e8_causal_enable)
    model.config.pgot_e8_causal_margin = float(model_args.pgot_e8_causal_margin)
    model.config.pgot_e8_register_margin = float(model_args.pgot_e8_register_margin)
    model.config.pgot_e8_need_weight = float(model_args.pgot_e8_need_weight)
    model.config.pgot_e8_local_weight = float(model_args.pgot_e8_local_weight)
    model.config.pgot_e8_register_bg_weight = float(
        model_args.pgot_e8_register_bg_weight
    )
    model.config.pgot_e8_register_fg_weight = float(
        model_args.pgot_e8_register_fg_weight
    )
    model.config.pgot_e8_causal_batch_probability = float(
        model_args.pgot_e8_causal_batch_probability
    )
    model.config.pgot_e8_causal_ramp_steps = int(
        model_args.pgot_e8_causal_ramp_steps
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
    model.config.pgot_dit_soft_routing_enable = bool(model_args.pgot_dit_soft_routing_enable)
    model.config.pgot_dit_soft_routing_scale = float(model_args.pgot_dit_soft_routing_scale)
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
    model.config.pgot_latent_distill_ramp_steps = int(model_args.pgot_latent_distill_ramp_steps)
    model.config.pgot_latent_distill_mse_weight = float(model_args.pgot_latent_distill_mse_weight)
    model.config.pgot_latent_distill_cos_weight = float(model_args.pgot_latent_distill_cos_weight)
    model.config.pgot_latent_distill_l1_weight = float(model_args.pgot_latent_distill_l1_weight)
    model.config.pgot_cfg_drop_rate = float(model_args.pgot_cfg_drop_rate)
    model.config.pgot_rae_attends_caption = bool(model_args.pgot_rae_attends_caption)
    model.config.pgot_unfreeze_mm_projector = bool(model_args.pgot_unfreeze_mm_projector)
    model.config.pgot_diffusion_force_fp32 = bool(model_args.pgot_diffusion_force_fp32)
    model.config.image_preprocess_mode = str(data_args.image_preprocess_mode)
    model.config.coda_crop_size = int(data_args.coda_crop_size)
    if one_shot_memory:
        logger.info(
            "[PGOT/Memory] mode=%s; memories object=%d/register=%d; direct RAE query=%s; RAE adapter=%s/%d; contrastive=%s lambda=%.4f mix=%.2f warmup=%d; owner loss=%.2f; Reader loss object=%.2f/background=%.2f",
            model.config.pgot_one_shot_readout_mode,
            model.config.pgot_e11_object_memories_per_owner,
            model.config.pgot_e11_register_memories_per_owner,
            model.config.pgot_one_shot_direct_rae_query,
            model.config.pgot_one_shot_rae_query_adapter_enable,
            model.config.pgot_one_shot_rae_query_adapter_bottleneck,
            model.config.pgot_one_shot_memory_contrastive_enable,
            model.config.pgot_one_shot_memory_contrastive_target_weight,
            model.config.pgot_contrastive_sampling_rate,
            model.config.pgot_one_shot_memory_contrastive_warmup_steps,
            model.config.pgot_e8_owner_weight,
            model.config.pgot_e8_reader_object_weight,
            model.config.pgot_e8_reader_background_weight,
        )
    else:
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
            f"isolated_attention={model.config.pgot_ovt_isolated_attention}, "
            f"own_caption={model.config.pgot_ovt_attends_own_caption}) "
            f"e5=(forcing_p={model.config.pgot_e5_forcing_probability}) "
            f"fvw=(enable={getattr(model.config, 'pgot_fvw_enable', False)}, "
            f"layers={getattr(model.config, 'pgot_fvw_layers', '0,8,16,24,27')}, "
            f"heads={getattr(model.config, 'pgot_fvw_num_heads', 8)}, "
            f"temp={getattr(model.config, 'pgot_fvw_temperature', 1.0)}, "
            f"write_out_w={getattr(model.config, 'pgot_fvw_write_outside_weight', 0.0)}, "
            f"full_p={getattr(model.config, 'pgot_fvw_full_probability', 0.5)}, "
            f"ovt_p={getattr(model.config, 'pgot_fvw_ovt_only_probability', 0.25)}) "
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
        if bool(model_args.pgot_e6_enable) or bool(model_args.pgot_e7_enable):
            # Direct SD experiments remove the unused Scale-RAE DiT so it
            # cannot become a reconstruction path or enter checkpoints.
            model.diff_head = None
            model.diff_head_projector = None
            logger.info("[PGOT/direct-SD] vision tower initialized; Scale-RAE DiT removed")
        else:
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
        # Persist the exact PEFT structure so standalone evaluation can rebuild
        # the adapter before loading base_layer/lora_* checkpoint keys.
        model.config.pgot_lora_enable = True
        model.config.pgot_lora_r = int(training_args.pgot_lora_r)
        model.config.pgot_lora_alpha = int(training_args.pgot_lora_alpha)
        model.config.pgot_lora_dropout = float(training_args.pgot_lora_dropout)
        model.config.pgot_lora_target_modules = ",".join(lora_targets)
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
    model.config.pgot_trainable_fp32 = bool(training_args.pgot_trainable_fp32)
    if (
        bool(training_args.gradient_checkpointing)
        and getattr(model, "diff_head", None) is not None
        and getattr(model.diff_head, "model", None) is not None
    ):
        model.diff_head.model.gradient_checkpointing = True
        logger.info("[PGOT/precision] enabled activation checkpointing for DiT blocks")

    # ---- Move to device (vision tower already moved above)
    model.to(training_args.device)
    if bool(training_args.pgot_trainable_fp32):
        converted_parameters = 0
        converted_numel = 0
        seen = set()
        for parameter in model.parameters():
            if (
                parameter.requires_grad
                and parameter.dtype != torch.float32
                and id(parameter) not in seen
            ):
                seen.add(id(parameter))
                parameter.data = parameter.data.float()
                converted_parameters += 1
                converted_numel += parameter.numel()
        logger.info(
            "[PGOT/precision] frozen weights retain %s; converted %d trainable "
            "tensors (%s params) to FP32",
            str(compute_dtype).replace("torch.", ""),
            converted_parameters,
            f"{converted_numel:,}",
        )
    if parsed_towers is not None:
        for vt in model.get_vision_tower_aux_list():
            vt.to(dtype=compute_dtype, device=training_args.device)

    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    model = _manual_fsdp_wrap(model, training_args)

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
    if bool(training_args.pgot_parameter_update_audit):
        trainer.add_callback(
            PGOTParameterUpdateAuditCallback(
                output_dir=training_args.output_dir,
                max_samples=training_args.pgot_parameter_update_audit_samples,
            )
        )

    # ---- Device sanity check log
    def _first_param_device(m):
        if m is None:
            return "removed"
        try:
            return str(next(m.parameters()).device)
        except StopIteration:
            return "none"
    inner = model.module if hasattr(model, "module") else model
    logger.info("[PGOT/device-check]")
    logger.info(f"  backbone: {_first_param_device(inner.model)}")
    logger.info(f"  lm_head:  {_first_param_device(inner.lm_head)}")
    logger.info(f"  diff_head: {_first_param_device(inner.diff_head)}")
    logger.info(f"  diff_head_projector: {_first_param_device(inner.diff_head_projector)}")
    logger.info(f"  pgot_register: {inner.pgot_register_embeddings.device}")
    logger.info(f"  latent_queries: {inner.get_model().latent_queries.device}")
    if bool(getattr(inner.config, "pgot_e6_enable", False)):
        logger.info(
            f"  e6_context_projector: "
            f"{_first_param_device(inner.pgot_e6_decoder.context_projector)}"
        )
        logger.info(
            f"  e6_coda_unet: {_first_param_device(inner.pgot_e6_decoder.unet)}"
        )
        logger.info(f"  e6_coda_vae: {_first_param_device(inner.pgot_e6_decoder.vae)}")
    if bool(getattr(inner.config, "pgot_e7_enable", False)):
        logger.info(
            f"  e7_owner_write: {_first_param_device(inner.pgot_e7_decoder.owner_write)}"
        )
        logger.info(
            f"  e7_context_projector: "
            f"{_first_param_device(inner.pgot_e7_decoder.context_projector)}"
        )
        logger.info(
            f"  e7_sd_unet: {_first_param_device(inner.pgot_e7_decoder.unet)}"
        )
        logger.info(f"  e7_sd_vae: {_first_param_device(inner.pgot_e7_decoder.vae)}")
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
