"""PGOT trainer — extends HF Trainer with PGOT-specific optimizer, scheduler, loss logging."""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

logger = logging.getLogger(__name__)


# ============================================================================
# Argument dataclasses
# ============================================================================
@dataclass
class PGOTModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen2-1.5B")
    version: str = field(default="qwen2")
    use_pgot: bool = field(default=True)

    # PGOT-specific config
    pgot_n_register: int = field(default=64)
    pgot_n_null_bg: int = field(default=1)
    pgot_n_ovt_per_object: int = field(default=2)
    pgot_max_objects: int = field(default=50)
    pgot_rae_bidirectional: bool = field(default=False)
    pgot_attention_use_layer_norm: bool = field(default=True)
    pgot_attention_temperature: float = field(default=1.0)

    # Loss weights
    pgot_lm_loss_weight: float = field(default=1.0)
    pgot_mask_loss_weight: float = field(default=0.5)
    pgot_recon_loss_weight: float = field(default=1.0)
    pgot_contrastive_loss_weight: float = field(default=0.0)  # config baseline (kept for back-compat)
    pgot_contrastive_sampling_rate: float = field(default=0.5)

    # Mask supervision. Primary: CODA-style per-patch softmax CE (exclusive
    # competition over thing + stuff OVTs -> good fARI). BCE/Tversky kept for
    # back-compat but default to 0 weight.
    pgot_mask_ce_weight: float = field(default=1.0)
    pgot_mask_ce_temperature: float = field(default=1.0)
    pgot_mask_ce_merge: str = field(default="max")
    pgot_use_null_bg_competition: bool = field(default=False)
    pgot_mask_fg_weight: float = field(default=0.0)
    pgot_mask_outside_weight: float = field(default=0.0)
    # v6a: auxiliary competition CE on last-layer q_proj/k_proj projections.
    # Gradient flows back into the LLM's own attention projections (LoRA), nudging
    # them toward competition-friendly Q/K geometry. 0 disables; v6 default ~0.5.
    pgot_mask_aux_competition_weight: float = field(default=0.0)
    pgot_mask_bce_weight: float = field(default=0.0)
    # V13: V3 sigmoid BCE's negative term only:
    # -(1-y)log(1-sigmoid(dot(OVT, patch))).
    pgot_mask_sigmoid_outside_weight: float = field(default=0.0)
    # V13: apply the same negative term to register-vs-patch scores on the
    # union of annotated object/stuff masks so registers remain residual/bg.
    pgot_register_foreground_suppression_weight: float = field(default=0.0)
    pgot_mask_object_balanced_bce_weight: float = field(default=0.0)
    pgot_mask_tversky_weight: float = field(default=0.0)
    pgot_mask_tversky_alpha: float = field(default=0.5)
    pgot_mask_tversky_beta: float = field(default=0.5)
    # v8: spatial softmax over patches per OVT, penalizing attention mass on
    # every other annotated thing/stuff region.
    pgot_mask_spatial_outside_weight: float = field(default=0.0)
    pgot_mask_spatial_temperature: float = field(default=1.0)
    # v8.2: patch-axis softmax + outside-only log penalty:
    # -sum_{outside target mask} log(1 - attention).
    pgot_mask_spatial_outside_log_weight: float = field(default=0.0)
    pgot_mask_spatial_outside_log_temperature: float = field(default=1.0)
    # v8.1: same outside objective, but scores come from selected LLM layers'
    # own Q/K projections: OVT query -> image-patch key, softmax over patches.
    pgot_mask_llm_qk_outside_weight: float = field(default=0.0)
    pgot_mask_llm_qk_outside_temperature: float = field(default=1.0)
    pgot_mask_llm_qk_outside_layers: str = field(default="last4")
    # v8.4: exact LLM attention outside-only log loss. Unlike the v8.1 Q/K
    # diagnostic, this reproduces RoPE, the full allowed-key softmax denominator,
    # and the model attention bias before selecting OVT -> image probabilities.
    pgot_mask_llm_attention_outside_weight: float = field(default=0.0)
    pgot_mask_llm_attention_outside_layers: str = field(default="last4")
    pgot_mask_llm_attention_void_weight: float = field(default=1.0)
    # v8.5: use the selected layers' real post-RoPE Q/K scores, but normalize
    # only over image patches for spatial supervision. A separate weak hinge on
    # the real full-key attention mass prevents escape to caption/self tokens.
    pgot_mask_llm_patch_outside_weight: float = field(default=0.0)
    pgot_mask_llm_patch_outside_layers: str = field(default="last4")
    pgot_mask_llm_patch_outside_temperature: float = field(default=1.0)
    pgot_mask_llm_patch_void_weight: float = field(default=1.0)
    pgot_mask_llm_image_use_weight: float = field(default=0.0)
    pgot_mask_llm_image_use_margin: float = field(default=0.05)
    # Core experiment: exact post-RoPE OVT->image attention, normalized only
    # across the 1024 image patches. Penalize raw probability outside each GT
    # segment at every selected layer and every head. Tail is an ablation and
    # defaults to zero weight.
    pgot_core_outside_weight: float = field(default=0.0)
    pgot_core_outside_layers: str = field(default="all")
    pgot_core_outside_temperature: float = field(default=1.0)
    pgot_core_void_weight: float = field(default=1.0)
    pgot_core_tail_weight: float = field(default=0.0)
    pgot_core_tail_fraction: float = field(default=0.1)
    # E1 background registers use the same exact post-RoPE, image-patch-only
    # attention as OVT supervision, but their forbidden target is the union of
    # every visible COCO instance mask.
    pgot_core_register_outside_weight: float = field(default=0.0)
    # E3: patch ownership competition inside the MLLM.  At every selected
    # layer, the head-averaged OVT maps and the four registers compete over
    # {object_1, ..., object_K, residual-background}.  The implementation
    # jointly computes this CE and the two core outside losses so Q/K/RoPE are
    # reconstructed only once per layer.
    pgot_e3_attention_competition_weight: float = field(default=0.0)
    pgot_e3_attention_competition_layers: str = field(default="all")
    pgot_e3_attention_competition_temperature: float = field(default=1.0)
    pgot_e3_attention_competition_bg_weight: float = field(default=0.25)
    # E4: keep the E3 object-centric MLLM supervision, but require actual
    # full-softmax attention mass on the matching image region and bind each
    # 16x16 Scale-RAE spatial query to its matching OVT/register owner.
    pgot_e4_rae_isolated: bool = field(default=False)
    pgot_e4_full_inside_weight: float = field(default=0.0)
    pgot_e4_full_inside_target: float = field(default=0.30)
    pgot_e4_rae_bind_weight: float = field(default=0.0)
    pgot_e4_rae_bind_layers: str = field(default="last8")
    # E2 diagnostic: during training, make every register -> foreground image
    # patch edge structurally impossible in the shared attention bias.  The
    # bias is broadcast to every head and reused by every LLM layer.  Eval is
    # GT-free by default; the explicit eval switch is oracle-only.
    pgot_register_hard_gt_mask: bool = field(default=False)
    pgot_register_hard_gt_mask_eval: bool = field(default=False)
    pgot_register_hard_gt_mask_threshold: float = field(default=0.0)
    # False for E1: registers read image patches and other registers, never
    # caption/OVT values. Legacy checkpoints default to True.
    pgot_register_attends_caption: bool = field(default=True)
    # Make layer-0 OVT queries object-specific using their own preceding caption
    # span. The projection is initialized to identity and trained with PGOT.
    pgot_ovt_caption_init: bool = field(default=False)
    pgot_ovt_caption_init_scale: float = field(default=1.0)
    # After the own-caption-conditioned layer-0 initialization, restrict every
    # OVT query row to image patches + itself in every LLM layer. This prevents
    # later caption, previous-OVT, register, and RAE value shortcuts.
    pgot_ovt_isolated_attention: bool = field(default=False)
    # E5 restores only the current object's caption span for an otherwise
    # isolated OVT row. Previous object captions and OVTs remain blocked.
    pgot_ovt_attends_own_caption: bool = field(default=False)
    # Fraction of training samples whose RAE attention update is forced to use
    # OVT values only; their diffusion loss is normalized over foreground
    # target tokens. Evaluation always uses the ordinary full condition.
    pgot_e5_forcing_probability: float = field(default=0.0)

    # V12: OVT-style owner competition injected between selected LLM layers.
    pgot_v12_enable: bool = field(default=False)
    pgot_v12_layers: str = field(default="12,16,20,24")
    pgot_v12_ovt_temperature: Optional[float] = field(default=None)
    pgot_v12_slot_temperature: float = field(default=1.0)
    pgot_v12_owner_temperature: float = field(default=1.0)
    pgot_v12_owner_weight: float = field(default=1.0)

    # V14: OVT bottleneck route. Diffusion condition is built from OVT/void only.
    pgot_v14_enable: bool = field(default=False)
    pgot_v14_route_temperature: float = field(default=1.0)
    pgot_v14_route_weight: float = field(default=1.0)
    pgot_v14_void_weight: float = field(default=0.5)
    pgot_v14_position_weight: float = field(default=1.0)
    pgot_v14_router_depth: int = field(default=1)
    pgot_v14_router_mlp_ratio: int = field(default=4)

    # V20: keep V15 router AdaLN conditioning, and additionally expose
    # OVT/void states to DiT through zero-init sparse slot cross-attention.
    pgot_dit_ovt_cross_attn_enable: bool = field(default=False)
    pgot_dit_ovt_cross_attn_start_block: int = field(default=25)
    pgot_dit_ovt_cross_attn_every_n_blocks: int = field(default=1)

    # V17: Generative Binding. Diffusion condition comes from the final-layer
    # RAE query hidden states; ownership CE supervises RAE query -> OVT attention.
    pgot_v17_enable: bool = field(default=False)
    pgot_v17_ownership_weight: float = field(default=0.0)
    pgot_v17_ownership_layers: str = field(default="last4")

    # V21: Group-grounded generative OVT. Same object OVTs form one group
    # responsibility map that is both grounded and used to compose DiT condition.
    pgot_v21_enable: bool = field(default=False)
    pgot_v21_ground_weight: float = field(default=0.0)
    pgot_v21_ground_final_weight: float = field(default=-1.0)
    pgot_v21_ground_anneal_steps: int = field(default=0)
    pgot_v21_temperature: float = field(default=1.0)
    pgot_v21_position_weight: float = field(default=1.0)
    pgot_v21_code_dim: int = field(default=0)

    # V22a: keep V15/V15.1 BCE grounding, but supervise the LLM-internal
    # OVT -> image attention maps to become object-local. The outside term
    # reuses the v8.5 patch-normalized attention loss; this weak competition CE
    # asks foreground patches to be owned by the matching object OVT group.
    pgot_v22_attention_competition_weight: float = field(default=0.0)
    pgot_v22_attention_competition_layers: str = field(default="26,27")
    pgot_v22_attention_competition_temperature: float = field(default=1.0)
    pgot_v22_attention_competition_include_void: bool = field(default=False)
    pgot_v22_attention_competition_bg_weight: float = field(default=0.25)

    # V18: V15 object-causal bottleneck + direct decoder-native latent distill.
    # The head predicts SigLIP decoder latents from the same bottleneck condition
    # used by DiT, so we can optimize toward the decoder_gt oracle directly.
    pgot_latent_distill_enable: bool = field(default=False)
    pgot_latent_distill_weight: float = field(default=0.0)
    pgot_latent_distill_mse_weight: float = field(default=1.0)
    pgot_latent_distill_cos_weight: float = field(default=1.0)
    pgot_latent_distill_l1_weight: float = field(default=0.0)

    # CFG: randomly drop the rae_hidden condition during training so diff_head learns
    # an unconditional path; at inference we use guidance_scale > 1.
    pgot_cfg_drop_rate: float = field(default=0.1)

    # Whether rae_query attends to whole caption (legacy). When False (default), rae_query
    # only attends OVT positions inside the caption + register + self.
    pgot_rae_attends_caption: bool = field(default=False)
    # NOTE: pgot_contrastive_loss_target_weight + pgot_contrastive_warmup_steps now live in
    # PGOTTrainingArguments so that compute_loss (which reads self.args) can see them.

    # Vision tower (reuse scale-rae args)
    vision_tower_aux_list: Optional[str] = field(default=None)
    vision_tower_aux_token_len_list: Optional[str] = field(default=None)
    image_feature_token_len: Optional[int] = field(default=None)
    diffusion_target_token_len: Optional[int] = field(default=None)
    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_use_im_start_end: bool = field(default=True)
    mm_use_im_patch_token: bool = field(default=False)
    vision_hidden_size: Optional[int] = field(default=1024)
    connector_only: bool = field(default=True)
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_adapter_and_vision_head: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_adapter_and_vision_head: Optional[str] = field(default=None)

    # Diffusion head (passed through to ScaleRAEQwenConfig)
    vision_loss: str = field(default="diffusion-loss")
    vision_loss_mode: str = field(default="query")
    vision_coef: float = field(default=1.0)
    diffusion_model_hidden_size: int = field(default=1152)
    diffusion_model_channels: int = field(default=1152)
    diffusion_model_depth: int = field(default=12)
    diffusion_model_heads: int = field(default=16)
    diffusion_model_z_channels: int = field(default=0)
    dit_cls: str = field(default="LightningDiT")
    diffusion_norm_stats_path: Optional[str] = field(default=None)

    # Image processing
    image_aspect_ratio: str = field(default="square")
    si_token_len: int = field(default=0)
    miv_token_len: int = field(default=0)

    # Frozen/Trainable policy
    freeze_dit_body: bool = field(default=True)  # True = only AdaLN trainable in DiT
    freeze_vision_tower: bool = field(default=True)
    # V15.1: adapt the Scale-RAE input projector to the 512px/1024-token
    # SigLIP feature distribution while keeping the vision tower frozen.
    pgot_unfreeze_mm_projector: bool = field(default=False)
    # Partial DiT unfreeze: unfreeze the last N dit_blocks fully (in addition to AdaLN-everywhere).
    # 0 = behave like freeze_dit_body. N > 0 implies AdaLN of all blocks PLUS last N blocks self-attn/MLP.
    pgot_dit_unfreeze_last_n_blocks: int = field(default=0)


@dataclass
class PGOTDataArguments:
    train_jsonl: str = field(default="/home/jovyan/PGOT/data/pgot_train.jsonl")
    val_jsonl: str = field(default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    max_caption_tokens: int = field(default=2048)
    grid_size: int = field(default=16)
    eval_num_images: int = field(default=128)
    image_preprocess_mode: str = field(default="default")
    coda_crop_size: int = field(default=512)
    dataset_format: str = field(default="pix2cap")
    is_multimodal: bool = field(default=True)
    # image_processor_aux_list / vision_tower_aux_token_len_list are populated
    # at runtime by train.py (they are not CLI args).


@dataclass
class PGOTTrainingArguments(transformers.TrainingArguments):
    # Tokenizer
    model_max_length: int = field(default=4096)

    # LR groups
    diff_head_lr: Optional[float] = field(default=None)
    pgot_register_lr: Optional[float] = field(default=None)
    pgot_rae_query_lr: Optional[float] = field(default=None)
    pgot_llm_lr: Optional[float] = field(default=None)
    pgot_mm_projector_lr: Optional[float] = field(default=None)
    # LR for unfrozen DiT body params (non-AdaLN, non-projector). Falls back to diff_head_lr.
    pgot_dit_body_lr: Optional[float] = field(default=None)

    # Eval-time image reconstruction logging (wandb)
    pgot_eval_log_recon_images: int = field(default=0)  # 0 = disabled; >0 = N images per eval
    pgot_eval_decoder_repo: str = field(default="nyu-visionx/siglip2_decoder")

    # Contrastive (mirrored here so compute_loss can read from self.args)
    pgot_contrastive_loss_target_weight: float = field(default=0.0)
    pgot_contrastive_warmup_steps: int = field(default=0)
    pgot_e4_loss_warmup_steps: int = field(default=0)

    # LoRA
    pgot_lora_enable: bool = field(default=True)
    pgot_lora_r: int = field(default=16)
    pgot_lora_alpha: int = field(default=32)
    pgot_lora_dropout: float = field(default=0.05)
    pgot_lora_target_modules: str = field(default="q_proj,k_proj,v_proj,o_proj")

    # LR schedule
    pgot_use_wsd_schedule: bool = field(default=False)
    pgot_wsd_decay_fraction: float = field(default=0.15)
    pgot_use_cosine_min_lr_schedule: bool = field(default=True)
    pgot_min_lr_ratio: float = field(default=0.10)


# ============================================================================
# Freeze policy
# ============================================================================
def freeze_for_pgot(
    model,
    freeze_dit_body: bool = True,
    freeze_vision_tower: bool = True,
    unfreeze_mm_projector: bool = False,
    dit_unfreeze_last_n_blocks: int = 0,
):
    """Apply PGOT freeze policy.

    Trainable (default):
      - LoRA params on Qwen2 LLM (lora_*)
      - <ovt>, <scene_end> embedding rows (we'll mark embed_tokens trainable
        BUT only those rows get nonzero gradient since other tokens are masked
        in the LM loss)
      - register embedding (full)
      - latent_queries (rae_query, full fine-tune)
      - diff_head_projector (full fine-tune)
      - mm_projector if unfreeze_mm_projector=True
      - diff_head AdaLN modulation only (if freeze_dit_body=True)
    Frozen:
      - Vision tower
      - RAE decoder (not in model)
      - DiT body (if freeze_dit_body=True; only adaLN trainable)
    """
    # Start by freezing everything
    model.requires_grad_(False)

    n_trainable = 0
    def _unfreeze(module_or_param, name=""):
        nonlocal n_trainable
        if isinstance(module_or_param, nn.Parameter):
            module_or_param.requires_grad_(True)
            n_trainable += module_or_param.numel()
        else:
            for p in module_or_param.parameters():
                p.requires_grad_(True)
                n_trainable += p.numel()

    # LoRA params
    n_lora = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
            n_lora += param.numel()
            n_trainable += param.numel()

    # PGOT register
    if (
        hasattr(model, "pgot_register_embeddings")
        and model.pgot_register_embeddings.numel() > 0
    ):
        model.pgot_register_embeddings.requires_grad_(True)
        n_trainable += model.pgot_register_embeddings.numel()

    # V7 null-bg segmentation owner
    if hasattr(model, "pgot_null_bg_embeddings"):
        model.pgot_null_bg_embeddings.requires_grad_(True)
        n_trainable += model.pgot_null_bg_embeddings.numel()

    # E1 layer-0 caption-conditioned OVT initialization.
    if (
        hasattr(model, "pgot_ovt_caption_projector")
        and model.pgot_ovt_caption_projector is not None
    ):
        for module in (model.pgot_ovt_caption_norm, model.pgot_ovt_caption_projector):
            for p in module.parameters():
                p.requires_grad_(True)
                n_trainable += p.numel()

    # V12 OVT-update / owner heads
    if hasattr(model, "pgot_v12_slot_update_blocks"):
        for p in model.pgot_v12_slot_update_blocks.parameters():
            p.requires_grad_(True)
            n_trainable += p.numel()
    if hasattr(model, "pgot_v12_owner_head") and model.pgot_v12_owner_head is not None:
        for p in model.pgot_v12_owner_head.parameters():
            p.requires_grad_(True)
            n_trainable += p.numel()

    # V14 OVT bottleneck router
    if hasattr(model, "pgot_v14_router") and model.pgot_v14_router is not None:
        for p in model.pgot_v14_router.parameters():
            p.requires_grad_(True)
            n_trainable += p.numel()

    # V21 group-grounded router
    if hasattr(model, "pgot_v21_router") and model.pgot_v21_router is not None:
        for p in model.pgot_v21_router.parameters():
            p.requires_grad_(True)
            n_trainable += p.numel()

    # V18 latent distillation head
    if hasattr(model, "pgot_latent_head") and model.pgot_latent_head is not None:
        for p in model.pgot_latent_head.parameters():
            p.requires_grad_(True)
            n_trainable += p.numel()

    # rae_query (latent_queries inside model.get_model())
    inner = model.get_model() if hasattr(model, "get_model") else model
    if hasattr(inner, "latent_queries") and inner.latent_queries is not None:
        inner.latent_queries.requires_grad_(True)
        n_trainable += inner.latent_queries.numel()

    n_mm_projector = 0
    if bool(unfreeze_mm_projector) and hasattr(inner, "mm_projector") and inner.mm_projector is not None:
        for p in inner.mm_projector.parameters():
            p.requires_grad_(True)
            n_mm_projector += p.numel()
            n_trainable += p.numel()

    # diff_head_projector
    if hasattr(model, "diff_head_projector") and model.diff_head_projector is not None:
        for p in model.diff_head_projector.parameters():
            p.requires_grad_(True)
            n_trainable += p.numel()

    # DiT body
    n_dit_adaln = 0
    n_dit_body_blocks = 0
    if hasattr(model, "diff_head") and model.diff_head is not None:
        if not freeze_dit_body:
            # Full unfreeze (legacy path)
            for p in model.diff_head.parameters():
                p.requires_grad_(True)
                n_trainable += p.numel()
        else:
            # AdaLN-modulation of ALL blocks (incl. final_layer) is always trainable.
            for name, p in model.diff_head.named_parameters():
                if "adaLN_modulation" in name:
                    p.requires_grad_(True)
                    n_trainable += p.numel()
                    n_dit_adaln += p.numel()

            # Optionally also unfreeze the LAST N dit_blocks fully (self-attn/MLP/etc.).
            n_unfreeze_last = max(int(dit_unfreeze_last_n_blocks), 0)
            if n_unfreeze_last > 0:
                # diff_head.model.dit_blocks is nn.ModuleList of depth blocks
                dit_body = getattr(model.diff_head, "model", None)
                if dit_body is not None and hasattr(dit_body, "dit_blocks"):
                    total_blocks = len(dit_body.dit_blocks)
                    start_idx = max(0, total_blocks - n_unfreeze_last)
                    for bi in range(start_idx, total_blocks):
                        for name, p in dit_body.dit_blocks[bi].named_parameters():
                            if not p.requires_grad:  # adaLN already True; avoid double-count
                                p.requires_grad_(True)
                                n_trainable += p.numel()
                                n_dit_body_blocks += p.numel()
                    logger.info(
                        "[PGOT/Freeze] DiT last %d blocks unfrozen (idx %d..%d of %d) | extra params=%d",
                        n_unfreeze_last, start_idx, total_blocks - 1, total_blocks, n_dit_body_blocks,
                    )

    # Embedding rows for <ovt> and <scene_end> — we unfreeze the full embed_tokens
    # weight (the LM loss only generates gradient at <ovt>/<scene_end> rows in
    # practice because the rest of vocab is supervised by next-token labels via
    # LM loss, but those rows are usually well-trained already). To match the
    # constraint "LoRA only", we keep embed_tokens frozen and rely on LoRA in
    # the attention layers. New rows for <ovt>, <scene_end> are unfrozen
    # selectively below.
    new_tokens = []
    if hasattr(model, "pgot_ovt_token_id") and model.pgot_ovt_token_id is not None:
        new_tokens.append(model.pgot_ovt_token_id)
    if hasattr(model, "pgot_scene_end_token_id") and model.pgot_scene_end_token_id is not None:
        new_tokens.append(model.pgot_scene_end_token_id)
    if hasattr(model, "pgot_thing_token_id") and model.pgot_thing_token_id is not None:
        new_tokens.append(model.pgot_thing_token_id)
    if hasattr(model, "pgot_stuff_token_id") and model.pgot_stuff_token_id is not None:
        new_tokens.append(model.pgot_stuff_token_id)
    if new_tokens:
        # Mark the full embed_tokens AND lm_head trainable; we will mask out
        # gradients of non-new rows in a hook below.
        embed = inner.embed_tokens
        embed.weight.requires_grad_(True)
        if hasattr(model, "lm_head") and model.lm_head is not None:
            model.lm_head.weight.requires_grad_(True)

        # Register a backward hook that zeros out gradient for all rows except
        # the new tokens.
        vocab_size = embed.weight.shape[0]
        new_token_ids = set(new_tokens)
        keep_mask = torch.zeros(vocab_size, dtype=torch.bool, device=embed.weight.device)
        for tid in new_token_ids:
            keep_mask[int(tid)] = True

        def _embed_hook(grad):
            if grad is None:
                return grad
            mask = keep_mask.to(device=grad.device, dtype=grad.dtype).unsqueeze(-1)
            return grad * mask

        embed.weight.register_hook(_embed_hook)
        if hasattr(model, "lm_head") and model.lm_head is not None:
            # lm_head.weight has shape (vocab, hidden) — same mask works
            model.lm_head.weight.register_hook(_embed_hook)

    # Vision tower: leave frozen (already requires_grad_(False))

    logger.info(
        "[PGOT/Freeze] Total trainable params: %s  (LoRA: %s, mm_projector: %s)",
        f"{n_trainable:,}",
        f"{n_lora:,}",
        f"{n_mm_projector:,}",
    )
    return model


# ============================================================================
# Trainer
# ============================================================================
class PGOTTrainer(Trainer):
    """Trainer with PGOT-specific optimizer, scheduler, contrastive warmup, logging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_loss_buffer: Dict[str, float] = {}
        self._loss_count_buffer: int = 0
        self._step_global: int = 0
        # Per-eval-call running sums of sub-losses (loss_lm, loss_mask, loss_recon, ...)
        self._eval_sub_loss_sums: Dict[str, float] = {}
        self._eval_sub_loss_count: int = 0
        self._last_eval_sub_metrics: Dict[str, float] = {}
        self._eval_decoder = None  # lazy-loaded siglip2 decoder for image recon logging

    # ----- Optimizer with grouped LRs -----
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        diff_head_lr = (
            self.args.diff_head_lr if self.args.diff_head_lr is not None else self.args.learning_rate
        )
        register_lr = (
            self.args.pgot_register_lr if self.args.pgot_register_lr is not None else self.args.learning_rate
        )
        rae_query_lr = (
            self.args.pgot_rae_query_lr if self.args.pgot_rae_query_lr is not None else self.args.learning_rate
        )
        llm_lr = (
            self.args.pgot_llm_lr if self.args.pgot_llm_lr is not None else self.args.learning_rate
        )
        mm_projector_lr = (
            self.args.pgot_mm_projector_lr
            if getattr(self.args, "pgot_mm_projector_lr", None) is not None
            else self.args.learning_rate
        )
        dit_body_lr = (
            self.args.pgot_dit_body_lr if getattr(self.args, "pgot_dit_body_lr", None) is not None else diff_head_lr
        )

        mm_projector_names = {n for n, p in opt_model.named_parameters()
                              if p.requires_grad and "mm_projector" in n}
        projector_names = {n for n, p in opt_model.named_parameters()
                           if p.requires_grad and "diff_head_projector" in n}
        # Split diff_head.model.* into AdaLN (faster, diff_head_lr) vs body (slower, dit_body_lr).
        dit_adaln_names = {n for n, p in opt_model.named_parameters()
                           if p.requires_grad and "diff_head.model." in n and "adaLN_modulation" in n}
        dit_body_names = {n for n, p in opt_model.named_parameters()
                          if p.requires_grad and "diff_head.model." in n and "adaLN_modulation" not in n}
        diff_body_names = dit_adaln_names | dit_body_names  # for assigned-set bookkeeping below
        register_names = {
            n
            for n, p in opt_model.named_parameters()
            if p.requires_grad and p.numel() > 0 and "pgot_register_embeddings" in n
        }
        null_bg_names = {n for n, p in opt_model.named_parameters()
                         if p.requires_grad and "pgot_null_bg_embeddings" in n}
        rae_query_names = {n for n, p in opt_model.named_parameters()
                           if p.requires_grad and "latent_queries" in n}
        v14_names = {n for n, p in opt_model.named_parameters()
                     if p.requires_grad and "pgot_v14_router" in n}
        v21_names = {n for n, p in opt_model.named_parameters()
                     if p.requires_grad and "pgot_v21_router" in n}
        latent_head_names = {n for n, p in opt_model.named_parameters()
                             if p.requires_grad and "pgot_latent_head" in n}
        llm_names = {n for n, p in opt_model.named_parameters()
                     if p.requires_grad and (n.startswith("model.layers.")
                                             or n.startswith("model.model.layers.")
                                             or "lora_" in n)}

        assigned = (
            projector_names
            | mm_projector_names
            | diff_body_names
            | register_names
            | null_bg_names
            | rae_query_names
            | v14_names
            | v21_names
            | latent_head_names
            | llm_names
        )

        groups = []
        used_param_ids = set()

        def _take_params(names=None):
            params = []
            for n, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue
                if names is None:
                    if n in assigned:
                        continue
                elif n not in names:
                    continue
                pid = id(p)
                if pid in used_param_ids:
                    continue
                used_param_ids.add(pid)
                params.append(p)
            return params

        # default group (e.g., embed_tokens trainable rows, lm_head)
        default = _take_params(None)
        if default:
            groups.append({"params": default, "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate})

        params = _take_params(mm_projector_names)
        if params:
            groups.append({"params": params, "weight_decay": self.args.weight_decay, "lr": mm_projector_lr})
        params = _take_params(projector_names)
        if params:
            groups.append({"params": params, "weight_decay": self.args.weight_decay, "lr": diff_head_lr})
        params = _take_params(dit_adaln_names)
        if params:
            groups.append({"params": params, "weight_decay": self.args.weight_decay, "lr": diff_head_lr})
        params = _take_params(dit_body_names)
        if params:
            groups.append({"params": params, "weight_decay": self.args.weight_decay, "lr": dit_body_lr})
        params = _take_params(register_names)
        if params:
            groups.append({"params": params, "weight_decay": 0.0, "lr": register_lr})
        params = _take_params(null_bg_names)
        if params:
            groups.append({"params": params, "weight_decay": 0.0, "lr": register_lr})
        params = _take_params(rae_query_names)
        if params:
            groups.append({"params": params, "weight_decay": 0.0, "lr": rae_query_lr})
        params = _take_params(v14_names)
        if params:
            groups.append({"params": params, "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate})
        params = _take_params(v21_names)
        if params:
            groups.append({"params": params, "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate})
        params = _take_params(latent_head_names)
        if params:
            groups.append({"params": params, "weight_decay": self.args.weight_decay, "lr": diff_head_lr})
        params = _take_params(llm_names)
        if params:
            groups.append({"params": params, "weight_decay": self.args.weight_decay, "lr": llm_lr})

        logger.info(
            "[PGOT] optimizer | base=%g mm_projector=%g diff_head=%g dit_body=%g register=%g rae_query=%g llm=%g | "
            "mm_projector=%d projector=%d dit_adaln=%d dit_body=%d register=%d null_bg=%d rae_query=%d v14=%d v21=%d latent_head=%d llm=%d",
            self.args.learning_rate, mm_projector_lr, diff_head_lr, dit_body_lr, register_lr, rae_query_lr, llm_lr,
            len(mm_projector_names), len(projector_names), len(dit_adaln_names), len(dit_body_names), len(register_names),
            len(null_bg_names), len(rae_query_names), len(v14_names), len(v21_names), len(latent_head_names), len(llm_names),
        )

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        return self.optimizer

    # ----- Scheduler -----
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if bool(getattr(self.args, "pgot_use_cosine_min_lr_schedule", False)):
            if self.lr_scheduler is not None:
                return self.lr_scheduler

            target = optimizer if optimizer is not None else self.optimizer
            warmup_steps = int(self.args.get_warmup_steps(num_training_steps))
            min_ratio = float(getattr(self.args, "pgot_min_lr_ratio", 0.10))

            logger.info(
                "[PGOT] cosine-min scheduler: total=%d warmup=%d min_ratio=%.3f",
                num_training_steps, warmup_steps, min_ratio,
            )

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_ratio + (1.0 - min_ratio) * cosine

            from torch.optim.lr_scheduler import LambdaLR
            self.lr_scheduler = LambdaLR(target, lr_lambda)
            self._created_lr_scheduler = True
            return self.lr_scheduler

        return super().create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    # ----- compute_loss + contrastive gating -----
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Decide whether to enable contrastive at this step
        global_step = int(self.state.global_step)
        warmup = int(getattr(self.args, "pgot_contrastive_warmup_steps", 0) or 0)
        target_w = float(getattr(self.args, "pgot_contrastive_loss_target_weight", 0.0) or 0.0)
        if warmup <= 0 or global_step >= warmup:
            contrastive_w = target_w
        else:
            contrastive_w = 0.0

        inner_model = model.module if hasattr(model, "module") else model
        if bool(getattr(inner_model.config, "pgot_v21_enable", False)):
            start_w = float(getattr(inner_model.config, "pgot_v21_ground_weight", 0.0))
            final_w = float(getattr(inner_model.config, "pgot_v21_ground_final_weight", -1.0))
            anneal_steps = int(getattr(inner_model.config, "pgot_v21_ground_anneal_steps", 0) or 0)
            if final_w >= 0.0 and anneal_steps > 0:
                t = min(max(float(global_step) / float(max(anneal_steps, 1)), 0.0), 1.0)
                eff_w = start_w + (final_w - start_w) * t
            else:
                eff_w = start_w
            inner_model.config.pgot_v21_ground_weight_effective = float(eff_w)

        e4_warmup = int(getattr(self.args, "pgot_e4_loss_warmup_steps", 0) or 0)
        if e4_warmup > 0:
            e4_scale = min(
                max(float(global_step + 1) / float(e4_warmup), 0.0),
                1.0,
            )
        else:
            e4_scale = 1.0
        e4_full_target_w = float(
            getattr(inner_model.config, "pgot_e4_full_inside_weight", 0.0)
        )
        e4_bind_target_w = float(
            getattr(inner_model.config, "pgot_e4_rae_bind_weight", 0.0)
        )
        inner_model.config.pgot_e4_full_inside_weight_effective = (
            e4_full_target_w * e4_scale
        )
        inner_model.config.pgot_e4_rae_bind_weight_effective = (
            e4_bind_target_w * e4_scale
        )

        images = inputs.pop("images")
        target_images = inputs.pop("target_images")
        outputs = model(
            images=images,
            target_images=target_images,
            caption_input_ids=inputs["caption_input_ids"],
            caption_attention_mask=inputs["caption_attention_mask"],
            caption_labels=inputs.get("caption_labels"),
            ovt_positions_in_caption=inputs["ovt_positions_in_caption"],
            ovt_valid_mask=inputs["ovt_valid_mask"],
            ovt_is_thing=inputs.get("ovt_is_thing"),
            gt_masks_per_ovt=inputs["gt_masks_per_ovt"],
            gt_rae_masks_per_ovt=inputs.get("gt_rae_masks_per_ovt"),
            pgot_contrastive_weight=contrastive_w,
        )
        loss = outputs.loss

        # Record per-loss metrics for logging
        inner = inner_model
        for key, attr in [
            ("loss_lm", "pgot_loss_lm"),
            ("loss_mask", "pgot_loss_mask"),
            ("loss_recon", "pgot_loss_recon"),
            ("loss_contrastive", "pgot_loss_contrastive"),
            ("n_objects_mean", "pgot_n_objects_mean"),
        ]:
            val = getattr(inner, attr, None)
            if val is not None and torch.is_tensor(val):
                self._custom_loss_buffer[key] = float(val.detach().cpu().item())
        for key, val in getattr(inner, "pgot_loss_details", {}).items():
            if val is not None and torch.is_tensor(val):
                self._custom_loss_buffer[key] = float(val.detach().cpu().item())
        if target_w > 0.0 or contrastive_w > 0.0:
            self._custom_loss_buffer["contrastive_w"] = contrastive_w
        if e4_full_target_w > 0.0 or e4_bind_target_w > 0.0:
            self._custom_loss_buffer["e4_loss_warmup_scale"] = e4_scale
        self._loss_count_buffer += 1

        return (loss, outputs) if return_outputs else loss

    # ----- log() override to add custom metrics -----
    def log(self, logs: Dict[str, float], *args, **kwargs):
        if self._loss_count_buffer > 0:
            for k, v in self._custom_loss_buffer.items():
                logs[k] = v
            self._custom_loss_buffer.clear()
            self._loss_count_buffer = 0
        return super().log(logs, *args, **kwargs)

    # ----- prediction_step override: HF's default skips loss because our model
    #       doesn't use the "labels" key. We always call compute_loss to get
    #       a proper eval_loss tensor, and we accumulate per-sub-loss metrics.
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                # compute_loss pops images/target_images and populates
                # self._custom_loss_buffer with sub-loss scalars.
                loss = self.compute_loss(model, inputs, return_outputs=False)
            # Hijack the train-side buffer for eval aggregation (and clear it so
            # it doesn't leak into the next training log).
            for k, v in list(self._custom_loss_buffer.items()):
                if k == "contrastive_w":
                    continue
                self._eval_sub_loss_sums[k] = self._eval_sub_loss_sums.get(k, 0.0) + float(v)
            self._custom_loss_buffer.clear()
            self._loss_count_buffer = 0
            self._eval_sub_loss_count += 1
        loss = loss.detach()
        return (loss, None, None)

    # ----- evaluation_loop override: inject per-sub-loss averages into metrics
    #       BEFORE HF Trainer logs them, and trigger image-recon logging on rank 0.
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None,
                        ignore_keys=None, metric_key_prefix="eval"):
        # Reset accumulators for this eval call
        self._eval_sub_loss_sums = {}
        self._eval_sub_loss_count = 0

        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if self._eval_sub_loss_count > 0:
            for k, total in self._eval_sub_loss_sums.items():
                output.metrics[f"{metric_key_prefix}_{k}"] = total / self._eval_sub_loss_count
            # Keep a copy so evaluate() can re-inject the metrics right before
            # Trainer.log(). This is defensive for HF internals/callbacks that
            # may snapshot metrics before our evaluation_loop override returns.
            self._last_eval_sub_metrics = {
                f"{metric_key_prefix}_{k}": total / self._eval_sub_loss_count
                for k, total in self._eval_sub_loss_sums.items()
            }
        else:
            self._last_eval_sub_metrics = {}

        # Image recon logging (rank 0 only, wandb only, best-effort)
        try:
            self._log_eval_recon_images(dataloader, step_prefix=metric_key_prefix)
        except Exception as exc:
            logger.warning("[PGOT/eval] recon image logging failed (non-fatal): %s", exc)

        return output

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        # Explicitly log eval sub-losses once more after the parent evaluate()
        # path. Duplicate keys at the same step are harmless in wandb and keep
        # train-time eval panels populated even if HF internals/callbacks change.
        extra = {
            k: v for k, v in getattr(self, "_last_eval_sub_metrics", {}).items()
            if k.startswith(f"{metric_key_prefix}_")
        }
        if extra:
            metrics.update(extra)
            self.log(extra)
        return metrics

    # ----- Helper: load siglip2 decoder lazily -----
    def _get_eval_decoder(self):
        if self._eval_decoder is not None:
            return self._eval_decoder
        try:
            from huggingface_hub import hf_hub_download
            from scale_rae.model.multimodal_decoder import MultimodalDecoder
        except Exception as exc:
            logger.warning("[PGOT/eval] decoder import failed: %s", exc)
            self._eval_decoder = False
            return None
        try:
            inner = self.model.module if hasattr(self.model, "module") else self.model
            vision_towers = list(getattr(
                inner.config,
                "mm_vision_tower_aux_list",
                ["google/siglip2-so400m-patch14-224"],
            ))
            encoder_path = vision_towers[1] if len(vision_towers) > 1 else vision_towers[0]
            encoder_path = encoder_path.split("-interp")[0]
            num_patches = int(getattr(inner.config, "diffusion_target_token_len", 256))
            repo_id = getattr(self.args, "pgot_eval_decoder_repo", "nyu-visionx/siglip2_decoder")
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
            ckpt_path = hf_hub_download(repo_id=repo_id, filename="model.pt")
            self._eval_decoder = MultimodalDecoder(
                pretrained_encoder_path=encoder_path,
                general_decoder_config=config_path,
                num_patches=num_patches,
                drop_cls_token=True,
                decoder_path=ckpt_path,
            )
            logger.info("[PGOT/eval] decoder loaded from %s", repo_id)
            return self._eval_decoder
        except Exception as exc:
            logger.warning("[PGOT/eval] decoder load failed: %s", exc)
            self._eval_decoder = False
            return None

    # ----- Helper: build per-sample attention panels (returned as numpy arrays
    # so the caller can put them in a wandb.Table cell). BCE and spatial losses
    # use different normalizations, so log their views separately. -----
    def _build_pgot_attention_overlays_pack(self, inner, batch, n_images: int, source_pixels: torch.Tensor):
        from pgot.eval.pgot_inference import pgot_forward_eval
        device = next(inner.parameters()).device

        def _slice(x):
            return x[:n_images] if torch.is_tensor(x) else x
        ovt_pos = _slice(batch["ovt_positions_in_caption"])
        ovt_valid = _slice(batch["ovt_valid_mask"])
        internal_attention_enabled = (
            float(
                getattr(
                    inner.config, "pgot_mask_llm_attention_outside_weight", 0.0
                )
            ) > 0.0
            or float(
                getattr(
                    inner.config, "pgot_mask_llm_patch_outside_weight", 0.0
                )
            ) > 0.0
            or float(
                getattr(inner.config, "pgot_mask_llm_image_use_weight", 0.0)
            ) > 0.0
            or float(getattr(inner.config, "pgot_core_outside_weight", 0.0)) > 0.0
            or float(getattr(inner.config, "pgot_core_tail_weight", 0.0)) > 0.0
            or float(getattr(inner.config, "pgot_core_register_outside_weight", 0.0)) > 0.0
        )
        patch_internal_attention_enabled = (
            float(
                getattr(
                    inner.config, "pgot_mask_llm_patch_outside_weight", 0.0
                )
            ) > 0.0
            or float(
                getattr(inner.config, "pgot_mask_llm_image_use_weight", 0.0)
            ) > 0.0
            or float(getattr(inner.config, "pgot_core_outside_weight", 0.0)) > 0.0
            or float(getattr(inner.config, "pgot_core_tail_weight", 0.0)) > 0.0
            or float(getattr(inner.config, "pgot_core_register_outside_weight", 0.0)) > 0.0
        )
        out = pgot_forward_eval(
            inner,
            images=_slice(batch["images"]).to(device),
            target_images=_slice(batch["target_images"]).to(device),
            caption_input_ids=_slice(batch["caption_input_ids"]).to(device),
            caption_attention_mask=_slice(batch["caption_attention_mask"]).to(device),
            ovt_positions_in_caption=ovt_pos.to(device),
            ovt_valid_mask=ovt_valid.to(device),
            return_llm_qk_maps=False,
            return_llm_attention_maps=internal_attention_enabled,
        )
        ovt_logits = out["ovt_logits"]
        n_per_obj = int(inner.pgot_n_ovt_per_object)
        B, M, P = ovt_logits.shape
        side = int(round(P ** 0.5))
        K = M // n_per_obj
        if K * n_per_obj < M:
            ovt_logits = ovt_logits[:, : K * n_per_obj]

        v14_owner_enabled = (
            bool(getattr(inner.config, "pgot_v14_enable", False))
            and float(getattr(inner.config, "pgot_v14_route_weight", 1.0)) > 0.0
        )
        v21_owner_enabled = bool(getattr(inner.config, "pgot_v21_enable", False))
        owner_enabled = bool(getattr(inner.config, "pgot_v12_enable", False)) or v14_owner_enabled or v21_owner_enabled
        bce_enabled = (
            float(getattr(inner.config, "pgot_mask_bce_weight", 0.0)) > 0.0
            or float(getattr(inner.config, "pgot_mask_object_balanced_bce_weight", 0.0)) > 0.0
            or float(getattr(inner.config, "pgot_mask_sigmoid_outside_weight", 0.0)) > 0.0
        )
        spatial_enabled = (
            float(getattr(inner.config, "pgot_mask_spatial_outside_weight", 0.0)) > 0.0
            or float(getattr(inner.config, "pgot_mask_spatial_outside_log_weight", 0.0)) > 0.0
        )
        map_modes = {}
        void_maps_by_mode = {}
        if owner_enabled and out.get("ovt_object_probs") is not None:
            map_modes["ovt_owner"] = out["ovt_object_probs"].float()
            if out.get("ovt_void_probs") is not None and out["ovt_void_probs"].shape[1] > 0:
                void_maps_by_mode["ovt_owner"] = out["ovt_void_probs"].float().sum(dim=1)
            per_obj_valid = out.get("ovt_object_valid")
        else:
            per_obj_valid = ovt_valid.reshape(B, K, n_per_obj).any(dim=2)
        if (not owner_enabled) and (bce_enabled or not (spatial_enabled or internal_attention_enabled)):
            map_modes["sigmoid"] = (
                torch.sigmoid(ovt_logits.float())
                .reshape(B, K, n_per_obj, P)
                .mean(dim=2)
            )
        if (not owner_enabled) and spatial_enabled:
            spatial_temp = float(
                getattr(
                    inner.config,
                    "pgot_mask_spatial_outside_log_temperature",
                    getattr(inner.config, "pgot_mask_spatial_temperature", 1.0),
                )
            )
            map_modes["spatial"] = (
                torch.softmax(ovt_logits.float() / max(spatial_temp, 1e-6), dim=-1)
                .reshape(B, K, n_per_obj, P)
                .mean(dim=2)
            )
        if (not owner_enabled) and internal_attention_enabled and out.get("llm_attention_maps") is not None:
            internal_maps = out["llm_attention_maps"][:, : K * n_per_obj]
            internal_mode = (
                "llm_patch_attention"
                if patch_internal_attention_enabled
                else "llm_attention"
            )
            map_modes[internal_mode] = (
                internal_maps.float()
                .reshape(B, K, n_per_obj, P)
                .mean(dim=2)
            )
            if (
                out.get("llm_attention_void_maps") is not None
                and out["llm_attention_void_maps"].numel() > 0
            ):
                void_maps_by_mode[internal_mode] = (
                    out["llm_attention_void_maps"].float().mean(dim=1)
                )
            elif out.get("llm_attention_register_maps") is not None:
                # E1 has no semantic void. Multiple residual registers are
                # averaged into one background diagnostic tile.
                void_maps_by_mode[internal_mode] = (
                    out["llm_attention_register_maps"].float().mean(dim=1)
                )

        chunk_labels = self._decode_chunk_labels(
            tokenizer=self.tokenizer,
            caption_input_ids=_slice(batch["caption_input_ids"]),
            ovt_positions=ovt_pos,
            ovt_valid=ovt_valid,
            n_per_obj=n_per_obj,
            max_chars=40,
        )

        H, W = source_pixels.shape[-2:]
        images_by_mode = {mode: {} for mode in map_modes}
        labels_by_sample = []
        map_stats_by_sample = [[] for _ in range(min(B, n_images))]

        def _normalize_attention_for_viz(raw: torch.Tensor) -> torch.Tensor:
            """Contrast-normalize a probability map without turning it black.

            Internal attention values are normally O(1/1024), so direct image
            casting is nearly black.  Percentile normalization exposes spatial
            structure while a uniform-map fallback uses mid gray/red rather
            than falsely looking like an all-zero map.
            """
            x = torch.nan_to_num(raw.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
            flat = x.flatten()
            lo = torch.quantile(flat, 0.02)
            hi = torch.quantile(flat, 0.98)
            if float((hi - lo).abs()) < 1e-10:
                if float(x.abs().max()) < 1e-12:
                    return torch.zeros_like(x)
                return torch.full_like(x, 0.5)
            return ((x - lo) / (hi - lo)).clamp(0.0, 1.0)

        for b in range(min(B, n_images)):
            valid_obj_idx = per_obj_valid[b].nonzero(as_tuple=False).flatten().tolist()
            for mode, per_obj in map_modes.items():
                tiles = [source_pixels[b]]
                for k in valid_obj_idx[:8]:
                    mode_side = int(round(float(per_obj.shape[-1]) ** 0.5))
                    m = per_obj[b, k].reshape(mode_side, mode_side)
                    m = torch.nn.functional.interpolate(
                        m.unsqueeze(0).unsqueeze(0), size=(H, W),
                        mode="bilinear", align_corners=False,
                    )[0, 0].cpu()
                    if mode in {
                        "spatial", "llm_attention", "llm_patch_attention", "ovt_owner"
                    }:
                        raw_min, raw_max = float(m.min()), float(m.max())
                        map_stats_by_sample[b].append(
                            f"{mode}[{k}]={raw_min:.2e}..{raw_max:.2e}"
                        )
                        # Visualization-only contrast. Loss/eval still consume
                        # the untouched probabilities above.
                        m = _normalize_attention_for_viz(m)
                    m = m.clamp(0.0, 1.0)
                    color = torch.tensor([1.0, 0.15, 0.15]).view(3, 1, 1)
                    overlay = source_pixels[b] * (1 - 0.6 * m) + color * 0.6 * m
                    tiles.append(overlay.clamp(0.0, 1.0))
                if mode in void_maps_by_mode:
                    mode_side = int(round(float(void_maps_by_mode[mode].shape[-1]) ** 0.5))
                    m = void_maps_by_mode[mode][b].reshape(mode_side, mode_side)
                    m = torch.nn.functional.interpolate(
                        m.unsqueeze(0).unsqueeze(0), size=(H, W),
                        mode="bilinear", align_corners=False,
                    )[0, 0].cpu()
                    raw_min, raw_max = float(m.min()), float(m.max())
                    map_stats_by_sample[b].append(
                        f"{mode}[register]={raw_min:.2e}..{raw_max:.2e}"
                    )
                    m = _normalize_attention_for_viz(m)
                    color = torch.tensor([0.1, 0.35, 1.0]).view(3, 1, 1)
                    overlay = source_pixels[b] * (1 - 0.6 * m) + color * 0.6 * m
                    tiles.append(overlay.clamp(0.0, 1.0))
                panel = torch.cat(tiles, dim=2)
                images_by_mode[mode][b] = panel.permute(1, 2, 0).numpy()
            captions = " | ".join(chunk_labels[b][:8]) if b < len(chunk_labels) else ""
            if void_maps_by_mode:
                bg_label = "void" if int(getattr(inner, "pgot_n_null_bg", 0)) > 0 else "register-bg"
                captions = f"{captions} | {bg_label}" if captions else bg_label
            if map_stats_by_sample[b]:
                captions = f"{captions} || " + "; ".join(map_stats_by_sample[b][:10])
            labels_by_sample.append(captions)
        return {"images_by_mode": images_by_mode, "labels_by_sample": labels_by_sample}

    # Kept for back-compat with any caller that still uses the old per-key path.
    def _build_pgot_attention_overlays(self, inner, batch, n_images: int, source_pixels: torch.Tensor):
        """Run pgot_forward_eval-equivalent and produce overlay images for each OVT.

        Returns a dict like {'attn/example_0': wandb.Image, ...} where each image
        is a grid showing source + per-OVT attention overlays with chunk captions.
        """
        import wandb
        from pgot.eval.pgot_inference import pgot_forward_eval
        device = next(inner.parameters()).device

        # Slice to n_images
        def _slice(x):
            return x[:n_images] if torch.is_tensor(x) else x
        ovt_pos = _slice(batch["ovt_positions_in_caption"])
        ovt_valid = _slice(batch["ovt_valid_mask"])
        out = pgot_forward_eval(
            inner,
            images=_slice(batch["images"]).to(device),
            target_images=_slice(batch["target_images"]).to(device),
            caption_input_ids=_slice(batch["caption_input_ids"]).to(device),
            caption_attention_mask=_slice(batch["caption_attention_mask"]).to(device),
            ovt_positions_in_caption=ovt_pos.to(device),
            ovt_valid_mask=ovt_valid.to(device),
            return_llm_qk_maps=True,
        )
        ovt_logits = out["ovt_logits"]            # (B, M, P)
        attn_maps = out.get("llm_qk_attn_maps", None)
        n_per_obj = int(inner.pgot_n_ovt_per_object)
        B, M, P = ovt_logits.shape
        side = int(round(P ** 0.5))

        # Per-object map (mean over n_per_obj OVTs)
        K = M // n_per_obj
        if K * n_per_obj < M:
            ovt_logits = ovt_logits[:, : K * n_per_obj]
            if attn_maps is not None:
                attn_maps = attn_maps[:, : K * n_per_obj]
        uses_spatial_softmax = (
            float(getattr(inner.config, "pgot_mask_spatial_outside_weight", 0.0)) > 0.0
            or float(getattr(inner.config, "pgot_mask_spatial_outside_log_weight", 0.0)) > 0.0
        )
        if attn_maps is not None:
            per_obj = attn_maps.float().reshape(B, K, n_per_obj, P).mean(dim=2)  # (B, K, P)
        elif uses_spatial_softmax:
            spatial_temp = float(
                getattr(
                    inner.config,
                    "pgot_mask_spatial_outside_log_temperature",
                    getattr(inner.config, "pgot_mask_spatial_temperature", 1.0),
                )
            )
            per_ovt = torch.softmax(
                ovt_logits.float() / max(spatial_temp, 1e-6),
                dim=-1,
            )
            per_obj = per_ovt.reshape(B, K, n_per_obj, P).mean(dim=2)
        else:
            per_obj = torch.sigmoid(ovt_logits.float()).reshape(B, K, n_per_obj, P).mean(dim=2)  # (B, K, P)
        per_obj_valid = ovt_valid.reshape(B, K, n_per_obj).any(dim=2)  # (B, K)

        # Decode caption chunks (one short label per object) using tokenizer
        chunk_labels = self._decode_chunk_labels(
            tokenizer=self.tokenizer,
            caption_input_ids=_slice(batch["caption_input_ids"]),
            ovt_positions=ovt_pos,
            ovt_valid=ovt_valid,
            n_per_obj=n_per_obj,
            max_chars=40,
        )

        H, W = source_pixels.shape[-2:]
        log = {}
        for b in range(min(B, n_images)):
            valid_obj_idx = per_obj_valid[b].nonzero(as_tuple=False).flatten().tolist()
            tiles = [source_pixels[b]]
            captions = ["source"]
            # Up to 8 objects per panel
            for k in valid_obj_idx[:8]:
                m = per_obj[b, k].reshape(side, side)
                m = torch.nn.functional.interpolate(
                    m.unsqueeze(0).unsqueeze(0), size=(H, W),
                    mode="bilinear", align_corners=False,
                )[0, 0].cpu().clamp(0.0, 1.0)
                color = torch.tensor([1.0, 0.15, 0.15]).view(3, 1, 1)
                overlay = source_pixels[b] * (1 - 0.6 * m) + color * 0.6 * m
                tiles.append(overlay.clamp(0.0, 1.0))
                lbl = chunk_labels[b][k] if k < len(chunk_labels[b]) else f"obj{k}"
                captions.append(lbl[:32])
            if out.get("null_bg_logits") is not None:
                bg = torch.sigmoid(out["null_bg_logits"][b, 0].float()).reshape(side, side)
                bg = torch.nn.functional.interpolate(
                    bg.unsqueeze(0).unsqueeze(0), size=(H, W),
                    mode="bilinear", align_corners=False,
                )[0, 0].cpu().clamp(0.0, 1.0)
                color = torch.tensor([0.1, 0.35, 1.0]).view(3, 1, 1)
                overlay = source_pixels[b] * (1 - 0.55 * bg) + color * 0.55 * bg
                tiles.append(overlay.clamp(0.0, 1.0))
                captions.append("null-bg")

            # Concatenate horizontally
            panel = torch.cat(tiles, dim=2)
            cap_str = " | ".join(captions)
            log[f"attn/example_{b}"] = wandb.Image(
                panel.permute(1, 2, 0).numpy(),
                caption=f"step={self.state.global_step} | {cap_str}",
            )
        return log

    @staticmethod
    def _decode_chunk_labels(tokenizer, caption_input_ids, ovt_positions, ovt_valid, n_per_obj, max_chars=40):
        """For each sample, decode the tokens preceding each first-OVT position into a short label."""
        B = caption_input_ids.shape[0]
        labels: List[List[str]] = []
        for b in range(B):
            ids = caption_input_ids[b].tolist()
            valid_b = ovt_valid[b].tolist()
            ovt_pos_b = ovt_positions[b].tolist()
            # Pair OVTs in groups of n_per_obj
            sample_labels: List[str] = []
            K = len(valid_b) // n_per_obj
            for k in range(K):
                first = k * n_per_obj
                if not valid_b[first]:
                    continue
                # decode tokens from previous OVT-end (or 0) up to this first OVT position
                this_pos = ovt_pos_b[first]
                if k > 0 and valid_b[(k - 1) * n_per_obj + (n_per_obj - 1)]:
                    prev_end = ovt_pos_b[(k - 1) * n_per_obj + (n_per_obj - 1)] + 1
                else:
                    prev_end = 0
                start = max(prev_end, 0)
                end = max(this_pos, start)
                chunk = ids[start:end]
                text = tokenizer.decode(chunk, skip_special_tokens=True).strip()
                # Trim "<scene_end>", common punctuation, etc.
                text = text.replace("<ovt>", "").strip(" .:")
                if len(text) > max_chars:
                    text = text[:max_chars - 1] + "…"
                sample_labels.append(text if text else f"obj{k}")
            labels.append(sample_labels)
        return labels

    # ----- Helper: decode latent tokens (B, T, D) -> pixels (B, 3, H, W) -----
    @staticmethod
    def _decode_latents(decoder, latent, device):
        decoder = decoder.to(device=device)
        dec_dtype = next(decoder.parameters()).dtype
        if hasattr(decoder, "image_mean") and hasattr(decoder, "image_std"):
            decoder.image_mean = decoder.image_mean.to(device=device, dtype=dec_dtype)
            decoder.image_std = decoder.image_std.to(device=device, dtype=dec_dtype)
        if latent.dtype != dec_dtype:
            latent = latent.to(dtype=dec_dtype)
        empty_cls = torch.zeros((latent.shape[0], 1, latent.shape[-1]),
                                device=device, dtype=dec_dtype)
        feats = torch.cat([empty_cls, latent], dim=1)
        recon = decoder(feats)
        recon = torch.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=0.0)
        return recon.clamp(0.0, 1.0).detach().cpu().float()

    # ----- Helper: sample + decode + log to wandb -----
    def _log_eval_recon_images(self, dataloader, step_prefix="eval"):
        n_images = int(getattr(self.args, "pgot_eval_log_recon_images", 0) or 0)
        if n_images <= 0:
            return
        if self.args.local_rank not in (-1, 0):
            return
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        decoder = self._get_eval_decoder()
        if decoder is None or decoder is False:
            return

        # Pull one batch from the eval dataloader (small)
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            return
        # Truncate to n_images
        def _slice(x):
            return x[:n_images] if torch.is_tensor(x) else x
        batch = {k: _slice(v) for k, v in batch.items()}

        inner = self.model.module if hasattr(self.model, "module") else self.model
        was_training = inner.training
        inner.eval()
        device = next(inner.parameters()).device
        try:
            out = inner.pgot_sample_recon_latents(
                images=batch["images"].to(device),
                caption_input_ids=batch["caption_input_ids"].to(device),
                caption_attention_mask=batch["caption_attention_mask"].to(device),
                target_images=batch["target_images"].to(device),
                ovt_positions_in_caption=batch["ovt_positions_in_caption"].to(device),
                ovt_valid_mask=batch["ovt_valid_mask"].to(device),
            )
        finally:
            if was_training:
                inner.train()

        pred_latent = out["pred_latent"]
        gt_siglip = out["gt_siglip"]
        direct_latent = out.get("direct_latent")

        recon_pixels = self._decode_latents(decoder, pred_latent, device)
        gt_pixels = self._decode_latents(decoder, gt_siglip, device)
        direct_pixels = (
            self._decode_latents(decoder, direct_latent, device)
            if direct_latent is not None
            else None
        )

        # Denormalize source target_images from SigLIP-target normalization to [0,1].
        # We use the second image processor (siglip2_decoder-aligned target) if any.
        try:
            vt_list = inner.get_vision_tower_aux_list()
            tp = vt_list[1].image_processor if len(vt_list) > 1 else vt_list[0].image_processor
            mean = torch.tensor(tp.image_mean).view(1, -1, 1, 1)
            std = torch.tensor(tp.image_std).view(1, -1, 1, 1)
        except Exception:
            mean = torch.zeros(1, 3, 1, 1)
            std = torch.ones(1, 3, 1, 1)
        src_norm = batch["target_images"][:n_images].detach().cpu().float()
        source_pixels = (src_norm * std + mean).clamp(0.0, 1.0)

        # Resize the three so they share the same H, W (recon side wins).
        target_hw = recon_pixels.shape[-2:]
        def _match(x):
            if x.shape[-2:] != target_hw:
                x = torch.nn.functional.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
            return x
        source_pixels = _match(source_pixels)
        gt_pixels = _match(gt_pixels)
        recon_pixels = _match(recon_pixels)
        if direct_pixels is not None:
            direct_pixels = _match(direct_pixels)

        # Build ONE wandb.Table with columns: step, sample, source, gt_decoded, our_recon, attn_overlay, caption.
        # Each row is a sample at the current step. wandb UI shows two-axis navigation:
        # the run-history step slider + the table's sample column → "step × sample" grid.
        n = min(recon_pixels.shape[0], gt_pixels.shape[0], source_pixels.shape[0])
        if direct_pixels is not None:
            n = min(n, direct_pixels.shape[0])
        images_by_mode = {}
        chunk_labels_by_sample = [""] * n
        try:
            overlays_pack = self._build_pgot_attention_overlays_pack(
                inner=inner, batch=batch, n_images=n_images, source_pixels=source_pixels,
            )
            images_by_mode = overlays_pack["images_by_mode"]
            chunk_labels_by_sample = overlays_pack["labels_by_sample"]
        except Exception as e:
            logger.warning(f"[PGOT/viz] attention overlay failed: {e}")

        overlay_modes = list(images_by_mode.keys())
        columns = ["step", "sample", "source", "gt_decoded", "our_recon"]
        if direct_pixels is not None:
            columns.append("direct_recon")
        columns.extend(f"{mode}_overlay" for mode in overlay_modes)
        columns.append("chunks")
        table = wandb.Table(columns=columns)
        for i in range(n):
            src_np = source_pixels[i].permute(1, 2, 0).numpy()
            gt_np  = gt_pixels[i].permute(1, 2, 0).numpy()
            rec_np = recon_pixels[i].permute(1, 2, 0).numpy()
            row = [
                int(self.state.global_step),
                i,
                wandb.Image(src_np),
                wandb.Image(gt_np),
                wandb.Image(rec_np),
            ]
            if direct_pixels is not None:
                direct_np = direct_pixels[i].permute(1, 2, 0).numpy()
                row.append(wandb.Image(direct_np))
            for mode in overlay_modes:
                overlay = images_by_mode[mode].get(i, None)
                row.append(wandb.Image(overlay) if overlay is not None else None)
            row.append(
                chunk_labels_by_sample[i] if i < len(chunk_labels_by_sample) else ""
            )
            table.add_data(*row)

        wandb.log({f"{step_prefix}/eval_table": table}, step=int(self.state.global_step))
