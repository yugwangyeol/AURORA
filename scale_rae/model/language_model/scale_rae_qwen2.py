#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import List, Optional, Tuple, Union, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import os
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2Model, Qwen2ForCausalLM
# import torch_xla.amp as amp # Can use torch.autocast as well

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)

from ..scale_rae_arch import ScaleRAEMetaModel, ScaleRAEMetaForCausalLM, apply_custom_kernel
from ..object_centric import (
    build_aurora_v2_attention_mask,
    sample_k_for_batch,
    hungarian_match,
    compute_mask_loss,
    compute_diversity_loss,
    extract_attention_logits,
    extract_attention_maps,
)

from scale_rae.utils import IS_XLA_AVAILABLE

if IS_XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm # <-- Import XLA model




def ensure_float32(module: nn.Module):
    """Convert all floating-point parameters and buffers to float32."""
    for param in module.parameters(recurse=True):
        if param.is_floating_point() and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
    for buf in module.buffers(recurse=True):
        if buf.is_floating_point() and buf.dtype != torch.float32:
            buf.data = buf.data.to(torch.float32)





class ScaleRAEQwenConfig(Qwen2Config):
    model_type = "cambrian_qwen"
    #@Peter: Hardcode diffusion loss for now, need to be changed later
    vision_loss = "regression-loss"
    vision_loss_mode = "causal"
    vision_tower_aux_token_len_list = [256]  # Default vision token length
    use_aurora = False
    aurora_max_slots = 10        # K_max: object embedding pool size
    aurora_n_register = 8        # R: number of register tokens
    aurora_cmd_length = 8        # C: number of cmd tokens
    aurora_mask_loss_weight = 1.0    # λ_mask
    aurora_diversity_loss_weight = 0.1  # λ_div
    aurora_inpaint_weight = 0.5
    aurora_fail_on_nan = True
    aurora_training_stage = 1    # 1 = decomposition, 2 = inpainting
    aurora_grad_clip_max_norm = 1.0


class ScaleRAEQwenModel(ScaleRAEMetaModel, Qwen2Model):
    config_class = ScaleRAEQwenConfig

    def __init__(self, config: Qwen2Config):
        if IS_XLA_AVAILABLE:
            config._attn_implementation = "eager"
        super(ScaleRAEQwenModel, self).__init__(config)


    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vision_tower_aux_feature_list: Optional[List[torch.FloatTensor]] = None,
        vision_tower_aux_attention_masks_list: Optional[List[torch.Tensor]] = None,
        final_vision_feature_size: Optional[List[tuple]] = None,
        global_context_feature: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:





        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_bias is not None:
            attention_mask = attention_bias
        elif attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if attention_bias is None:
            if self._attn_implementation == "flash_attention_2":
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            elif self._attn_implementation == "sdpa" and not output_attentions:
                # output_attentions=True can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            else:
                    if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
                        # ! NOTE@shusheng: this is a hack to speed up the training
                        # ! NOTE@shusheng: we use torch_xla's flash attention which does not require mask
                        attention_mask = None
                    else:
                        # 4d mask is passed through the layers
                        attention_mask = _prepare_4d_causal_attention_mask(
                            attention_mask,
                            (batch_size, seq_length),
                            inputs_embeds,
                            past_key_values_length,
                            sliding_window=self.config.sliding_window,
                        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

def compute_feature_loss(predictions, targets, valid_positions, loss_type='l2'):
    """
    Compute per-token feature prediction loss, normalized by feature dimension.
    
    Args:
        predictions: Tensor of shape [bs, seq_len, dim]
        targets: Tensor of shape [bs, seq_len, dim]
        valid_positions: Tensor of shape [bs, seq_len] - binary mask (1 for valid tokens)
        loss_type: One of 'l1', 'l2', or 'smooth_l1'
    
    Returns:
        Mean loss per valid token, normalized by feature dimension
    """
    # Expand mask to feature dimension
    mask_expanded = valid_positions.unsqueeze(-1).expand_as(predictions)
    
    # Get feature dimension for normalization
    feature_dim = predictions.size(-1)
    
    # Apply loss function
    if loss_type == 'l1':
        # L1 loss (Mean Absolute Error)
        diff = torch.abs(predictions - targets)
    elif loss_type == 'smooth_l1':
        # Smooth L1 loss (Huber loss)
        diff = torch.nn.functional.smooth_l1_loss(
            predictions, targets, reduction='none')
    else:  # default to 'l2'
        # L2 loss (Mean Squared Error)
        diff = (predictions - targets)**2
    
    # Mask the differences
    masked_diff = diff * mask_expanded
    
    # Sum across feature dimension and divide by feature dim to get average per dimension
    # This gives us per-token loss normalized by feature dimension
    per_token_loss = masked_diff.sum(dim=-1) / feature_dim  # [bs, seq_len] 
    
    # Get mean of only valid tokens in a TPU-compatible way
    epsilon = 1e-8
    total_loss = per_token_loss.sum()
    num_valid = valid_positions.sum() + epsilon
    mean_loss = total_loss / num_valid
    
    return mean_loss

from ..diffusion_loss.diffloss import create_rf_projector

class ScaleRAEQwenForCausalLM(Qwen2ForCausalLM, ScaleRAEMetaForCausalLM):
    config_class = ScaleRAEQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "cambrian_qwen"
        config.rope_scaling = None

        self.model = ScaleRAEQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing

        self.vision_loss = getattr(config, 'vision_loss', 'diffusion-loss')
        self.vision_loss_mode = getattr(config, 'vision_loss_mode', 'causal')
        self.vision_coef = getattr(config, 'vision_coef', 1.0)
        self.vision_tower_aux_token_len_list = getattr(config, 'vision_tower_aux_token_len_list', [256])  # Default vision token length
        self.diffusion_model_channels = getattr(config, 'diffusion_model_channels', 1152)
        self.num_image_tokens = 256 # fixed
        self.debug = False
        if self.vision_loss == 'diffusion-loss' or self.vision_loss == 'ddt-loss':
            if self.vision_loss_mode == 'causal':
                self.diff_head_config =  {
                    "diffusion_tokens": 1, # default = 1, we actually hardcode causal diffusion to 1 token
                    "diffusion_channels": self.diffusion_model_channels,
                    "z_channels": config.hidden_size, # Qwen-2.5 7B 3584
                    "model_hidden_size": config.diffusion_model_hidden_size, # default = 1152
                    "model_depth": config.diffusion_model_depth, # default = 12
                    "model_heads": config.diffusion_model_heads,
                    "guidance_scale": 2.0,
                }
                
                self.diff_head_config["use_mlp"] = True
                # NEW: optional normalization stats path
                if hasattr(config, 'diffusion_norm_stats_path') and config.diffusion_norm_stats_path:
                    self.diff_head_config["batchnorm_path"] = config.diffusion_norm_stats_path

            elif self.vision_loss_mode == 'ar-ddt':
                print("Using AR-DDT diffusion-loss for image feature prediction, split_per_token, model_hidden_size, model_depth and diffusion_model_z_channels:", config.diffusion_split_per_token, config.diffusion_model_hidden_size, config.diffusion_model_depth, config.diffusion_model_z_channels)
                
                # Check if we should use DDT architecture
                use_ddt = (self.vision_loss == "ddt-loss")
                print(f"AR-DDT mode with vision_loss='{self.vision_loss}', use_DDT={use_ddt}")
                
                self.diff_head_config = {
                    "diffusion_tokens": self.vision_tower_aux_token_len_list[0], # default = 256
                    "diffusion_channels": self.diffusion_model_channels,
                    "z_channels": config.hidden_size if config.diffusion_model_z_channels == 0 else config.diffusion_model_z_channels, # Qwen-2.5 7B 3584
                    "model_hidden_size": config.diffusion_model_hidden_size, # default = 1152
                    "model_depth": config.diffusion_model_depth, # default = 12
                    "model_heads": config.diffusion_model_heads,
                    "guidance_scale": 1.0,
                    "use_mlp": False,
                    "use_DDT": use_ddt,  # Enable DDT when vision_loss == "ddt-loss"
                }
                if hasattr(config, 'diffusion_base_dim') and config.diffusion_base_dim is not None:
                    self.diff_head_config["base_dim"] = config.diffusion_base_dim
                # NEW: optional normalization stats path
                if hasattr(config, 'diffusion_norm_stats_path') and config.diffusion_norm_stats_path:
                    self.diff_head_config["batchnorm_path"] = config.diffusion_norm_stats_path
                
                # Add DDT-specific parameters when using DDT
                if use_ddt:
                    try:
                        cls_prob = config.diffusion_class_dropout_prob
                    except:
                        cls_prob = 0.0
                        print("diffusion_class_dropout_prob not found in config, using default 0.0")
                        
                    self.diff_head_config.update({
                        "DDT_encoder_depth": config.ddt_encoder_depth,
                        "class_dropout_prob": cls_prob
                    })

                if config.diffusion_model_z_channels != 0:
                    self.diff_head_projector = nn.Linear(config.hidden_size, config.diffusion_model_z_channels)
                    self.use_diff_head_projector = True
                else:
                    self.use_diff_head_projector = False



            elif self.vision_loss_mode == 'query' or self.vision_loss_mode == 'query-block':

                if self.vision_loss == "ddt-loss":

                    try:
                        cls_prob = config.diffusion_class_dropout_prob
                    except:
                        cls_prob = 0.0
                        print("diffusion_class_dropout_prob not found in config, using default 0.0")

                    self.diff_head_config =  {
                        "diffusion_tokens": self.vision_tower_aux_token_len_list[0], # default = 1
                        "diffusion_channels": self.diffusion_model_channels,
                        "z_channels": config.hidden_size if config.diffusion_model_z_channels == 0 else config.diffusion_model_z_channels, # QEwen-2.5 7B 3584
                        "model_hidden_size": config.diffusion_model_hidden_size, # default = 1152
                        "model_depth": config.diffusion_model_depth, # default = 12
                        "model_heads": config.diffusion_model_heads,
                        "guidance_scale": 1.0,
                        "use_mlp": False,
                        "class_dropout_prob": cls_prob
                    }
                    if type(config.diffusion_model_hidden_size) != list:
                        # This means the classic DDT
                        self.diff_head_config["use_DDT"] = True
                        self.diff_head_config["DDT_encoder_depth"] = config.ddt_encoder_depth
                                        # optional: pass base dimension for diffusion scaling if defined
                    if hasattr(config, 'diffusion_base_dim') and config.diffusion_base_dim is not None:
                        self.diff_head_config["base_dim"] = config.diffusion_base_dim
                    # NEW: optional normalization stats path
                    if hasattr(config, 'diffusion_norm_stats_path') and config.diffusion_norm_stats_path:
                        self.diff_head_config["batchnorm_path"] = config.diffusion_norm_stats_path


                
                
                else:
                    
                    try:
                        cls_prob = config.diffusion_class_dropout_prob
                    except:
                        cls_prob = 0.0
                        print("diffusion_class_dropout_prob not found in config, using default 0.0")


                    self.diff_head_config =  {
                        "diffusion_tokens": self.vision_tower_aux_token_len_list[0], # default = 1
                        "diffusion_channels": self.diffusion_model_channels,
                        "z_channels": config.hidden_size if config.diffusion_model_z_channels == 0 else config.diffusion_model_z_channels, # QEwen-2.5 7B 3584
                        "model_hidden_size": config.diffusion_model_hidden_size, # default = 1152
                        "model_depth": config.diffusion_model_depth, # default = 12
                        "model_heads": config.diffusion_model_heads,
                        "guidance_scale": 1.0,
                        "cond_silu": True, # assume the run is before 0711
                        "class_dropout_prob": cls_prob
                    }
                    if hasattr(config, 'diffusion_base_dim') and config.diffusion_base_dim is not None:
                        self.diff_head_config["base_dim"] = config.diffusion_base_dim
                    # NEW: optional normalization stats path
                    _norm_path = getattr(config, 'diffusion_norm_stats_path', None)
                    print(f"[AURORA-INIT] diffusion_norm_stats_path = {_norm_path}")
                    if _norm_path:
                        self.diff_head_config["batchnorm_path"] = _norm_path


                if config.diffusion_model_z_channels != 0:
                    self.diff_head_projector = nn.Linear(config.hidden_size, config.diffusion_model_z_channels)
                    self.use_diff_head_projector = True
                else:
                    self.use_diff_head_projector = False

                self.diff_head_config["use_mlp"] = False

            
            # Ensure diffusion backbone selection is propagated
            self.diff_head_config["dit_cls"] = getattr(config, 'dit_cls', 'DiT')

            # # Add use_mlp=True if diffusion_tokens is 1RF
            # if self.diff_head_config["split_per_token"] == 1:
            #     print("Calling from MLP")
            #     self.diff_head_config["use_mlp"] = True
            # else:
            #     print("Calling from DiT")
            #     self.diff_head_config["use_mlp"] = False


            self.diff_head = create_rf_projector(self.diff_head_config)
            print(f"[AURORA-INIT] diff_head.normalize_data = {getattr(self.diff_head, 'normalize_data', 'N/A')}")

            # self.conditioning_preprocessor = nn.Sequential(
            #     nn.LayerNorm(config.hidden_size),
            #     # nn.Linear(config.hidden_size, config.hidden_size), # Or project to diff_head.z_channels if different
            #     # nn.LayerNorm(config.hidden_size),
            #     # You could add an activation here too if desired, e.g., nn.GELU()
            # )

            self.set_to_fp32 = False

            # Optional: auxiliary regression head trained alongside diffusion
            self.aux_regression_enabled = getattr(config, 'aux_regression', False)
            self.aux_regression_coef = getattr(config, 'aux_regression_coef', 1.0)
            if self.aux_regression_enabled:
                self.aux_vision_head = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.GELU(),
                    nn.Linear(config.hidden_size, 1152),
                )




        elif self.vision_loss == 'regression-loss':
            print("Using regression-loss for image feature prediction")
            self.vision_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, 1152),
            )

        # exit()

        self.post_init()

        # AURORA: initialize after post_init so _init_weights does not reset zero-inits
        if getattr(config, 'use_aurora', False):
            self._init_aurora()

    def load_vision_head(self, model_args):
        pretrain_adapter_and_vision_head = getattr(model_args, 'pretrain_adapter_and_vision_head', None)
        print(f"pretrain_adapter_and_vision_head is: {pretrain_adapter_and_vision_head}")

        if pretrain_adapter_and_vision_head is not None and os.path.isfile(pretrain_adapter_and_vision_head):
            print(f"[DEBUG] Loading adapter and vision head weights from: {pretrain_adapter_and_vision_head}")
            adapter_vision_weights = torch.load(pretrain_adapter_and_vision_head, map_location='cpu')
            print(f"[DEBUG] Keys in loaded checkpoint: {list(adapter_vision_weights.keys())}")
            
            # Extract from 'model' key if it exists
            if 'model' in adapter_vision_weights:
                model_weights = adapter_vision_weights['model']
                print(f"[DEBUG] Keys in model: {list(model_weights.keys())}")
            else:
                model_weights = adapter_vision_weights

            def get_w(weights, keyword):
              return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword+'.' in k}
            
            
            # Load vision_head if present
            if hasattr(self, 'vision_head') and any('vision_head.' in k for k in model_weights.keys()):
                print("[DEBUG] Loading vision_head weights.")
                self.vision_head.load_state_dict(get_w(model_weights, 'vision_head'), strict=False)
            else:
                print("[DEBUG] No vision_head weights found in the checkpoint, skipping loading.")
            
            # Load diff_head if present
            if hasattr(self, 'diff_head') and any('diff_head.' in k for k in model_weights.keys()):
                print("[DEBUG] Loading diff_head weights.")
                self.diff_head.load_state_dict(get_w(model_weights, 'diff_head'), strict=False)
            else:
                print("[DEBUG] No diff_head weights found in the checkpoint, skipping loading.")
        elif pretrain_adapter_and_vision_head is not None:
            print(
                f"[DEBUG] Skipping vision/diff extra load because path is not a file: "
                f"{pretrain_adapter_and_vision_head}"
            )

    # <<< --- ADD THIS OVERRIDE --- >>>
    def _init_weights(self, module):
        """
        Override the base _init_weights. Initialize base model parts as default,
        but SKIP re-initialization for our custom diff_head.
        """
        # Check if the module belongs to the diff_head
        is_in_diff_head = False
        if hasattr(self, 'diff_head'):
            for name, child_module in self.diff_head.named_modules():
                if module is child_module:
                    is_in_diff_head = True
                    break

        if is_in_diff_head:
            # If it's part of diff_head, do nothing - let its own init stand.
            pass
        else:
            # If it's part of the base model, call the parent's _init_weights
            # (which contains the standard Qwen2 initialization logic)
            super()._init_weights(module) # Calls Qwen2ForCausalLM's _init_weights


    def set_diff_fp32(self):
        if not self.set_to_fp32:
            ensure_float32(self.diff_head.model)
            self.set_to_fp32 = True


    def get_model(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        vision_token_indices: Optional[torch.Tensor] = None,
        decoding: Optional[bool] = False,
        answer_img_mask: Optional[torch.Tensor] = None,
        reverse_vti: Optional[torch.Tensor] = None,
        answer_token_mask: Optional[torch.Tensor] = None,
        guidance_level: Optional[float] = 1.0,
        images_gen: Optional[torch.FloatTensor] = None,
        n_objects: Optional[torch.Tensor] = None,
        gt_image_features: Optional[torch.Tensor] = None,
        gt_masks_patches: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
        inpaint_mask_patches: Optional[torch.Tensor] = None,
        has_inpaint: Optional[torch.Tensor] = None,
        aurora_inpaint_weight_override: Optional[float] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # AURORA v2 routing
        if getattr(self.config, 'use_aurora', False) and not decoding and images is not None:
            aurora_loss, aurora_info = self._forward_aurora(
                images=images,
                n_objects=n_objects,
                gt_masks_patches=gt_masks_patches,
                target_images=target_images,
                inpaint_mask_patches=inpaint_mask_patches,
                has_inpaint=has_inpaint,
                aurora_inpaint_weight_override=aurora_inpaint_weight_override,
            )
            return CausalLMOutputWithPast(
                loss=aurora_loss,
                logits=torch.zeros(1, device=images.device),
            )
        
        selected_features, input_embed_mask, attention_bias, extra_mm = None, None, None, None

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                selected_features,
                input_embed_mask,
                attention_bias,
                extra_mm,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                vision_token_indices=vision_token_indices,
                answer_img_mask=answer_img_mask,
                reverse_vti=reverse_vti,
                answer_token_mask=answer_token_mask,
                images_gen=images_gen,
            )
        
        # Store attention bias for training patch to access
        if attention_bias is not None:
            # Store on all attention layers so the training patch can access it
            for layer in self.get_model().layers:
                layer.self_attn._current_attention_bias = attention_bias
        else:
            # Clear attention bias when not needed
            for layer in self.get_model().layers:
                if hasattr(layer.self_attn, '_current_attention_bias'):
                    layer.self_attn._current_attention_bias = None
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if IS_XLA_AVAILABLE:
            # Very Important for TorchXLA
            #self.model.gradient_checkpointing = False
                
            from torch_xla.utils.checkpoint import checkpoint
            self.model._gradient_checkpointing_func = checkpoint

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # training
        if IS_XLA_AVAILABLE:
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else: # inference
            if hasattr(self, "vision_tower_aux_feature_list"):
                raise NotImplementedError("vision_tower_aux_feature_list should not be set in inference mode")
            else:
                # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

        hidden_states = outputs[0]


        logits = self.lm_head(hidden_states)
        
        
        if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
            import torch_xla.distributed.spmd as xs
            
        # if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
        #     xs.mark_sharding(logits, xs.get_global_mesh(), ("fsdp", None, "mp"))


            
        logits = logits.float()


        if decoding:
            vision_loss_mode = self.vision_loss_mode
            use_query_mode = vision_loss_mode == "query" or vision_loss_mode == "half-query" or vision_loss_mode == "query-block"
            # Currently, this code assumes regression loss
            generated_token_length = 1 if not use_query_mode else self.num_image_tokens
            if self.vision_loss == 'regression-loss':
                pred_z = hidden_states[:, -generated_token_length:, :].squeeze(1)  # (B, hidden_dim) for next toekn, (B, L, hidden_dim) for multiple tokens 

                print(f"[DEBUG] Hidden states shape before regression loss: {hidden_states.shape}, pred_z shape: {pred_z.shape}")
                
                pred_z = self.vision_head(pred_z)

                prediction = self.get_model().mm_projector(pred_z)

                pred_patches = pred_z
                prediction_target = answer_token_mask if answer_token_mask is not None else None

                
                valid_mask_patch = torch.ones(prediction.shape[0], prediction.shape[1], dtype=torch.int, device=prediction.device)

                feature_loss_type = getattr(self.config, 'feature_loss_type', 'l2')
                feature_loss = compute_feature_loss(
                    pred_patches,                # (B,T,F)
                    prediction_target,           # (B,T,F)
                    valid_mask_patch,            # (B,T)
                    loss_type=feature_loss_type
                )

                eps = 1e-8

                # Cosine loss (patch-level)
                norm_pred = torch.nn.functional.normalize(pred_patches + eps, dim=-1)
                norm_tgt  = torch.nn.functional.normalize(prediction_target + eps, dim=-1)
                cosine_sim = (norm_pred * norm_tgt).sum(-1)  # (B,T)
                masked_cos = cosine_sim * valid_mask_patch
                avg_cosine = masked_cos.sum() / (valid_mask_patch.sum() + eps)
                cosine_loss = 1.0 - avg_cosine

                print("[DEBUG] Feature loss:", feature_loss.item(), "Cosine loss:", cosine_loss.item())
                # exit()


                hidden_states[:, -generated_token_length:, :] = prediction
                
                print(f"[DEBUG] Hidden states shape after regression loss: {hidden_states.shape}, pred_z shape: {pred_z.shape}, prediction shape: {prediction.shape}")
            elif self.vision_loss == 'diffusion-loss' or self.vision_loss == 'ddt-loss':
                pred_z = hidden_states[:, -generated_token_length:, :].squeeze(1) 
                hidden_pred_z = pred_z.clone().detach()
                 # (B, hidden_dim) for next toekn, (B, L, hidden_dim) for multiple tokens
                
                # Ensure diff_head is on the same device as pred_z (important for multi-GPU with accelerate)
                target_device = pred_z.device
                self.diff_head = self.diff_head.to(target_device)
                
                if self.use_diff_head_projector:
                    pred_z = self.diff_head_projector(pred_z)

                pred_z = self.diff_head.infer(pred_z, guidance_level=guidance_level)


                try:

                    prediction = self.get_model().mm_projector(pred_z)
                except:
                    prediction = hidden_pred_z.clone().detach()

                hidden_states[:, -generated_token_length:, :] = prediction
                
            
            else:
                raise NotImplementedError(f"Decoding mode not implemented for vision_loss type: {self.vision_loss}")

            

        if decoding:
            return CausalLMOutputWithPast(
                loss=pred_z,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=hidden_states,
                attentions=outputs.attentions,
            )






        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            loss = loss_fct(shift_logits, shift_labels)


            self.loss_language = loss



            vision_loss_mode_cfg = getattr(self.get_model().config, 'vision_loss_mode', 'causal')

            # ------------------------------------------------------------------
            # QUERY-MODE  (answer-image tokens are latent queries)
            # ------------------------------------------------------------------
            if vision_loss_mode_cfg == "query" or vision_loss_mode_cfg == "query-block":

                if self.vision_loss == 'regression-loss':
                    # Expecting extra_mm = (image_features, reverse_vti, answer_img_mask)
                    img_feats_raw, reverse_vti, answer_img_mask, prediction_target = extra_mm

                    B, T, feature_dim = img_feats_raw.shape
                    M = answer_img_mask.size(1)
                    tokens_per_image = T // M

                    hidden_dim = hidden_states.size(-1)

                    # 1. Hidden states aligned to sequence indices (pad dummy at t=0)
                    # hs_pred = hidden_states[:, :-1]  # (B, L-1, hidden)
                    # pad_zero_hs = torch.zeros(B, 1, hidden_dim, dtype=hs_pred.dtype, device=hs_pred.device)
                    # hs_full = torch.cat([pad_zero_hs, hs_pred], dim=1)  # (B, L, hidden)
                    hs_full  = hidden_states  # (B, L, hidden)

                    # 2. Build zeros placeholder for left chunk (B,T,hidden)
                    zeros_left_hs = torch.zeros(B, T, hidden_dim, dtype=hs_full.dtype, device=hs_full.device)

                    # 3. Gather hidden states using kernel-ready reverse_vti
                    patch_hs = apply_custom_kernel(
                        zeros_left_hs,  # length T (image side)
                        hs_full,        # length Lmax text side
                        reverse_vti     # kernel-ready indices (B,T)
                    )  # (B,T,hidden)

                    # # --------------------------- DEBUG SHAPES ---------------------------
                    # dbg_shape("img_feats_raw", img_feats_raw)
                    # dbg_shape("reverse_vti", reverse_vti)
                    # dbg_shape("answer_img_mask", answer_img_mask)
                    # dbg_shape("prediction_target", prediction_target)
                    # dbg_shape("patch_hs", patch_hs)

                    # 5. Predict features from gathered hidden states
                    pred_patches = self.vision_head(patch_hs)  # (B,T,feat)

                    # dbg_shape("pred_patches", pred_patches)

                    # ---- Patch-level mask identical to non-query branch ----
                    valid_mask_patch = answer_img_mask.unsqueeze(-1).expand(B, M, tokens_per_image)  # (B,M,P)
                    valid_mask_patch = valid_mask_patch.reshape(B, T).int()

                    eps = 1e-8

                    feature_loss_type = getattr(self.config, 'feature_loss_type', 'l2')
                    feature_loss = compute_feature_loss(
                        pred_patches,                # (B,T,F)
                        prediction_target,           # (B,T,F)
                        valid_mask_patch,            # (B,T)
                        loss_type=feature_loss_type
                    )

                    # Cosine loss (patch-level)
                    norm_pred = torch.nn.functional.normalize(pred_patches + eps, dim=-1)
                    norm_tgt  = torch.nn.functional.normalize(prediction_target + eps, dim=-1)
                    cosine_sim = (norm_pred * norm_tgt).sum(-1)  # (B,T)
                    masked_cos = cosine_sim * valid_mask_patch
                    avg_cosine = masked_cos.sum() / (valid_mask_patch.sum() + eps)
                    cosine_loss = 1.0 - avg_cosine

                    img_loss = feature_loss + cosine_loss
                    loss = loss + img_loss * self.vision_coef

                    self.loss_image_mse = feature_loss
                    self.loss_image_cos = cosine_loss

                elif self.vision_loss == 'diffusion-loss' or self.vision_loss == 'ddt-loss':
                    # --------------------------------------------------------------
                    # Diffusion loss (query-mode)
                    # --------------------------------------------------------------

                    # Unpack multimodal outputs
                    img_feats_raw, reverse_vti, answer_img_mask, prediction_target = extra_mm

                    B, T, feature_dim = prediction_target.shape
                    M = answer_img_mask.size(1)
                    tokens_per_image = T // M

                    hidden_dim = hidden_states.size(-1)

                    # Gather hidden states corresponding to each image patch via reverse_vti
                    hs_full = hidden_states  # (B, L, hidden)
                    zeros_left_hs = torch.zeros(B, T, hidden_dim, dtype=hs_full.dtype, device=hs_full.device)
                    patch_hs = apply_custom_kernel(zeros_left_hs, hs_full, reverse_vti)  # (B, T, hidden)


                    # 1. Reshape inputs for training_loss: (B, T, dim) -> (B*M, tokens_per_image, dim)
                    # patch_hs: (B, T, hidden_dim) -> (B, M, tokens_per_image, hidden_dim) -> (B*M, tokens_per_image, hidden_dim)
                    patch_hs_reshaped = patch_hs.view(B, M, tokens_per_image, hidden_dim).view(B*M, tokens_per_image, hidden_dim)
                    if self.use_diff_head_projector:
                        # Project to diffusion model z_channels if needed
                        patch_hs_reshaped = self.diff_head_projector(patch_hs_reshaped)

                    
                    # prediction_target: (B, T, feature_dim) -> (B, M, tokens_per_image, feature_dim) -> (B*M, tokens_per_image, feature_dim)
                    prediction_target_reshaped = prediction_target.view(B, M, tokens_per_image, feature_dim).view(B*M, tokens_per_image, feature_dim)

                    # Compute diffusion loss per image, with optional K-tiling of timesteps per sample.
                    # We reuse internal timestep sampling by passing t=None.
                    K = int(getattr(self.config, 'diffusion_timesteps_per_sample', 1) or 1)
                    if K > 1:
                        # Tile along the image batch dimension
                        z_tiled = torch.repeat_interleave(patch_hs_reshaped, repeats=K, dim=0)              # (B*M*K, P, hidden)
                        x_tiled = torch.repeat_interleave(prediction_target_reshaped, repeats=K, dim=0)     # (B*M*K, P, feat)
                        loss_vec = self.diff_head.training_loss(z=z_tiled, x=x_tiled)                        # (B*M*K,)
                        diffusion_loss_per_image = loss_vec.view(B * M, K).mean(dim=1)                       # (B*M,)
                    else:
                        diffusion_loss_per_image = self.diff_head.training_loss(
                            z=patch_hs_reshaped, x=prediction_target_reshaped
                        )  # (B*M,)
                    
                    # 2. Reshape back to (B, M) for image-level masking
                    diffusion_loss_per_image = diffusion_loss_per_image.view(B, M)
                    
                    # Apply image-level masking using answer_img_mask (B, M)
                    masked_loss_per_image = diffusion_loss_per_image * answer_img_mask.float()


                    eps = 1e-8
                    mean_diffusion_loss = masked_loss_per_image.sum() / (answer_img_mask.sum() + eps)

                    # Aggregate into total loss
                    loss = loss + mean_diffusion_loss * self.vision_coef

                    self.loss_image_diff = mean_diffusion_loss


                    # --------------------------------------------------------------
                    # Optional auxiliary regression loss (query-mode)
                    # --------------------------------------------------------------
                    if getattr(self, 'aux_regression_enabled', False):
                        # Reuse gathered hidden states and targets
                        pred_patches_aux = self.aux_vision_head(patch_hs)  # (B,T,feat)
                        valid_mask_patch = answer_img_mask.unsqueeze(-1).expand(B, M, tokens_per_image)
                        valid_mask_patch = valid_mask_patch.reshape(B, T).int()

                        feature_loss_type = getattr(self.config, 'feature_loss_type', 'l2')
                        aux_feature_loss = compute_feature_loss(
                            pred_patches_aux,
                            prediction_target,
                            valid_mask_patch,
                            loss_type=feature_loss_type
                        )
                        # optional cosine term mirroring regression branch
                        eps = 1e-8
                        norm_pred = torch.nn.functional.normalize(pred_patches_aux + eps, dim=-1)
                        norm_tgt  = torch.nn.functional.normalize(prediction_target + eps, dim=-1)
                        cosine_sim = (norm_pred * norm_tgt).sum(-1)
                        masked_cos = cosine_sim * valid_mask_patch
                        avg_cosine = masked_cos.sum() / (valid_mask_patch.sum() + eps)
                        aux_cosine_loss = 1.0 - avg_cosine

                        aux_img_loss = aux_feature_loss + aux_cosine_loss
                        loss = loss + self.aux_regression_coef * aux_img_loss
                        self.loss_image_aux_reg = aux_img_loss
                        # Log aux components under standard names for callbacks
                        self.loss_image_mse = aux_feature_loss
                        self.loss_image_cos = aux_cosine_loss


                else:
                    raise ValueError(f"Unsupported vision_loss '{self.vision_loss}' in query mode")

            # ------------------------------------------------------------------
            # AR-DDT MODE (noisy patches fed to LLM for enhanced conditioning)
            # ------------------------------------------------------------------
            elif vision_loss_mode_cfg == "ar-ddt":
                # Expecting extra_mm = (x_t, t_ar, answer_img_mask, pred_image_features, reverse_vti)
                x_t, t_ar, answer_img_mask, pred_image_features, reverse_vti = extra_mm

                # Shapes
                #   pred_image_features : (B, T_total, feat_dim)
                #   x_t                 : (B, T_total, feat_dim)
                #   t_ar               : (B_img,) where B_img = B * M (image-level batch)

                B, T_total, feature_dim = pred_image_features.shape
                M = answer_img_mask.size(1)                               # images per sample
                tokens_per_image = T_total // M                           # patches per image

                hidden_dim = hidden_states.size(-1)

                # ------------------------------------------------------
                # 1) Gather hidden states for *each patch* via reverse_vti
                # ------------------------------------------------------
                hs_full = hidden_states                                   # (B, L, hidden)
                zeros_left_hs = torch.zeros(B, T_total, hidden_dim,
                                            dtype=hs_full.dtype, device=hs_full.device)
                patch_hs = apply_custom_kernel(zeros_left_hs, hs_full, reverse_vti)  # (B, T_total, hidden)

                # ------------------------------------------------------
                # 2)  Optional projection to z_channels
                # ------------------------------------------------------
                if self.use_diff_head_projector:
                    patch_hs = self.diff_head_projector(patch_hs)         # (B, T_total, z_channels)

                # ------------------------------------------------------
                # 3)  Reshape to image-level batches that the diffusion
                #     head expects:  (B*M, tokens_per_image, dim)
                # ------------------------------------------------------
                patch_hs_img = patch_hs.view(B, M, tokens_per_image, -1).view(B * M, tokens_per_image, -1)
                x_clean_img  = pred_image_features.view(B, M, tokens_per_image, feature_dim).view(B * M, tokens_per_image, feature_dim)
                x_t_img      = x_t.view(B, M, tokens_per_image, feature_dim).view(B * M, tokens_per_image, feature_dim)

                # t_ar is already on image-level (B*M,)
                t_img = t_ar.view(-1)

                # ------------------------------------------------------
                # 4)  Diffusion loss with external t and x_t
                # ------------------------------------------------------
                diffusion_loss_per_img = self.diff_head.training_loss(
                    z=patch_hs_img,
                    x=x_clean_img,
                    t=t_img,
                    x_t=x_t_img,
                )  # (B*M,)

                # ------------------------------------------------------
                # 5)  Mask & aggregate over answer images only
                # ------------------------------------------------------
                diffusion_loss_per_img = diffusion_loss_per_img.view(B, M)  # (B, M)

                # answer_img_mask: 1 for answer images, 0 for context images
                masked_loss_per_img = diffusion_loss_per_img * answer_img_mask.float()

                eps = 1e-8
                mean_diffusion_loss = masked_loss_per_img.sum() / (answer_img_mask.sum() + eps)

                # ------------------------------------------------------
                # 6)  Add to total loss & log
                # ------------------------------------------------------
                loss = loss + mean_diffusion_loss * self.vision_coef
                self.loss_image_diff = mean_diffusion_loss
 

                # Optional auxiliary regression loss (AR-DDT): use clean x as target
                if getattr(self, 'aux_regression_enabled', False):
                    pred_patches_aux = self.aux_vision_head(patch_hs)  # (B, T_total, feat)
                    # Mask to answer images' tokens
                    valid_mask_patch = answer_img_mask.unsqueeze(-1).expand(B, M, tokens_per_image)
                    valid_mask_patch = valid_mask_patch.reshape(B, T_total).int()
                    feature_loss_type = getattr(self.config, 'feature_loss_type', 'l2')
                    aux_feature_loss = compute_feature_loss(
                        pred_patches_aux,
                        x_clean_img.view(B, T_total, feature_dim),
                        valid_mask_patch,
                        loss_type=feature_loss_type
                    )
                    eps = 1e-8
                    norm_pred = torch.nn.functional.normalize(pred_patches_aux + eps, dim=-1)
                    norm_tgt  = torch.nn.functional.normalize(x_clean_img.view(B, T_total, feature_dim) + eps, dim=-1)
                    cosine_sim = (norm_pred * norm_tgt).sum(-1)
                    masked_cos = cosine_sim * valid_mask_patch
                    avg_cosine = masked_cos.sum() / (valid_mask_patch.sum() + eps)
                    aux_cosine_loss = 1.0 - avg_cosine
                    aux_img_loss = aux_feature_loss + aux_cosine_loss
                    loss = loss + self.aux_regression_coef * aux_img_loss
                    self.loss_image_aux_reg = aux_img_loss
                    # Log aux components under standard names
                    self.loss_image_mse = aux_feature_loss
                    self.loss_image_cos = aux_cosine_loss

            # ------------------------------------------------------------------
            # NON-QUERY modes   →  keep existing logic unchanged
            # ------------------------------------------------------------------
            else:
                hidden_states_for_prediction = hidden_states[:, :-1]  # States 0 to SeqLen-2
                
                if self.vision_loss == 'diffusion-loss' and self.debug == False:
                    # (existing diffusion branch stays verbatim)

                # Pad the hidden states to match the sequence length of targets/mask
                    bs, seq_len, _ = selected_features.shape  # Use seq_len from selected_features
                    hidden_dim = hidden_states_for_prediction.shape[2]
                    padding_tensor = torch.zeros(hidden_states_for_prediction.shape[0], 1, hidden_states_for_prediction.shape[2],
                                            dtype=hidden_states_for_prediction.dtype, device=hidden_states_for_prediction.device)
                    hidden_states_padded = torch.cat((padding_tensor, hidden_states_for_prediction), dim=1)

                    valid_positions = input_embed_mask.int()
                    reshaped_hidden_states = hidden_states_padded.view(bs * seq_len, hidden_dim)
                    reshaped_hidden_states = reshaped_hidden_states.to(dtype=selected_features.dtype)
                    feature_dim = selected_features.shape[2]
                    diffusion_tokens = self.diff_head_config['diffusion_tokens']
                    reshaped_target_features = selected_features.view(bs * seq_len, 1, feature_dim)
                    diffusion_loss_flat = self.diff_head.training_loss(
                        z=reshaped_hidden_states,
                        x=reshaped_target_features
                        )
                    diffusion_loss_flat = diffusion_loss_flat.squeeze()
                    diffusion_loss_reshaped = diffusion_loss_flat.view(bs, seq_len)
                    masked_loss = diffusion_loss_reshaped * valid_positions.float()
                    epsilon = 1e-8
                    mean_diffusion_loss = masked_loss.sum() / (valid_positions.sum() + epsilon)
                    self.loss_image_diff = mean_diffusion_loss
                    loss = loss + mean_diffusion_loss * self.vision_coef

                else:
                    # existing regression branch (causal / block modes)
                    predicted_features_raw = self.vision_head(hidden_states_for_prediction)
                    padding_tensor = torch.zeros(predicted_features_raw.shape[0], 1, predicted_features_raw.shape[2],
                                                dtype=predicted_features_raw.dtype, device=predicted_features_raw.device)
                    all_predicted_features = torch.cat((padding_tensor, predicted_features_raw), dim=1)
                    valid_positions = input_embed_mask.int()
                    feature_loss_type = getattr(self.config, 'feature_loss_type', 'l2')
                    feature_loss = compute_feature_loss(all_predicted_features, selected_features, valid_positions, loss_type=feature_loss_type)
                    epsilon = 1e-8
                    norm_pred = torch.nn.functional.normalize(all_predicted_features + epsilon, dim=-1)
                    norm_target = torch.nn.functional.normalize(selected_features + epsilon, dim=-1)
                    cosine_sim = (norm_pred * norm_target).sum(dim=-1)
                    masked_cosine_sim = cosine_sim * valid_positions
                    num_valid = valid_positions.sum() + epsilon
                    avg_cosine_sim = masked_cosine_sim.sum() / num_valid
                    cosine_loss = 1.0 - avg_cosine_sim
                    img_loss = feature_loss + cosine_loss
                    loss = loss + img_loss * self.vision_coef
                    self.loss_image_mse = feature_loss
                    self.loss_image_cos = cosine_loss


        

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )


    def greedy_decode(self, position_ids, attention_mask, inputs_embeds, start_image_token_id, end_image_token_id, eos_token_id, do_sample=None, temperature=None, top_p=None, num_beams=None, max_new_tokens=1024, use_cache=None, output_image=False, extra_mm=None, guidance_level=None):
        
        # Convert eos_token_id to list if it's not already a list
        if eos_token_id is not None and not isinstance(eos_token_id, list):
            eos_token_id = [eos_token_id]
        past_key_values = None
        in_image_mode = False
        generated_ids_list = []
        total_image_tokens = 0
        total_output_tokens = 0
        
        use_cache = False
        
        image_embeds_list = []


        # Initialize attention_mask if it's None
        if attention_mask is None:
            _, num_tokens, _ = inputs_embeds.shape
            attention_mask = torch.ones((1, num_tokens), dtype=torch.long, device=inputs_embeds.device)

        num_image_tokens = self.num_image_tokens 
        vision_loss_mode = self.vision_loss_mode
        use_query_mode = vision_loss_mode == "query" or vision_loss_mode == "half-query" or vision_loss_mode == "query-block"
        while True:
           
            attention_mask = torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)

            outputs = self.forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                return_dict=True,
                decoding=in_image_mode, 
                answer_token_mask=extra_mm, # Debugging note, we pass extra_mm here
                guidance_level=guidance_level,
            )

            image_embed = outputs.loss
            if in_image_mode:
                generated_token_length = 1 if not use_query_mode else num_image_tokens
            else:
                generated_token_length = 1
            next_token_logits = outputs.logits[:, -generated_token_length:, :].squeeze(1) # backward compatibility
            next_embed = outputs.hidden_states[:, -generated_token_length:, :].squeeze(1)

            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            next_token_embed = self.model.embed_tokens(next_token)
           

            if (not in_image_mode) and next_token.item() == start_image_token_id:
                in_image_mode = True
                #generated_ids_list.append(next_token.item())
                generated_ids_list.extend(next_token.squeeze(0).tolist())
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)
                if use_query_mode:
                    latent_query = self.get_model().latent_queries.unsqueeze(0)
                    expanded_latent_query = latent_query.expand(inputs_embeds.size(0), -1, -1)
                    inputs_embeds = torch.cat((inputs_embeds, expanded_latent_query), dim=1)


            elif (in_image_mode) and (total_image_tokens<num_image_tokens):

                total_image_tokens += generated_token_length # directly generate all tokens at once if use queries

                #image_embeds_list.append(image_embed)
                image_embeds_list.extend(image_embed)

                if use_query_mode:
                    inputs_embeds[:, -self.num_image_tokens:, :] = next_embed
                else:
                    inputs_embeds = torch.cat((inputs_embeds, next_embed), dim=1)
                    
                # inputs_embeds = torch.cat((inputs_embeds, next_embed), dim=1)

                if total_image_tokens==num_image_tokens:
                    in_image_mode = False
                    if use_query_mode:
                        break


            elif next_token.item() == end_image_token_id:
                in_image_mode = False
                total_image_tokens = 0
                #generated_ids_list.append(next_token.item())
                generated_ids_list.extend(next_token.squeeze(0).tolist())
                
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)

            
            else:
                # Append token embeddings
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)
                generated_ids_list.extend(next_token.squeeze(0).tolist())

            total_output_tokens += 1

            if next_token.numel() == 1 and next_token.item() in eos_token_id: # only judge eos for single token pred
                break

            if total_output_tokens > max_new_tokens:
                break

            past_key_values = outputs.past_key_values
          
        if image_embeds_list:
            image_embeds_tensor = torch.cat(image_embeds_list, dim=0)
        else:
            image_embeds_tensor = torch.tensor([], dtype=torch.float32, device=inputs_embeds.device)

        output = [torch.tensor(generated_ids_list, dtype=torch.int32, device=inputs_embeds.device)]


         # Perform random cropping and compute cosine similarity
        if output_image:
            return output, image_embeds_tensor

        return output, None


    def greedy_decode_with_logits(self, position_ids, attention_mask, inputs_embeds, start_image_token_id, end_image_token_id, eos_token_id, do_sample=None, temperature=None, top_p=None, num_beams=None, max_new_tokens=1024, use_cache=None, output_image=False, extra_mm=None, guidance_level=None):
        


        # Convert eos_token_id to list if it's not already a list
        if eos_token_id is not None and not isinstance(eos_token_id, list):
            eos_token_id = [eos_token_id]
        past_key_values = None
        in_image_mode = False
        generated_ids_list = []
        total_image_tokens = 0
        total_output_tokens = 0
        
        use_cache = False
        
        image_embeds_list = []
        logits_list = []  # Store logits for each generated token
        conf_scores_list = []  # Store confidence scores for each generated token


        # Initialize attention_mask if it's None
        if attention_mask is None:
            _, num_tokens, _ = inputs_embeds.shape
            attention_mask = torch.ones((1, num_tokens), dtype=torch.long, device=inputs_embeds.device)

        num_image_tokens = self.num_image_tokens 
        vision_loss_mode = self.vision_loss_mode
        use_query_mode = vision_loss_mode == "query" or vision_loss_mode == "half-query" or vision_loss_mode == "query-block"
        while True:
           
            attention_mask = torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)

            outputs = self.forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                return_dict=True,
                decoding=in_image_mode, 
                answer_token_mask=extra_mm, # Debugging note, we pass extra_mm here
                guidance_level=guidance_level,
            )

            image_embed = outputs.loss
            if in_image_mode:
                generated_token_length = 1 if not use_query_mode else num_image_tokens
            else:
                generated_token_length = 1
            next_token_logits = outputs.logits[:, -generated_token_length:, :].squeeze(1) # backward compatibility
            next_embed = outputs.hidden_states[:, -generated_token_length:, :].squeeze(1)

            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)            # Extract logit value for the selected token only
            selected_token_logit = torch.gather(next_token_logits, -1, next_token.squeeze(-1).unsqueeze(-1)).squeeze(-1)
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token_conf_scores = -torch.log(self.config.vocab_size * next_token_probs).mean()

            next_token_embed = self.model.embed_tokens(next_token)
           

            if (not in_image_mode) and next_token.item() == start_image_token_id:
                print("Enter image mode")
                in_image_mode = True
                #generated_ids_list.append(next_token.item())
                print("I am entering image mode!!!, this image sequence has number of tokens:", total_image_tokens)
                print('next token is:', next_token, next_token.squeeze(0).tolist())
                generated_ids_list.extend(next_token.squeeze(0).tolist())
                # Store logit for the selected token only (start_image_token)
                logits_list.append(selected_token_logit.unsqueeze(0))
                conf_scores_list.append(next_token_conf_scores.unsqueeze(0))
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)
                if use_query_mode:
                    latent_query = self.get_model().latent_queries.unsqueeze(0)
                    expanded_latent_query = latent_query.expand(inputs_embeds.size(0), -1, -1)
                    print("[Appending Query] inputs_embeds shape before appending:", inputs_embeds.shape, 'query shape:', expanded_latent_query.shape)
                    inputs_embeds = torch.cat((inputs_embeds, expanded_latent_query), dim=1)


            elif (in_image_mode) and (total_image_tokens<num_image_tokens):

                total_image_tokens += generated_token_length # directly generate all tokens at once if use queries

                #image_embeds_list.append(image_embed)
                print("I am in image mode!!!, this image sequence has number of tokens:", total_image_tokens, 'image embed shape:', image_embed.shape, 'next embed shape:', next_embed.shape)
                image_embeds_list.extend(image_embed)
                print('image_embeds_list shape:', len(image_embeds_list))

                if use_query_mode:
                    inputs_embeds[:, -self.num_image_tokens:, :] = next_embed
                else:
                    inputs_embeds = torch.cat((inputs_embeds, next_embed), dim=1)
                    
                # inputs_embeds = torch.cat((inputs_embeds, next_embed), dim=1)

                if total_image_tokens==num_image_tokens:
                    in_image_mode = False
                    if use_query_mode:
                        print("I am leaving image mode!!!, this image sequence has number of tokens:", total_image_tokens)
                        break


            elif next_token.item() == end_image_token_id:
                print("I am leaving image mode!!!, this image sequence has number of tokens:", total_image_tokens)
                in_image_mode = False
                total_image_tokens = 0
                #generated_ids_list.append(next_token.item())
                generated_ids_list.extend(next_token.squeeze(0).tolist())
                # Store logit for the selected token only (end_image_token)
                logits_list.append(selected_token_logit.unsqueeze(0))
                conf_scores_list.append(next_token_conf_scores.unsqueeze(0))
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)

            
            else:
                # Append token embeddings
                inputs_embeds = torch.cat((inputs_embeds, next_token_embed), dim=1)
                generated_ids_list.extend(next_token.squeeze(0).tolist())
                # Store logit for the selected token only (regular text tokens)
                logits_list.append(selected_token_logit.unsqueeze(0))
                conf_scores_list.append(next_token_conf_scores.unsqueeze(0))

            total_output_tokens += 1

            if next_token.numel() == 1 and next_token.item() in eos_token_id: # only judge eos for single token pred
                break

            if total_output_tokens > max_new_tokens:
                break

            past_key_values = outputs.past_key_values
          
        if image_embeds_list:
            image_embeds_tensor = torch.cat(image_embeds_list, dim=0)
        else:
            image_embeds_tensor = torch.tensor([], dtype=torch.float32, device=inputs_embeds.device)

        output = [torch.tensor(generated_ids_list, dtype=torch.int32, device=inputs_embeds.device)]
        
        # Convert logits_list to tuple of tensors (similar to HuggingFace format)
        logits = tuple(logits_list) if logits_list else None
        conf_scores = tuple(conf_scores_list) if conf_scores_list else None

         # Perform random cropping and compute cosine similarity
        if output_image:
            # print('final image embed shape:', image_embeds_tensor.shape)
            # print("output, image embeds are:", image_embeds_tensor)
            return output, image_embeds_tensor, logits, conf_scores

        return output, None, logits, conf_scores


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        use_customize_greedy: Optional[bool] = False,
        return_scores: Optional[bool] = False,
        start_image_token_id = None,
        end_image_token_id = None,
        eos_token_id=None,
        guidance_level=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")


        extra_mm = kwargs.pop("extra_mm", None) # extra_mm is only passed when debug_with_vision is True
        if images is not None or image_embeds is not None:
            (
                # inputs,
                # position_ids,
                # attention_mask,
                # _,
                # inputs_embeds,
                # _,
                # vision_tower_aux_feature_list,
                # vision_tower_aux_attention_masks_list,
                # final_vision_feature_size,
                # global_context_feature,


                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                selected_features,
                input_embed_mask,
                attention_bias,
                extra_mm,


            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images=images,
                image_embeds=image_embeds,
            )
            # self.vision_tower_aux_feature_list = vision_tower_aux_feature_list
            # self.vision_tower_aux_attention_masks_list = vision_tower_aux_attention_masks_list
            # self.final_vision_feature_size = final_vision_feature_size
            # self.global_context_feature = global_context_feature
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)


        if use_customize_greedy:
            if return_scores:
                return self.greedy_decode_with_logits(position_ids=position_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    start_image_token_id=start_image_token_id,
                    end_image_token_id=end_image_token_id,
                    eos_token_id=eos_token_id,
                    extra_mm=extra_mm,
                    guidance_level=guidance_level,
                    **kwargs
                )
            else:
                return self.greedy_decode(position_ids=position_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    start_image_token_id=start_image_token_id,
                    end_image_token_id=end_image_token_id,
                    eos_token_id=eos_token_id,
                    extra_mm=extra_mm,
                    guidance_level=guidance_level,
                    **kwargs
                )
        else:
            return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images

        return inputs


    # ================================================================
    # AURORA — Autoregressive Object-Centric Slot Discovery
    # ================================================================

    def _init_aurora(self):
        """Initialise AURORA v2 learnable tokens and freeze base model."""
        D = self.config.hidden_size
        embed_std = 1.0 / math.sqrt(D)
        self.aurora_C = getattr(self.config, 'aurora_cmd_length', 8)
        self.aurora_K_max = getattr(self.config, 'aurora_max_slots', 10)
        self.aurora_N_reg = getattr(self.config, 'aurora_n_register', 8)

        # Learnable tokens (v2)
        self.aurora_cmd_embeddings = nn.Parameter(torch.randn(self.aurora_C, D) * embed_std)
        self.aurora_obj_embedding_pool = nn.Parameter(torch.randn(self.aurora_K_max, D) * embed_std)
        self.aurora_reg_embeddings = nn.Parameter(torch.randn(self.aurora_N_reg, D) * embed_std)

        self._aurora_warned_nan_diff_loss = False
        self._aurora_warned_nan_total_loss = False
        self._aurora_global_step = 0

        # Freeze rae_query (latent_queries) — used as frozen conditioning for DiT
        stage = int(getattr(self.config, 'aurora_training_stage', 1))
        if self.get_model().latent_queries is not None:
            self.get_model().latent_queries.requires_grad_(stage >= 2)

        print(
            f"[AURORA v2] Initialised — D={D}, C={self.aurora_C}, "
            f"K_max={self.aurora_K_max}, N_reg={self.aurora_N_reg}, "
            f"stage={stage}"
        )

    def _aurora_check_finite(self, tensor: Optional[torch.Tensor], name: str):
        if tensor is None or not torch.is_tensor(tensor):
            return
        if torch.isfinite(tensor).all():
            return
        message = f"[AURORA] Non-finite values detected in {name}."
        if getattr(self.config, 'aurora_fail_on_nan', True):
            raise FloatingPointError(message)
        logger.warning("%s Falling back to torch.nan_to_num.", message)

    def _encode_images_aurora(self, images: torch.Tensor):
        vision_tower = self.get_model().get_vision_tower_aux_list()[0]
        if images.dim() == 3:
            images = images.unsqueeze(0)

        vt_param = next(vision_tower.parameters(), None)
        vt_device = vt_param.device if vt_param is not None else images.device
        vt_dtype = vt_param.dtype if vt_param is not None and vt_param.is_floating_point() else images.dtype
        projector_param = next(self.get_model().mm_projector.parameters(), None)
        projector_device = projector_param.device if projector_param is not None else vt_device
        projector_dtype = (
            projector_param.dtype
            if projector_param is not None and projector_param.is_floating_point()
            else vt_dtype
        )

        with torch.no_grad():
            raw_features = vision_tower(images.to(device=vt_device, dtype=vt_dtype))
            raw_features = raw_features.to(device=projector_device, dtype=projector_dtype)
            img_features = self.get_model().mm_projector(raw_features)

        gt_siglip = raw_features.detach()
        return raw_features.detach(), img_features.detach(), gt_siglip

    def _aurora_model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def _aurora_compute_diffusion_loss(
        self,
        hidden: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        self._aurora_check_finite(hidden, "rae_hidden")
        cond = hidden
        if getattr(self, 'use_diff_head_projector', False):
            cond = self.diff_head_projector(cond)
        # Normalize conditioning to prevent DiT AdaLN explosion from
        # out-of-distribution inputs during early AURORA training.
        cond = F.layer_norm(cond, (cond.shape[-1],))
        self._aurora_check_finite(cond, "diffusion_condition")
        self.diff_head = self.diff_head.to(cond.device)
        diff_dtype = next(self.diff_head.parameters()).dtype
        cond_input = cond.to(device=cond.device, dtype=diff_dtype)
        target_input = target_features.to(device=cond.device, dtype=diff_dtype)
        loss = self.diff_head.training_loss(z=cond_input, x=target_input)
        if not torch.isfinite(loss).all():
            self._aurora_check_finite(loss, "diffusion_loss")
            if torch.isnan(loss).any() and not getattr(self, "_aurora_warned_nan_diff_loss", False):
                logger.warning("[AURORA] diffusion loss produced NaN; clamping to a finite penalty.")
                self._aurora_warned_nan_diff_loss = True
            loss = torch.nan_to_num(loss, nan=1e4, posinf=1e4, neginf=1e4)
        if torch.is_tensor(loss) and loss.dim() > 0:
            loss = loss.mean()
        return loss.float()

    def _aurora_positions(self, K: int) -> Dict[str, int]:
        N_img = self.num_image_tokens
        N_rae = self.get_model().latent_queries.shape[0]
        img_s, img_e = 0, N_img
        obj_s = N_img + self.aurora_C
        obj_e = obj_s + K
        reg_s = obj_e
        reg_e = reg_s + self.aurora_N_reg
        rae_s = reg_e
        rae_e = rae_s + N_rae
        return {
            "img_s": img_s,
            "img_e": img_e,
            "obj_s": obj_s,
            "obj_e": obj_e,
            "reg_s": reg_s,
            "reg_e": reg_e,
            "rae_s": rae_s,
            "rae_e": rae_e,
        }

    def _aurora_run_pass(
        self,
        img_features: torch.Tensor,
        obj_embeds: torch.Tensor,
        reg_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Run one AURORA pass with explicit object/register token inputs."""
        B = img_features.shape[0]
        device = img_features.device
        N_img = self.num_image_tokens
        N_rae = self.get_model().latent_queries.shape[0]
        K = obj_embeds.shape[1]

        cmd_embeds = self.aurora_cmd_embeddings.unsqueeze(0).expand(B, -1, -1)
        rae_embeds = self.get_model().latent_queries.unsqueeze(0).expand(B, -1, -1)
        inputs_embeds = torch.cat(
            [img_features, cmd_embeds, obj_embeds, reg_embeds, rae_embeds],
            dim=1,
        ).to(dtype=self._aurora_model_dtype())

        attn_bias = build_aurora_v2_attention_mask(
            n_img=N_img,
            n_cmd=self.aurora_C,
            n_obj=K,
            n_reg=self.aurora_N_reg,
            n_rae=N_rae,
            device=device,
            dtype=inputs_embeds.dtype,
        ).expand(B, -1, -1, -1)

        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_bias=attn_bias,
            use_cache=False,
            return_dict=True,
        )
        return out.last_hidden_state

    def _aurora_pick_slot_from_masks(
        self,
        pred_maps: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pick the object slot with the highest soft IoU against a target mask."""
        if pred_maps.dim() != 3:
            raise ValueError(f"pred_maps must be [B, K, P], got {tuple(pred_maps.shape)}")
        if target_masks.dim() != 2:
            raise ValueError(f"target_masks must be [B, P], got {tuple(target_masks.shape)}")
        if pred_maps.shape[0] != target_masks.shape[0]:
            raise ValueError(
                f"Batch mismatch: pred_maps={tuple(pred_maps.shape)} target_masks={tuple(target_masks.shape)}"
            )

        pred = pred_maps.float().clamp(0.0, 1.0)
        target = target_masks.float().unsqueeze(1).clamp(0.0, 1.0)
        intersection = torch.minimum(pred, target).sum(dim=-1)
        union = pred.sum(dim=-1) + target.sum(dim=-1) - intersection
        scores = intersection / union.clamp_min(1e-6)
        return scores.argmax(dim=1), scores

    def _aurora_drop_slots(
        self,
        obj_hidden: torch.Tensor,
        remove_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Drop one object slot per sample while keeping sequence length aligned."""
        B, K, D = obj_hidden.shape
        if K == 0:
            return obj_hidden
        remove_indices = remove_indices.to(device=obj_hidden.device, dtype=torch.long).clamp(0, max(K - 1, 0))
        if K == 1:
            return obj_hidden.new_empty(B, 0, D)

        keep_mask = torch.ones(B, K, dtype=torch.bool, device=obj_hidden.device)
        keep_mask.scatter_(1, remove_indices.unsqueeze(1), False)
        return obj_hidden[keep_mask].view(B, K - 1, D)

    def _aurora_normalize_slot_indices(
        self,
        slot_indices,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if slot_indices is None:
            raise ValueError("slot_indices must not be None when normalization is requested.")
        if isinstance(slot_indices, int):
            return torch.full((batch_size,), slot_indices, device=device, dtype=torch.long)
        if torch.is_tensor(slot_indices):
            slot_tensor = slot_indices.to(device=device, dtype=torch.long).view(-1)
        else:
            slot_tensor = torch.as_tensor(slot_indices, device=device, dtype=torch.long).view(-1)
        if slot_tensor.numel() == 1 and batch_size > 1:
            slot_tensor = slot_tensor.expand(batch_size)
        if slot_tensor.numel() != batch_size:
            raise ValueError(
                f"Expected {batch_size} slot indices, got {slot_tensor.numel()} from {slot_indices!r}"
            )
        return slot_tensor

    def _aurora_swap_slots(
        self,
        target_hidden: torch.Tensor,
        source_hidden: torch.Tensor,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        mixed = target_hidden.clone()
        batch_idx = torch.arange(target_hidden.shape[0], device=target_hidden.device)
        mixed[batch_idx, target_indices] = source_hidden[batch_idx, source_indices]
        return mixed

    def _aurora_decode_output(
        self,
        lm_output: torch.Tensor,
        K: int,
        guidance_level: float = 1.0,
        return_generated: bool = True,
    ) -> Dict[str, torch.Tensor]:
        positions = self._aurora_positions(K)
        pred_maps = extract_attention_maps(
            lm_output,
            list(range(positions["obj_s"], positions["obj_e"])),
            positions["img_s"],
            positions["img_e"],
        )
        rae_hidden = lm_output[:, positions["rae_s"]:positions["rae_e"], :]
        rae_cond = rae_hidden
        generated = None
        if return_generated:
            if getattr(self, 'use_diff_head_projector', False):
                projector_dtype = next(self.diff_head_projector.parameters()).dtype
                rae_cond = self.diff_head_projector(rae_cond.to(dtype=projector_dtype))
            rae_cond = F.layer_norm(rae_cond, (rae_cond.shape[-1],))
            self.diff_head = self.diff_head.to(rae_cond.device)
            diff_dtype = next(self.diff_head.parameters()).dtype
            generated = self.diff_head.infer(
                rae_cond.to(dtype=diff_dtype),
                guidance_level=guidance_level,
            )
        return {
            "generated": generated,
            "attn_maps": pred_maps,
            "rae_cond": rae_cond,
            "obj_hidden": lm_output[:, positions["obj_s"]:positions["obj_e"], :],
            "reg_hidden": lm_output[:, positions["reg_s"]:positions["reg_e"], :],
            "K": K,
        }

    def _forward_aurora(
        self,
        images: torch.Tensor,
        n_objects: Optional[torch.Tensor] = None,
        gt_masks_patches: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
        inpaint_mask_patches: Optional[torch.Tensor] = None,
        has_inpaint: Optional[torch.Tensor] = None,
        aurora_inpaint_weight_override: Optional[float] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """AURORA v2 single-forward training pass."""
        model_device = self.aurora_cmd_embeddings.device
        images = images.to(model_device)
        if n_objects is not None:
            n_objects = n_objects.to(model_device)
        if gt_masks_patches is not None:
            gt_masks_patches = gt_masks_patches.to(model_device)
        if target_images is not None:
            target_images = target_images.to(model_device)
        if inpaint_mask_patches is not None:
            inpaint_mask_patches = inpaint_mask_patches.to(model_device)
        if has_inpaint is not None:
            has_inpaint = has_inpaint.to(model_device)

        B = images.shape[0]
        device = images.device
        N_img = self.num_image_tokens
        zero = torch.zeros((), device=device, dtype=torch.float32)

        if n_objects is None:
            n_objects = torch.full((B,), self.aurora_K_max, device=device, dtype=torch.long)
        else:
            n_objects = n_objects.to(device=device, dtype=torch.long).clamp(min=1, max=self.aurora_K_max)

        if gt_masks_patches is None:
            gt_masks_patches = images.new_zeros(B, self.aurora_K_max, N_img)
        else:
            gt_masks_patches = gt_masks_patches.to(device=device, dtype=torch.float32)

        # ============ 1. Image Encoding (Frozen) ============
        _, img_features, gt_siglip = self._encode_images_aurora(images)

        # ============ 2. Random K Sampling (batch-wide) ============
        n_objects_list = n_objects.tolist()
        K = sample_k_for_batch(n_objects_list, self.aurora_K_max)

        # ============ 3. Build Input Sequence ============
        obj_embeds = self.aurora_obj_embedding_pool[:K].unsqueeze(0).expand(B, -1, -1)
        reg_embeds = self.aurora_reg_embeddings.unsqueeze(0).expand(B, -1, -1)
        lm_output = self._aurora_run_pass(img_features, obj_embeds, reg_embeds)

        # ============ 4. Position Indices ============
        positions = self._aurora_positions(K)
        img_s, img_e = positions["img_s"], positions["img_e"]
        obj_s, obj_e = positions["obj_s"], positions["obj_e"]
        reg_s, reg_e = positions["reg_s"], positions["reg_e"]
        rae_s, rae_e = positions["rae_s"], positions["rae_e"]
        obj_positions = list(range(obj_s, obj_e))

        # ============ 7. Attention Maps ============
        pred_logits = extract_attention_logits(lm_output, obj_positions, img_s, img_e)  # (B, K, N_img)
        pred_maps = torch.sigmoid(pred_logits)

        # ============ 8. Hungarian Matching (per sample) ============
        all_matchings = []
        for b in range(B):
            n_obj_b = min(int(n_objects[b].item()), gt_masks_patches.shape[1])
            if n_obj_b > 0 and K > 0:
                matching_b = hungarian_match(pred_logits[b, :K], gt_masks_patches[b, :n_obj_b])
            else:
                matching_b = []
            all_matchings.append(matching_b)

        # ============ 9. Losses ============
        # 9a. Reconstruction Loss (diffusion)
        rae_hidden = lm_output[:, rae_s:rae_e, :]
        L_recon = self._aurora_compute_diffusion_loss(rae_hidden, gt_siglip)

        # 9b. Mask Supervision Loss
        L_mask = compute_mask_loss(
            pred_logits,
            gt_masks_patches, all_matchings,
        )

        # 9c. Diversity Loss
        L_div = compute_diversity_loss(lm_output, obj_positions)

        # 9d. Optional stage-2 inpainting loss
        L_inpaint = zero
        inpaint_weight = 0.0
        remove_indices = None
        if target_images is not None and inpaint_mask_patches is not None:
            if has_inpaint is None:
                has_inpaint = torch.ones(B, device=device, dtype=torch.bool)
            else:
                has_inpaint = has_inpaint.to(device=device, dtype=torch.bool)
            if bool(has_inpaint.any().item()):
                if aurora_inpaint_weight_override is not None:
                    inpaint_weight = float(aurora_inpaint_weight_override)
                elif int(getattr(self.config, "aurora_training_stage", 1)) >= 2:
                    inpaint_weight = float(getattr(self.config, "aurora_inpaint_weight", 0.5))

                if inpaint_weight > 0.0:
                    inpaint_idx = torch.nonzero(has_inpaint, as_tuple=False).flatten()
                    _, _, target_siglip = self._encode_images_aurora(target_images[inpaint_idx])
                    obj_hidden = lm_output[inpaint_idx, obj_s:obj_e, :]
                    reg_hidden = lm_output[inpaint_idx, reg_s:reg_e, :]
                    remove_indices, _ = self._aurora_pick_slot_from_masks(
                        pred_maps[inpaint_idx],
                        inpaint_mask_patches[inpaint_idx].to(device=device, dtype=torch.float32),
                    )
                    kept_obj_hidden = self._aurora_drop_slots(obj_hidden, remove_indices)
                    edited_output = self._aurora_run_pass(
                        img_features[inpaint_idx],
                        kept_obj_hidden,
                        reg_hidden,
                    )
                    edited_positions = self._aurora_positions(kept_obj_hidden.shape[1])
                    edited_rae_hidden = edited_output[:, edited_positions["rae_s"]:edited_positions["rae_e"], :]
                    L_inpaint = self._aurora_compute_diffusion_loss(edited_rae_hidden, target_siglip)

        # 9e. Total
        lambda_mask = float(getattr(self.config, 'aurora_mask_loss_weight', 1.0))
        lambda_div = float(getattr(self.config, 'aurora_diversity_loss_weight', 0.1))
        total_loss = L_recon + lambda_mask * L_mask + lambda_div * L_div + inpaint_weight * L_inpaint

        if not torch.isfinite(total_loss).all():
            self._aurora_check_finite(total_loss, "total_loss")
            if torch.isnan(total_loss).any() and not getattr(self, "_aurora_warned_nan_total_loss", False):
                logger.warning("[AURORA v2] total loss produced NaN; clamping.")
                self._aurora_warned_nan_total_loss = True
            total_loss = torch.nan_to_num(total_loss, nan=1e4, posinf=1e4, neginf=1e4)

        # Store for logging
        self.loss_image_diff = L_recon.detach()
        self.aurora_loss_recon = L_recon.detach()
        self.aurora_loss_mask = L_mask.detach()
        self.aurora_loss_div = L_div.detach()
        self.aurora_loss_inpaint = L_inpaint.detach()
        self.aurora_inpaint_weight = torch.tensor(inpaint_weight, device=device)
        self.aurora_K_sampled = torch.tensor(K, device=device)

        info = {
            "loss_recon": L_recon.detach(),
            "loss_mask": L_mask.detach(),
            "loss_div": L_div.detach(),
            "loss_inpaint": L_inpaint.detach(),
            "inpaint_weight": torch.tensor(inpaint_weight, device=device),
            "K_sampled": torch.tensor(K, device=device),
        }
        if remove_indices is not None:
            info["removed_slots"] = remove_indices.detach()

        return total_loss, info

    def _aurora_forward_single(
        self,
        img_features: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """Single forward pass for inference — returns lm_output."""
        obj_embeds = self.aurora_obj_embedding_pool[:K].unsqueeze(0).expand(img_features.shape[0], -1, -1)
        reg_embeds = self.aurora_reg_embeddings.unsqueeze(0).expand(img_features.shape[0], -1, -1)
        return self._aurora_run_pass(
            img_features=img_features,
            obj_embeds=obj_embeds,
            reg_embeds=reg_embeds,
        )

    @torch.no_grad()
    def generate_aurora(
        self,
        images: torch.Tensor,
        K: Optional[int] = None,
        guidance_level: float = 1.0,
        return_generated: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """AURORA v2 inference — reconstruction with K object prompts."""
        model_device = self.aurora_cmd_embeddings.device
        images = images.to(model_device)
        B = images.shape[0]
        N_img = self.num_image_tokens
        N_rae = self.get_model().latent_queries.shape[0]

        K = K if K is not None else self.aurora_K_max

        _, img_features, _ = self._encode_images_aurora(images)
        lm_output = self._aurora_forward_single(img_features, K)
        return self._aurora_decode_output(
            lm_output=lm_output,
            K=K,
            guidance_level=guidance_level,
            return_generated=return_generated,
        )

    @torch.no_grad()
    def edit_aurora(
        self,
        images: torch.Tensor,
        removal_masks: Optional[torch.Tensor] = None,
        remove_slot_indices: Optional[List[int]] = None,
        K: Optional[int] = None,
        guidance_level: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """AURORA v2 object removal via explicit slot indices or removal masks."""
        model_device = self.aurora_cmd_embeddings.device
        images = images.to(model_device)
        B = images.shape[0]
        K_orig = K if K is not None else self.aurora_K_max

        _, img_features, _ = self._encode_images_aurora(images)

        # First forward — get attention maps for slot identification
        lm_output_orig = self._aurora_forward_single(img_features, K_orig)
        decoded_orig = self._aurora_decode_output(
            lm_output=lm_output_orig,
            K=K_orig,
            guidance_level=guidance_level,
            return_generated=False,
        )
        pred_maps = decoded_orig["attn_maps"]
        obj_hidden = decoded_orig["obj_hidden"]
        reg_hidden = decoded_orig["reg_hidden"]

        if removal_masks is not None and remove_slot_indices is None:
            removal_masks = removal_masks.to(model_device, dtype=torch.float32)
            remove_indices, remove_scores = self._aurora_pick_slot_from_masks(pred_maps, removal_masks)
        else:
            remove_indices = self._aurora_normalize_slot_indices(
                remove_slot_indices if remove_slot_indices is not None else 0,
                batch_size=B,
                device=model_device,
            )
            remove_scores = None

        kept_obj_hidden = self._aurora_drop_slots(obj_hidden, remove_indices)
        edited_output = self._aurora_run_pass(img_features, kept_obj_hidden, reg_hidden)
        decoded_edit = self._aurora_decode_output(
            lm_output=edited_output,
            K=kept_obj_hidden.shape[1],
            guidance_level=guidance_level,
            return_generated=True,
        )

        return {
            "generated": decoded_edit["generated"],
            "original_attn_maps": pred_maps,
            "edited_attn_maps": decoded_edit["attn_maps"],
            "removed_indices": remove_indices.detach().cpu(),
            "removal_scores": remove_scores.detach().cpu() if remove_scores is not None else None,
            "rae_cond": decoded_edit["rae_cond"],
        }

    @torch.no_grad()
    def transfer_aurora(
        self,
        source_images: torch.Tensor,
        target_images: torch.Tensor,
        source_slot_indices=None,
        target_slot_indices=None,
        source_masks: Optional[torch.Tensor] = None,
        target_masks: Optional[torch.Tensor] = None,
        K: Optional[int] = None,
        guidance_level: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Inject one object slot from the source image into the target image."""
        model_device = self.aurora_cmd_embeddings.device
        source_images = source_images.to(model_device)
        target_images = target_images.to(model_device)
        if source_images.shape[0] != target_images.shape[0]:
            raise ValueError(
                f"Batch mismatch between source and target images: {source_images.shape[0]} vs {target_images.shape[0]}"
            )

        B = source_images.shape[0]
        K = K if K is not None else self.aurora_K_max
        _, source_img_features, _ = self._encode_images_aurora(source_images)
        _, target_img_features, _ = self._encode_images_aurora(target_images)

        source_output = self._aurora_forward_single(source_img_features, K)
        target_output = self._aurora_forward_single(target_img_features, K)
        decoded_source = self._aurora_decode_output(source_output, K, guidance_level, return_generated=False)
        decoded_target = self._aurora_decode_output(target_output, K, guidance_level, return_generated=False)

        if source_masks is not None and source_slot_indices is None:
            source_slot_indices, source_scores = self._aurora_pick_slot_from_masks(
                decoded_source["attn_maps"],
                source_masks.to(model_device, dtype=torch.float32),
            )
        else:
            source_slot_indices = self._aurora_normalize_slot_indices(
                source_slot_indices if source_slot_indices is not None else 0,
                batch_size=B,
                device=model_device,
            )
            source_scores = None

        if target_masks is not None and target_slot_indices is None:
            target_slot_indices, target_scores = self._aurora_pick_slot_from_masks(
                decoded_target["attn_maps"],
                target_masks.to(model_device, dtype=torch.float32),
            )
        else:
            target_slot_indices = self._aurora_normalize_slot_indices(
                target_slot_indices if target_slot_indices is not None else 0,
                batch_size=B,
                device=model_device,
            )
            target_scores = None

        mixed_obj_hidden = self._aurora_swap_slots(
            decoded_target["obj_hidden"],
            decoded_source["obj_hidden"],
            source_slot_indices,
            target_slot_indices,
        )
        edited_output = self._aurora_run_pass(
            target_img_features,
            mixed_obj_hidden,
            decoded_target["reg_hidden"],
        )
        decoded_edit = self._aurora_decode_output(
            edited_output,
            K,
            guidance_level,
            return_generated=True,
        )

        return {
            "generated": decoded_edit["generated"],
            "source_attn_maps": decoded_source["attn_maps"],
            "target_attn_maps": decoded_target["attn_maps"],
            "edited_attn_maps": decoded_edit["attn_maps"],
            "source_indices": source_slot_indices.detach().cpu(),
            "target_indices": target_slot_indices.detach().cpu(),
            "source_scores": source_scores.detach().cpu() if source_scores is not None else None,
            "target_scores": target_scores.detach().cpu() if target_scores is not None else None,
            "rae_cond": decoded_edit["rae_cond"],
        }




AutoConfig.register("cambrian_qwen", ScaleRAEQwenConfig)
AutoModelForCausalLM.register(ScaleRAEQwenConfig, ScaleRAEQwenForCausalLM)
