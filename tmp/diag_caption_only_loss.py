"""Diagnose where the 100K recon loss comes from in caption_only forward."""
import os, sys, json, torch
sys.path.insert(0, '/home/jovyan/AURORA')

import transformers
from scale_rae.model.language_model.scale_rae_qwen2 import ScaleRAEQwenForCausalLM
from scale_rae.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

MODEL_PATH = "/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B"
BN_PATH = "/home/jovyan/data/siglip2_bn_stats.pt"
ANN_PATH = "/home/jovyan/AURORA/outputs/stagea_captionslot_train_cache_train2017/captionslot_annotations.json"
IMG_FOLDER = "/home/jovyan/data/coco/train2017"

device = torch.device("cuda:0")
dtype = torch.bfloat16

# Build config mirroring the train script
cfg = transformers.AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
cfg.diffusion_norm_stats_path = BN_PATH
cfg.use_captionslot = True
cfg.captionslot_max_slots = 1
cfg.captionslot_n_register = 1
cfg.captionslot_cmd_length = 8
cfg.captionslot_training_stage = 1
cfg.captionslot_train_latent_queries = False
cfg.captionslot_control_mode = "caption_only"
cfg.captionslot_condition_gate_init = 1.0
cfg.captionslot_attention_use_layer_norm = True
cfg.captionslot_attention_temperature = 1.0
cfg.captionslot_prior_bias_scale = 0.0
cfg.captionslot_recon_loss_weight = 1.0
cfg.captionslot_caption_loss_weight = 0.0
cfg.captionslot_diversity_loss_weight = 0.0
cfg.vision_loss = "diffusion-loss"
cfg.vision_loss_mode = "query"
cfg.vision_coef = 1.0
cfg.diffusion_model_hidden_size = 2048
cfg.diffusion_model_channels = 1152
cfg.diffusion_model_z_channels = 2048
cfg.diffusion_model_depth = 32
cfg.diffusion_model_heads = 32
cfg.dit_cls = "DiT"
cfg.mm_projector_type = "mlp2x_gelu"
cfg.mm_use_im_start_end = True
cfg.mm_use_im_patch_token = False

print("[diag] loading model...")
model = ScaleRAEQwenForCausalLM.from_pretrained(
    MODEL_PATH, config=cfg, torch_dtype=dtype, ignore_mismatched_sizes=True,
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    MODEL_PATH, model_max_length=2048, padding_side="right", use_fast=False,
)
tokenizer.pad_token = "<|endoftext|>"
tokenizer.pad_token_id = 151643

# Vision modules
class MA:
    pass
model_args = MA()
model_args.vision_tower_aux_list = ["google/siglip2-so400m-patch14-224"]
model_args.vision_tower_aux_token_len_list = [256]
model_args.unfreeze_mm_vision_tower = False
model_args.mm_projector_type = "mlp2x_gelu"
model_args.mm_use_im_start_end = True
model_args.mm_use_im_patch_token = False
model_args.si_token_len = 0
model_args.miv_token_len = 0
model_args.pretrain_mm_mlp_adapter = None
model_args.pretrain_adapter_and_vision_head = None
model_args.vision_hidden_size = 1152
model_args.mm_vision_select_layer = -1
model_args.mm_vision_select_feature = "patch"

model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)
model.load_vision_head(model_args=model_args)
vts = model.get_vision_tower_aux_list()
for vt in vts:
    vt.to(dtype=dtype, device=device)

# Register token ids (same as trainer)
im_start_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
im_end_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
model.im_start_id = im_start_id
model.im_end_id = im_end_id
model.config.im_start_id = im_start_id
model.config.im_end_id = im_end_id

block_map = {
    "captionslot_system_prefix_ids": "<|im_start|>system\nYou are a helpful assistant.",
    "captionslot_system_suffix_ids": "<|im_end|>\n",
    "captionslot_user_prefix_ids": "<|im_start|>user\n",
    "captionslot_user_text_prefix_ids": "",
    "captionslot_user_suffix_ids": "<|im_end|>\n",
    "captionslot_assistant_prefix_ids": "<|im_start|>assistant\n",
    "captionslot_assistant_suffix_ids": "<|im_end|>",
}
for attr_name, text in block_map.items():
    ids = tokenizer.encode(text, add_special_tokens=False) if text else []
    setattr(model, attr_name, ids)
    setattr(model.config, attr_name, ids)

model = model.to(device=device)
model.eval()

# Verify pretrained weights loaded
print("\n[diag] pretrained weight check:")
lq = model.get_model().latent_queries
print(f"  latent_queries: shape={tuple(lq.shape)}, mean={lq.float().mean().item():.4e}, std={lq.float().std().item():.4e}")
proj_w = model.diff_head_projector.weight
print(f"  diff_head_projector.weight: mean={proj_w.float().mean().item():.4e}, std={proj_w.float().std().item():.4e}")
# Check AdaLN layer 0 block 0 weight
for n, p in model.diff_head.model.named_parameters():
    if 'dit_blocks.0.adaLN_modulation.1.weight' in n:
        print(f"  diff_head.{n}: mean={p.float().mean().item():.4e}, std={p.float().std().item():.4e}, is_zero={torch.allclose(p, torch.zeros_like(p))}")
        break
for n, p in model.diff_head.model.named_parameters():
    if 'final_layer.linear.weight' in n:
        print(f"  diff_head.{n}: mean={p.float().mean().item():.4e}, std={p.float().std().item():.4e}, is_zero={torch.allclose(p, torch.zeros_like(p))}")
        break

# Check BN stats
print(f"  diff_head.normalize_data = {getattr(model.diff_head, 'normalize_data', 'N/A')}")
if hasattr(model.diff_head, 'data_mean'):
    print(f"  diff_head.data_mean: shape={tuple(model.diff_head.data_mean.shape)}, mean={model.diff_head.data_mean.mean().item():.4e}")
    print(f"  diff_head.data_std:  shape={tuple(model.diff_head.data_std.shape)},  mean={model.diff_head.data_std.mean().item():.4e}")

# Build a minimal forward with a dummy caption + image
from PIL import Image
import torchvision.transforms as T

with open(ANN_PATH) as f:
    ann = json.load(f)
# ann structure: list or dict? try both
if isinstance(ann, dict):
    entries = list(ann.values())[:1]
else:
    entries = ann[:1]
entry = entries[0]
print(f"\n[diag] sample entry keys: {list(entry.keys()) if isinstance(entry, dict) else type(entry)}")

caption = entry.get("caption") or entry.get("captions", [""])[0] if isinstance(entry, dict) else "A photo."
print(f"  caption: {caption!r}")

# Tokenize caption
cap_ids = tokenizer(caption, return_tensors="pt", max_length=64, truncation=True, padding="max_length").input_ids.to(device)
cap_mask = (cap_ids != tokenizer.pad_token_id).to(device)
print(f"  caption token len: {cap_mask.sum().item()}")

# Load image
img_name = entry.get("image") or entry.get("file_name") if isinstance(entry, dict) else None
if img_name:
    img_path = os.path.join(IMG_FOLDER, img_name)
    img = Image.open(img_path).convert("RGB")
    # Simple 224x224 center crop + to tensor
    from scale_rae.mm_utils import expand2square
    img_p = image_processor = vts[0].image_processor if hasattr(vts[0], 'image_processor') else None
    if image_processor is not None:
        img = expand2square(img, tuple(int(x*255) for x in image_processor.image_mean))
        img_t = image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
    else:
        img_t = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])(img)
    img_t = img_t.unsqueeze(0).to(device=device, dtype=dtype)
else:
    img_t = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)

print(f"  image shape: {tuple(img_t.shape)}")

# Hook into forward to capture intermediates
with torch.no_grad():
    # Manually replicate _forward_captionslot_caption_only but capture tensors
    _, _, gt_siglip = model._encode_images_aurora(img_t)
    print(f"\n[diag] gt_siglip (SigLIP target): shape={tuple(gt_siglip.shape)}, mean={gt_siglip.float().mean().item():.4e}, std={gt_siglip.float().std().item():.4e}, min={gt_siglip.float().min().item():.4e}, max={gt_siglip.float().max().item():.4e}")

    # After BN normalization (same logic as training_loss)
    dm = model.diff_head.data_mean.to(gt_siglip.device).float()
    ds = model.diff_head.data_std.to(gt_siglip.device).float()
    gt_norm = (gt_siglip.float() - dm) / ds
    print(f"[diag] gt_siglip after BN-normalize: mean={gt_norm.mean().item():.4e}, std={gt_norm.std().item():.4e}")

    # Full forward to get rae_hidden
    loss, metrics = model._forward_captionslot_caption_only(
        images=img_t, caption_input_ids=cap_ids, caption_attention_mask=cap_mask,
    )
    print(f"\n[diag] recon_loss = {loss.item():.4f}")
    print(f"[diag] metrics = {metrics}")
