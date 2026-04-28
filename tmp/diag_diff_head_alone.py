"""Test diff_head in isolation — does it give O(1) loss on properly-shaped inputs?"""
import os, sys, torch
sys.path.insert(0, '/home/jovyan/AURORA')

import transformers
from scale_rae.model.language_model.scale_rae_qwen2 import ScaleRAEQwenForCausalLM

MODEL_PATH = "/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B"
BN_PATH = "/home/jovyan/data/siglip2_bn_stats.pt"

device = torch.device("cuda:0")
dtype = torch.bfloat16

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

print("[diag] loading model (cpu first)...")
model = ScaleRAEQwenForCausalLM.from_pretrained(
    MODEL_PATH, config=cfg, torch_dtype=dtype, ignore_mismatched_sizes=True,
)

# Check weights BEFORE moving to device
print("\n[weight sanity]")
lq = model.get_model().latent_queries
print(f"  latent_queries: shape={tuple(lq.shape)}, mean={lq.float().mean().item():.4e}, std={lq.float().std().item():.4e}, norm={lq.float().norm().item():.4e}")
proj_w = model.diff_head_projector.weight
print(f"  diff_head_projector.weight: shape={tuple(proj_w.shape)}, mean={proj_w.float().mean().item():.4e}, std={proj_w.float().std().item():.4e}")
proj_b = model.diff_head_projector.bias
print(f"  diff_head_projector.bias: shape={tuple(proj_b.shape)}, mean={proj_b.float().mean().item():.4e}, std={proj_b.float().std().item():.4e}")

# Check several AdaLN and final_layer weights
nonzero_adaLN = 0
zero_adaLN = 0
for n, p in model.diff_head.model.named_parameters():
    if 'adaLN_modulation.1.weight' in n:
        if p.float().abs().max().item() < 1e-8:
            zero_adaLN += 1
        else:
            nonzero_adaLN += 1
print(f"  AdaLN modulation.1.weight: {nonzero_adaLN} non-zero, {zero_adaLN} zero (out of {nonzero_adaLN + zero_adaLN})")

for n, p in model.diff_head.model.named_parameters():
    if 'final_layer.linear.weight' in n:
        print(f"  {n}: shape={tuple(p.shape)}, max_abs={p.float().abs().max().item():.4e}, mean={p.float().mean().item():.4e}")
    if 'final_layer.linear.bias' in n:
        print(f"  {n}: shape={tuple(p.shape)}, max_abs={p.float().abs().max().item():.4e}")
    if 'final_layer.adaLN_modulation.1.weight' in n:
        print(f"  {n}: shape={tuple(p.shape)}, max_abs={p.float().abs().max().item():.4e}")

print(f"  diff_head.normalize_data = {getattr(model.diff_head, 'normalize_data', 'N/A')}")
if hasattr(model.diff_head, 'data_mean'):
    print(f"  diff_head.data_mean: shape={tuple(model.diff_head.data_mean.shape)}, mean={model.diff_head.data_mean.float().mean().item():.4e}, range=[{model.diff_head.data_mean.float().min().item():.2f}, {model.diff_head.data_mean.float().max().item():.2f}]")
    print(f"  diff_head.data_std:  shape={tuple(model.diff_head.data_std.shape)},  mean={model.diff_head.data_std.float().mean().item():.4e}, range=[{model.diff_head.data_std.float().min().item():.2f}, {model.diff_head.data_std.float().max().item():.2f}]")

# Move diff_head + projector to GPU
model.diff_head = model.diff_head.to(device)
model.diff_head_projector = model.diff_head_projector.to(device)

# Test 1: feed PROPERLY normalized random condition + random target (simulating siglip features)
print("\n[test 1] random Gaussian cond + properly-normalized target")
torch.manual_seed(0)
B = 4
# rae_hidden from Qwen is typically LayerNorm'd hidden; mimic with unit Gaussian
rae_hidden_fake = torch.randn(B, 256, 1536, device=device, dtype=torch.float32)
# Project through diff_head_projector then LayerNorm (mimics _captionslot_prepare_diffusion_condition)
proj_dtype = next(model.diff_head_projector.parameters()).dtype
cond = model.diff_head_projector(rae_hidden_fake.to(dtype=proj_dtype))
cond = torch.nn.functional.layer_norm(cond.float(), (cond.shape[-1],))
print(f"  cond after projector+LN: mean={cond.mean().item():.4e}, std={cond.std().item():.4e}, shape={tuple(cond.shape)}")

# Build a realistic target (denormalized SigLIP-like features): Gaussian in the data stats
# Since normalize_data=True, diff_head will normalize using data_mean/data_std internally.
target = torch.randn(B, 256, 1152, device=device, dtype=torch.float32)
# Realistic: de-standardize so that after diff_head's internal BN, values are ~N(0,1)
dm = model.diff_head.data_mean.to(device).float()
ds = model.diff_head.data_std.to(device).float()
target_realistic = target * ds + dm
print(f"  target (pre-BN): mean={target_realistic.mean().item():.4e}, std={target_realistic.std().item():.4e}")

# Compute loss
from scale_rae.model.language_model.scale_rae_qwen2 import ensure_float32
ensure_float32(model.diff_head.model)
with torch.no_grad():
    loss = model.diff_head.training_loss(z=cond, x=target_realistic)
    print(f"  loss (realistic target): {loss.mean().item():.4e}")

# Test 2: target that is already normalized (pre-normalized = mean 0 std 1 fed directly)
target_already_norm = torch.randn(B, 256, 1152, device=device, dtype=torch.float32)
with torch.no_grad():
    loss2 = model.diff_head.training_loss(z=cond, x=target_already_norm)
    print(f"  loss (already-norm target, treated as raw): {loss2.mean().item():.4e}")

# Test 3: WITHOUT normalize_data — bypass BN and use LayerNorm directly
orig_norm_flag = model.diff_head.normalize_data
model.diff_head.normalize_data = False
target_ln = torch.nn.functional.layer_norm(target_realistic, (target_realistic.shape[-1],))
with torch.no_grad():
    loss3 = model.diff_head.training_loss(z=cond, x=target_ln)
    print(f"  loss (normalize_data=False, target pre-LN'd): {loss3.mean().item():.4e}")
model.diff_head.normalize_data = orig_norm_flag

# Test 4: zero condition (pure unconditional) — tests diff_head alone
print("\n[test 4] zero condition, realistic target")
cond_zero = torch.zeros_like(cond)
with torch.no_grad():
    loss4 = model.diff_head.training_loss(z=cond_zero, x=target_realistic)
    print(f"  loss (zero cond, realistic target BN-norm): {loss4.mean().item():.4e}")
