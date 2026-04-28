"""Test whether the extra F.layer_norm in caption_only path is the root cause.

Pretrain path (line 986): patch_hs_reshaped = diff_head_projector(patch_hs_reshaped)  # NO LayerNorm
CaptionSlot path (1836):  cond = F.layer_norm(diff_head_projector(hidden))           # EXTRA LayerNorm

Compare loss: projector-only vs projector+LN, on realistic Qwen-like hidden states.
"""
import sys, torch
sys.path.insert(0, '/home/jovyan/AURORA')

import transformers
from scale_rae.model.language_model.scale_rae_qwen2 import ScaleRAEQwenForCausalLM, ensure_float32

MODEL_PATH = "/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B"
BN_PATH = "/home/jovyan/data/siglip2_bn_stats.pt"

device = torch.device("cuda:0")
dtype = torch.bfloat16

cfg = transformers.AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
cfg.diffusion_norm_stats_path = BN_PATH
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
model.diff_head = model.diff_head.to(device)
model.diff_head_projector = model.diff_head_projector.to(device)
ensure_float32(model.diff_head.model)

# Simulate Qwen-produced rae_hidden (LayerNorm'd LLM output ~N(0,1))
torch.manual_seed(0)
B = 4
rae_hidden = torch.randn(B, 256, 1536, device=device, dtype=torch.float32)

# Target: realistic SigLIP features (de-BN-normalized)
target = torch.randn(B, 256, 1152, device=device, dtype=torch.float32)
dm = model.diff_head.data_mean.to(device).float()
ds = model.diff_head.data_std.to(device).float()
target_realistic = target * ds + dm

# Path A: projector ONLY (pretrain path, line 986)
proj_dtype = next(model.diff_head_projector.parameters()).dtype
cond_noln = model.diff_head_projector(rae_hidden.to(dtype=proj_dtype)).float()
print(f"\n[Path A — projector only] cond: mean={cond_noln.mean():.4e}, std={cond_noln.std():.4e}, shape={tuple(cond_noln.shape)}")

with torch.no_grad():
    loss_A = model.diff_head.training_loss(z=cond_noln, x=target_realistic)
    print(f"  loss (no LN): {loss_A.mean().item():.4e}")

# Path B: projector + F.layer_norm (caption_only path, line 1836)
cond_ln = torch.nn.functional.layer_norm(cond_noln, (cond_noln.shape[-1],))
print(f"\n[Path B — projector + LN] cond: mean={cond_ln.mean():.4e}, std={cond_ln.std():.4e}")

with torch.no_grad():
    loss_B = model.diff_head.training_loss(z=cond_ln, x=target_realistic)
    print(f"  loss (with LN): {loss_B.mean().item():.4e}")

# Path C: zero condition for baseline
with torch.no_grad():
    loss_C = model.diff_head.training_loss(z=torch.zeros_like(cond_noln), x=target_realistic)
    print(f"\n[Path C — zero cond] loss: {loss_C.mean().item():.4e}")

# Path D: projector output SCALED to roughly match a trained LLM's residual stream (try scaling x 5)
cond_scaled = cond_noln * 5.0
with torch.no_grad():
    loss_D = model.diff_head.training_loss(z=cond_scaled, x=target_realistic)
    print(f"\n[Path D — projector x5] loss: {loss_D.mean().item():.4e}")
