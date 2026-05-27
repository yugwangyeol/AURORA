"""PGOT-specific token & loss key constants.

We only add TWO new special tokens to the Qwen2 vocabulary:
  - <ovt>        : Object Visual Token (object representation, 2 per object by default)
  - <scene_end>  : End-of-scene marker, signals "no more objects".

Category names (e.g., Person, Giraffe) are NOT new tokens — they are encoded
with the existing Qwen2 tokenizer so that LoRA-only LLM fine-tuning works.
"""

# New special tokens (added to tokenizer)
OVT_TOKEN = "<ovt>"
SCENE_END_TOKEN = "<scene_end>"
NEW_SPECIAL_TOKENS = [OVT_TOKEN, SCENE_END_TOKEN]

# Default formatting
N_OVT_PER_OBJECT_DEFAULT = 2

# ChatML system / user / assistant template strings used to build the sequence.
PGOT_SYSTEM_PROMPT = "You are a vision assistant that describes scenes with grounded objects."
PGOT_USER_INSTRUCTION = "\nDescribe all objects and regions in this scene with grounded tokens."

# Loss bookkeeping keys (for trainer logging)
LOSS_KEY_LM = "loss_lm"
LOSS_KEY_MASK = "loss_mask"
LOSS_KEY_RECON = "loss_recon"
LOSS_KEY_CONTRASTIVE = "loss_contrastive"
LOSS_KEY_TOTAL = "loss"
