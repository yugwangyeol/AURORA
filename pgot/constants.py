"""PGOT-specific token & loss key constants.

The clean object-centric format uses four new special tokens:
  - <thing>      : countable object marker
  - <stuff>      : amorphous region/background-stuff marker
  - <ovt>        : Object Visual Token (one per segment in the core experiment)
  - <scene_end>  : End-of-scene marker, signals "no more objects".

Category names (e.g., Person, Giraffe) are NOT new tokens — they are encoded
with the existing Qwen2 tokenizer so that LoRA-only LLM fine-tuning works.
"""

# New special tokens (added to tokenizer)
OVT_TOKEN = "<ovt>"
SCENE_END_TOKEN = "<scene_end>"
THING_TOKEN = "<thing>"
STUFF_TOKEN = "<stuff>"
NEW_SPECIAL_TOKENS = [OVT_TOKEN, SCENE_END_TOKEN, THING_TOKEN, STUFF_TOKEN]

# Default formatting
N_OVT_PER_OBJECT_DEFAULT = 2

# ChatML system / user / assistant template strings used to build the sequence.
PGOT_SYSTEM_PROMPT = "You are a vision assistant that describes scenes with grounded objects."
PGOT_USER_INSTRUCTION = "\nDescribe all objects and regions in this scene with grounded tokens."

# Instance-only E1 prompt.  Keep the legacy prompt above unchanged so older
# Pix2Cap checkpoints can still be evaluated with the exact template on which
# they were trained.
PGOT_INSTANCE_SYSTEM_PROMPT = (
    "You are a vision assistant that decomposes an image into countable object "
    "instances and grounded object visual tokens."
)
PGOT_INSTANCE_USER_INSTRUCTION = (
    "\nDescribe every visible countable object instance one at a time, in "
    "descending order of visible area. After each object description, output "
    "exactly one <ovt>. Do not describe background, stuff, or uncountable "
    "regions. After the last object, output <scene_end>."
)


def get_pgot_prompts(dataset_format: str):
    """Return the checkpoint-stable prompt pair for a dataset format."""
    if str(dataset_format).lower() == "coco_instance":
        return PGOT_INSTANCE_SYSTEM_PROMPT, PGOT_INSTANCE_USER_INSTRUCTION
    return PGOT_SYSTEM_PROMPT, PGOT_USER_INSTRUCTION

# Loss bookkeeping keys (for trainer logging)
LOSS_KEY_LM = "loss_lm"
LOSS_KEY_MASK = "loss_mask"
LOSS_KEY_RECON = "loss_recon"
LOSS_KEY_CONTRASTIVE = "loss_contrastive"
LOSS_KEY_TOTAL = "loss"
