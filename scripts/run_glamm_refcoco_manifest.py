"""Run released GLaMM-RefSeg on the exact DeOR RefCOCO manifest.

This adapter disables the unused ROI-input branch but otherwise retains the
released global encoder, grounding encoder, text projection, and mask decoder.
No ground-truth annotations are loaded during inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, CLIPImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mllm_refcoco_common import (  # noqa: E402
    group_samples_by_image,
    load_samples,
    prediction_complete,
    save_prediction,
    write_json,
)


PROMPT_TEMPLATE = "What is {expression} in this image? Please output segmentation mask."
ASSISTANT_RESPONSE = "[SEG]."
IMAGE_PREAMBLE = "The <image> provides an overview of the picture.\n"


def repo_provenance(repo):
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    diff = subprocess.check_output(["git", "-C", str(repo), "diff", "--no-ext-diff"])
    return commit, bool(diff), hashlib.sha256(diff).hexdigest()


def preprocess_sam(image, transform, image_size=1024):
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    resized = transform.apply_image(image)
    resize = resized.shape[:2]
    tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous()
    tensor = (tensor - pixel_mean) / pixel_std
    tensor = F.pad(tensor, (0, image_size - tensor.shape[-1], 0, image_size - tensor.shape[-2]))
    return tensor, resize


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", default="/home/jovyan/baselines/groundingLMM")
    parser.add_argument("--model", default="MBZUAI/GLaMM-RefSeg")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--subset", choices=("full", "nonoverlap"), default="nonoverlap")
    parser.add_argument("--splits", nargs="+", default=("val", "testA", "testB"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--queries-per-pass", type=int, default=4)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    if args.max_samples < 0 or args.queries_per_pass < 1:
        parser.error("invalid sample/query limit")
    repo = Path(args.repo).resolve()
    if not (repo / "model/GLaMM.py").is_file():
        raise FileNotFoundError(f"GLaMM repository is missing: {repo}")
    if not Path(args.sam_checkpoint).is_file():
        raise FileNotFoundError(f"SAM checkpoint is missing: {args.sam_checkpoint}")
    sys.path.insert(0, str(repo))

    from model.GLaMM import GLaMMForCausalLM
    from model.SAM.utils.transforms import ResizeLongestSide
    from model.llava import conversation as conversation_lib
    from model.llava.mm_utils import tokenizer_image_token
    from tools.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN

    _, samples = load_samples(args.manifest, args.subset, args.train, args.max_samples, args.splits)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pending = [sample for sample in samples if not prediction_complete(output, sample)]
    if not pending:
        print(json.dumps({"status": "complete", "expressions": len(samples), "output": str(output)}, indent=2))
        return
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, model_max_length=1536, padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    # ``GLaMM-RefSeg`` stores these as added tokens.  With current
    # SentencePiece tokenizers, encoding the bare spelling may prepend a
    # whitespace token, so taking input_ids[0] silently selects the wrong
    # token (and produces one mask for every ordinary whitespace token).
    bbox_token_idx = tokenizer.convert_tokens_to_ids("<bbox>")
    seg_token_idx = tokenizer.convert_tokens_to_ids("[SEG]")
    bop_token_idx = tokenizer.convert_tokens_to_ids("<p>")
    eop_token_idx = tokenizer.convert_tokens_to_ids("</p>")
    config = AutoConfig.from_pretrained(args.model)
    config.with_region = False
    model = GLaMMForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        train_mask_decoder=True,
        out_dim=256,
        ce_loss_weight=1.0,
        dice_loss_weight=0.5,
        bce_loss_weight=2.0,
        seg_token_idx=seg_token_idx,
        vision_pretrained=args.sam_checkpoint,
        vision_tower=args.vision_tower,
        vision_module=args.vision_tower,
        use_mm_start_end=True,
        mm_vision_select_layer=-2,
        pretrain_mm_mlp_adapter="",
        tune_mm_mlp_adapter=False,
        freeze_mm_mlp_adapter=False,
        with_region=False,
        bbox_token_idx=bbox_token_idx,
        eop_token_idx=eop_token_idx,
        bop_token_idx=bop_token_idx,
        num_level_reg_features=4,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().get_vision_tower().to(device=device, dtype=dtype)
    model.to(device=device, dtype=dtype).eval()
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    clip_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
    sam_transform = ResizeLongestSide(1024)

    commit, dirty, diff_sha256 = repo_provenance(repo)
    info = {
        "model": args.model,
        "model_family": "GLaMM",
        "repo": str(repo),
        "repo_commit": commit,
        "repo_has_runtime_patch": dirty,
        "repo_diff_sha256": diff_sha256,
        "subset": args.subset,
        "expressions": len(samples),
        "prompt_template": IMAGE_PREAMBLE + PROMPT_TEMPLATE,
        "assistant_response": ASSISTANT_RESPONSE,
        "expression_rewriting": False,
        "gt_inputs": [],
        "mask_threshold": 0.0,
        "precision": args.precision,
        "with_region": False,
        "queries_per_pass": args.queries_per_pass,
        "splits": list(args.splits),
    }
    write_json(output / "model_info.json", info)

    pending_ids = {sample["id"] for sample in pending}
    groups = [[sample for sample in group if sample["id"] in pending_ids]
              for group in group_samples_by_image(samples)]
    groups = [group for group in groups if group]
    started = time.monotonic()
    for group in tqdm(groups, desc="GLaMM RefCOCO images"):
        image = cv2.cvtColor(cv2.imread(group[0]["image_path"]), cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (group[0]["height"], group[0]["width"]):
            raise ValueError(f"Image size mismatch: {group[0]['image_path']}")
        image_clip = clip_processor.preprocess(image, return_tensors="pt")["pixel_values"].to(device, dtype)
        image_sam, resize = preprocess_sam(image, sam_transform)
        image_sam = image_sam.unsqueeze(0).to(device, dtype)
        for block in chunks(group, args.queries_per_pass):
            prompts = []
            for sample in block:
                conv = conversation_lib.conv_templates["llava_v1"].copy()
                conv.messages = []
                question = IMAGE_PREAMBLE + PROMPT_TEMPLATE.format(expression=sample["shared_text"])
                question = question.replace(
                    DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], ASSISTANT_RESPONSE)
                prompts.append(conv.get_prompt())
            ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in prompts]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                ids, batch_first=True, padding_value=tokenizer.pad_token_id
            ).to(device)
            attention = input_ids.ne(tokenizer.pad_token_id)
            dummy_masks = torch.zeros(
                (len(block), image.shape[0], image.shape[1]), device=device, dtype=torch.float32
            )
            label = torch.full((image.shape[0], image.shape[1]), 255.0, device=device)
            with torch.inference_mode():
                result = model(
                    global_enc_images=image_clip,
                    grounding_enc_images=image_sam,
                    bboxes=None,
                    input_ids=input_ids,
                    labels=input_ids,
                    attention_masks=attention,
                    offset=torch.tensor([0, len(block)], device=device),
                    masks_list=[dummy_masks],
                    label_list=[label],
                    resize_list=[resize],
                    inference=True,
                )
            logits = result["pred_masks"][0]
            if logits.shape[0] != len(block):
                raise ValueError(f"GLaMM produced {logits.shape[0]} masks for {len(block)} expressions")
            for sample, logit in zip(block, logits):
                save_prediction(
                    output,
                    sample,
                    (logit > 0).detach().cpu().numpy(),
                    "glamm",
                    threshold=0.0,
                    prompt_template=PROMPT_TEMPLATE,
                )
            del result, logits, input_ids, attention, dummy_masks
    info["elapsed_seconds"] = time.monotonic() - started
    info["completed_expressions"] = sum(prediction_complete(output, sample) for sample in samples)
    write_json(output / "model_info.json", info)
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
