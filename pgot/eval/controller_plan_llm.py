"""Controller stage: an instruction-tuned LLM turns edit instructions into PGOT token plans.

Runs in an env with a recent transformers (e.g. the ``t2i`` conda env with the
cached Qwen/Qwen3-VL-8B-Instruct, used text-only).  Input is ``pairs.json`` from
``controller_edit_pipeline.py --stage captions``; output ``plans.json`` holds,
per instruction, the raw completion, the parsed plan and whether its core ops
(background, removed image-1 ids, added image-2 ids) match the expected plan.

    python pgot/eval/controller_plan_llm.py --pairs .../pairs.json --output .../plans.json
"""
import argparse
import json
import re
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

SYSTEM = (
    "You plan object-level image edits. A vision model decomposed each of two images into numbered object "
    "tokens plus background tokens. The edited image always shows: image-1 objects NOT in remove_image1, "
    "image-2 objects listed in add_image2, and the background of the chosen image.\n"
    "Rules:\n"
    "1. Put an image-1 id in remove_image1 ONLY if the instruction explicitly asks to remove, delete, drop or "
    "replace that object.\n"
    "2. Put an image-2 id in add_image2 ONLY if the instruction explicitly asks to bring, add or use that object.\n"
    "3. Never include objects the instruction does not name, even if related (a frisbee with a dog, a couch with "
    "a cat, skis with a skier).\n"
    "4. background is \"image2\" only if the instruction asks for the scene, place or background of image 2; "
    "otherwise \"image1\".\n"
    "5. Use only ids from the lists. Reply with JSON only, no prose: "
    '{"background": "image1" or "image2", "remove_image1": [ids], "add_image2": [ids]}\n'
    "Example: image1: 0 Dog 1 | 1 Frisbee 1 ; image2: 0 Cat 1 | 1 Couch 1 ; Instruction: Swap the dog for the cat. "
    '-> {"background": "image1", "remove_image1": [0], "add_image2": [0]}\n'
    "Example: image1: 0 Person 1 | 1 Skis 1 ; image2: 0 Couch 1 | 1 Table 1 ; Instruction: Put the table from the "
    'second picture next to the person. -> {"background": "image1", "remove_image1": [], "add_image2": [1]}'
)


def object_list(tag, objects):
    return f"{tag}: " + " | ".join(f"{o['id']} {o['label']}: {o['description']}" for o in objects)


def core(plan):
    return (plan.get("background"), sorted(int(i) for i in plan.get("remove_image1", [])),
            sorted(int(j) for j in plan.get("add_image2", [])))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--max_new_tokens", type=int, default=160)
    args = p.parse_args()

    start = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    pairs = json.load(open(args.pairs))
    correct = total = 0
    by_type = {}
    for pair in pairs:
        lists = object_list("image1", pair["image1"]["objects"]) + "\n" + object_list("image2", pair["image2"]["objects"])
        for inst in pair["instructions"]:
            messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                        {"role": "user", "content": [{"type": "text", "text": f"{lists}\nInstruction: {inst['text']}"}]}]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            raw = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            match = re.search(r"\{.*\}", raw, re.S)
            try:
                plan = json.loads(match.group(0)) if match else None
            except json.JSONDecodeError:
                plan = None
            ok = plan is not None and core(plan) == core(inst["expected"])
            inst.update({"raw": raw.strip(), "plan": plan, "correct": bool(ok)})
            total += 1
            correct += int(ok)
            stats = by_type.setdefault(inst["type"], [0, 0])
            stats[0] += int(ok)
            stats[1] += 1
            print(f"[pair {pair['pair_id']} {inst['type']}] {inst['text']}\n   -> {plan} | expected {inst['expected']} | ok={ok}")
    with open(args.output, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"\nplan accuracy {correct}/{total}; by type: { {k: f'{v[0]}/{v[1]}' for k, v in by_type.items()} } "
          f"({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
