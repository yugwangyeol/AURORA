#!/usr/bin/env bash
# 8-image smoke for the zero-shot VOC baselines: loads each COCO checkpoint,
# runs its native preprocessing and mask read-out, scores with PGOT metrics,
# and checks the CODA row against CODA's own metric implementation.
#
#   bash scripts/smoke_baselines_voc_zeroshot.sh              # coda spot metaslot adaslot
#   bash scripts/smoke_baselines_voc_zeroshot.sh metaslot     # subset
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SMOKE_ROOT="${SMOKE_ROOT:-${PROJECT_ROOT}/outputs/_smoke_baselines_voc_zeroshot}"

if (( $# == 0 )); then
    models=(coda spot metaslot adaslot)
else
    models=("$@")
fi
export MAX_SAMPLES="${MAX_SAMPLES:-8}"

BATCH_SIZE="${BATCH_SIZE:-4}" NUM_WORKERS="${NUM_WORKERS:-2}" OUTPUT_ROOT="${SMOKE_ROOT}" \
    bash "${PROJECT_ROOT}/scripts/eval_baselines_voc_zeroshot.sh" "${models[@]}"

SMOKE_ROOT="${SMOKE_ROOT}" EXPECT="${models[*]}" "${PYTHON}" - <<'EOF'
import json, math, os, sys

root = os.environ["SMOKE_ROOT"]
expected = os.environ["EXPECT"].split()
want_n = int(os.environ["MAX_SAMPLES"])
native = {"coda": [7, 32, 32], "spot": [7, 14, 14], "metaslot": [7, 16, 16],
          "adaslot": [33, 224, 224]}
problems = []
for model in expected:
    path = f"{root}/{model}/summary.json"
    if not os.path.exists(path):
        problems.append(f"{model}: summary.json missing")
        continue
    s = json.load(open(path))
    if s["num_samples"] != want_n:
        problems.append(f"{model}: num_samples={s['num_samples']} != {want_n}")
    if s["native_mask_shape"] != native[model]:
        problems.append(f"{model}: native mask shape {s['native_mask_shape']} != {native[model]}")
    for key in ("fARI", "mBO", "mIoU", "sMBO", "sMIOU"):
        value = s.get(key)
        lower = -1 if key == "fARI" else 0
        if not isinstance(value, float) or not math.isfinite(value) or not lower <= value <= 1:
            problems.append(f"{model}: bad {key}={value}")
    if not s["pred_segments_mean"] >= 2:
        problems.append(f"{model}: degenerate masks, pred_segments_mean={s['pred_segments_mean']}")
    if sum(group["n"] for group in s.get("by_object_count", {}).values()) != want_n:
        problems.append(f"{model}: object-count groups do not cover all smoke images")
    if sum(group["n"] for group in s.get("by_exact_object_count", {}).values()) != want_n:
        problems.append(f"{model}: exact object-count groups do not cover all smoke images")
    if model == "adaslot":
        report = s["model_info"]["load_report"]
        if not report["strict"] or report["missing"] or report["unexpected"]:
            problems.append("adaslot: checkpoint did not load strictly")
        if s.get("device") != "cuda":
            problems.append("adaslot: smoke must use CUDA")
        if s.get("zero_kept_slots_count", 1) != 0:
            problems.append("adaslot: empty slot selection in smoke")
        if s["model_info"].get("resize_antialias") is not False:
            problems.append("adaslot: preprocessing does not reproduce torchvision 0.13")
        records = [json.loads(line) for line in open(f"{root}/{model}/per_image.jsonl") if line.strip()]
        if len(records) != want_n:
            problems.append("adaslot: per-image result count mismatch")
        for row in records:
            if not 1 <= row["pred_segments"] <= row["kept_slots"] <= 33:
                problems.append(f"adaslot: invalid kept-slot readout for {row['id']}")
    if model == "coda":
        check = s.get("coda_metric_cross_check") or {}
        for key in ("fARI", "mBO", "mIoU", "sMBO", "sMIOU"):
            if key not in check or abs(check[key] - s[key]) > 5e-3:
                problems.append(f"coda: PGOT vs CODA metric code {key}: {s[key]} vs {check.get(key)}")
    peak = s.get("peak_gpu_memory_GiB")
    print(f"{model:<9} n={s['num_samples']} mask={s['native_mask_shape']} "
          f"segs={s['pred_segments_mean']:.2f} fARI={s['fARI']:.3f} mBO^i={s['mBO']:.3f} "
          f"mBO^c={s['sMBO']:.3f} peak={(peak if peak is not None else float('nan')):.1f}GiB "
          f"load={s['model_info']['load_report']}")
if problems:
    print("SMOKE FAIL:\n  " + "\n  ".join(problems))
    sys.exit(1)
print("SMOKE PASS")
EOF
echo "smoke outputs: ${SMOKE_ROOT}"
