#!/usr/bin/env bash
# End-to-end smoke for compositional generation on 8 images (2 mixing batches).
# Checks that the rebuilt identity slot set reproduces the model's own Reader
# condition, that mixing actually changes the condition, and that FID/KID and
# example grids are written.  Metric values on 8 images are meaningless.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SMOKE_DIR="${SMOKE_DIR:-${PROJECT_ROOT}/outputs/_smoke_compgen_$(date +%Y%m%d_%H%M%S)}"

MAX_SAMPLES="${MAX_SAMPLES:-8}" BATCH_SIZE="${BATCH_SIZE:-4}" NUM_WORKERS="${NUM_WORKERS:-2}" \
KID_SUBSETS=10 KID_SUBSET_SIZE=4 SAVE_EXAMPLES=2 OUTPUT_DIR="${SMOKE_DIR}" \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_compositional_gen.sh" "$@"

SUMMARY="${SMOKE_DIR}/summary.json" "${PYTHON}" - <<'EOF'
import json, math, os, sys

path = os.environ["SUMMARY"]
s = json.load(open(path))
problems = []
ident = s["identity_check"]
if not ident["passed"]:
    problems.append(f"identity condition mismatch: {ident}")
if s["raw_rae_query_batch_std"] > 1e-3:
    problems.append(f"RAE queries vary across images (std={s['raw_rae_query_batch_std']:.2e})")
if s["n_generated"] != s["n_real"] or s["n_generated"] < 2:
    problems.append(f"sample counts: generated={s['n_generated']} real={s['n_real']}")
if not s["mix_stats"]["mixed_condition_mean_abs_delta"] > 0:
    problems.append("mixing did not change the Reader condition")
if s["mix_stats"]["foreign_object_slot_fraction"] <= 0:
    problems.append("no object slot came from another image")
for key in ("FID", "KID_x1e3_mean", "KID_x1e3_std"):
    if s.get(key) is None or not isinstance(s[key], (int, float)):
        problems.append(f"missing metric {key}")
    elif not math.isfinite(s[key]):
        print(f"WARNING: {key} is not finite on the smoke subset: {s[key]}")
examples = s.get("example_files", [])
if not examples or not all(os.path.exists(p) for p in examples):
    problems.append(f"example grids missing: {examples}")
if problems:
    print("SMOKE FAIL:\n  " + "\n  ".join(problems))
    sys.exit(1)
print("SMOKE PASS")
print(json.dumps({k: s[k] for k in ("identity_check", "raw_rae_query_batch_std", "n_generated",
                                    "FID", "KID_x1e3_mean", "mix_stats", "peak_gpu_memory_GiB")}, indent=1))
EOF
echo "smoke outputs: ${SMOKE_DIR}"
