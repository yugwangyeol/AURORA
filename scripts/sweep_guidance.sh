#!/bin/bash
# =============================================================================
# PGOT v3 — CFG guidance_scale sweep.
# Runs the full eval (with rFID) at multiple guidance scales, then prints a
# comparison table at the end. Each run uses COCO-instance GT + rFID.
#
# Usage:
#   bash scripts/sweep_guidance.sh                 # default sweep
#   GUIDANCE_LIST="1.0 1.5 2.0" bash scripts/sweep_guidance.sh
#   MAX_SAMPLES=500 bash scripts/sweep_guidance.sh    # faster diagnostic
# =============================================================================
set -e

MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_main_v3/checkpoint-14000}"
CKPT_TAG="$(basename ${MODEL_PATH})"
GUIDANCE_LIST="${GUIDANCE_LIST:-1.0 1.5 2.0 3.0 4.0 5.0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"             # empty -> full 2338 sample
SWEEP_ROOT="${SWEEP_ROOT:-/home/jovyan/PGOT/outputs/sweep_v3_${CKPT_TAG}_guidance}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${SWEEP_ROOT}"

echo "========================================================="
echo " CFG guidance_scale sweep"
echo "   Model       : ${MODEL_PATH}"
echo "   Guidance    : ${GUIDANCE_LIST}"
echo "   Max samples : ${MAX_SAMPLES:-(full 2338)}"
echo "   Output root : ${SWEEP_ROOT}"
echo "========================================================="

if [ -n "${MAX_SAMPLES}" ]; then
    export MAX_SAMPLES
fi
export MODEL_PATH
export GT_SOURCE=coco_instance
export COMPUTE_RFID=1

for G in ${GUIDANCE_LIST}; do
    TAG="g${G//./_}"
    OUT="${SWEEP_ROOT}/${TAG}"
    if [ -f "${OUT}/summary.json" ]; then
        echo ""
        echo ">>> ${TAG}: already evaluated (skipping) — ${OUT}/summary.json"
        continue
    fi
    echo ""
    echo "========================================================="
    echo ">>> Running eval @ guidance_scale=${G}"
    echo "========================================================="
    GUIDANCE_SCALE="${G}" OUTPUT_DIR="${OUT}" \
        bash "${SCRIPT_DIR}/run_eval_pgot.sh"
done

# =============================================================================
# Comparison table
# =============================================================================
echo ""
echo "========================================================="
echo " SUMMARY (guidance_scale sweep)"
echo "========================================================="

SCALE_RAE_ENV="${HOME}/.conda/envs/scale_rae"
"${SCALE_RAE_ENV}/bin/python" - <<EOF
import json, os, glob
root = "${SWEEP_ROOT}"
rows = []
for d in sorted(glob.glob(os.path.join(root, "g*"))):
    fp = os.path.join(d, "summary.json")
    if not os.path.exists(fp):
        continue
    s = json.load(open(fp))
    rows.append({
        "tag": os.path.basename(d),
        "n": s.get("num_samples", 0),
        "fARI": s.get("fARI", float("nan")),
        "mBO":  s.get("mBO",  float("nan")),
        "mIoU": s.get("mIoU", float("nan")),
        "rFID": s.get("rFID", float("nan")),
        "PSNR": s.get("recon_psnr", float("nan")),
        "SSIM": s.get("recon_ssim", float("nan")),
        "Active": s.get("active_count_mean", float("nan")),
    })

if not rows:
    print("No results to summarize.")
else:
    print(f"  {'guidance':>10}  {'n':>5}  {'fARI':>7}  {'mBO':>7}  {'mIoU':>7}  {'rFID':>8}  {'PSNR':>7}  {'SSIM':>7}  {'Active':>7}")
    print("  " + "-" * 90)
    for r in rows:
        g = r["tag"].replace("g","").replace("_",".")
        print(f"  {g:>10}  {r['n']:>5d}  {r['fARI']:>7.4f}  {r['mBO']:>7.4f}  {r['mIoU']:>7.4f}  "
              f"{r['rFID']:>8.3f}  {r['PSNR']:>7.3f}  {r['SSIM']:>7.4f}  {r['Active']:>7.2f}")
    # Best per metric
    best_rfid = min(rows, key=lambda r: r["rFID"])
    best_mbo  = max(rows, key=lambda r: r["mBO"])
    best_miou = max(rows, key=lambda r: r["mIoU"])
    best_fari = max(rows, key=lambda r: r["fARI"])
    print()
    print(f"  ★ Best rFID  (lowest): {best_rfid['tag']} = {best_rfid['rFID']:.3f}")
    print(f"  ★ Best mBO   (highest): {best_mbo['tag']} = {best_mbo['mBO']:.4f}")
    print(f"  ★ Best mIoU  (highest): {best_miou['tag']} = {best_miou['mIoU']:.4f}")
    print(f"  ★ Best fARI  (highest): {best_fari['tag']} = {best_fari['fARI']:.4f}")

    # Save consolidated CSV
    csv_path = os.path.join(root, "sweep_summary.csv")
    import csv as _csv
    with open(csv_path, "w") as f:
        w = _csv.writer(f)
        w.writerow(["guidance","n","fARI","mBO","mIoU","rFID","PSNR","SSIM","Active"])
        for r in rows:
            g = r["tag"].replace("g","").replace("_",".")
            w.writerow([g, r["n"], r["fARI"], r["mBO"], r["mIoU"], r["rFID"],
                        r["PSNR"], r["SSIM"], r["Active"]])
    print(f"\n  CSV: {csv_path}")
EOF
