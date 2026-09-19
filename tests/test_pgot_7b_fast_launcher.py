import json
import os
from pathlib import Path
import subprocess


def test_fast_launcher_preserves_train_tf_ar_order(tmp_path):
    """Exercise real orchestration without loading a model or launching GPUs."""
    root = Path(__file__).resolve().parents[1]
    fake_bash = tmp_path / "bash"
    fake_bash.write_text('''#!/bin/bash
case "$1" in
  */run_pgot_oneshot_memory8_register32_noprior_7b_fp32.sh)
    exec /bin/bash "$@" ;;
esac
printf 'ACTION:%s|%s|%s|%s|%s|%s|%s|%s\\n' "${1##*/}" "$NUM_GPUS" "$PER_DEVICE_TRAIN_BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS" "$GLOBAL_BATCH_SIZE" "$CUDA_VISIBLE_DEVICES" "${OUTPUT_DIR:-}" "${MODEL_PATH:-}"
''')
    fake_bash.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
               OUTPUT_DIR=str(tmp_path / "checkpoint"), EVAL_ROOT=str(tmp_path / "eval"),
               CUDA_VISIBLE_DEVICES="0,1,2,3")
    result = subprocess.run(["/bin/bash", str(root / "scripts/run_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast.sh")],
                            env=env, text=True, capture_output=True, check=True)
    actions = [line.split("|") for line in result.stdout.splitlines() if line.startswith("ACTION:")]
    assert len(actions) == 3, result.stdout
    assert [a[0].split(":")[1] for a in actions] == [
        "train_pgot_oneshot_memory8_register32_noprior_7b_fp32.sh",
        "eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_tf.sh",
        "eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_ar.sh"]
    assert all(a[1:5] == ["4", "4", "2", "32"] for a in actions)
    assert [a[5] for a in actions] == ["0,1,2,3", "0", "1"]
    assert actions[1][6] == str(tmp_path / "eval/tf")
    assert actions[2][6] == str(tmp_path / "eval/ar")
    assert actions[1][7] == actions[2][7] == str(tmp_path / "checkpoint")
    config = json.loads((root / "configs/fsdp_7b_full_shard_post.json").read_text())
    assert config["fsdp_backward_prefetch"] == "backward_post"
    assert config["use_orig_params"] == "true"
    logfile = tmp_path / "eval/train.log"
    original_log = logfile.read_bytes()
    rerun = subprocess.run(["/bin/bash", str(root / "scripts/run_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast.sh")],
                           env=env, text=True, capture_output=True)
    assert rerun.returncode == 2
    assert logfile.read_bytes() == original_log
