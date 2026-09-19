import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pgot.eval.verify_woreader_config import verify


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts/train_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader.sh"
PARALLEL = ROOT / "scripts/eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader_parallel.sh"
RUN = ROOT / "scripts/run_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader.sh"


CONFIG = dict(
    pgot_e8_owner_weight=1.0,
    pgot_e8_reader_supervision_mode="none",
    pgot_e8_reader_object_weight=0.0,
    pgot_e8_reader_background_weight=0.0,
    pgot_e11_object_memories_per_owner=8,
    pgot_e11_register_memories_per_owner=32,
    pgot_n_register=4,
    pgot_one_shot_reader_enable=True,
    pgot_one_shot_readout_mode="memory_content",
    pgot_one_shot_writer_softmax_axis="patch",
    pgot_one_shot_writer_owner_prior=False,
    pgot_lm_loss_weight=1.0,
    pgot_recon_loss_weight=1.5,
    hidden_size=3584,
    diffusion_model_hidden_size=4096,
    diffusion_model_z_channels=4096,
)


@pytest.mark.parametrize("change", [
    {},
    {"pgot_e8_reader_supervision_mode": "writer"},
    {"pgot_e8_reader_object_weight": 0.5},
    {"pgot_e8_reader_background_weight": 0.25},
    {"hidden_size": 1536},
    {"diffusion_model_hidden_size": 2048},
    {"pgot_one_shot_reader_enable": False},
])
def test_7b_checkpoint_audit(tmp_path, change):
    (tmp_path / "config.json").write_text(json.dumps(dict(CONFIG, **change)))
    if change:
        with pytest.raises(ValueError):
            verify(tmp_path, model_size="7b")
    else:
        verify(tmp_path, model_size="7b")


def make_fake_commands(tmp_path):
    fake_bash = tmp_path / "bash"
    fake_bash.write_text(
        """#!/bin/bash
name="${1##*/}"
printf 'EVENT:%s|%s|%s|%s|%s|%s|%s|%s\n' "$name" "${CUDA_VISIBLE_DEVICES:-}" \
  "${E8_READER_SUPERVISION_MODE:-}" "${E8_READER_OBJECT_WEIGHT:-}" \
  "${E8_READER_BACKGROUND_WEIGHT:-}" "${PER_DEVICE_TRAIN_BATCH_SIZE:-}" \
  "${GRADIENT_ACCUMULATION_STEPS:-}" "${MODEL_PATH:-}"
if [[ "${FAIL_SCRIPT:-}" == "$name" ]]; then exit 9; fi
case "$name" in
  train_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader.sh)
    folder="${OUTPUT_DIR}/checkpoint-${MAX_STEPS:-5000}"
    mkdir -p "$folder"
    cp "${TEST_CONFIG_PATH}" "$folder/config.json" ;;
  eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader_ar.sh)
    mkdir -p "${OUTPUT_DIR}"; echo '{}' > "${OUTPUT_DIR}/summary.json" ;;
  eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader_voc_ar.sh)
    mkdir -p "${OUTPUT_ROOT}/voc_ar"; echo '{}' > "${OUTPUT_ROOT}/voc_ar/summary.json" ;;
  eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader_compgen.sh)
    mkdir -p "${OUTPUT_DIR}"; echo '{}' > "${OUTPUT_DIR}/summary.json" ;;
  eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader_parallel.sh)
    exit 0 ;;
  train_pgot_oneshot_memory8_register32_noprior_7b_fp32.sh)
    exit 0 ;;
esac
"""
    )
    fake_bash.chmod(0o755)
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/bin/bash\nexit 0\n")
    fake_python.chmod(0o755)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(CONFIG))
    return fake_bash, fake_python, config


def events(stdout):
    return [line[6:].split("|") for line in stdout.splitlines() if line.startswith("EVENT:")]


def test_train_wrapper_changes_only_reader_supervision_env(tmp_path):
    _, _, config = make_fake_commands(tmp_path)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
               CUDA_VISIBLE_DEVICES="0,1,2,3", OUTPUT_DIR=str(tmp_path / "out"),
               TEST_CONFIG_PATH=str(config))
    result = subprocess.run(["/bin/bash", str(TRAIN)], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    row = events(result.stdout)[0]
    assert row[0] == "train_pgot_oneshot_memory8_register32_noprior_7b_fp32.sh"
    assert row[1:7] == ["0,1,2,3", "none", "0.0", "0.0", "4", "2"]


def test_parallel_eval_maps_three_protocols_to_distinct_gpus(tmp_path):
    _, fake_python, config = make_fake_commands(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(config.read_text())
    output = tmp_path / "eval"
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
               CUDA_VISIBLE_DEVICES="0,1,2,3", MODEL_PATH=str(model),
               EVAL_ROOT=str(output), AUDIT_PYTHON=str(fake_python),
               TEST_CONFIG_PATH=str(config))
    result = subprocess.run(["/bin/bash", str(PARALLEL)], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = {row[0]: row for row in events(result.stdout)}
    assert rows["eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader_ar.sh"][1] == "0"
    assert rows["eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader_voc_ar.sh"][1] == "1"
    assert rows["eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader_compgen.sh"][1] == "2"
    assert (output / "ar/summary.json").is_file()
    assert (output / "voc_ar/summary.json").is_file()
    assert (output / "compgen/summary.json").is_file()


def test_training_failure_prevents_evaluation_transition(tmp_path):
    _, fake_python, config = make_fake_commands(tmp_path)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
               CUDA_VISIBLE_DEVICES="0,1,2,3", OUTPUT_DIR=str(tmp_path / "checkpoint"),
               EVAL_ROOT=str(tmp_path / "eval"), AUDIT_PYTHON=str(fake_python),
               TEST_CONFIG_PATH=str(config),
               FAIL_SCRIPT="train_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader.sh")
    result = subprocess.run(["/bin/bash", str(RUN)], env=env, text=True, capture_output=True)
    assert result.returncode != 0
    names = [row[0] for row in events(result.stdout)]
    assert names == ["train_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast_woreader.sh"]


@pytest.mark.parametrize("gpus", ["0", "0,1", "0,1,2", "0,1,2,2"])
def test_invalid_gpu_sets_fail_before_training(tmp_path, gpus):
    _, _, config = make_fake_commands(tmp_path)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
               CUDA_VISIBLE_DEVICES=gpus, OUTPUT_DIR=str(tmp_path / "out"),
               TEST_CONFIG_PATH=str(config))
    result = subprocess.run(["/bin/bash", str(TRAIN)], env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert not events(result.stdout)
