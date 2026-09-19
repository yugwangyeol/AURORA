import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_gqa_deor_cache_4gpu.sh"


def make_fixture(tmp_path):
    prepared = tmp_path / "prepared/manifests"
    prepared.mkdir(parents=True)
    for split in ("train", "val", "testdev"):
        (prepared / f"images_{split}.jsonl").write_text('{"image_id":"x"}\n')
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({}))

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        "#!/bin/bash\n"
        "printf 'EXTRACT:%s:%s:%s:%s:%s\\n' \"$GPU\" \"$SPLIT\" \"$NUM_SHARDS\" \"$SHARD_INDEX\" \"$BATCH_SIZE\"\n"
    )
    fake_bash.chmod(0o755)
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/bin/bash\nprintf 'VERIFY:%s\\n' \"$*\"\n")
    fake_python.chmod(0o755)
    return prepared.parent, model, fake_bin, fake_python


def test_four_gpu_launcher_maps_shards_and_advances_splits(tmp_path):
    prepared, model, fake_bin, fake_python = make_fixture(tmp_path)
    env = dict(
        os.environ,
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        CUDA_VISIBLE_DEVICES="3,2,1,0",
        PREPARED_ROOT=str(prepared),
        CACHE_ROOT=str(tmp_path / "cache"),
        MODEL_PATH=str(model),
        DEOR_PYTHON=str(fake_python),
        SPLITS="train val",
    )
    result = subprocess.run(
        ["/bin/bash", str(LAUNCHER)], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    extracts = [line for line in result.stdout.splitlines() if line.startswith("EXTRACT:")]
    assert set(extracts[:4]) == {
        "EXTRACT:3:train:4:0:8", "EXTRACT:2:train:4:1:8",
        "EXTRACT:1:train:4:2:8", "EXTRACT:0:train:4:3:8",
    }
    assert set(extracts[4:]) == {
        "EXTRACT:3:val:4:0:8", "EXTRACT:2:val:4:1:8",
        "EXTRACT:1:val:4:2:8", "EXTRACT:0:val:4:3:8",
    }
    assert result.stdout.count("VERIFY:") == 2


def test_four_gpu_launcher_rejects_duplicate_devices(tmp_path):
    prepared, model, fake_bin, fake_python = make_fixture(tmp_path)
    env = dict(
        os.environ,
        PATH=f"{fake_bin}:{os.environ['PATH']}",
        CUDA_VISIBLE_DEVICES="0,1,2,2",
        PREPARED_ROOT=str(prepared),
        CACHE_ROOT=str(tmp_path / "cache"),
        MODEL_PATH=str(model),
        DEOR_PYTHON=str(fake_python),
    )
    result = subprocess.run(
        ["/bin/bash", str(LAUNCHER)], env=env, text=True, capture_output=True
    )
    assert result.returncode == 2
    assert "EXTRACT:" not in result.stdout
