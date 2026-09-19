import json
from pathlib import Path
from types import SimpleNamespace

import torch

from pgot.eval.gqa_object_cache import cache_path, compact_object_bundle, iter_jsonl


def test_cache_path_has_stable_fanout(tmp_path):
    first = cache_path(tmp_path, "train", "n123")
    second = cache_path(tmp_path, "train", "n123")
    assert first == second
    assert first.parent.parent.name == "train"
    assert first.name == "n123.pt"


def test_manifest_loader_rejects_non_objects(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"image_id":"a"}\n')
    assert list(iter_jsonl(path)) == [{"image_id": "a"}]


def test_compaction_excludes_registers_and_keeps_eight_memories():
    k, registers, patches, dim = 4, 32, 16, 6
    result = {
        "semantic_slots": torch.arange(2 * (k + registers) * dim).reshape(2, k + registers, dim).float(),
        "visual_memory": torch.arange(2 * (k + registers) * 32 * dim).reshape(2, k + registers, 32, dim).float(),
        "ovt_object_valid": torch.tensor([[True, False, False, False], [True, False, True, False]]),
        "ovt_object_probs": torch.rand(2, k, patches),
        "ovt_void_probs": torch.rand(2, registers, patches),
    }
    caption = "<thing> cat: black cat <ovt><stuff> sky: blue sky <ovt><scene_end>"
    value = compact_object_bundle(result, caption, 2, cache_dtype="float16", batch_index=1)
    assert value["semantic"].shape == (2, dim)
    assert value["visual_memory"].shape == (2, 8, dim)
    assert value["owner_probabilities"].shape == (2, patches)
    assert value["ownership_labels"].shape == (4, 4)
    assert value["background_register_owners_excluded"] == registers
    assert value["register_memories_per_owner_excluded"] == 32
    assert [obj["source_slot_index"] for obj in value["objects"]] == [0, 2]
    assert all(t.dtype == torch.float16 for t in (
        value["semantic"], value["visual_memory"], value["owner_probabilities"]
    ))
