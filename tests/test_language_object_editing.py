import json
from types import SimpleNamespace

import pytest

from pgot.eval.language_object_editing import (
    SYSTEM, apply_selected, candidates_for, checked_selection, digest,
    messages_for, parse_plan, prepare, resolve_plan, write_json,
)


def candidate(name, slot):
    return dict(id=name, slot_index=slot, prediction_label=slot + 1,
                category="Dog", description="A brown dog.", gt="SECRET_GT", expected=True)


def test_selector_whitelist_and_no_reference_slots():
    messages = messages_for("Remove the dog.", [candidate("opaque", 4)], [])
    payload = json.loads(messages[1]["content"])
    assert payload == {"request": "Remove the dog.", "image_A_objects": [
        {"id": "opaque", "category": "Dog", "description": "A brown dog."}], "image_B_objects": []}
    assert "SECRET_GT" not in json.dumps(messages)
    assert "slot_index" not in json.dumps(messages)
    assert messages[0]["content"] == SYSTEM


@pytest.mark.parametrize("raw,status", [
    ('{"operation":"replace","target_id":"a","donor_id":"b"}', "ok"),
    ('{"operation":"transfer","target_id":"a","donor_id":"b"}', "ok"),
    ('{"operation":"remove","target_id":"a","donor_id":null}', "ok"),
    ('{"operation":"none","target_id":null,"donor_id":null}', "abstain"),
    ('{"operation":"remove","target_id":"b","donor_id":null}', "invalid_target_id"),
    ('{"operation":"replace","target_id":"a","donor_id":"a"}', "invalid_donor_id"),
    ('{"operation":"remove","target_id":"a","donor_id":"b"}', "invalid_schema"),
    ('{"operation":"none","target_id":"a","donor_id":null}', "invalid_schema"),
    ('{"operation":"add","target_id":"a","donor_id":"b"}', "invalid_operation"),
    ('{"operation":"remove","target_id":0,"donor_id":null}', "invalid_target_id"),
    ('{"operation":"remove","target_id":"a","donor_id":null,"expected":0}', "invalid_schema"),
    ('not JSON', "invalid_json"),
    ('```json\n{"operation":"remove","target_id":"a","donor_id":null}\n```', "invalid_json"),
])
def test_strict_external_plan_and_no_fallback(raw, status):
    a, b = [candidate("a", 2)], [candidate("b", 0)]
    plan, got = parse_plan(raw, a, b)
    assert got == status
    resolved = resolve_plan(plan, got, a, b)
    assert (resolved is not None) == (status == "ok")
    if resolved:
        assert resolved["target"]["slot_index"] == 2


def test_execute_selected_donor_not_manual_reference():
    objects = [("s0", "m0"), ("s1", "m1"), ("s2", "m2")]
    donors = [("ds0", "dm0"), ("ds1", "dm1")]
    registers = [("reg", "regmem")]
    # Intentionally choose an arbitrary nonzero slot; no category/request lookup.
    resolved = {"operation": "replace", "target": candidate("a", 2), "donor": candidate("b", 1)}
    units, backgrounds = apply_selected(objects, registers, resolved, donors)
    assert units == [("s0", "m0"), ("s1", "m1"), ("s2", "dm1")]
    assert backgrounds is registers
    assert units[0] is objects[0] and units[1] is objects[1]
    assert objects[2] == ("s2", "m2")


def test_remove_only_llm_selected_owner_with_rewritten_memories():
    objects = [("s0", "m0"), ("s1", "m1"), ("s2", "m2")]
    rewritten = [("s0", "rw0"), ("s1", "rw1"), ("s2", "rw2")]
    registers, background = [("r", "m")], [("r", "rw")]
    resolved = {"operation": "remove", "target": candidate("a", 1), "donor": None}
    units, backgrounds = apply_selected(objects, registers, resolved, rewritten=(rewritten, background))
    assert units == [rewritten[0], rewritten[2]]
    assert backgrounds is background
    with pytest.raises(ValueError, match="requires"):
        apply_selected(objects, registers, resolved)


def test_recorded_plan_cannot_be_manually_corrected(tmp_path):
    query = {"id": "test", "image_A": 1, "image_B": 2, "request": "Replace the dog."}
    m = {"fingerprint": "fp", "entries": {"1": {"candidates": [candidate("a", 0)]},
                                             "2": {"candidates": [candidate("b", 1)]}}}
    a, b = candidates_for(m, query)
    messages = messages_for(query["request"], a, b)
    raw = '{"operation":"replace","target_id":"a","donor_id":"b"}'
    plan, status = parse_plan(raw, a, b)
    record = dict(query=query, selection_fingerprint="selection", messages=messages,
                  raw_completion=raw, plan=plan, status=status, prompt_sha256=digest(messages),
                  resolved=resolve_plan(plan, status, a, b))
    write_json(tmp_path / "selector_info.json", {"selection_fingerprint": "selection", "spec": {"fingerprint": "fp"}})
    write_json(tmp_path / "plans/test.json", record)
    assert checked_selection(tmp_path, m, query)[1]["donor"]["slot_index"] == 1
    record = json.loads(json.dumps(record))  # Tamper disk record, NOT shared manifest objects.
    record["resolved"]["donor"]["slot_index"] = 0
    write_json(tmp_path / "plans/test.json", record)
    with pytest.raises(ValueError, match="corrected or altered"):
        checked_selection(tmp_path, m, query)


def test_prepare_uses_generated_captions_only_and_rejects_expected_slots(tmp_path, monkeypatch):
    from pgot.eval import run_natural_object_selection
    monkeypatch.setattr(run_natural_object_selection, "checkpoint_signature", lambda path: {"test": True})
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image bytes")
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(json.dumps({"image_id": 1, "image_path": str(image),
        "caption": "SECRET_GT", "objects": [{"description": "SECRET_GT"}]}) + "\n")
    captions = tmp_path / "captions.jsonl"
    captions.write_text(json.dumps({"image_id": 1, "text": "<thing> Dog 1: A brown dog. <ovt>. <thing> Cup 1: A cup. <ovt>. <scene_end>",
                                   "ovt_count": 2}) + "\n")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"caption_mode": "autoregressive", "ckpt": str(tmp_path)}))
    requests = tmp_path / "requests.jsonl"
    query = {"id": "test", "image_A": 1, "image_B": None, "request": "Remove the cup."}
    requests.write_text(json.dumps(query) + "\n")
    args = SimpleNamespace(model_path=str(tmp_path), output_dir=str(tmp_path / "out"), requests=str(requests),
        annotations=str(annotations), captions=str(captions), caption_summary=str(summary),
        selector_model=str(tmp_path), seed=1234, diffusion_inference_steps=10, guidance_scale=1.,
        max_caption_tokens=4096, selector_max_new_tokens=160, selector_max_input_tokens=8192)
    prepare(args)
    m = json.loads((tmp_path / "out/manifest.json").read_text())
    assert "SECRET_GT" not in json.dumps(m)
    assert len(m["entries"]["1"]["candidates"]) == 2  # NOT target-only shortlisting.
    assert m["config"]["selector_expected_id_input"] is False
    prepare(args)  # Exact same manifest resumes safely.
    query["expected_slot"] = 1
    requests.write_text(json.dumps(query) + "\n")
    with pytest.raises(ValueError, match="no expected IDs"):
        prepare(args)


def test_full_bundle_replace_and_appearance_transfer_are_distinct():
    objects = [("source_semantic", "source_memory"), ("other", "other_memory")]
    donors = [("donor_semantic", "donor_memory")]
    registers = [("background", "background_memory")]
    resolved = dict(operation="replace", target=candidate("a", 0), donor=candidate("b", 0))
    units, background = apply_selected(objects, registers, resolved, donors, replacement_mode="full_bundle")
    assert units == [donors[0], objects[1]]
    assert background is registers
    resolved["operation"] = "transfer"
    units, background = apply_selected(objects, registers, resolved, donors, replacement_mode="full_bundle")
    assert units == [("source_semantic", "donor_memory"), objects[1]]
    assert objects[0] == ("source_semantic", "source_memory")
    assert background is registers
