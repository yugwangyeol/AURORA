import json

import pytest
from PIL import Image

from pgot.eval.language_object_editing import (
    digest, file_digest, messages_for, parse_plan, resolve_plan, write_json,
)
from pgot.eval.render_language_object_editing import OUTLINE_COLOR, compose, outlined


def test_require_full_execution_audit_before_figure(tmp_path):
    write_json(tmp_path/"manifest.json", {"requests": [{"id": "q"}]})
    write_json(tmp_path/"validation.json", {"exact_llm_selection_execution_validated": True, "checked_requests": 0})
    with pytest.raises(ValueError, match="full exact-selection audit"):
        compose(tmp_path)


def test_figure_uses_actual_plan_images_and_keeps_abstention(tmp_path, monkeypatch):
    # Image-generation model is never involved in renderer-only unit tests.
    from matplotlib.figure import Figure
    saves = []
    monkeypatch.setattr(Figure, "savefig", lambda self, path, **kwargs: saves.append(str(path)))
    a = {"id": "opaque_a", "category": "Dog", "description": "A brown dog.", "slot_index": 0}
    b = {"id": "opaque_b", "category": "Dog", "description": "A white dog.", "slot_index": 0}
    queries = []
    m = {"fingerprint": "fp", "entries": {"1": {"candidates": [a]}, "2": {"candidates": [b]}}}
    write_json(tmp_path/"selector_info.json", {"selection_fingerprint": "sel", "spec": {"fingerprint": "fp"}})
    for index in range(6):
        query = dict(id=f"q{index}", image_A=1, image_B=2 if index % 2 else None,
                     request="Replace A's dog with B's dog." if index % 2 else "Remove the dog from image A.")
        queries.append(query)
        ca, cb = [a], [b] if index % 2 else []
        # Explicitly include a controller abstention rather than hiding that case.
        operation = "none" if index == 0 else "replace" if index % 2 else "remove"
        raw = json.dumps(dict(operation=operation, target_id=None if index == 0 else "opaque_a",
                              donor_id="opaque_b" if index % 2 else None))
        plan, status = parse_plan(raw, ca, cb)
        resolved = resolve_plan(plan, status, ca, cb)
        messages = messages_for(query["request"], ca, cb)
        selection_path = tmp_path/"plans"/(query["id"]+".json")
        write_json(selection_path, dict(query=query, messages=messages, prompt_sha256=digest(messages),
            raw_completion=raw, plan=plan, status=status, resolved=resolved, selection_fingerprint="sel"))
        folder = tmp_path/"edits"/query["id"]
        write_json(folder/"metadata.json", dict(executed=resolved, status="generated" if index else "not_generated_abstain",
                                                selection_sha256=file_digest(selection_path)))
        if index:
            for name in ("source_A.png", "reconstruction_native.png", "edited_native.png", "donor_B.png"):
                Image.new("RGB", (32, 32), (index*20, 50, 60)).save(folder/name)
            for name in ("target_prediction_mask.png", "donor_prediction_mask.png"):
                Image.new("1", (32, 32), 0).save(folder/name)
    m["requests"] = queries
    write_json(tmp_path/"manifest.json", m)
    write_json(tmp_path/"validation.json", {"exact_llm_selection_execution_validated": True, "checked_requests": 6})
    compose(tmp_path)
    manifest = json.loads((tmp_path/"figure/manifest.json").read_text())
    assert manifest["actual_recorded_llm_decisions"] is True
    assert manifest["manual_corrections"] is False
    assert len(manifest["figure_requests"]) == 6
    assert manifest["figure_requests"][0]["status"] == "not_generated_abstain"
    assert manifest["figure_requests"][0]["native_panels"] == []  # NO invented fallback.
    assert len(manifest["figure_requests"][1]["native_panels"]) == 4
    assert len(manifest["figure_requests"][2]["native_panels"]) == 3
    assert len(saves) == 4  # PNG, PDF, SVG and preview.
    assert "none/invalid" in (tmp_path/"selection_report.md").read_text()
    assert (tmp_path/"review_all_requests.jpg").is_file()


def test_predicted_overlay_thicker_contour_and_fill_are_display_only(tmp_path):
    import numpy as np
    image_path, mask_path = tmp_path/"image.png", tmp_path/"mask.png"
    Image.new("RGB", (40,40), (100,100,100)).save(image_path)
    mask = np.zeros((40,40), dtype=bool)
    mask[10:30,10:30] = True
    Image.fromarray(mask).save(mask_path)
    image_hash, mask_hash = file_digest(image_path), file_digest(mask_path)
    line = np.asarray(outlined(image_path, mask_path))
    filled = np.asarray(outlined(image_path, mask_path, fill_alpha=0.38))
    assert tuple(line[20,20]) == (100,100,100)  # Replace: no interior tint.
    assert tuple(filled[20,20]) == tuple(np.rint(0.62*100+0.38*np.asarray(OUTLINE_COLOR)).astype(np.uint8))
    assert tuple(line[20,13]) == OUTLINE_COLOR  # Visible inward contour, not just one pixel.
    assert tuple(line[20,8]) == OUTLINE_COLOR  # Two-pixel outward contour.
    assert tuple(filled[0,0]) == (100,100,100)
    assert file_digest(image_path) == image_hash
    assert file_digest(mask_path) == mask_hash
