import json

import pytest
from PIL import Image

from pgot.eval.compose_object_transfer_figure import compose, extract_panels


def test_exact_overview_columns(tmp_path):
    image = Image.new("RGB", (60, 15), "white")
    for column in range(6):
        image.paste((column * 30, 5, 10), (column * 10, 5, column * 10 + 10, 15))
    path = tmp_path / "figure.png"
    image.save(path)
    panels, boxes = extract_panels(path)
    assert [p.getpixel((0, 0))[0] for p in panels] == [0, 30, 60, 120]
    assert boxes == [(0, 5, 10, 15), (10, 5, 20, 15),
                     (20, 5, 30, 15), (40, 5, 50, 15)]


def test_six_case_render_and_provenance(tmp_path):
    entries = []
    for index in range(6):
        name = f"pair{index}"
        folder = tmp_path / "pairs" / name
        folder.mkdir(parents=True)
        Image.new("RGB", (60, 15), (index * 20, 50, 80)).save(folder / "figure.png")
        (folder / "metadata.json").write_text(json.dumps({"instruction": "Replace A's dog with B's brown dog."}))
        entries.append({"source_root": str(tmp_path), "pair_id": name})
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"examples": entries}))
    manifest = compose(selection, tmp_path / "out", dpi=25)
    assert [r["grid_position"] for r in manifest["examples"]] == [
        [1, 1], [1, 2], [2, 1], [2, 2], [3, 1], [3, 2]]
    assert all(r["overview_columns"] == [1, 2, 3, 5] for r in manifest["examples"])
    assert all(r["result_condition"] == "visual_only" for r in manifest["examples"])
    assert manifest["prompt_font_pt"] == 23
    assert manifest["freeform_instruction_following_claim"] is False
    assert Image.open(tmp_path / "out" / "object_transfer_3x2.png").size == (600, 325)
    for suffix in ("pdf", "svg"):
        assert (tmp_path / "out" / f"object_transfer_3x2.{suffix}").is_file()


def test_reject_incorrect_case_count(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"examples": []}))
    with pytest.raises(ValueError, match="six distinct"):
        compose(selection, tmp_path / "out")


def test_remove_left_replace_right_preserves_operation_provenance(tmp_path):
    entries = []
    for index, category in enumerate(("cup", "bench", "giraffe")):
        removal = tmp_path / f"remove{index}"
        removal.mkdir()
        for name in ("original.png", "reconstruction_native.png", "edited_native.png"):
            Image.new("RGB", (10, 10), (index * 30, 90, 20)).save(removal / name)
        (removal / "metadata.json").write_text(json.dumps({"mode": "remove", "target_category": category}))
        entries.append({"mode": "remove", "pair_id": f"remove{index}", "source_dir": str(removal)})
        folder = tmp_path / "pairs" / f"replace{index}"
        folder.mkdir(parents=True)
        Image.new("RGB", (60, 15), (index * 30, 20, 90)).save(folder / "figure.png")
        (folder / "metadata.json").write_text(json.dumps({"instruction": "Replace A's dog with B's dog."}))
        entries.append({"pair_id": f"replace{index}", "source_root": str(tmp_path)})
    selection = tmp_path / "mixed.json"
    selection.write_text(json.dumps({"column_titles": ["Remove", "Replace"],
                                    "filename": "mixed", "examples": entries,
                                    "compact": True, "center_line": True,
                                    "red_operation_verbs": True}))
    manifest = compose(selection, tmp_path / "out", dpi=25)
    assert manifest["editing_protocols_differ_by_operation"] is True
    assert manifest["outer_margin_inches"] == 0
    assert manifest["center_gap_inches"] == 0.28
    assert manifest["center_line"] is True
    assert manifest["red_operation_verbs"] is True
    assert manifest["canvas_inches"][0] < 21
    assert (tmp_path / "out" / "mixed.pdf").exists()
    for left, right in zip(manifest["examples"][::2], manifest["examples"][1::2]):
        assert left["grid_position"][1] == 1
        assert left["panel_count"] == 3
        assert left["writer_rewrite"] is True
        assert len(left["source_panels"]) == 3
        assert left["prompt"].startswith("Remove the ")
        assert right["grid_position"][1] == 2
        assert right["panel_count"] == 4
        assert right["result_condition"] == "visual_only"
        assert right["writer_rewrite"] is False
