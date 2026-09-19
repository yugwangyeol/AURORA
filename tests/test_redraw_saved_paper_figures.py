import hashlib
import json

import pytest
from PIL import Image

from pgot.eval.language_object_editing import file_digest, write_json
from pgot.eval.redraw_saved_paper_figures import redraw


def test_frozen_panel_pixels_and_verbatim_descriptions_preserved(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda self, path, **kwargs: saved.append(str(path)))
    source = tmp_path/"source";source.mkdir()
    original = Image.new("RGB", (1240,1000), "white")
    groups = []
    for i in range(4):
        x,y = (i%2)*640, 200+(i//2)*500
        entries = []
        for j in range(3):
            box = [x+j*200,y,x+(j+1)*200,y+200]
            tile = Image.new("RGB", (200,200), (i*40,j*50,30))
            original.paste(tile, (box[0],box[1]))
            # Deliberately stale/absent source path; saved embedded PNG is authoritative.
            entries.append({"source": "/not/an/available/panel.png", "box": box})
        groups.append({"rank":i+1,"descriptions":["A dog.","A cup."],"panels":entries})
    path = source/"deor_grounding_2x2.png";original.save(path)
    write_json(source/"manifest.json", {"canvas_size":[1240,1000],"sets":groups})
    before = file_digest(path)
    manifest = redraw(source,tmp_path/"output")
    assert file_digest(path) == before
    assert manifest["ai_image_editing"] is False
    assert manifest["masks_recolored_or_modified"] is False
    assert manifest["descriptions_paraphrased"] is False
    for group,record in zip(groups,manifest["panels"]):
        assert record["descriptions"] == group["descriptions"]
        for entry in record["panels"]:
            expected = original.crop(entry["box"]).tobytes()
            assert entry["embedded_tile_rgb_sha256"] == hashlib.sha256(expected).hexdigest()
    assert len(saved) == 4
    assert json.loads((tmp_path/"output/manifest.json").read_text())["renderer"] == "matplotlib"


def test_refuse_original_output_overwrite(tmp_path):
    with pytest.raises(ValueError, match="Preserve original figures"):
        redraw(tmp_path,tmp_path)
