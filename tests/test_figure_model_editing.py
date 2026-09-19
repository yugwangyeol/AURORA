import tempfile
import unittest
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image

from pgot.eval.figure_model_editing import binary_iou, overlay, edit_stats, panel_grid, select


class ModelEditingFigureTests(unittest.TestCase):
    def test_iou_and_empty(self):
        a = np.array([[1, 0], [0, 0]], bool)
        self.assertEqual(binary_iou(a, a), 1.)
        self.assertEqual(binary_iou(a*0, a*0), 0.)

    def test_stats_are_only_change_not_semantic_success(self):
        ref = np.zeros((2, 2, 3))
        edit = ref.copy()
        edit[0, 0] = 1.
        region = np.array([[1, 0], [0, 0]], bool)
        stats = edit_stats(edit, ref, region)
        self.assertEqual(stats["inside_change"], 1.)
        self.assertEqual(stats["outside_change"], 0.)
        self.assertEqual(stats["change_energy_inside_fraction"], 1.)
        self.assertNotIn("success", stats)

    def test_overlay_and_export(self):
        image = Image.fromarray(np.full((2, 2, 3), 100, np.uint8))
        mask = np.array([[1, 0], [0, 0]], bool)
        before = mask.copy()
        result = overlay(image, mask)
        np.testing.assert_array_equal(mask, before)
        np.testing.assert_array_equal(np.asarray(result)[~mask], np.asarray(image)[~mask])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "figure.png"
            grid = panel_grid([image, result, image], ["Original", "Mask", "Edit"], path, size=32)
            self.assertEqual(grid.size, (96, 70))
            self.assertTrue(path.exists())

    def test_review_priority_and_distinct_target_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reviews = {}
            for task, target, score, status in [("a", "1", 9., "success"),
                    ("b", "1", 8., "success"), ("c", "2", .5, "success"),
                    ("d", "3", 100., "failure")]:
                dest = root / "candidates/remove" / task
                dest.mkdir(parents=True)
                (dest / "metadata.json").write_text(json.dumps({"task_id": task,
                    "target_id": target, "stats": {"screening_score": score}}))
                Image.new("RGB", (96, 70), "white").save(dest / "figure.png")
                reviews[task] = {"status": status}
            (root / "visual_review.json").write_text(json.dumps(reviews))
            select(SimpleNamespace(output_dir=directory, examples=2, selection_subdir="curated"))
            records = json.loads((root / "curated_selection.json").read_text())["groups"]["remove"]
            self.assertEqual([r["target_id"] for r in records], ["1", "2"])
            self.assertTrue(all(r["visual_success_verified"] for r in records))


if __name__ == "__main__":
    unittest.main()
