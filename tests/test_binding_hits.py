"""CPU identity/matching and COCO class-point query tests."""
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch

from pgot.eval.binding_hits import score_binding_hits
from pgot.eval.binding_coco_dataset import (BindingCocoDataset, BindingCollator, QUERY_MODES,
                                           build_query_text, natural_position)
from pgot.eval.run_binding_hits import aggregate


class Tokenizer:
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {"<ovt>": 2, "<scene_end>": 3}.get(token, 1)

    def encode(self, text, add_special_tokens=False):
        return [self.convert_tokens_to_ids(token.rstrip(".")) for token in text.split()]


class Processor:
    def preprocess(self, image, return_tensors="pt"):
        return {"pixel_values": torch.zeros(1, 3, 4, 4)}


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.targets = np.array([[[1, 1], [0, 0]], [[0, 0], [1, 1]]], dtype=bool)

    def test_natural_position_grid_and_boundaries(self):
        self.assertEqual(natural_position(.1, .9), "lower-left part of the image")
        self.assertEqual(natural_position(.9, .1), "upper-right part of the image")
        self.assertEqual(natural_position(.5, .5), "center of the image")
        self.assertEqual(natural_position(1 / 3, 1 / 3), "center of the image")
        self.assertEqual(natural_position(2 / 3, 2 / 3), "lower-right part of the image")
        self.assertEqual(natural_position(0, 0), "upper-left part of the image")
        self.assertEqual(natural_position(1, 1), "lower-right part of the image")
        with self.assertRaises(ValueError):
            natural_position(-.1, .5)

    def test_prompt_variants_preserve_query_identity_and_legacy_text(self):
        objects = [dict(annotation_id=11, category="person", center_normalized_xy=[.25, .60]),
                   dict(annotation_id=12, category="person", center_normalized_xy=[.75, .10])]
        legacy, reference = build_query_text(objects, [1, 0], "class_point")
        self.assertTrue(legacy.startswith("<thing> Person 1: a person centered at normalized image coordinates (x=0.7500, y=0.1000), with x measured from the left and y from the top. <ovt>."))
        for mode in QUERY_MODES:
            caption, queries = build_query_text(objects, [1, 0], mode)
            self.assertEqual([{k: v for k, v in q.items() if k != "text"} for q in queries],
                             [{k: v for k, v in q.items() if k != "text"} for q in reference])
            self.assertEqual(caption.count("<ovt>"), 2)
            self.assertEqual(caption.count("<scene_end>"), 1)
            if mode != "class_only":
                self.assertIn("(x=0.7500, y=0.1000)", caption)
                self.assertIn("(x=0.2500, y=0.6000)", caption)
            self.assertEqual(build_query_text(objects, [], mode), ("<scene_end>", []))
        short, _ = build_query_text(objects, [0], "class_point_short")
        self.assertIn("a person at normalized coordinates", short)
        natural, _ = build_query_text(objects, [0], "class_point_natural")
        self.assertIn("middle-left part of the image", natural)
        with self.assertRaises(ValueError):
            build_query_text(objects, [0], "invalid")

    def test_perfect_and_swapped_queries(self):
        pred = np.array([[1, 1], [2, 2]])
        self.assertTrue(all(r["hit"] for r in score_binding_hits(pred, self.targets, [0, 1])))
        swapped = score_binding_hits(pred, self.targets, [1, 0])
        self.assertFalse(any(r["hit"] for r in swapped))
        self.assertTrue(all(r["target_iou"] == 0 for r in swapped))

    def test_distractor_is_not_removed_and_unmatched_query_is_miss(self):
        # The only queried slot covers the OTHER instance; correct target goes to BG.
        rows = score_binding_hits([[0, 0], [1, 1]], self.targets, [0])
        self.assertFalse(rows[0]["hit"])
        self.assertEqual(rows[0]["matched_gt_index"], 1)
        # With one GT, an empty query must not be accidentally hit by row-index compaction.
        rows = score_binding_hits(np.zeros((2, 2)), self.targets[:1], [0])
        self.assertIsNone(rows[0]["matched_gt_index"])
        self.assertFalse(rows[0]["hit"])

    def test_overlap_ignore_and_identity_validation(self):
        targets = np.ones((1, 2, 2), dtype=bool)
        row = score_binding_hits([[0, 1], [1, 1]], targets, [0], [[1, 0], [0, 0]])[0]
        self.assertEqual(row["target_iou"], 1.)
        with self.assertRaises(ValueError):
            score_binding_hits([[1, 1], [2, 2]], self.targets, [0, 0])
        with self.assertRaises(ValueError):
            score_binding_hits([[3, 3], [3, 3]], self.targets, [0])

    def test_zero_queries_and_object_weighted_aggregation(self):
        self.assertEqual(score_binding_hits(np.zeros((2, 2)), np.zeros((0, 2, 2)), []), [])
        hit = {"hit": True, "target_iou": 1., "hit_iou50": True, "positive_iou_hit": True}
        miss = {"hit": False, "target_iou": 0., "hit_iou50": False, "positive_iou_hit": False}
        summary = aggregate([{"queries": []}, {"queries": [hit]}, {"queries": [miss, miss]}])
        self.assertEqual(summary["num_images"], 3)
        self.assertEqual(summary["num_queries"], 3)
        self.assertAlmostEqual(summary["binding_hits"], 1 / 3)
        self.assertEqual(summary["binding_hits_image_mean"], .5)
        self.assertIsNone(aggregate([{"queries": []}])["binding_hits"])

    def test_annotation_query_pairs_same_class_empty_scene_and_collation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "annotations").mkdir()
            (root / "val2017").mkdir()
            for iid in (1, 2):
                Image.new("RGB", (4, 4)).save(root / "val2017" / f"{iid:012d}.jpg")
            annotations = {"images": [{"id": i, "height": 4, "width": 4, "file_name": f"{i:012d}.jpg"} for i in (1, 2)],
                           "categories": [{"id": 1, "name": "person"}],
                           "annotations": [{"id": i, "image_id": 1, "category_id": 1, "iscrowd": 0,
                                            "area": 8, "bbox": [x, 0, 2, 4],
                                            "segmentation": [[x, 0, x+2, 0, x+2, 4, x, 4]]}
                                           for i, x in [(11, 0), (12, 2)]]}
            (root / "annotations/instances_val2017.json").write_text(json.dumps(annotations))
            kwargs = dict(root=root, tokenizer=Tokenizer(), image_processor=Processor(), target_processor=Processor(),
                          eval_size=4, max_bindings=1, max_objects=4, require_full_split=False)
            dataset = BindingCocoDataset(**kwargs)
            sample, empty = dataset[0], dataset[1]
            self.assertEqual(len(sample["targets"]), 2)  # unqueried distractor remains
            self.assertEqual(len(sample["queries"]), 1)
            self.assertEqual(sample["queries"][0]["annotation_id"], sample["objects"][sample["selected"][0]]["annotation_id"])
            self.assertIn("x=", sample["caption"])
            class_only = BindingCocoDataset(**kwargs, query_mode="class_only")[0]
            self.assertEqual(class_only["selected"], sample["selected"])
            self.assertNotIn("x=", class_only["caption"])
            for mode in ("class_point_short", "class_point_natural"):
                variant = BindingCocoDataset(**kwargs, query_mode=mode)[0]
                self.assertEqual(variant["selected"], sample["selected"])
                self.assertEqual(variant["objects"], sample["objects"])
                np.testing.assert_array_equal(variant["targets"], sample["targets"])
                torch.testing.assert_close(variant["ovt_valid"], sample["ovt_valid"])
                self.assertIn("x=", variant["caption"])
            self.assertEqual(empty["queries"], [])
            self.assertFalse(empty["ovt_valid"].any())
            batch = BindingCollator(0)([sample, empty])
            self.assertEqual(batch["ovt_valid_mask"].sum().item(), 1)
            self.assertEqual(batch["images"].shape[0], 2)
            self.assertFalse(batch["caption_attention_mask"][1, -1])
            # Only scalar identity/category/point metadata is in the query, never GT pixels.
            self.assertNotIn("mask", sample["queries"][0])
            # All targets are selected in all-instance mode; overflow is an error.
            all_data = BindingCocoDataset(**dict(kwargs, max_bindings=0))
            self.assertEqual(len(all_data[0]["queries"]), 2)
            with self.assertRaisesRegex(ValueError, "silently truncate"):
                BindingCocoDataset(**dict(kwargs, max_bindings=0, max_objects=1))[0]
            with self.assertRaisesRegex(ValueError, "5000-image"):
                BindingCocoDataset(**dict(kwargs, require_full_split=True))


if __name__ == "__main__":
    unittest.main()
