import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
from PIL import Image

from pgot.eval.coco_baseline_dataset import COCOBaselineDataset
from pgot.eval.summarize_coco_checkpoint_baselines import matched_summary
from pgot.eval.slotdiffusion_baseline import SlotDiffusionBaseline


class COCOBaselineDatasetTests(unittest.TestCase):
    def test_slotdiffusion_rejects_non_coco_checkpoint_before_model_import(self):
        args = SimpleNamespace(slotdiffusion_root="missing", slotdiffusion_ckpt="voc.pth")
        with patch("pgot.eval.slotdiffusion_baseline.file_sha256", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "COCO"):
                SlotDiffusionBaseline(args, "cpu")

    def test_cache_labels_preserved_and_overlaps_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "val2017").mkdir()
            Image.new("RGB", (4, 2)).save(root / "val2017/000000000139.jpg")
            instance = np.array([[[0, 1], [3, 3]]], dtype=np.uint16)
            category = np.array([[[0, 5], [5, 5]]], dtype=np.uint16)
            overlap = np.array([[[0, 1], [0, 0]]], dtype=np.uint8)
            for kind, masks, ov in [("instance", instance, overlap), ("category", category, np.zeros_like(overlap))]:
                path = root / kind
                path.mkdir()
                np.save(path / "masks.npy", masks)
                np.save(path / "overlap_masks.npy", ov)
                np.save(path / "image_ids.npy", np.array([139]))
                (path / "meta.json").write_text(json.dumps({"mode": "coda", "size": 2, "gt_granularity": kind}))
            dataset = COCOBaselineDataset(root, root / "instance", root / "category", eval_size=2)
            row = dataset[0]
            np.testing.assert_array_equal(row["instance"].numpy(), instance[0])
            np.testing.assert_array_equal(row["semantic"].numpy(), category[0])
            self.assertEqual(row["overlap"].sum().item(), 1)
            self.assertEqual(row["semantic_overlap"].sum().item(), 0)
            self.assertEqual(row["id"], 139)

    def test_matched_summary_is_nan_safe_and_requires_complete_ids(self):
        summary = {"object_count_source": "cache", "model": "SPOT"}
        metrics = {"fARI": 0.5, "mBO": float("nan"), "mIoU": 0.3, "sMBO": 0.2, "sMIOU": 0.4}
        records = [{"id": 139, "gt_objects": 2, "pred_segments": 7, **metrics}]
        result = matched_summary(summary, records, {139}, "manifest")
        self.assertEqual(result["num_samples"], 1)
        self.assertIsNone(result["mBO"])
        self.assertEqual(result["mBO_valid_n"], 0)
        self.assertEqual(result["mIoU_i"], 0.3)
        with self.assertRaises(ValueError):
            matched_summary(summary, records, {139, 285}, "manifest")


if __name__ == "__main__":
    unittest.main()
