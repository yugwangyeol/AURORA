"""CPU tests for full coverage, GT-free placeholders, and empty-scene readout."""
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from pgot.constants import OVT_TOKEN, SCENE_END_TOKEN, THING_TOKEN
from pgot.eval.coda_datasets import CodaEvalPGOTDataset
from pgot.eval.voc_object_counts import summarize_object_counts
from pgot.train.pgot_dataset import PGOTDataCollator


class FakeTokenizer:
    def convert_tokens_to_ids(self, token):
        return {OVT_TOKEN: 2, SCENE_END_TOKEN: 3, THING_TOKEN: 4}.get(token, 1)

    def encode(self, text, add_special_tokens=False):
        return [self.convert_tokens_to_ids(token) for token in text.split()]


class FakeProcessor:
    def preprocess(self, image, return_tensors="pt"):
        return {"pixel_values": torch.zeros(1, 3, 2, 2)}


class FullARDataTests(unittest.TestCase):
    def test_coco_5000_and_empty_images_and_no_gt_count_in_caption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "annotations").mkdir()
            (root / "val2017").mkdir()
            cache = root / "cache"
            cache.mkdir()
            (root / "annotations/instances_val2017.json").write_text(json.dumps({"images": [{"id": i} for i in range(5000)]}))
            (cache / "meta.json").write_text(json.dumps({"mode": "coda", "size": 2}))
            masks = np.zeros((5000, 2, 2), dtype=np.uint16)
            masks[1] = [[0, 1], [3, 3]]
            np.save(cache / "masks.npy", masks)
            np.save(cache / "overlap_masks.npy", np.zeros_like(masks, dtype=np.uint8))
            np.save(cache / "image_ids.npy", np.arange(5000))
            for i in (0, 1):
                Image.new("RGB", (4, 3)).save(root / "val2017" / f"{i:012d}.jpg")
            # Only bypass file-existence preflight for this tiny fixture.
            with patch.object(Path, "is_file", return_value=True):
                dataset = CodaEvalPGOTDataset("coco", root, FakeTokenizer(), FakeProcessor(),
                                              eval_size=2, coco_mask_cache=cache)
            self.assertEqual(len(dataset), 5000)
            empty, nonempty = dataset[0], dataset[1]
            self.assertEqual(empty["n_objects"], 0)
            self.assertEqual(nonempty["n_objects"], 2)  # count IDs, not max label=3
            torch.testing.assert_close(empty["caption_input_ids"], nonempty["caption_input_ids"])
            self.assertFalse(nonempty["ovt_valid_mask"].any())
            self.assertEqual(empty["caption_input_ids"].tolist(), [3])
            self.assertEqual(nonempty["image_id"], 1)
            batch = PGOTDataCollator(pad_token_id=0)([empty, nonempty])
            self.assertEqual(batch["n_objects_list"], [0, 2])
            self.assertEqual(batch["image_ids"], [0, 1])
            # Incomplete coverage must never silently fall back to 4720 images.
            np.save(cache / "image_ids.npy", np.arange(4720))
            with self.assertRaisesRegex(ValueError, "5,000"):
                CodaEvalPGOTDataset("coco", root, FakeTokenizer(), FakeProcessor(),
                                    eval_size=2, coco_mask_cache=cache)

    def test_voc_retains_gt_empty_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for folder in ("sets", "images", "SegmentationObject", "SegmentationClass"):
                (root / folder).mkdir()
            (root / "sets/val.txt").write_text("empty\nnonempty\n")
            for name, label in (("empty", 0), ("nonempty", 1)):
                Image.new("RGB", (3, 2)).save(root / "images" / f"{name}.jpg")
                for folder in ("SegmentationObject", "SegmentationClass"):
                    Image.fromarray(np.full((2, 3), label, dtype=np.uint8)).save(root / folder / f"{name}.png")
            dataset = CodaEvalPGOTDataset("voc", root, FakeTokenizer(), FakeProcessor(), eval_size=2)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]["n_objects"], 0)
            self.assertEqual(dataset[1]["n_objects"], 1)

    def test_zero_bucket_and_nan_safe_valid_counts(self):
        scores = summarize_object_counts([{"gt_objects": 0, "fARI": 1., "mBO": float("nan"), "mIoU": .1}])
        zero = scores["by_object_count"]["0"]
        self.assertEqual(zero["n"], 1)
        self.assertEqual(zero["mBO_valid_n"], 0)
        self.assertIsNone(zero["mBO"])

    def test_no_generated_ovts_can_read_out_register_background(self):
        from pgot.model.pgot_utils import build_pred_mask_ovt_owner_eval
        pred = build_pred_mask_ovt_owner_eval(torch.zeros(1, 50, 4), torch.ones(1, 4, 4) / 4,
                                             torch.zeros(1, 50, dtype=torch.bool), torch.zeros(1, 50, dtype=torch.bool),
                                             target_size=4, n_ovt_per_object=1, patch_grid=2)
        self.assertEqual(tuple(pred.shape), (1, 4, 4))
        self.assertEqual(pred.count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
