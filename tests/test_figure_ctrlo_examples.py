import unittest
import numpy as np

from pgot.eval.figure_ctrlo_examples import queries, iou, overlay
from PIL import Image


class CTRLOFigureTests(unittest.TestCase):
    def test_thing_compaction_and_unique_category_only(self):
        entry = {"ar_caption": {"text": "<stuff>Sky: Blue<ovt><thing>Dog: Brown<ovt><thing>Person 1: Left<ovt><thing>Person 2: Right<ovt>", "ovt_count": 4}}
        pred = np.array([[1, 1, 2, 3]])
        result = queries(entry, pred, {"dog": None, "person": None})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["prediction_label"], 1)
        self.assertEqual(result[0]["query"], "dog")

    def test_ignore_and_empty_union(self):
        a, b = np.array([1, 1, 0]), np.array([1, 0, 0])
        self.assertEqual(iou(a, b, np.array([1, 0, 1], bool)), 1.)
        self.assertEqual(iou(a*0, b*0, np.ones(3, bool)), 0.)

    def test_overlay_does_not_modify_mask_or_unmasked_image(self):
        image = Image.fromarray(np.full((2, 2, 3), 100, np.uint8))
        mask = np.array([[1, 0], [0, 0]], bool)
        before = mask.copy()
        result = np.asarray(overlay(image, mask))
        np.testing.assert_array_equal(mask, before)
        np.testing.assert_array_equal(result[~mask], np.asarray(image)[~mask])


if __name__ == "__main__":
    unittest.main()
