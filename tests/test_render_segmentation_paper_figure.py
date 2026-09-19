import unittest

import numpy as np

from pgot.eval.render_segmentation_paper_figure import PALETTE, blend, label_colors


class PaperSegmentationRenderTests(unittest.TestCase):
    def test_custom_background_changes_only_background_display_color(self):
        gt = np.array([[0, 0, 0], [1, 1, 1]])
        pred = np.array([[5, 5, 5], [2, 2, 3]])
        ignore = np.zeros_like(gt, bool)
        _, old = label_colors(pred, gt, ignore)
        _, new = label_colors(pred, gt, ignore, (128, 142, 160))
        self.assertEqual(new[5], (128, 142, 160))
        self.assertEqual(new[2], old[2])
        self.assertEqual(new[3], old[3])

    def test_background_and_permuted_labels(self):
        gt = np.array([[0, 0], [1, 1]])
        pred = np.array([[4, 4], [2, 2]])
        before = pred.copy()
        rgb, colors = label_colors(pred, gt, np.zeros_like(gt, bool))
        self.assertEqual(colors[4], PALETTE[0])
        self.assertEqual(colors[2], PALETTE[1])
        np.testing.assert_array_equal(pred, before)
        np.testing.assert_array_equal(rgb[0, 0], PALETTE[0])

    def test_oversegmented_object_keeps_different_colors(self):
        gt = np.array([[0, 0, 0], [1, 1, 1]])
        pred = np.array([[5, 5, 5], [2, 2, 3]])
        _, colors = label_colors(pred, gt, np.zeros_like(gt, bool))
        self.assertEqual(len(set(colors.values())), 3)
        self.assertNotEqual(colors[2], colors[3])

    def test_ignored_region_and_overlay_strength(self):
        gt = np.array([[0, 1]])
        ignore = np.array([[False, True]])
        rgb, _ = label_colors(gt, gt, ignore)
        np.testing.assert_array_equal(rgb[0, 1], [160, 160, 160])
        image = np.zeros_like(rgb)
        np.testing.assert_array_equal(blend(image, rgb, 1), rgb)
        np.testing.assert_array_equal(blend(image, rgb, 0), image)
        with self.assertRaises(ValueError):
            blend(image, rgb, 2)


if __name__ == "__main__":
    unittest.main()
