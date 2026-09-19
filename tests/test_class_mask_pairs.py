import unittest
from unittest.mock import patch

import numpy as np

from pgot.eval.class_mask_pairs import ar_pairs, ctrlo_pair_masks, masks_for_pairs, rates, score_pairs
from pgot.eval.run_ar_text_pair_eval import center_crop
from PIL import Image


class ClassMaskPairTests(unittest.TestCase):
    def test_original_ovt_identity_and_thing_compaction(self):
        pairs = ar_pairs('<stuff>Sky: Blue<ovt><thing>Dog 1: Brown. Sitting.<ovt><thing>Person 1: Left<ovt>', 3)
        self.assertEqual([p['slot_index'] for p in pairs], [1, 2])
        self.assertEqual([p['prediction_label'] for p in pairs], [1, 2])
        self.assertEqual(pairs[0]['shared_text'], 'Dog: Brown. Sitting.')

    def test_duplicates_and_unknown_names_not_filtered(self):
        pairs = ar_pairs('<thing>Dog 1: Left<ovt><thing>Dog 2: Right<ovt><thing>Alien 1: Green<ovt>', 3)
        self.assertEqual(len(pairs), 3)
        self.assertEqual([p['normalized_category'] for p in pairs], ['dog', 'dog', 'alien'])

    def test_malformed_and_unbound_predictions_remain(self):
        pairs = ar_pairs('<thing>bad text<ovt><thing>Dog: Brown<scene_end>', 1)
        self.assertEqual(len(pairs), 2)
        self.assertFalse(pairs[0]['parsed'])
        self.assertTrue(pairs[1]['unbound_text'])
        self.assertIsNone(pairs[1]['prediction_label'])
        masks = masks_for_pairs(np.ones((2, 2), int), pairs)
        self.assertTrue(masks[0].all())
        self.assertFalse(masks[1].any())

    def test_empty_seed_marker_is_not_a_generated_object(self):
        self.assertEqual(ar_pairs('<thing> <scene_end>', 0), [])
        self.assertEqual(ar_pairs('<scene_end>', 0), [])

    def test_wrong_class_empty_and_duplicate_predictions_are_fp(self):
        gt = np.ones((1, 2, 2), bool)
        pred = np.array([np.ones((2, 2)), np.ones((2, 2)), np.zeros((2, 2)), np.ones((2, 2))], bool)
        result = score_pairs(pred, ['dog', 'dog', 'dog', 'cat'], gt, ['dog'], np.ones((2, 2), bool))['0.50']
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 3, 0))

    def test_missing_gt_objects_remain_fn(self):
        result = score_pairs(np.zeros((0, 2, 2), bool), [], np.ones((2, 2, 2), bool), ['dog', 'cat'], np.ones((2, 2), bool))['0.50']
        self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 0, 2))

    def test_zero_gt_images_false_positives_and_undefined_rates(self):
        result = score_pairs(np.ones((1, 2, 2), bool), ['dog'], np.zeros((0, 2, 2), bool), [], np.ones((2, 2), bool))['0.50']
        self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 1, 0))
        self.assertEqual(rates(0, 0, 0)['precision'], None)
        self.assertEqual(rates(0, 0, 0)['recall'], None)

    def test_match_cardinality_precedes_sum_iou(self):
        with patch('pgot.eval.class_mask_pairs.iou_matrix', return_value=np.array([[.99, .5], [.5, 0.]])):
            result = score_pairs(np.ones((2, 2, 2), bool), ['dog']*2, np.ones((2, 2, 2), bool), ['dog']*2, np.ones((2, 2), bool))['0.50']
        self.assertEqual(result['tp'], 2)

    def test_ignore_pixels_shared(self):
        pred = np.array([[[1, 1], [0, 0]]], bool)
        gt = np.array([[[1, 0], [0, 0]]], bool)
        valid = np.array([[1, 0], [1, 1]], bool)
        self.assertEqual(score_pairs(pred, ['dog'], gt, ['dog'], valid)['0.50']['tp'], 1)

    def test_overflow_all_query_identities_preserved(self):
        maps = np.array([[[0, 1, 5, 6]], [[0, 1, 6, 6]]], np.uint8)
        masks = ctrlo_pair_masks(maps, 8, 6)
        self.assertEqual(len(masks), 8)
        np.testing.assert_array_equal(masks[6], maps[1] == 0)
        np.testing.assert_array_equal(masks[7], maps[1] == 1)
        with self.assertRaises(ValueError):
            ctrlo_pair_masks(maps[:1], 8, 6)

    def test_square_shared_center_crop(self):
        image = Image.fromarray(np.full((4, 8, 3), 123, np.uint8))
        result = center_crop(image, 4)
        self.assertEqual(result.size, (4, 4))
        self.assertTrue((np.asarray(result) == 123).all())


if __name__ == '__main__':
    unittest.main()
