"""CPU regressions for GT-free AR parsing, routing, and routed scoring."""
import unittest

import numpy as np

from pgot.eval.ar_binding_router import parse_ar_slots, route_ar_queries, routed_prediction, normalize_category
from pgot.eval.binding_hits import score_binding_hits


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.pred = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
        self.slots = parse_ar_slots('<thing> Man 1: a man. <ovt>. <thing> Person 2: a person. <ovt>. <scene_end>', 2)

    def test_same_class_point_selection_keeps_query_identity(self):
        inputs = [{'category': 'person', 'center_normalized_xy': [.75, .5]},
                  {'category': 'person', 'center_normalized_xy': [.25, .5]}]
        labels, rows, _ = route_ar_queries(self.pred, self.slots, inputs)
        self.assertEqual(labels, [2, 1])
        self.assertEqual(rows[0]['point_distance_squared'], 0)
        routed = routed_prediction(self.pred, labels)
        np.testing.assert_array_equal(routed, [[2, 2, 1, 1], [2, 2, 1, 1]])
        gt = np.stack([self.pred == 2, self.pred == 1])
        self.assertTrue(all(r['hit'] for r in score_binding_hits(routed, gt, [0, 1])))
        self.assertTrue(all(r['hit'] for r in score_binding_hits(self.pred, gt, [0, 1], query_labels=labels, num_predictions=2)))

    def test_unmatched_and_one_to_one_shortage_are_misses(self):
        slots = parse_ar_slots('<thing> Person 1: person <ovt>', 1)
        inputs = [{'category': 'person', 'center_normalized_xy': [.25, .5]},
                  {'category': 'person', 'center_normalized_xy': [.75, .5]},
                  {'category': 'dog', 'center_normalized_xy': [.5, .5]}]
        labels, rows, _ = route_ar_queries(np.ones((2, 2)), slots, inputs)
        self.assertEqual(sum(x is not None for x in labels), 1)
        self.assertIsNone(labels[2])
        self.assertEqual(rows[2]['unmatched_reason'], 'no_compatible_nonempty_slot')
        gt = np.stack([self.pred == 1, self.pred == 2])
        rows = score_binding_hits(np.zeros((2, 4)), gt, [0, 1], query_labels=[None, None])
        self.assertFalse(any(r['hit'] for r in rows))
        self.assertTrue(all(r['target_iou'] == 0 for r in rows))
        no_ar_slots = score_binding_hits(np.zeros((2, 4)), gt, [0, 1],
                                         query_labels=[None, None], num_predictions=0)
        self.assertFalse(any(r['hit'] for r in no_ar_slots))

    def test_class_constraint_and_fixed_gt_target_survive_nearby_distractors(self):
        slots = parse_ar_slots('<thing> Person 1: person <ovt> <thing> Dog 1: dog <ovt>', 2)
        inputs = [{'category': 'dog', 'center_normalized_xy': [.25, .5]}]
        labels, _, _ = route_ar_queries(self.pred, slots, inputs)
        self.assertEqual(labels, [2])  # class wins over the nearer wrong-class region
        gt = np.stack([self.pred == 1, self.pred == 2])
        scored = score_binding_hits(self.pred, gt, [0], query_labels=labels, num_predictions=2)
        self.assertFalse(scored[0]['hit'])  # scoring never rewrites expected identity
        self.assertEqual(scored[0]['matched_gt_index'], 1)

    def test_empty_stuff_malformed_slots_preserve_ovt_indices(self):
        slots = parse_ar_slots('<stuff> Sky 1: sky <ovt> junk <ovt> <thing> Dog 1: dog <ovt>', 3)
        self.assertEqual([s['prediction_label'] for s in slots], [1, 2, 3])
        self.assertFalse(slots[0]['is_thing'])
        self.assertFalse(slots[1]['parsed'])
        labels, _, _ = route_ar_queries(np.full((2, 2), 3), slots,
                        [{'category': 'dog', 'center_normalized_xy': [.5, .5]}])
        self.assertEqual(labels, [3])
        labels, _, candidates = route_ar_queries(np.zeros((2, 2)), self.slots,
                        [{'category': 'person', 'center_normalized_xy': [.5, .5]}])
        self.assertEqual(labels, [None])
        self.assertEqual(candidates, [])
        self.assertEqual(parse_ar_slots('<scene_end>', 0), [])
        self.assertEqual(route_ar_queries(np.zeros((2, 2)), [], [])[0], [])

    def test_gt_scoring_metadata_rejected_and_point_validation(self):
        query = {'category': 'person', 'center_normalized_xy': [.5, .5]}
        for key in ('targets', 'annotation_id', 'target_gt_index', 'ignore', 'category_id'):
            with self.assertRaises(ValueError):
                route_ar_queries(self.pred, self.slots, [dict(query, **{key: 0})])
        with self.assertRaises(ValueError):
            route_ar_queries(self.pred, self.slots, [dict(query, center_normalized_xy=[float('nan'), .5])])
        with self.assertRaises(ValueError):
            parse_ar_slots('<thing> Dog 1: dog <ovt>', 2)

    def test_residual_background_and_class_only(self):
        routed = routed_prediction(self.pred, [2, None])
        np.testing.assert_array_equal(routed, [[0, 0, 1, 1], [0, 0, 1, 1]])
        inputs = [{'category': 'person', 'center_normalized_xy': [.1, .1]}]
        labels, rows, _ = route_ar_queries(self.pred, self.slots, inputs, use_point=False)
        self.assertIsNotNone(labels[0])
        self.assertIsNone(rows[0]['point_distance_squared'])
        self.assertEqual(normalize_category('A sofa'), 'couch')
        with self.assertRaises(ValueError):
            routed_prediction(self.pred, [1, 1])


if __name__ == '__main__':
    unittest.main()
