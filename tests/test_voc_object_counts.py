import math
import unittest

from pgot.eval.voc_object_counts import METRICS, count_bucket, summarize_object_counts


class VOCObjectCountTest(unittest.TestCase):
    def test_bucket_boundaries(self):
        expected = {0: "0", 1: "1-3", 3: "1-3", 4: "4-6", 6: "4-6",
                    7: "7-10", 10: "7-10", 11: "11+", 19: "11+"}
        for count, label in expected.items():
            self.assertEqual(count_bucket(count), label)
        with self.assertRaises(ValueError):
            count_bucket(-1)

    def test_gt_not_prediction_grouping_and_nan(self):
        rows = [
            {"gt_objects": 2, "pred_segments": 30, **dict.fromkeys(METRICS, 0.5)},
            {"gt_objects": 3, "pred_segments": 1, **dict.fromkeys(METRICS, math.nan)},
            {"gt_objects": 7, "pred_segments": 2, **dict.fromkeys(METRICS, 0.8)},
        ]
        result = summarize_object_counts(rows)
        group = result["by_object_count"]["1-3"]
        self.assertEqual(group["n"], 2)
        self.assertEqual(group["fARI_valid_n"], 1)
        self.assertEqual(group["fARI"], 0.5)
        self.assertEqual(result["by_exact_object_count"]["3"]["fARI"], None)
        self.assertEqual(sum(g["n"] for g in result["by_object_count"].values()), 3)

    def test_empty_and_kept_slot_statistics(self):
        self.assertEqual(summarize_object_counts([])["by_object_count"], {})
        result = summarize_object_counts([{"gt_objects": 1, "kept_slots": 5},
                                          {"gt_objects": 2, "kept_slots": 7}])
        self.assertEqual(result["by_object_count"]["1-3"]["kept_slots_mean"], 6)


if __name__ == "__main__":
    unittest.main()
