import unittest
import numpy as np
from PIL import Image

from pgot.eval.figure_ar_grounding_examples import first_sentence, overlay, thing_slots


class ARGroundingFigureTests(unittest.TestCase):
    def test_thing_label_compaction_retains_original_slot_identity(self):
        entry = {"ar_caption": {"text": "<stuff>Sky: Blue.<ovt><thing>Dog 1: Brown dog.<ovt><thing>Person: Red jacket.<ovt>", "ovt_count": 3}}
        slots = thing_slots(entry)
        self.assertEqual([s['prediction_label'] for s in slots], [1, 2])
        self.assertEqual([s['slot_index'] for s in slots], [1, 2])
        self.assertEqual(slots[0]['description'], 'Brown dog.')

    def test_first_sentence_is_verbatim_not_a_rewritten_caption(self):
        text = 'A person holds a phone. The phone is black.'
        self.assertEqual(first_sentence(text), 'A person holds a phone.')
        self.assertTrue(text.startswith(first_sentence(text)))

    def test_overlay_preserves_mask_and_unselected_pixels(self):
        image = Image.fromarray(np.full((2, 2, 3), 100, dtype=np.uint8))
        mask = np.array([[True, False], [False, False]])
        before = mask.copy()
        result = np.asarray(overlay(image, mask, (235, 65, 70)))
        np.testing.assert_array_equal(mask, before)
        np.testing.assert_array_equal(result[~mask], np.asarray(image)[~mask])


if __name__ == '__main__':
    unittest.main()
