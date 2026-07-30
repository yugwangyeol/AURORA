import unittest

import torch

from pgot.model.pgot_utils import (
    apply_e5_rae_ovt_forcing_mask,
    build_pgot_attention_mask,
)


class TestOVTIsolatedAttention(unittest.TestCase):
    def setUp(self):
        self.positions = {
            "sys_s": 0,
            "sys_e": 2,
            "user_prefix_s": 2,
            "user_prefix_e": 3,
            "img_s": 3,
            "img_e": 7,
            "user_suffix_s": 7,
            "user_suffix_e": 8,
            "assistant_prefix_s": 8,
            "assistant_prefix_e": 9,
            "cap_s": 9,
            "cap_e": 15,
            "assistant_suffix_s": 15,
            "assistant_suffix_e": 16,
            "null_bg_s": 16,
            "null_bg_e": 16,
            "reg_s": 16,
            "reg_e": 18,
            "rae_s": 18,
            "rae_e": 21,
            "total_len": 21,
        }
        self.caption_mask = torch.ones(1, 6, dtype=torch.bool)
        self.ovt_positions = torch.tensor([[10, 13]], dtype=torch.long)
        self.ovt_valid = torch.tensor([[True, True]])

    @staticmethod
    def allowed(mask: torch.Tensor, row: int):
        return set((mask[0, 0, row] == 0).nonzero(as_tuple=False).flatten().tolist())

    def build(self, isolated: bool, own_caption: bool = False):
        return build_pgot_attention_mask(
            positions=self.positions,
            caption_padding_mask=self.caption_mask,
            device=torch.device("cpu"),
            ovt_absolute_positions=self.ovt_positions,
            ovt_valid_mask=self.ovt_valid,
            register_attends_caption=False,
            ovt_isolated=isolated,
            ovt_attends_own_caption=own_caption,
        )

    def test_each_ovt_sees_only_image_patches_and_itself(self):
        mask = self.build(isolated=True)
        image_columns = set(range(3, 7))
        self.assertEqual(self.allowed(mask, 10), image_columns | {10})
        self.assertEqual(self.allowed(mask, 13), image_columns | {13})

    def test_ordinary_caption_and_e2_causal_rae_paths_are_unchanged(self):
        mask = self.build(isolated=True)
        self.assertEqual(self.allowed(mask, 11), set(range(12)))
        self.assertEqual(self.allowed(mask, 18), {10, 13, 16, 17, 18})
        self.assertEqual(self.allowed(mask, 20), {10, 13, 16, 17, 18, 19, 20})

    def test_disabled_flag_preserves_legacy_ovt_caption_path(self):
        mask = self.build(isolated=False)
        self.assertEqual(self.allowed(mask, 10), set(range(11)))
        self.assertEqual(self.allowed(mask, 13), set(range(14)))

    def test_each_isolated_ovt_can_restore_only_its_own_caption_span(self):
        mask = self.build(isolated=True, own_caption=True)
        image_columns = set(range(3, 7))
        self.assertEqual(self.allowed(mask, 10), image_columns | {9, 10})
        self.assertEqual(self.allowed(mask, 13), image_columns | {11, 12, 13})

    def test_e5_forcing_removes_register_and_all_rae_keys_only_for_selected_sample(self):
        base = self.build(isolated=True).expand(2, -1, -1, -1).clone()
        forced = apply_e5_rae_ovt_forcing_mask(
            base,
            positions=self.positions,
            forcing_sample_mask=torch.tensor([True, False]),
        )
        for row in range(18, 21):
            self.assertTrue(torch.isneginf(forced[0, 0, row, 16:21]).all())
        self.assertTrue(torch.equal(forced[1], base[1]))


if __name__ == "__main__":
    unittest.main()
