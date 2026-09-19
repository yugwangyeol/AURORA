"""Opt-in official-checkpoint GPU mask and preprocessing parity check.

SLOTDIFFUSION_INTEGRATION=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=0 \
 /home/jovyan/.venvs/slotdiffusion_eval/bin/python -m unittest discover \
 -s tests -p test_slotdiffusion_baseline.py -v
"""
import os
from types import SimpleNamespace
import unittest


@unittest.skipUnless(os.environ.get("SLOTDIFFUSION_INTEGRATION") == "1", "requires official COCO weights and CUDA")
class SlotDiffusionIntegrationTest(unittest.TestCase):
    def test_official_forward_and_opencv_preprocessing(self):
        import cv2
        import numpy as np
        import torch
        from pgot.eval.slotdiffusion_baseline import SlotDiffusionBaseline

        torch.set_num_threads(3)
        torch.manual_seed(0)
        args = SimpleNamespace(slotdiffusion_root="/home/jovyan/baselines/SlotDiffusion",
                               slotdiffusion_ckpt="/home/jovyan/baselines/SlotDiffusion/pretrained/sa_ldm_dino_coco_params-res224.pth",
                               dataset="coco")
        baseline = SlotDiffusionBaseline(args, "cuda")
        image = np.random.default_rng(0).integers(0, 256, (317, 521, 3), dtype=np.uint8)
        factor = 224 / 317
        resized = cv2.resize(image, (round(521 * factor), 224), interpolation=cv2.INTER_LINEAR)
        offset = (resized.shape[1] - 224) // 2
        expected = (resized[:, offset:offset + 224].astype(np.float32) / 255. - .5) / .5
        expected = torch.from_numpy(expected.copy()).permute(2, 0, 1).unsqueeze(0).cuda()
        torch.testing.assert_close(baseline.preprocess([image]), expected, rtol=0, atol=0)
        masks = baseline([image])
        with torch.no_grad():
            native_masks = baseline.model({"img": expected})["masks"]
        torch.testing.assert_close(masks, native_masks, rtol=0, atol=0)
        self.assertEqual(tuple(masks.shape), (1, 7, 224, 224))
        self.assertTrue(torch.isfinite(masks).all())
        torch.testing.assert_close(masks.sum(1), torch.ones_like(masks[:, 0]), rtol=1e-5, atol=1e-6)
        self.assertTrue(baseline.info["load_report"]["strict"])
        self.assertFalse(baseline.info["diffusion_sampling"])
        self.assertFalse(any(p.requires_grad for p in baseline.model.encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
