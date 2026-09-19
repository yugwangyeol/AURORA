"""Opt-in GPU integration test using the downloaded official COCO checkpoint.

ADASLOT_INTEGRATION=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=0 \
  /home/jovyan/.venvs/adaslot_eval/bin/python -m unittest discover \
  -s tests -p test_adaslot_voc_adapter.py -v
"""
import os
from types import SimpleNamespace
import unittest


@unittest.skipUnless(os.environ.get("ADASLOT_INTEGRATION") == "1", "requires AdaSlot checkpoint and CUDA")
class AdaSlotVOCAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from pgot.eval.adaslot_baseline import AdaSlotBaseline

        if not torch.cuda.is_available():
            raise RuntimeError("Requested AdaSlot GPU integration test, but CUDA is unavailable")
        torch.set_num_threads(3)
        torch.manual_seed(0)
        args = SimpleNamespace(
            adaslot_root=os.environ.get("ADASLOT_ROOT", "/home/jovyan/baselines/AdaSlot"),
            adaslot_ckpt=os.environ.get("ADASLOT_CKPT", "/home/jovyan/baselines/ckpt/adaslot-coco.ckpt"),
        )
        cls.baseline = AdaSlotBaseline(args, "cuda")

    def test_strict_complete_checkpoint_and_native_forward_parity(self):
        import numpy as np
        from PIL import Image
        import torch

        report = self.baseline.info["load_report"]
        self.assertTrue(report["strict"])
        self.assertEqual(report["missing"], 0)
        self.assertEqual(report["unexpected"], 0)
        self.assertTrue(report["frozen_backbone_included"])
        self.assertFalse(any(p.requires_grad for p in self.baseline.model.feature_extractor.parameters()))
        image_path = "/home/jovyan/data/voc/images/2007_000033.jpg"
        image = np.asarray(Image.open(image_path).convert("RGB")).copy()
        torch.manual_seed(123)
        masks = self.baseline([image])
        diagnostics = self.baseline.last_diagnostics
        torch.manual_seed(123)
        with torch.no_grad():
            x = self.baseline.transform(image).unsqueeze(0).cuda()
            native = self.baseline.model({"input": {"image": x}})
        torch.testing.assert_close(masks, native["object_decoder"].masks_as_image, rtol=0, atol=0)
        kept = native["hard_keep_decision"]
        self.assertEqual(diagnostics[0]["kept_slots"], int(kept.sum().item()))
        self.assertGreater(int(kept.sum().item()), 0)
        self.assertEqual(int(torch.count_nonzero(masks[kept == 0]).item()), 0)
        self.assertFalse(self.baseline.info["resize_antialias"])


if __name__ == "__main__":
    unittest.main()
