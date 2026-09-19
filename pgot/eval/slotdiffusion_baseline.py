"""Unmodified official SlotDiffusion, using only the COCO configuration/weights."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


class SlotDiffusionBaseline:
    name = "SlotDiffusion"

    def __init__(self, args, device):
        root = Path(args.slotdiffusion_root)
        digest = file_sha256(args.slotdiffusion_ckpt)
        if digest != "653c7e539aae123b3b9051124414626909d1f1f35106a7cff521ee2afd9e56ca":
            raise ValueError("Expected verified official COCO SlotDiffusion checkpoint, not a VOC/other checkpoint")
        sys.path.insert(0, str(root))
        from slotdiffusion.img_based.models import build_model
        from slotdiffusion.img_based.datasets.coco_transforms import ResizeMinShape as COCOResize
        from slotdiffusion.img_based.datasets.coco_transforms import CenterCrop as COCOCrop
        from slotdiffusion.img_based.datasets.coco_transforms import Normalize as COCONorm
        from slotdiffusion.img_based.datasets.voc_transforms import ResizeMinShape, CenterCrop, Normalize

        config = root / "slotdiffusion/img_based/configs/sa_ldm/sa_ldm_dino_coco_params-res224.py"
        spec = importlib.util.spec_from_file_location("slotdiffusion_coco_params", config)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        params = module.SlotAttentionParams()
        params.slot_dict = copy.deepcopy(params.slot_dict)
        params.enc_dict = copy.deepcopy(params.enc_dict)
        params.dec_dict = copy.deepcopy(params.dec_dict)
        params.dec_dict["vae_dict"]["vqvae_ckp_path"] = str(root / "pretrained/vqvae_coco_params-res224.pth")
        model = build_model(params)
        checkpoint = torch.load(args.slotdiffusion_ckpt, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state, strict=True)
        self.model = model.to(device).eval()
        self.model.testing = True
        self.device = device
        self.dataset = args.dataset
        if args.dataset == "coco":
            self.transforms = (COCOResize(params.resolution), COCOCrop(params.resolution), COCONorm(params.norm_mean, params.norm_std))
        else:
            self.transforms = (ResizeMinShape(params.resolution), CenterCrop(params.resolution), Normalize(params.norm_mean, params.norm_std))
        self.info = {
            "checkpoint": str(args.slotdiffusion_ckpt), "checkpoint_sha256": digest,
            "config": str(config), "train_dataset": "COCO",
            "repository_revision": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
            "num_slots": int(params.slot_dict["num_slots"]), "model_input_resolution": 224,
            "encoder": params.enc_dict["dino"], "decoder_patch_grid": list(model.visual_resolution),
            "mask_source": "official SADiffusion.forward masks: final slot-attention, bilinear 28 -> 224",
            "preprocessing": f"official {args.dataset.upper()} val: OpenCV ResizeMinShape(224), CenterCrop(224), normalize mean/std=0.5",
            "diffusion_sampling": False, "dtype": "float32", "torch_version": torch.__version__,
            "load_report": {"strict": True, "loaded_tensors": len(state), "missing": 0, "unexpected": 0,
                            "frozen_backbone_included": True},
        }

    def preprocess(self, images):
        batch = []
        for image in images:
            h, w = image.shape[:2]
            sample = {"image": image, "masks": np.zeros((h, w), dtype=np.uint8)}
            if self.dataset == "coco":
                sample.update(annos=np.zeros((0, 5), dtype=np.float32), scale=1., size=[h, w])
            for transform in self.transforms:
                sample = transform(sample)
            batch.append(torch.from_numpy(np.ascontiguousarray(sample["image"])).permute(2, 0, 1))
        return torch.stack(batch).float().to(self.device)

    @torch.no_grad()
    def __call__(self, images):
        return self.model({"img": self.preprocess(images)})["masks"]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()
