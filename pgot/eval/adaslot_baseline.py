"""Inference-only adapter for the official AdaSlot COCO checkpoint.

Constructs the architecture in the official COCO eval YAML directly, avoiding
the legacy Lightning training/data plugins. All model operations come from the
unmodified AdaSlot repository; all checkpoint tensors must load strictly.
"""

from functools import partial
from pathlib import Path
import subprocess
import sys

import torch


class AdaSlotBaseline:
    name = "AdaSlot"

    def __init__(self, args, device):
        root = Path(args.adaslot_root)
        checkpoint_path = Path(args.adaslot_ckpt)
        if not (root / "ocl" / "models" / "image_grouping_adaslot.py").is_file():
            raise FileNotFoundError(
                f"Missing AdaSlot repository: {root}; run scripts/setup_adaslot_eval.sh"
            )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing AdaSlot checkpoint: {checkpoint_path}; run scripts/setup_adaslot_eval.sh"
            )
        sys.path.insert(0, str(root))
        import timm
        from torchvision import transforms
        from ocl.conditioning import RandomConditioning
        from ocl.decoding import PatchDecoderGumbelV1
        from ocl.feature_extractors import TimmFeatureExtractor
        from ocl.models.image_grouping_adaslot import GroupingImgGumbel
        from ocl.neural_networks import build_mlp, build_two_layer_mlp
        from ocl.neural_networks.positional_embedding import DummyPositionEmbed
        from ocl.neural_networks.wrappers import Sequential
        from ocl.perceptual_grouping import SlotAttentionGroupingGumbelV1

        if timm.__version__ != "0.6.7":
            raise RuntimeError(
                f"AdaSlot requires its inference venv (timm 0.6.7), got {timm.__version__}"
            )
        self.model = GroupingImgGumbel(
            conditioning=RandomConditioning(object_dim=256, n_slots=33),
            # The released checkpoint includes the complete frozen DINO model.
            # Never download a second backbone or allow random missing weights.
            feature_extractor=TimmFeatureExtractor(
                model_name="vit_base_patch16_224_dino", feature_level=12,
                pretrained=False, freeze=True,
            ),
            perceptual_grouping=SlotAttentionGroupingGumbelV1(
                input_dim=768, feature_dim=256, object_dim=256,
                iters=3, low_bound=0, use_projection_bias=False,
                positional_embedding=Sequential(
                    DummyPositionEmbed(),
                    build_two_layer_mlp(768, 256, 768, initial_layer_norm=True),
                ),
                ff_mlp=build_two_layer_mlp(
                    256, 256, 1024, initial_layer_norm=True, residual=True,
                ),
                single_gumbel_score_network=build_two_layer_mlp(
                    256, 2, 1024, initial_layer_norm=True,
                ),
            ),
            object_decoder=PatchDecoderGumbelV1(
                object_dim=256, output_dim=768, num_patches=196,
                decoder=partial(build_mlp, features=[2048, 2048, 2048]),
                mask_type="mask_normalized",
            ),
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw_state = checkpoint["state_dict"]
        if not raw_state or any(not key.startswith("models.") for key in raw_state):
            raise RuntimeError("Unexpected AdaSlot checkpoint state_dict namespace")
        state = {key.removeprefix("models."): value for key, value in raw_state.items()}
        report = self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.device = device
        self.last_diagnostics = []
        # Official _preprocessing_coco_dino_feature_recon_ccrop eval transform:
        # tensor bicubic resize(min side 224), clamp, center crop, normalize.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # torchvision 0.13 (official dependency) defaulted to no tensor
            # antialiasing; modern torchvision changed that default to True.
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=False),
            transforms.Lambda(lambda image: image.clamp(0.0, 1.0)),
            transforms.CenterCrop(224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.info = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_global_step": int(checkpoint["global_step"]),
            "config": str(root / "configs/experiment/projects/bridging/dinosaur/"
                          "coco_feat_rec_dino_base16_adaslot_eval.yaml"),
            "repository_revision": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
            ).strip(),
            "num_slots": 33,
            "model_input_resolution": 224,
            "decoder_patch_grid": [14, 14],
            "mask_source": "official object_decoder.masks_as_image (224x224), kept-slot masked alpha",
            "preprocessing": "official COCO eval: tensor bicubic Resize(224), clamp, CenterCrop(224), ImageNet normalize",
            "resize_antialias": False,
            "slot_selection": "official stochastic random conditioning + hard Gumbel softmax (tau=1); no GT count",
            "dtype": "float32",
            "torch_version": torch.__version__,
            "timm_version": timm.__version__,
            "load_report": {
                "strict": True, "loaded_tensors": len(state),
                "missing": len(report.missing_keys), "unexpected": len(report.unexpected_keys),
                "frozen_backbone_included": True,
            },
        }
        del state, raw_state, checkpoint

    @torch.no_grad()
    def __call__(self, images):
        x = torch.stack([self.transform(image) for image in images]).to(self.device)
        outputs = self.model({"input": {"image": x}})
        masks = outputs["object_decoder"].masks_as_image
        kept = outputs["hard_keep_decision"]
        probabilities = outputs["slots_keep_prob"]
        if masks.shape != (len(images), 33, 224, 224) or not torch.isfinite(masks).all():
            raise RuntimeError(f"Invalid AdaSlot mask output: {tuple(masks.shape)}")
        self.last_diagnostics = [
            {"kept_slots": int(count), "slot_keep_probability_mean": float(probability)}
            for count, probability in zip(
                kept.sum(-1).cpu().tolist(), probabilities.mean(-1).cpu().tolist()
            )
        ]
        return masks.float()
