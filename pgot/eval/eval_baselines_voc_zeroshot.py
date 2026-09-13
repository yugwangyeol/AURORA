"""Zero-shot PASCAL VOC segmentation of COCO-trained object-centric baselines.

CODA, SPOT and MetaSlot checkpoints trained on COCO are scored on VOC2012 val
with exactly the ground truth and metric code behind PGOT's VOC row
(``run_eval --dataset voc``):

* samples: ``<voc_root>/sets/val.txt`` in file order (1,449 ids);
* GT: CODA 512 center crop (``_coda_center_crop_image``); pixels labelled 255
  in either map form the overlap mask and are zeroed in both GT maps;
* metrics: ``pgot.eval.pgot_metrics`` fARI / mBO / mIoU on instance masks and
  mBO / mIoU on semantic masks, per image, NaN-skipped mean.

Each baseline sees the ORIGINAL image through its own repository's validation
preprocessing and contributes only the soft slot masks its paper evaluates.
Those masks are upsampled bilinearly to 512 and arg-maxed -- the read-out every
one of these papers uses, at PGOT's metric resolution.

Run one model per process (``--model``): the three repositories ship clashing
top-level module names (SPOT's ``datasets.py`` shadows HuggingFace ``datasets``).
"""
import argparse
import copy
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, "/home/jovyan/PGOT")
from pgot.eval.coda_datasets import _remap_contiguous
from pgot.eval.pgot_metrics import fari_metric, mbo_metric, miou_metric
from pgot.train.pgot_dataset import _coda_center_crop_image, _pil_resample

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pgot.voc_zeroshot_baselines")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VOCZeroShot(Dataset):
    """PGOT's VOC GT (``CodaEvalPGOTDataset._load_voc``) plus the raw image."""

    def __init__(self, root: str, eval_size: int = 512, max_samples=None):
        self.root = Path(root)
        split_file = self.root / "sets" / "val.txt"
        ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        self.ids = ids[: int(max_samples)] if max_samples else ids
        self.eval_size = int(eval_size)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        image = Image.open(self.root / "images" / f"{sample_id}.jpg").convert("RGB")
        instance = Image.open(self.root / "SegmentationObject" / f"{sample_id}.png")
        semantic = Image.open(self.root / "SegmentationClass" / f"{sample_id}.png")
        nearest = _pil_resample("NEAREST")
        instance_np = np.asarray(
            _coda_center_crop_image(instance, self.eval_size, resample=nearest), dtype=np.uint8
        ).copy()
        semantic_np = np.asarray(
            _coda_center_crop_image(semantic, self.eval_size, resample=nearest), dtype=np.uint8
        ).copy()
        overlap = (instance_np == 255) | (semantic_np == 255)
        instance_np[overlap] = 0
        semantic_np[overlap] = 0
        return {
            "id": sample_id,
            "image": np.asarray(image, dtype=np.uint8).copy(),
            "instance": torch.from_numpy(_remap_contiguous(instance_np)).long(),
            "semantic": torch.from_numpy(_remap_contiguous(semantic_np)).long(),
            "overlap": torch.from_numpy(overlap.astype(np.int64)),
        }


def collate(items):
    return {
        "id": [it["id"] for it in items],
        "image": [it["image"] for it in items],  # variable-size originals
        "instance": torch.stack([it["instance"] for it in items]),
        "semantic": torch.stack([it["semantic"] for it in items]),
        "overlap": torch.stack([it["overlap"] for it in items]),
    }


# ---------------------------------------------------------------------------
# Baselines: each returns soft slot masks [B, K, h, w] for original RGB images.
# ---------------------------------------------------------------------------
class CodaBaseline:
    """CODA encoder only; masks = slot-attention maps (CODA ``measure_segmentation``)."""

    name = "CODA"

    def __init__(self, args, device):
        root = Path(args.coda_root)
        ckpt = Path(args.coda_ckpt)
        sys.path.insert(0, str(root))
        import diffusers

        # src/model/encoder.py imports StableDiffusion3Pipeline at module level for an
        # SD3 helper that segmentation never calls; diffusers 0.27 lacks the symbol.
        if not hasattr(diffusers, "StableDiffusion3Pipeline"):
            diffusers.StableDiffusion3Pipeline = None
        from omegaconf import OmegaConf
        from safetensors.torch import load_file

        from experiment.dataset.voc import VOCTransforms
        from src.model.encoder import RegisterSlotDiffusion

        conf = OmegaConf.load(ckpt / "config.yaml")
        enc_kwargs = OmegaConf.to_container(conf.encoder, resolve=True)
        target = enc_kwargs.pop("_target_")
        if not target.endswith("RegisterSlotDiffusion"):
            raise ValueError(f"Unexpected CODA encoder class {target}")
        self.encoder = RegisterSlotDiffusion(**enc_kwargs)
        state = load_file(ckpt / "encoder" / "diffusion_pytorch_model.safetensors")
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        # CODA saves the frozen DINOv2 outside the encoder state (DINOEncoder.state_dict),
        # and loads with strict=False itself; anything else missing is an error.
        bad_missing = [k for k in missing if not k.startswith("backbone.dinov2.")]
        if bad_missing or unexpected:
            raise RuntimeError(f"CODA encoder load: missing={bad_missing[:8]} unexpected={unexpected[:8]}")
        self.encoder.to(device).eval()
        self.transform = VOCTransforms((512, 512), norm_mean=0.5, norm_std=0.5, val=True)
        self.device = device
        self.info = {
            "checkpoint": str(ckpt),
            "num_slots": int(enc_kwargs["slot_n_slots"]),
            "model_input_resolution": 512,
            "mask_source": "encoder slot attention (dino_sample_size x dino_sample_size)",
            "preprocessing": "CODA experiment.dataset.voc.VOCTransforms(512, val=True)",
            "load_report": {"missing_frozen_dinov2_keys": len(missing), "other_missing": 0, "unexpected": 0},
        }

    @torch.no_grad()
    def __call__(self, images):
        batch = []
        for img in images:
            h, w = img.shape[:2]
            out = self.transform({"image": img, "masks": np.zeros((h, w), dtype=np.uint8)})
            batch.append(torch.from_numpy(np.ascontiguousarray(out["image"])).permute(2, 0, 1))
        x = torch.stack(batch).float().to(self.device)
        attn = self.encoder(x)["attn"]  # [B, heads=1, M, N]
        b, heads, tokens, slots = attn.shape
        side = int(round(math.sqrt(tokens)))
        return attn.mean(1).permute(0, 2, 1).reshape(b, slots, side, side)


class SpotBaseline:
    """SPOT (dino_vitb16, transformer decoder); masks = decoder slot attention (as eval_spot.py)."""

    name = "SPOT"

    def __init__(self, args, device):
        root = Path(args.spot_root)
        # Append, never prepend: SPOT's datasets.py would shadow HuggingFace datasets.
        sys.path.append(str(root))
        from torchvision import transforms

        from spot import SPOT

        spot_args = argparse.Namespace(
            image_size=224, val_image_size=224, num_dec_blocks=4, d_model=768, num_heads=6,
            dropout=0.0, num_iterations=3, num_slots=args.spot_num_slots, slot_size=256,
            mlp_hidden_size=1024, img_channels=3, pos_channels=4, num_cross_heads=None,
            dec_type="transformer", cappa=-1, mlp_dec_hidden=2048, use_slot_proj=True,
            which_encoder="dino_vitb16", finetune_blocks_after=100, encoder_final_norm=False,
            truncate="bi-level", init_method="embedding", use_second_encoder=True,
            train_permutations="random", eval_permutations=args.spot_eval_permutations,
        )
        spot_args.max_tokens = int((spot_args.val_image_size / 16) ** 2)
        spot_args.num_cross_heads = spot_args.num_heads
        encoder = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").eval()
        second = copy.deepcopy(encoder).eval()
        model = SPOT(encoder, spot_args, second)
        checkpoint = torch.load(args.spot_ckpt, map_location="cpu", weights_only=False)
        state = {k.replace("tf_dec.", "dec."): v for k, v in checkpoint["model"].items()}
        model.load_state_dict(state, strict=True)
        self.model = model.to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.device = device
        self.info = {
            "checkpoint": str(args.spot_ckpt),
            "num_slots": int(args.spot_num_slots),
            "model_input_resolution": 224,
            "eval_permutations": args.spot_eval_permutations,
            "mask_source": "transformer-decoder slot attention (14x14), mean over permutations",
            "preprocessing": "SPOT PascalVOC val: Resize(224) + CenterCrop(224) + ImageNet normalize",
            "load_report": "strict=True",
        }

    @torch.no_grad()
    def __call__(self, images):
        x = torch.stack([self.transform(Image.fromarray(img)) for img in images]).to(self.device)
        _, _, dec_slots_attns, _, _, _ = self.model(x)
        return dec_slots_attns  # [B, K, 14, 14]


class MetaSlotBaseline:
    """MetaSlot-DINOSAUR (MLP decoder); masks = decoder alpha ``attent2`` (config after_forward)."""

    name = "MetaSlot"

    def __init__(self, args, device):
        root = Path(args.metaslot_root)
        sys.path.insert(0, str(root))
        import object_centric_bench.datum  # noqa: F401  registers transform types (CenterCrop, ...)
        from object_centric_bench.model import ModelWrap
        from object_centric_bench.utils import Config, build_from_config

        cfg = Config.fromfile(Path(args.metaslot_cfg))
        model = ModelWrap(build_from_config(cfg.model), cfg.model_imap, cfg.model_omap)
        state = torch.load(args.metaslot_ckpt, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        model.load_state_dict(state, strict=True)
        self.model = model.to(device).eval()
        self.omap = list(cfg.model_omap)
        self.transform = build_from_config(dict(type="Compose", transforms=cfg.transform_v))
        self.device = device
        self.info = {
            "checkpoint": str(args.metaslot_ckpt),
            "config": str(args.metaslot_cfg),
            "num_slots": int(cfg.max_num),
            "model_input_resolution": list(cfg.resolut0),
            "mask_source": "BroadcastMLPDecoder alpha masks (output.attent2)",
            "preprocessing": "MetaSlot transform_v: CenterCrop(max square) + Resize + ImageNet normalize (0-255)",
            "autocast": "float16 (as MetaSlot eval.py)",
            "load_report": "strict=True",
        }

    @torch.no_grad()
    def __call__(self, images):
        batch = []
        for img in images:
            h, w = img.shape[:2]
            pack = {
                "image": torch.from_numpy(img).permute(2, 0, 1).contiguous(),
                "segment": torch.zeros((h, w), dtype=torch.uint8),
            }
            batch.append(self.transform(**pack)["image"].float())
        x = torch.stack(batch).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
            outputs = self.model.m(x)
        return outputs[self.omap.index("attent2")].float()  # [B, K, 16, 16]


BASELINES = {"coda": CodaBaseline, "spot": SpotBaseline, "metaslot": MetaSlotBaseline}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(BASELINES), required=True)
    p.add_argument("--voc_root", default="/home/jovyan/data/voc")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--eval_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--coda_root", default="/home/jovyan/coda")
    p.add_argument("--coda_ckpt", default="/home/jovyan/coda/pretrained/coda/coco")
    p.add_argument("--coda_metric_cross_check", action="store_true",
                   help="CODA only: also score with CODA's own src.metric.segmentation functions.")
    p.add_argument("--spot_root", default="/home/jovyan/baselines/spot")
    p.add_argument("--spot_ckpt", default="/home/jovyan/baselines/ckpt/spot_coco_checkpoint.pt.tar")
    p.add_argument("--spot_num_slots", type=int, default=7)
    p.add_argument("--spot_eval_permutations", choices=["standard", "all"], default="all")
    p.add_argument("--metaslot_root", default="/home/jovyan/baselines/MetaSlot")
    p.add_argument("--metaslot_cfg",
                   default="/home/jovyan/baselines/MetaSlot/Config/config-metaslot/dinosaur_r-coco.py")
    p.add_argument("--metaslot_ckpt", default="/home/jovyan/baselines/ckpt/metaslot-dinosaur-coco.pth")
    return p.parse_args()


def _mean(values):
    valid = [v for v in values if not math.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def main():
    args = parse_args()
    # The container's CPU quota is 3 cores while torch/OpenCV see 288 host cores;
    # uncapped pools make CPU tensor ops ~100x slower (see scripts/eval_baselines_voc_zeroshot.sh).
    import os

    threads = int(os.environ.get("OMP_NUM_THREADS", "3"))
    torch.set_num_threads(threads)
    try:
        import cv2

        cv2.setNumThreads(threads)
    except ImportError:
        pass
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    start = time.time()

    baseline = BASELINES[args.model](args, device)
    log.info("%s loaded: %s", baseline.name, baseline.info)

    coda_metrics = None
    if args.coda_metric_cross_check:
        if args.model != "coda":
            raise ValueError("--coda_metric_cross_check is only defined for --model coda")
        from src.metric.segmentation import fARI_metric as coda_fari
        from src.metric.segmentation import mbo_metric as coda_mbo
        from src.metric.segmentation import miou_metric as coda_miou

        coda_metrics = {"fARI": 0.0, "mBO": 0.0, "mIoU": 0.0, "sMBO": 0.0, "sMIOU": 0.0, "n": 0}

    dataset = VOCZeroShot(args.voc_root, args.eval_size, args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)
    log.info("VOC val: %d images | eval_size=%d | batch=%d", len(dataset), args.eval_size, args.batch_size)

    scores = {k: [] for k in ("fARI", "mBO", "mIoU", "sMBO", "sMIOU")}
    records = []
    mask_shape = None
    segments = []
    model_sec = metric_sec = 0.0
    for batch in tqdm(loader, desc=baseline.name):
        t0 = time.time()
        soft = baseline(batch["image"])
        mask_shape = list(soft.shape[1:])
        pred = F.interpolate(soft.float(), size=(args.eval_size, args.eval_size),
                             mode="bilinear", align_corners=False).argmax(dim=1).long()
        if pred.is_cuda:
            torch.cuda.synchronize()
        t1 = time.time()
        model_sec += t1 - t0
        # Score on the model device, as run_eval does: the one-hot ARI and
        # bitmask-DP mIoU at 512x512 are far slower on CPU.
        inst = batch["instance"].to(pred.device)
        sem = batch["semantic"].to(pred.device)
        overlap = batch["overlap"].to(pred.device)
        for b, sample_id in enumerate(batch["id"]):
            gt_b, pr_b, ov_b, sem_b = inst[b:b + 1], pred[b:b + 1], overlap[b:b + 1], sem[b:b + 1]
            row = {
                "id": sample_id,
                "fARI": fari_metric(gt_b, pr_b, ov_b),
                "mBO": mbo_metric(gt_b, pr_b, ov_b),
                "mIoU": miou_metric(gt_b, pr_b, ov_b),
                "sMBO": mbo_metric(sem_b, pr_b, ov_b),
                "sMIOU": miou_metric(sem_b, pr_b, ov_b),
                "pred_segments": int(torch.unique(pr_b).numel()),
                "gt_objects": int(gt_b.max().item()),
            }
            for key in scores:
                scores[key].append(row[key])
            segments.append(row["pred_segments"])
            records.append(row)
        if coda_metrics is not None:
            n = pred.shape[0]
            coda_metrics["fARI"] += float(coda_fari(inst, pred, overlap)) * n
            coda_metrics["mBO"] += float(coda_mbo(inst, pred, overlap)) * n
            coda_metrics["mIoU"] += float(coda_miou(inst, pred, overlap)) * n
            coda_metrics["sMBO"] += float(coda_mbo(sem, pred, overlap)) * n
            coda_metrics["sMIOU"] += float(coda_miou(sem, pred, overlap)) * n
            coda_metrics["n"] += n
        if pred.is_cuda:
            torch.cuda.synchronize()
        metric_sec += time.time() - t1
    baseline.info["timing_sec"] = {
        "model_and_preprocess": round(model_sec, 2),
        "metrics": round(metric_sec, 2),
    }

    summary = {
        "task": "voc_zero_shot_segmentation",
        "model": baseline.name,
        "train_dataset": "COCO",
        "dataset": "voc",
        "caption_mode": "not_applicable",
        "protocol": ("PGOT VOC protocol: sets/val.txt order, CODA 512 center crop GT, 255 -> overlap, "
                     "pgot_metrics per-image NaN-skipped means; model masks bilinear-upsampled to "
                     "eval_size then argmax"),
        "num_samples": len(records),
        "metric_resolution": args.eval_size,
        "fARI": _mean(scores["fARI"]),
        "mBO": _mean(scores["mBO"]),
        "mIoU": _mean(scores["mIoU"]),
        "mBO_i": _mean(scores["mBO"]),
        "mIoU_i": _mean(scores["mIoU"]),
        "sMBO": _mean(scores["sMBO"]),
        "sMIOU": _mean(scores["sMIOU"]),
        "mBO_variant": "mBO^i",
        "mIoU_variant": "mIoU^i",
        "pred_segments_mean": float(np.mean(segments)) if segments else float("nan"),
        "native_mask_shape": mask_shape,
        "model_info": baseline.info,
        "elapsed_sec": time.time() - start,
        "peak_gpu_memory_GiB": (torch.cuda.max_memory_allocated() / 2 ** 30
                                if torch.cuda.is_available() else None),
    }
    if coda_metrics is not None and coda_metrics["n"]:
        n = coda_metrics.pop("n")
        summary["coda_metric_cross_check"] = {k: v / n for k, v in coda_metrics.items()}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_image.jsonl", "w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    log.info("%s | n=%d fARI=%.4f mBO^i=%.4f mBO^c=%.4f mIoU^i=%.4f mIoU^c=%.4f -> %s",
             baseline.name, len(records), summary["fARI"], summary["mBO"], summary["sMBO"],
             summary["mIoU"], summary["sMIOU"], out_dir)


if __name__ == "__main__":
    main()
