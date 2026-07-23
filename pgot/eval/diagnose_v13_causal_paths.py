"""Causal information-path diagnostics for PGOT V13-style checkpoints.

This script tests whether reconstruction actually uses OVT/register content:
  - RAE access ablation: baseline / OVT-only / register-only / self-only.
  - Input zero ablation: zero initial OVT embeddings and/or register embeddings.
  - OVT swap: replace one object's final OVT hidden with another image's OVT.
  - RAE query source mass: Q/K softmax with the PGOT attention bias over
    OVT/register/RAE-self/image/text sources.
  - Gradient comparison: L_recon vs V13 outside losses w.r.t. input tokens.
"""

import argparse
import gc
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from pgot.eval.pgot_inference import (
    generate_siglip_latent,
    ovt_swap_inference,
    pgot_forward_eval,
)
from pgot.eval.run_eval import decode_to_image, denormalize_images, load_rae_decoder
from pgot.eval.visualize_ovt_overlays import _load_model
from pgot.model.pgot_utils import (
    compute_mask_bce_loss,
    compute_per_ovt_mask_logits,
    compute_register_foreground_suppression_loss,
    compute_sigmoid_outside_bce_loss,
    gather_ovt_hidden_states,
)
from pgot.train.pgot_dataset import PGOTDataCollator, Pix2CapPGOTDataset

log = logging.getLogger('pgot.v13_causal_paths')


def parse_model(spec: str) -> tuple[str, str]:
    label, path = spec.split('|', 1)
    return label, path


def selected_layers(model, spec: str) -> list[int]:
    if hasattr(model, '_resolve_llm_qk_outside_layers'):
        return model._resolve_llm_qk_outside_layers(spec)
    n = len(model.model.layers)
    spec = str(spec)
    if spec.startswith('last'):
        count = int(spec[4:] or '1')
        return list(range(max(0, n - count), n))
    return [int(x) for x in spec.split(',') if x.strip()]


def fixed_recon_loss(model, rae_hidden, target, seed: int):
    devices = [rae_hidden.device.index] if rae_hidden.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return model._captionslot_compute_diffusion_loss(
            hidden=rae_hidden,
            target_features=target,
        )


def _source_masks(positions, ovt_abs_positions, ovt_valid_mask, device):
    total = int(positions['total_len'])
    masks = {}
    for name in ('image', 'ovt', 'register', 'rae_self', 'null_bg', 'caption_non_ovt', 'other'):
        masks[name] = torch.zeros(total, dtype=torch.bool, device=device)
    masks['image'][positions['img_s']:positions['img_e']] = True
    masks['register'][positions['reg_s']:positions['reg_e']] = True
    masks['rae_self'][positions['rae_s']:positions['rae_e']] = True
    masks['null_bg'][positions['null_bg_s']:positions['null_bg_e']] = True
    masks['caption_non_ovt'][positions['cap_s']:positions['cap_e']] = True
    valid_pos = ovt_abs_positions[0][ovt_valid_mask[0]]
    if valid_pos.numel() > 0:
        masks['ovt'][valid_pos] = True
        masks['caption_non_ovt'][valid_pos] = False
    used = torch.zeros(total, dtype=torch.bool, device=device)
    for name in ('image', 'ovt', 'register', 'rae_self', 'null_bg', 'caption_non_ovt'):
        used |= masks[name]
    masks['other'] = ~used
    return masks


def rae_qk_source_mass(model, hidden_states, attn_bias, positions, ovt_abs_positions, ovt_valid_mask, layer_ids):
    """Approximate source mass using the layer Q/K projections plus PGOT attention bias."""
    records = []
    device = attn_bias.device
    masks = _source_masks(positions, ovt_abs_positions, ovt_valid_mask, device)
    rae_s, rae_e = positions['rae_s'], positions['rae_e']
    for layer_idx in layer_ids:
        if layer_idx >= len(hidden_states) - 1:
            continue
        block = model.model.layers[layer_idx]
        attn = block.self_attn
        x = block.input_layernorm(hidden_states[layer_idx])
        rae = x[:, rae_s:rae_e]
        q_proj = attn.q_proj(rae)
        k_proj = attn.k_proj(x)
        n_heads = int(getattr(attn, 'num_heads', model.config.num_attention_heads))
        n_kv = int(getattr(attn, 'num_key_value_heads', model.config.num_key_value_heads))
        head_dim = int(getattr(attn, 'head_dim', q_proj.shape[-1] // n_heads))
        q = q_proj.reshape(q_proj.shape[0], q_proj.shape[1], n_heads, head_dim)
        k = k_proj.reshape(k_proj.shape[0], k_proj.shape[1], n_kv, head_dim)
        if n_kv != n_heads:
            k = k.repeat_interleave(max(n_heads // n_kv, 1), dim=2)[:, :, :n_heads]
        scores = torch.einsum('bqhd,blhd->bqhl', q.float(), k.float()) / math.sqrt(head_dim)
        bias = attn_bias[:, :, rae_s:rae_e, :].permute(0, 2, 1, 3).float()
        probs = (scores + bias).softmax(dim=-1)
        rec = {'layer': int(layer_idx)}
        for name, mask in masks.items():
            rec[name + '_mass'] = float(probs[..., mask].sum(dim=-1).mean().detach()) if mask.any() else 0.0
        allowed = torch.isfinite(attn_bias[:, :, rae_s:rae_e, :]).float()
        rec['allowed_token_fraction'] = float(allowed.mean().detach())
        records.append(rec)
    return records


def _grad_group_stats(grad, positions, ovt_abs_positions, ovt_valid_mask):
    valid_pos = ovt_abs_positions[0][ovt_valid_mask[0]]
    groups = {
        'image': grad[:, positions['img_s']:positions['img_e']],
        'ovt': grad[:, valid_pos] if valid_pos.numel() else grad[:, :0],
        'register': grad[:, positions['reg_s']:positions['reg_e']],
        'rae_query': grad[:, positions['rae_s']:positions['rae_e']],
    }
    out = {}
    for name, values in groups.items():
        if values.numel() == 0:
            out[name] = {'l2': 0.0, 'token_norm_mean': 0.0, 'token_norm_max': 0.0}
            continue
        token_norm = values.float().norm(dim=-1)
        out[name] = {
            'l2': float(values.float().flatten(1).norm(dim=-1).mean().detach()),
            'token_norm_mean': float(token_norm.mean().detach()),
            'token_norm_max': float(token_norm.max().detach()),
        }
    return out


def _group_cosine(grad_a, grad_b, positions, ovt_abs_positions, ovt_valid_mask):
    valid_pos = ovt_abs_positions[0][ovt_valid_mask[0]]
    groups = {
        'all_tokens': (grad_a, grad_b),
        'image': (grad_a[:, positions['img_s']:positions['img_e']], grad_b[:, positions['img_s']:positions['img_e']]),
        'ovt': ((grad_a[:, valid_pos], grad_b[:, valid_pos]) if valid_pos.numel() else (grad_a[:, :0], grad_b[:, :0])),
        'register': (grad_a[:, positions['reg_s']:positions['reg_e']], grad_b[:, positions['reg_s']:positions['reg_e']]),
        'rae_query': (grad_a[:, positions['rae_s']:positions['rae_e']], grad_b[:, positions['rae_s']:positions['rae_e']]),
    }
    out = {}
    for name, (a, b) in groups.items():
        if a.numel() == 0 or b.numel() == 0:
            out[name] = None
            continue
        af = a.float().flatten()
        bf = b.float().flatten()
        denom = af.norm() * bf.norm()
        out[name] = None if float(denom.detach()) == 0.0 else float((af @ bf / denom).detach())
    return out


def _resolve_mask_mode(model, mask_mode: str) -> str:
    if mask_mode != 'auto':
        return mask_mode
    outside_w = float(getattr(model.config, 'pgot_mask_sigmoid_outside_weight', 0.0))
    reg_w = float(getattr(model.config, 'pgot_register_foreground_suppression_weight', 0.0))
    return 'v13_outside' if (outside_w > 0.0 or reg_w > 0.0) else 'v3_bce'


def _forward_loss_and_input_grad(model, baseline, batch, seed: int, loss_kind: str, mask_mode: str):
    inputs = baseline['inputs_embeds'].detach().clone().requires_grad_(True)
    out = model.model(
        inputs_embeds=inputs,
        attention_bias=baseline['attn_bias'],
        use_cache=False,
        return_dict=True,
    )
    pos = baseline['positions']
    hidden = out.last_hidden_state
    img_hidden = hidden[:, pos['img_s']:pos['img_e']]
    rae_hidden = hidden[:, pos['rae_s']:pos['rae_e']]
    ovt_hidden = gather_ovt_hidden_states(
        hidden,
        baseline['ovt_abs_positions'],
        baseline['ovt_valid_mask'],
    )
    temp = float(getattr(model.config, 'pgot_attention_temperature', 1.0))
    use_ln = bool(getattr(model.config, 'pgot_attention_use_layer_norm', True))
    ovt_logits = compute_per_ovt_mask_logits(ovt_hidden, img_hidden, temp, use_ln)
    reg_hidden = hidden[:, pos['reg_s']:pos['reg_e']]
    reg_logits = compute_per_ovt_mask_logits(reg_hidden, img_hidden, temp, use_ln)

    gt_masks = batch['gt_masks_per_ovt'].to(inputs.device).float()
    ovt_valid = batch['ovt_valid_mask'].to(inputs.device, dtype=torch.bool)
    target = baseline['gt_siglip']

    recon = fixed_recon_loss(model, rae_hidden, target, seed)
    full_bce = compute_mask_bce_loss(ovt_logits, gt_masks, ovt_valid)
    outside = compute_sigmoid_outside_bce_loss(ovt_logits, gt_masks, ovt_valid)['loss']
    reg_sup = compute_register_foreground_suppression_loss(reg_logits, gt_masks, ovt_valid)['loss']
    outside_w = float(getattr(model.config, 'pgot_mask_sigmoid_outside_weight', 0.0))
    reg_w = float(getattr(model.config, 'pgot_register_foreground_suppression_weight', 0.0))
    legacy_bce_w = float(getattr(model.config, 'pgot_mask_loss_weight', 1.0))
    resolved_mode = _resolve_mask_mode(model, mask_mode)
    selected_mask = (
        legacy_bce_w * full_bce
        if resolved_mode == 'v3_bce'
        else outside_w * outside + reg_w * reg_sup
    )
    losses = {
        'recon': recon,
        'full_bce': full_bce,
        'sigmoid_outside': outside,
        'register_suppression': reg_sup,
        'weighted_mask_total': outside_w * outside + reg_w * reg_sup,
        'selected_mask': selected_mask,
    }
    loss = losses[loss_kind]
    grad = torch.autograd.grad(loss, inputs, retain_graph=False, allow_unused=False)[0].detach()
    scalar_losses = {k: float(v.detach()) for k, v in losses.items()}
    scalar_losses['mask_mode'] = resolved_mode
    scalar_losses['legacy_bce_weight'] = legacy_bce_w
    scalar_losses['sigmoid_outside_weight'] = outside_w
    scalar_losses['register_suppression_weight'] = reg_w
    return scalar_losses, grad


def gradient_comparison(model, baseline, batch, seed: int, mask_mode: str):
    old_flags = [p.requires_grad for p in model.parameters()]
    params = list(model.parameters())
    try:
        for p in params:
            p.requires_grad_(False)
        loss_values, grad_recon = _forward_loss_and_input_grad(model, baseline, batch, seed, 'recon', mask_mode)
        _, grad_full_bce = _forward_loss_and_input_grad(model, baseline, batch, seed, 'full_bce', mask_mode)
        _, grad_outside = _forward_loss_and_input_grad(model, baseline, batch, seed, 'sigmoid_outside', mask_mode)
        _, grad_reg = _forward_loss_and_input_grad(model, baseline, batch, seed, 'register_suppression', mask_mode)
        _, grad_mask = _forward_loss_and_input_grad(model, baseline, batch, seed, 'selected_mask', mask_mode)
    finally:
        for p, flag in zip(params, old_flags):
            p.requires_grad_(flag)

    pos = baseline['positions']
    ovt_pos = baseline['ovt_abs_positions']
    ovt_valid = baseline['ovt_valid_mask']
    return {
        'loss_values': loss_values,
        'grad_norms': {
            'recon': _grad_group_stats(grad_recon, pos, ovt_pos, ovt_valid),
            'full_bce': _grad_group_stats(grad_full_bce, pos, ovt_pos, ovt_valid),
            'sigmoid_outside': _grad_group_stats(grad_outside, pos, ovt_pos, ovt_valid),
            'register_suppression': _grad_group_stats(grad_reg, pos, ovt_pos, ovt_valid),
            'selected_mask': _grad_group_stats(grad_mask, pos, ovt_pos, ovt_valid),
        },
        'cosine_recon_vs_full_bce': _group_cosine(grad_recon, grad_full_bce, pos, ovt_pos, ovt_valid),
        'cosine_recon_vs_sigmoid_outside': _group_cosine(grad_recon, grad_outside, pos, ovt_pos, ovt_valid),
        'cosine_recon_vs_selected_mask': _group_cosine(grad_recon, grad_mask, pos, ovt_pos, ovt_valid),
    }


def add_label(tile, label: str):
    tile = tile.convert('RGB')
    canvas = Image.new('RGB', (tile.width, tile.height + 28), 'white')
    canvas.paste(tile, (0, 28))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 14)
    except Exception:
        font = None
    draw.text((6, 6), label, fill=(0, 0, 0), font=font)
    return canvas


def decode_variant_grid(model, decoder, variant_rae, target_image, target_proc, output_path, seed, guidance):
    mean = torch.tensor(target_proc.image_mean)
    std = torch.tensor(target_proc.image_std)
    source = denormalize_images(target_image.float(), mean, std)[0]
    tiles = [add_label(Image.fromarray((source.permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)), 'source')]
    for name, rae_hidden in variant_rae.items():
        torch.manual_seed(seed)
        latent = generate_siglip_latent(model, rae_hidden, guidance_level=guidance)
        img = decode_to_image(decoder, latent, rae_hidden.device)[0].cpu().clamp(0, 1)
        pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        tiles.append(add_label(pil, name))
    max_w = max(t.width for t in tiles)
    max_h = max(t.height for t in tiles)
    resized = []
    for tile in tiles:
        if tile.size != (max_w, max_h):
            tile = tile.resize((max_w, max_h), Image.BILINEAR)
        resized.append(tile)
    cols = min(3, len(resized))
    rows = math.ceil(len(resized) / cols)
    grid = Image.new('RGB', (cols * max_w, rows * max_h), 'white')
    for i, tile in enumerate(resized):
        grid.paste(tile, ((i % cols) * max_w, (i // cols) * max_h))
    grid.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='append', required=True, help='label|checkpoint')
    parser.add_argument('--sample_indices', default='0,1020,1407')
    parser.add_argument('--swap_sample_index', type=int, default=None)
    parser.add_argument('--swap_target_object', type=int, default=0)
    parser.add_argument('--swap_source_object', type=int, default=0)
    parser.add_argument('--val_jsonl', default='/home/jovyan/PGOT/data/pgot_val.jsonl')
    parser.add_argument('--image_preprocess_mode', default='default', choices=['default', 'coda_center_crop'])
    parser.add_argument('--coda_crop_size', type=int, default=512)
    parser.add_argument('--max_caption_tokens', type=int, default=2048)
    parser.add_argument('--n_ovt_per_object', type=int, default=2)
    parser.add_argument('--output_dir', default='/home/jovyan/PGOT/outputs/v13_causal_path_diagnostics')
    parser.add_argument('--layers', default='last4')
    parser.add_argument('--compute_gradients', action='store_true')
    parser.add_argument(
        '--mask_gradient_loss',
        default='auto',
        choices=['auto', 'v3_bce', 'v13_outside'],
        help='Mask loss used for selected-mask gradient comparison.',
    )
    parser.add_argument('--decode_recon', action='store_true')
    parser.add_argument('--diffusion_inference_steps', type=int, default=15)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dtype', choices=['fp32', 'bf16'], default='fp32')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s :: %(message)s')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = [int(x) for x in args.sample_indices.split(',') if x.strip()]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32 if args.dtype == 'fp32' else torch.bfloat16
    with open(args.val_jsonl) as f:
        raw_samples = [json.loads(line) for line in f]

    all_results = {}
    for label, model_path in map(parse_model, args.model):
        log.info('Loading %s: %s', label, model_path)
        model, tokenizer = _load_model(model_path, dtype, device)
        vt_list = model.get_vision_tower_aux_list()
        image_proc = vt_list[0].image_processor
        target_proc = vt_list[1].image_processor if len(vt_list) > 1 else image_proc
        dataset = Pix2CapPGOTDataset(
            jsonl_path=args.val_jsonl,
            tokenizer=tokenizer,
            image_processor=image_proc,
            target_image_processor=target_proc,
            grid_size=32,
            max_caption_tokens=args.max_caption_tokens,
            n_ovt_per_object=args.n_ovt_per_object,
            max_objects=50,
            panoptic_categories_json='/home/jovyan/data/coco/annotations/panoptic_val2017.json',
            image_preprocess_mode=args.image_preprocess_mode,
            coda_crop_size=args.coda_crop_size,
        )
        collator = PGOTDataCollator(pad_token_id=tokenizer.pad_token_id)
        decoder = load_rae_decoder(model, device, dtype) if args.decode_recon else None
        if args.decode_recon and args.diffusion_inference_steps != 50:
            from scale_rae.model.diffusion_loss.diffusion import create_diffusion
            inf = model.diff_head.inference_flow
            model.diff_head.inference_flow = create_diffusion(
                str(args.diffusion_inference_steps),
                noise_schedule='linear', use_kl=False, sigma_small=False,
                predict_xstart=False, learn_sigma=False, rescale_learned_sigmas=False,
                diffusion_steps=int(getattr(inf, 'diffusion_steps', 1000)),
                input_base_dimension_ratio=float(getattr(inf, 'size_ratio', 1.0)),
                diffusion_type='rf', use_loss_weighting=False,
            )
        layer_ids = selected_layers(model, args.layers)
        model_dir = output_dir / label
        model_dir.mkdir(exist_ok=True)
        model_results = []

        for sample_idx in sample_indices:
            batch = collator([dataset[sample_idx]])
            kwargs = dict(
                images=batch['images'],
                target_images=batch['target_images'],
                caption_input_ids=batch['caption_input_ids'],
                caption_attention_mask=batch['caption_attention_mask'],
                ovt_positions_in_caption=batch['ovt_positions_in_caption'],
                ovt_valid_mask=batch['ovt_valid_mask'],
            )
            variant_specs = {
                'baseline': {},
                'ovt_only_access': {'rae_access_mode': 'ovt_only'},
                'register_only_access': {'rae_access_mode': 'register_only'},
                'self_only_access': {'rae_access_mode': 'self_only'},
                'zero_ovt_inputs': {'zero_ovt_inputs': True},
                'zero_register_inputs': {'zero_register_inputs': True},
                'zero_ovt_and_register_inputs': {'zero_ovt_inputs': True, 'zero_register_inputs': True},
            }
            variants = {}
            for name, extra in variant_specs.items():
                variants[name] = pgot_forward_eval(
                    model,
                    **kwargs,
                    return_hidden_states=(name == 'baseline'),
                    **extra,
                )
            base = variants['baseline']
            losses = {}
            shifts = {}
            for name, out in variants.items():
                losses[name] = float(fixed_recon_loss(model, out['rae_hidden'], out['gt_siglip'], args.seed))
                shifts[name] = {
                    'cosine_to_baseline': float(F.cosine_similarity(
                        base['rae_hidden'].float().flatten(1),
                        out['rae_hidden'].float().flatten(1),
                    ).mean().detach()),
                    'relative_l2_to_baseline': float(
                        (out['rae_hidden'].float() - base['rae_hidden'].float()).norm()
                        / base['rae_hidden'].float().norm().clamp_min(1e-8)
                    ),
                    'delta_recon_loss': losses[name] - losses['baseline'] if 'baseline' in losses else 0.0,
                }

            record = {
                'sample_index': sample_idx,
                'image_id': raw_samples[sample_idx]['image_id'],
                'n_objects': int(batch['ovt_valid_mask'][0].sum().item() // args.n_ovt_per_object),
                'recon_loss': losses,
                'rae_hidden_shift': shifts,
                'rae_qk_source_mass_with_bias': rae_qk_source_mass(
                    model,
                    base['hidden_states'],
                    base['attn_bias'],
                    base['positions'],
                    base['ovt_abs_positions'],
                    base['ovt_valid_mask'],
                    layer_ids,
                ),
            }

            decode_rae = {name: out['rae_hidden'] for name, out in variants.items()}
            swap_source_idx = args.swap_sample_index
            if swap_source_idx is not None:
                if swap_source_idx == sample_idx:
                    swap_source_idx = next((x for x in sample_indices if x != sample_idx), None)
                if swap_source_idx is not None:
                    swap_batch = collator([dataset[swap_source_idx]])
                    swap_out = pgot_forward_eval(
                        model,
                        images=swap_batch['images'],
                        target_images=swap_batch['target_images'],
                        caption_input_ids=swap_batch['caption_input_ids'],
                        caption_attention_mask=swap_batch['caption_attention_mask'],
                        ovt_positions_in_caption=swap_batch['ovt_positions_in_caption'],
                        ovt_valid_mask=swap_batch['ovt_valid_mask'],
                    )
                    swapped_rae, _ = ovt_swap_inference(
                        model,
                        base,
                        swap_out,
                        [(args.swap_target_object, args.swap_source_object)],
                        args.n_ovt_per_object,
                    )
                    record['object_swap'] = {
                        'source_sample_index': int(swap_source_idx),
                        'target_object_index': int(args.swap_target_object),
                        'source_object_index': int(args.swap_source_object),
                        'recon_loss': float(fixed_recon_loss(model, swapped_rae, base['gt_siglip'], args.seed)),
                        'delta_recon_loss': float(fixed_recon_loss(model, swapped_rae, base['gt_siglip'], args.seed)) - losses['baseline'],
                        'relative_l2_to_baseline': float(
                            (swapped_rae.float() - base['rae_hidden'].float()).norm()
                            / base['rae_hidden'].float().norm().clamp_min(1e-8)
                        ),
                        'cosine_to_baseline': float(F.cosine_similarity(
                            base['rae_hidden'].float().flatten(1),
                            swapped_rae.float().flatten(1),
                        ).mean().detach()),
                    }
                    decode_rae['object_swap'] = swapped_rae

            if args.compute_gradients:
                record['gradient_comparison'] = gradient_comparison(
                    model, base, batch, args.seed, args.mask_gradient_loss
                )

            if args.decode_recon:
                path = model_dir / f'sample{sample_idx}_causal_recon_grid.png'
                decode_variant_grid(
                    model,
                    decoder,
                    decode_rae,
                    batch['target_images'].to(device),
                    target_proc,
                    path,
                    args.seed,
                    args.guidance_scale,
                )
                record['recon_grid'] = str(path)
                record['recon_grid_order'] = ['source', *decode_rae.keys()]

            model_results.append(record)
            with (model_dir / f'sample{sample_idx}.json').open('w') as f:
                json.dump(record, f, indent=2)
            log.info('%s sample=%s recon_loss=%s', label, sample_idx, losses)

        all_results[label] = model_results
        with (model_dir / 'summary.json').open('w') as f:
            json.dump(model_results, f, indent=2)
        del decoder, dataset, tokenizer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (output_dir / 'summary.json').open('w') as f:
        json.dump(all_results, f, indent=2)
    print(output_dir / 'summary.json')


if __name__ == '__main__':
    main()
