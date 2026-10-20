"""CPU-only cached-mask audit. TargetHit@1 is NOT full-slot Binding Hits."""

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
from pycocotools.coco import COCO


POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def target_hit(ious, target_index):
    """Positive unique best GT IoU; empty/zero-overlap predictions always miss."""
    value = float(ious[target_index])
    others = np.delete(ious, target_index)
    best_other = float(others.max()) if len(others) else -1.0
    return value > 0 and value > best_other + 1e-12


def same_category_target_hit(ious, target_index, category_ids):
    """Unique best IoU among GT instances sharing the target's category."""
    target_category = category_ids[target_index]
    competitors = [
        index for index, category_id in enumerate(category_ids)
        if index != target_index and category_id == target_category
    ]
    if not competitors:
        return None
    value = float(ious[target_index])
    best_other = float(np.asarray(ious)[competitors].max())
    return value > 0 and value > best_other + 1e-12


def audit_image(coco, root, records, models):
    iid = records[0]['image_id']
    annotations = sorted((coco.anns[i] for i in coco.getAnnIds(imgIds=[iid])
                          if not coco.anns[i].get('iscrowd', 0)), key=lambda a: a['id'])
    masks = [coco.annToMask(a).astype(bool) for a in annotations]
    retained = [(a, m) for a, m in zip(annotations, masks) if m.any()]
    ids = [a['id'] for a, _ in retained]
    category_ids = [int(a['category_id']) for a, _ in retained]
    masks = np.stack([m for _, m in retained])
    areas = masks.sum(axis=(1, 2), dtype=np.int64)
    packed = np.packbits(masks.reshape(len(masks), -1), axis=1)
    output = []
    for record in records:
        target = ids.index(record['annotation_id'])
        result = dict(record)
        result['num_gt_competitors'] = len(ids)
        target_category_id = category_ids[target]
        result['target_category_id'] = target_category_id
        result['same_category_gt_competitors'] = sum(
            index != target and category_id == target_category_id
            for index, category_id in enumerate(category_ids)
        )
        result['same_category_multiple'] = result['same_category_gt_competitors'] > 0
        for model, directory in models:
            folder = root / directory / record['split'] / str(record['ref_id']) / str(record['sent_id'])
            with np.load(folder / 'prediction.npz') as saved:
                mask = saved['mask'].astype(bool)
            if mask.shape != masks.shape[1:]:
                raise ValueError(f'Incorrect cached mask shape: {folder}')
            pred_area = int(mask.sum())
            pred_bits = np.packbits(mask.reshape(-1))
            intersection = POPCOUNT[packed & pred_bits].sum(axis=1, dtype=np.int64)
            union = pred_area + areas - intersection
            ious = np.divide(intersection, union, out=np.zeros(len(ids)), where=union > 0)
            previous = record[model]
            if int(intersection[target]) != previous['intersection'] or int(union[target]) != previous['union']:
                raise ValueError(f'Cached score verification failed: {folder}')
            others = np.delete(ious, target)
            same_category_hit = same_category_target_hit(ious, target, category_ids)
            result[model] = dict(previous, target_hit_at_1=bool(target_hit(ious, target)),
                same_category_target_hit_at_1=same_category_hit,
                positive_top_tie=bool(ious[target] > 0 and len(others) and
                                     abs(ious[target] - others.max()) <= 1e-12))
        output.append(result)
    return output


def aggregate(records, models):
    result = dict(expressions=len(records), images=len({r['image_id'] for r in records}),
                  references=len({r['ref_id'] for r in records}))
    for model, _ in models:
        values = [r[model] for r in records]
        same_category_values = [
            value['same_category_target_hit_at_1']
            for value in values
            if value.get('same_category_target_hit_at_1') is not None
        ]
        union = sum(v['union'] for v in values)
        result[model] = dict(
            mIoU=100 * float(np.mean([v['iou'] for v in values])),
            oIoU=100 * sum(v['intersection'] for v in values) / union if union else 0.,
            P50=100 * float(np.mean([v['iou'] >= .5 for v in values])),
            target_hit_at_1=100 * float(np.mean([v['target_hit_at_1'] for v in values])),
            same_category_target_hit_at_1=(
                100 * float(np.mean(same_category_values))
                if same_category_values else float('nan')
            ),
            same_category_expressions=len(same_category_values),
            empty_masks=sum(v['empty_mask'] for v in values),
            positive_top_ties=sum(v['positive_top_tie'] for v in values))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/jovyan/PGOT/outputs/eval_deor_ctrlo_refcoco_direct'))
    parser.add_argument('--output', type=Path, default=Path('/home/jovyan/PGOT/outputs/analysis_refcoco_cached_grounding'))
    parser.add_argument('--train', type=Path, default=Path('/home/jovyan/PGOT/data/pgot_pix2cap_thing_train.jsonl'))
    parser.add_argument('--annotations', default='/home/jovyan/data/coco/refcoco/instances.json')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument(
        '--models', nargs='+', choices=('deor', 'ctrlo'), default=None,
        help='Models to audit. By default, detect available ours/ and ctrlo/ caches.',
    )
    args = parser.parse_args()
    started = time.monotonic()
    model_directories = {'deor': 'ours', 'ctrlo': 'ctrlo'}
    selected_models = args.models or [
        model for model, directory in model_directories.items()
        if (args.root / directory).is_dir()
    ]
    if not selected_models:
        raise FileNotFoundError(f'No cached prediction directories found under {args.root}')
    models = [(model, model_directories[model]) for model in selected_models]
    train_ids = {int(json.loads(line)['image_id']) for line in args.train.open()}
    records = [json.loads(line) for line in (args.root / 'per_expression.jsonl').open()]
    groups = defaultdict(list)
    for r in records:
        groups[r['image_id']].append(r)
    coco = COCO(args.annotations)
    audited = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(audit_image, coco, args.root, rs, models)
                   for rs in groups.values()]
        for index, future in enumerate(as_completed(futures), 1):
            audited.extend(future.result())
            if index % 100 == 0:
                print(f'{index}/{len(groups)} images; {len(audited)} expressions; '
                      f'{time.monotonic() - started:.1f}s', flush=True)
    audited.sort(key=lambda r: (r['split'], r['ref_id'], r['sent_id']))
    summary = dict(
        protocol='RefCOCO one supplied human expression per pass; cached query mask, no new inference',
        target_hit_definition='Positive strictly highest query-mask IoU against all nonempty noncrowd '
            'GT instance masks in the image; ties within 1e-12 and zero overlap count as misses. '
            'NOT original CTRL-O all-slot Hungarian Binding Hits.',
        same_category_definition='Positive strictly highest query-mask IoU among the target and all '
            'other nonempty noncrowd GT instances with the same COCO category. Expressions without '
            'a same-category competitor are excluded from this metric.',
        subset_definition='Exclude image IDs in the actual DeOR Pix2Cap training JSONL; '
            'does not audit earlier pretraining or CTRL-O training exposure.',
        source=str(args.root), train=str(args.train), train_images=len(train_ids),
        aggregation='expression-weighted; no expressions dropped; percentages',
        evaluated_models=selected_models, subsets={})
    summary['source_fingerprint'] = json.load((args.root / 'manifest.json').open())['fingerprint']
    for name, predicate in [('full', lambda r: True),
                            ('same_category', lambda r: r['same_category_multiple']),
                            ('nonoverlap', lambda r: r['image_id'] not in train_ids),
                            ('overlap', lambda r: r['image_id'] in train_ids),
                            ('same_category_nonoverlap', lambda r: (
                                r['same_category_multiple'] and r['image_id'] not in train_ids))]:
        selected = [r for r in audited if predicate(r)]
        summary['subsets'][name] = {split: aggregate(
            [r for r in selected if r['split'] == split], models)
                                   for split in ('val', 'testA', 'testB')}
        summary['subsets'][name]['all'] = aggregate(selected, models)
    summary['elapsed_seconds'] = time.monotonic() - started
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'summary.json').open('w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (args.output / 'per_expression.jsonl').open('w') as f:
        for record in audited:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    with (args.output / 'comparison.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subset', 'split', 'images', 'expressions', 'method',
                         'mIoU', 'oIoU', 'P50', 'TargetHit@1', 'Same-category TargetHit@1'])
        for subset, splits in summary['subsets'].items():
            for split, stats in splits.items():
                for model, _ in models:
                    name = {'deor': 'DeOR', 'ctrlo': 'CTRL-O'}[model]
                    v = stats[model]
                    writer.writerow([subset, split, stats['images'], stats['expressions'], name,
                                     v['mIoU'], v['oIoU'], v['P50'], v['target_hit_at_1'],
                                     v['same_category_target_hit_at_1']])
    with (args.output / 'nonoverlap_table.md').open('w') as f:
        f.write('Subset excludes image IDs in DeOR actual Pix2Cap training JSONL. '
                'Percentages; expression-weighted. TargetHit@1 is NOT original Binding Hits.\n\n')
        f.write('| Split | Images | Expressions | Method | mIoU | oIoU | P@0.5 | TargetHit@1 |\n')
        f.write('|---|---:|---:|---|---:|---:|---:|---:|\n')
        for split, stats in summary['subsets']['nonoverlap'].items():
            for model, _ in models:
                name = {'deor': 'DeOR', 'ctrlo': 'CTRL-O'}[model]
                v = stats[model]
                f.write(f'| {split} | {stats["images"]} | {stats["expressions"]} | {name} | '
                        f'{v["mIoU"]:.2f} | {v["oIoU"]:.2f} | {v["P50"]:.2f} | '
                        f'{v["target_hit_at_1"]:.2f} |\n')
    print(json.dumps(summary['subsets']['nonoverlap'], indent=2), flush=True)


if __name__ == '__main__':
    main()
