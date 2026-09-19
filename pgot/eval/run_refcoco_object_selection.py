"""RefCOCO retrieval of image-only AR objects; never query-condition DeOR.

Models run in separate processes. Original-resolution masks are scored only
after prediction; the external selector sees only expressions and generated text.
"""
import argparse
from collections import OrderedDict
import csv
import json
import pickle
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from pgot.eval.natural_object_selection import (
    candidates_from_caption, digest, fixed_correspondence, query_kinds,
)
from pgot.eval.run_natural_object_selection import (
    checkpoint_signature, file_digest, jsonl, select as text_select, write_json,
)
from pgot.eval.run_refcoco_eval import (
    annotation_paths, completed, file_hash, load_full_image, make_queries,
    resolve_image, score_mask, selected_references,
)

VERSION = 'refcoco-cached-ar-object-retrieval-v1'
CONDITIONS = ('category', 'description', 'oracle_id')


def configuration(args):
    keys = ('model_path', 'data_root', 'image_root', 'dataset', 'split_by', 'splits',
            'max_refs', 'max_samples', 'eval_size', 'dtype', 'seed', 'max_objects',
            'max_caption_tokens', 'ar_max_new_tokens', 'cache_bundles', 'presence_iou',
            'with_ctrlo', 'ctrlo_repo', 'checkpoint_dir', 'llm2vec_base',
            'llm2vec_adapter', 'embedding_max_length', 'selector_model',
            'selector_max_new_tokens', 'selector_max_input_tokens')
    config = {k: getattr(args, k) for k in keys}
    config.update(version=VERSION, checkpoint_signature=checkpoint_signature(args.model_path),
        annotation_hashes=[file_digest(p) for p in annotation_paths(args)],
        train_annotations_sha256=file_digest(args.train_annotations),
        code_hashes={p.name: file_digest(p) for p in (
            Path(__file__), Path(__file__).with_name('natural_object_selection.py'),
            Path(__file__).with_name('run_natural_object_selection.py'),
            Path(__file__).with_name('run_eval.py'),
            Path(__file__).with_name('run_refcoco_eval.py'))})
    if args.with_ctrlo:
        config['ctrlo_checkpoint_sha256'] = file_digest(Path(args.checkpoint_dir)/'pretrained_model.ckpt')
        config['ctrlo_config_sha256'] = file_digest(Path(args.checkpoint_dir)/'config.yaml')
    return config


def manifest(args):
    root = Path(args.output_dir)
    value = json.loads((root/'manifest.json').read_text())
    if value['config'] != configuration(args):
        raise ValueError('Configuration/source/checkpoint changed; use a fresh OUTPUT_DIR')
    return root, value


def prepare(args):
    root = Path(args.output_dir)
    config = configuration(args)
    if (root/'manifest.json').exists():
        manifest(args)
        print('Identical retrieval manifest reused.', flush=True)
        return
    refs_path, instances_path = annotation_paths(args)
    # Only locally trusted official REFER pickle annotations are supported.
    with refs_path.open('rb') as stream:
        refs = pickle.load(stream, encoding='latin1')
    instances = json.loads(instances_path.read_text())
    images = {int(i['id']): i for i in instances['images']}
    annotations = {int(a['id']): a for a in instances['annotations']}
    queries = make_queries(selected_references(refs, args.splits, args.max_refs), args.max_samples)
    samples = {}
    for query in queries:
        iid = query['image_id']
        if int(annotations[query['annotation_id']]['image_id']) != iid:
            raise ValueError('RefCOCO reference/annotation image mismatch')
        if iid not in samples:
            info = images[iid]
            path = resolve_image(args.image_root, info)
            with Image.open(path) as image:
                if image.size != (int(info['width']), int(info['height'])):
                    raise ValueError('Image dimensions disagree with annotation')
                image.verify()
            samples[iid] = dict(image_id=iid, image_path=str(path),
                width=int(info['width']), height=int(info['height']), image_sha256=file_hash(path))
        query.update(samples[iid], dataset=args.dataset, request=query['shared_text'],
                     query_types=query_kinds(query['shared_text']))
        # The structural direct-conditioning caption is baseline compatibility
        # data only. It is NEVER fed into our DeOR extraction.
    train_ids = {int(r['image_id']) for r in jsonl(args.train_annotations)}
    overlap = sorted(train_ids & samples.keys())
    protocol = dict(task='natural_language_retrieval_of_cached_object_representations',
        query_source='original RefCOCO raw expressions, unchanged; every selected sentence',
        deor='image-only AR discovery + own-caption replay ONCE per image; no user expression input',
        selector_image_input=False, selector_gt_input=False, selector_region_coordinates=False,
        comparison='same shuffled generated IDs/category names; add full generated descriptions only',
        image='entire frame resized to 512 square, no crop; probabilities interpolated to original H,W before argmax',
        mask='native valid owners compete with summed void probability; immutable per-image regions',
        scoring='actual chosen mask against exact referred annotation; NO coverage override or GT repair',
        invalid_and_missing='null/invalid/empty/unbound candidates give empty mask, retain every expression',
        oracle='fixed class-compatible one-to-one IoU correspondence diagnostic ONLY; not a deployable method',
        presence_iou=args.presence_iou, inference_gt_inputs='none',
        full_split_evaluated=not (args.max_refs or args.max_samples),
        train_image_overlap_count=len(overlap), train_image_overlap_ids=overlap,
        limits='No additional RefCOCO training. COCO training overlap and unequal mask supervision; NOT unseen-image zero-shot. CTRL-O released VG+COCO checkpoint is not paper Table 3 reproduction.')
    value = dict(config=config, fingerprint=digest(config), protocol=protocol,
                 samples=list(samples.values()), queries=queries)
    write_json(root/'manifest.json', value)
    if args.with_ctrlo:
        baseline_config = {k: config[k] for k in ('model_path', 'data_root', 'image_root', 'dataset',
            'split_by', 'splits', 'max_refs', 'max_samples', 'eval_size', 'dtype', 'seed',
            'max_objects', 'max_caption_tokens', 'ctrlo_repo', 'checkpoint_dir',
            'llm2vec_base', 'llm2vec_adapter', 'embedding_max_length')}
        baseline_config.update(text_mode='description',
            annotation_hashes=config['annotation_hashes'],
            ctrlo_checkpoint_sha256=config['ctrlo_checkpoint_sha256'],
            ctrlo_config_sha256=config['ctrlo_config_sha256'])
        write_json(root/'baseline/manifest.json', dict(config=baseline_config,
            fingerprint=digest(baseline_config), samples=queries, protocol=protocol))
    print(f'Prepared {len(samples)} images / {len(queries)} expressions; train overlap={len(overlap)}.', flush=True)


def owner_prediction(object_probs, void_probs, valid, candidates, height, width):
    """Same native competition/thing ranks as build_pred_mask_ovt_owner_eval."""
    import torch
    import torch.nn.functional as F
    if object_probs.ndim != 3 or object_probs.shape[0] != 1:
        raise ValueError('Invalid object probability shape')
    k, patches = object_probs.shape[1:]
    grid = round(patches**.5)
    if grid*grid != patches or void_probs is None or void_probs.shape[1] < 1:
        raise ValueError('Invalid native ownership/void grid')
    maps = torch.cat([object_probs.float(), void_probs.float().sum(1, keepdim=True)], 1)
    if not torch.isfinite(maps).all():
        raise ValueError('Nonfinite native ownership probabilities')
    valid = valid[:, :k].to(maps.device)
    allowed = torch.cat([valid, torch.ones((1, 1), device=maps.device, dtype=torch.bool)], 1)
    up = F.interpolate(maps.reshape(1,k+1,grid,grid), (height,width), mode='bilinear', align_corners=False)
    assign = up.masked_fill(~allowed[:, :, None, None], torch.finfo(up.dtype).min).argmax(1)[0]
    prediction = torch.zeros((height,width), dtype=torch.int64, device=maps.device)
    for c in candidates:
        if c['slot_index'] is not None and c['prediction_label'] is not None:
            prediction[assign == c['slot_index']] = c['prediction_label']
    return prediction.cpu().numpy().astype(np.uint16)


def owner_cache(root, m, sample):
    folder = root/'owners'/str(sample['image_id'])
    if not (folder/'metadata.json').exists():
        return None
    meta = json.loads((folder/'metadata.json').read_text())
    if meta['fingerprint'] != m['fingerprint'] or meta['image_sha256'] != sample['image_sha256']:
        raise ValueError('Owner cache provenance mismatch')
    if not (folder/'prediction.npz').is_file():
        return None
    return meta


def extract(args):
    import torch
    from pgot.eval.eval_recon_oracles import load_model_and_tokenizer
    from pgot.eval.pgot_inference import pgot_forward_eval
    from pgot.eval.run_eval import generate_pgot_caption_batch
    root, m = manifest(args)
    pending = [s for s in m['samples'] if owner_cache(root,m,s) is None]
    if pending:
        if not torch.cuda.is_available():
            raise RuntimeError('Real extraction requires requested CUDA GPU')
        torch.set_num_threads(args.cpu_threads)
        torch.manual_seed(args.seed)
        model, tokenizer, device, _ = load_model_and_tokenizer(args)
        keys = set(json.loads((Path(args.model_path)/'model.safetensors.index.json').read_text())['weight_map'])
        params = dict(model.named_parameters(remove_duplicate=False))
        restored = {id(params[k]) for k in keys if k in params}
        missing = [k for k in params if k not in keys and id(params[k]) not in restored]
        unexpected = sorted(keys-set(model.state_dict()))
        if missing or unexpected:
            raise ValueError(f'Checkpoint restore failure: {missing=}, {unexpected=}')
        cfg = model.config
        if (int(cfg.pgot_e11_object_memories_per_owner), int(cfg.pgot_e11_register_memories_per_owner)) != (8,32):
            raise ValueError('Expected original Object8/Register32 checkpoint')
        if bool(getattr(cfg,'pgot_one_shot_direct_rae_query',False)) or bool(getattr(cfg,'pgot_one_shot_memory_contrastive_enable',False)):
            raise ValueError('Expected original MLLM-query/non-contrastive checkpoint')
        for name, token in (('ovt','<ovt>'),('scene_end','<scene_end>'),('thing','<thing>'),('stuff','<stuff>')):
            setattr(model, f'pgot_{name}_token_id', tokenizer.convert_tokens_to_ids(token))
        write_json(root/'deor_model_info.json',dict(checkpoint=args.model_path, missing_parameters=missing,
            unexpected=unexpected, dtype=args.dtype, object_memory=8, register_memory=32,
            query_in_mllm=True, ar_generation='native PGOT image-only greedy decoder',
            gpu=torch.cuda.get_device_name(0)))
        towers = model.get_vision_tower_aux_list()
        for index, sample in enumerate(pending,1):
            image = load_full_image(sample,args.eval_size)
            pixels = towers[0].image_processor.preprocess(image,return_tensors='pt')['pixel_values']
            target = (towers[1] if len(towers)>1 else towers[0]).image_processor.preprocess(image,return_tensors='pt')['pixel_values']
            started = time.perf_counter()
            with torch.inference_mode():
                ar = generate_pgot_caption_batch(model,tokenizer,pixels,
                    max_new_tokens=args.ar_max_new_tokens, max_objects=args.max_objects, n_ovt_per_object=1)
                record = ar['records'][0]
                candidates = candidates_from_caption(record['text'],record['ovt_count'],sample['image_id'],args.seed)
                result = None
                if record['ovt_count']:
                    if ar['caption_input_ids'].shape[1] > args.max_caption_tokens:
                        raise ValueError('Generated caption exceeds replay limit; do not truncate')
                    result = pgot_forward_eval(model,images=pixels,target_images=target,
                        **{k:ar[k] for k in ('caption_input_ids','caption_attention_mask','ovt_positions_in_caption','ovt_valid_mask')})
                    prediction = owner_prediction(result['ovt_object_probs'],result['ovt_void_probs'],
                        ar['ovt_valid_mask'],candidates,sample['height'],sample['width'])
                else:
                    prediction = np.zeros((sample['height'],sample['width']),np.uint16)
            folder = root/'owners'/str(sample['image_id']); folder.mkdir(parents=True,exist_ok=True)
            image.save(folder/'image.png')
            np.savez_compressed(folder/'prediction.tmp.npz',prediction=prediction)
            (folder/'prediction.tmp.npz').replace(folder/'prediction.npz')
            if args.cache_bundles and result is not None:
                bundles = {k:result[k].detach().cpu() for k in ('semantic_slots','visual_memory','ovt_object_valid','raw_rae_hidden')}
                if not all(torch.isfinite(v).all() for v in bundles.values()):
                    raise ValueError('Nonfinite extracted object bundles')
                torch.save(bundles,folder/'owner_bundles.pt')
            write_json(folder/'metadata.json',dict(fingerprint=m['fingerprint'], image_sha256=sample['image_sha256'],
                caption=record['text'],caption_sha256=digest(record['text']),ovt_count=record['ovt_count'],
                candidates=candidates,ar_record=record,seconds=time.perf_counter()-started,
                inference_inputs='image only for AR; image + own AR caption for ownership; no expressions/GT'))
            del result
            print(f'Extract {index}/{len(pending)} image={sample["image_id"]}: owners={record["ovt_count"]}, scene_end={record["scene_end_generated"]}',flush=True)
    else:
        print('All per-image AR/owner caches complete; no DeOR model loaded.',flush=True)
    for sample in m['samples']:
        meta = owner_cache(root,m,sample)
        sample.update({k:meta[k] for k in ('caption','caption_sha256','ovt_count','candidates')})
    write_json(root/'manifest.json',m)


def select(args):
    root, m = manifest(args)
    if not all(owner_cache(root,m,s) and 'candidates' in s for s in m['samples']):
        raise ValueError('Run extraction before selection')
    text_select(args)


def selected_mask(prediction, candidates, selected_id):
    candidate = next((c for c in candidates if c['id']==selected_id),None)
    if candidate is None or candidate['prediction_label'] is None:
        return np.zeros(prediction.shape,bool)
    return prediction == candidate['prediction_label']


def metrics(rows):
    if not rows:
        raise ValueError('Cannot score an empty split')
    present = [r for r in rows if r['target_present']]
    result = dict(num_expressions=len(rows),num_images=len({r['image_id'] for r in rows}),
        mIoU=sum(r['iou'] for r in rows)/len(rows),
        oIoU=sum(r['intersection'] for r in rows)/sum(r['union'] for r in rows),
        success_50=sum(r['iou']>=.5 for r in rows)/len(rows),
        success_75=sum(r['iou']>=.75 for r in rows)/len(rows),
        target_coverage=len(present)/len(rows),
        selection_accuracy_present=sum(r['selection_correct'] for r in present)/len(present) if present else None,
        empty_masks=sum(r['empty_mask'] for r in rows))
    if all(r.get('condition')=='ctrlo' for r in rows):
        # CTRL-O has a fixed expression-bound slot, not DeOR candidate IDs.
        result.update(target_coverage=None,selection_accuracy_present=None)
    return result


def score(args):
    from pycocotools.coco import COCO
    root, m = manifest(args)
    coco = COCO(str(annotation_paths(args)[1]))
    by_image = {s['image_id']:s for s in m['samples']}
    rows, diagnostics, saved = [], OrderedDict(), 0
    conditions = list(CONDITIONS)+(['ctrlo'] if args.with_ctrlo else [])
    infos = {c:json.loads((root/'selections'/c/'selector_info.json').read_text()) for c in ('category','description')}
    a, b = [infos[c]['spec'] for c in ('category','description')]
    if {k:v for k,v in a.items() if k!='condition'} != {k:v for k,v in b.items() if k!='condition'}:
        raise ValueError('Selector conditions used different models/settings')
    baseline = json.loads((root/'baseline/manifest.json').read_text()) if args.with_ctrlo else None
    for query in m['queries']:
        iid = query['image_id']; sample = by_image[iid]
        if iid not in diagnostics:
            if not owner_cache(root,m,sample):
                raise ValueError(f'Missing owner cache: {iid}')
            prediction = np.load(root/'owners'/str(iid)/'prediction.npz')['prediction']
            if prediction.shape != (sample['height'],sample['width']):
                raise ValueError('Owner prediction is not at original resolution')
            anns = sorted(coco.loadAnns(coco.getAnnIds(imgIds=[iid],iscrowd=False)),key=lambda ann:ann['id'])
            objects = [dict(segment_id=int(ann['id']),category=coco.cats[ann['category_id']]['name']) for ann in anns]
            targets = (np.stack([coco.annToMask(ann).astype(bool) for ann in anns]) if anns
                       else np.zeros((0,sample['height'],sample['width']),bool))
            mapping, _ = fixed_correspondence(prediction,sample['candidates'],targets,objects,args.presence_iou)
            diagnostics[iid] = (prediction,mapping)
            if len(diagnostics)>16: diagnostics.popitem(last=False)
        diagnostics.move_to_end(iid)
        prediction, mapping = diagnostics[iid]
        target = coco.annToMask(coco.anns[query['annotation_id']]).astype(bool)
        expected = mapping.get(str(query['annotation_id']))
        panels = []
        for condition in conditions:
            if condition == 'oracle_id':
                chosen, status = expected, 'fixed_GT_correspondence_diagnostic'
                mask = selected_mask(prediction,sample['candidates'],chosen)
            elif condition == 'ctrlo':
                folder = root/'baseline/ctrlo'/query['id']
                if not completed(folder,baseline,query):
                    raise ValueError(f'Missing CTRL-O prediction: {query["id"]}')
                mask = np.load(folder/'prediction.npz')['mask']; chosen, status = None, 'direct_expression_conditioning'
            else:
                path = root/'selections'/condition/'queries'/f'{query["id"]}.json'
                selection = json.loads(path.read_text())
                if (selection['selection_fingerprint'] != infos[condition]['selection_fingerprint'] or
                        selection['query_id'] != query['id'] or infos[condition]['spec']['fingerprint'] != m['fingerprint']):
                    raise ValueError('Selection provenance mismatch')
                chosen, status = selection['selected_id'],selection['status']
                mask = selected_mask(prediction,sample['candidates'],chosen)
            result = score_mask(mask,target)
            row = dict(condition=condition,query_id=query['id'],split=query['split'],
                image_id=iid,ref_id=query['ref_id'],sent_id=query['sent_id'],
                annotation_id=query['annotation_id'],request=query['request'],
                selected_id=chosen,status=status,expected_id=expected,target_present=expected is not None,
                selection_correct=expected is not None and chosen==expected,**result)
            rows.append(row)
            if saved < args.save_examples:
                with Image.open(sample['image_path']) as im:
                    im = im.convert('RGB'); array = np.asarray(im).copy()
                array[mask] = np.rint(.4*array[mask]+.6*np.array([255,60,60])).astype(np.uint8)
                panel = Image.new('RGB',(im.width,im.height+32),'white')
                panel.paste(Image.fromarray(array),(0,32))
                ImageDraw.Draw(panel).text((8,8),f'{condition}: IoU={result["iou"]:.3f}',fill='black')
                panels.append(panel)
        if panels:
            combined = Image.new('RGB',(sum(p.width for p in panels),panels[0].height),'white')
            for index,panel in enumerate(panels): combined.paste(panel,(index*panel.width,0))
            folder = root/'examples'/query['id']; folder.mkdir(parents=True,exist_ok=True)
            combined.save(folder/'selection_overlay.png')
            write_json(folder/'metadata.json',dict(query=query,results=rows[-len(conditions):]))
            saved += 1
    result = dict(protocol=m['protocol'],fingerprint=m['fingerprint'],
        selector=a,by_split={split:{c:metrics([r for r in rows if r['split']==split and r['condition']==c])
            for c in conditions} for split in args.splits},
        num_images=len(m['samples']),num_expressions=len(m['queries']),
        ar_incomplete_images=sum(not json.loads((root/'owners'/str(s['image_id'])/'metadata.json').read_text())['ar_record']['scene_end_generated'] for s in m['samples']))
    write_json(root/'summary.json',result)
    with (root/'per_query.jsonl').open('w') as stream:
        for row in rows: stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    fields = ['split','condition']+list(metrics(rows[:1]))
    with (root/'comparison.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for split, methods in result['by_split'].items():
            for c, stats in methods.items(): writer.writerow(dict(split=split,condition=c,**stats))
    lines = ['# RefCOCO cached-object retrieval','',
        'Scores below are percentages. Oracle is a GT-only diagnostic, not a method.','',
        '| Split | Method | Expressions | mIoU | oIoU | Success@0.5 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for split,methods in result['by_split'].items():
        for c,stats in methods.items():
            lines.append(f'| {split} | {c} | {stats["num_expressions"]} | {100*stats["mIoU"]:.2f} | {100*stats["oIoU"]:.2f} | {100*stats["success_50"]:.2f} |')
    (root/'results.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['prepare','extract','select','preflight','encode','ctrlo','score'])
    p.add_argument('--output_dir',required=True)
    p.add_argument('--model_path',default='/home/jovyan/PGOT/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000')
    p.add_argument('--data_root',default='/home/jovyan/data/coco')
    p.add_argument('--image_root',default='/home/jovyan/data/coco')
    p.add_argument('--dataset',choices=['refcoco','refcoco+','refcocog'],default='refcoco')
    p.add_argument('--split_by',default='unc')
    p.add_argument('--splits',nargs='+',default=['val'])
    p.add_argument('--max_refs',type=int,default=0)
    p.add_argument('--max_samples',type=int,default=0)
    p.add_argument('--train_annotations',default='/home/jovyan/PGOT/data/pgot_pix2cap_thing_train.jsonl')
    p.add_argument('--eval_size',type=int,default=512)
    p.add_argument('--dtype',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max_objects',type=int,default=50)
    p.add_argument('--max_caption_tokens',type=int,default=4096)
    p.add_argument('--ar_max_new_tokens',type=int,default=2048)
    p.add_argument('--cache_bundles',action='store_true')
    p.add_argument('--presence_iou',type=float,default=.1)
    p.add_argument('--cpu_threads',type=int,default=3)
    p.add_argument('--selector_model',default='/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17')
    p.add_argument('--selector_max_new_tokens',type=int,default=64)
    p.add_argument('--selector_max_input_tokens',type=int,default=8192)
    p.add_argument('--condition',choices=['category','description'],default='description')
    p.add_argument('--save_examples',type=int,default=24)
    p.add_argument('--with_ctrlo',action='store_true')
    p.add_argument('--ctrlo_repo',default='/home/jovyan/baselines/CTRL-O')
    p.add_argument('--checkpoint_dir',default='/home/jovyan/baselines/ckpt/ctrlo')
    p.add_argument('--llm2vec_base',default='McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp')
    p.add_argument('--llm2vec_adapter',default='McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse')
    p.add_argument('--embedding_batch_size',type=int,default=4)
    p.add_argument('--embedding_max_length',type=int,default=1024)
    args = p.parse_args(argv)
    allowed = {'val','test'} if args.dataset=='refcocog' else {'val','testA','testB'}
    if not set(args.splits)<=allowed or len(set(args.splits))!=len(args.splits): p.error('Invalid/duplicate evaluation splits')
    if args.max_refs and args.max_samples: p.error('Choose only one subset limit')
    if min(args.max_refs,args.max_samples,args.seed,args.save_examples)<0: p.error('Invalid negative limit')
    if min(args.eval_size,args.max_objects,args.max_caption_tokens,args.ar_max_new_tokens,
           args.cpu_threads,args.selector_max_new_tokens,args.selector_max_input_tokens,
           args.embedding_batch_size,args.embedding_max_length)<1: p.error('Sizes must be positive')
    if args.eval_size!=512: p.error('Protocol fixes full-frame preprocessing at 512')
    if not 0<args.presence_iou<=1: p.error('presence_iou must be in (0,1]')
    if args.dataset=='refcocog' and args.split_by!='umd': p.error('Use UMD val/test for RefCOCOg')
    args.diffusion_inference_steps = 10
    args.text_mode = 'description'
    return args


def main():
    args = parse_args()
    if args.stage in ('preflight','encode','ctrlo'):
        manifest(args)
        if not args.with_ctrlo: raise ValueError('CTRL-O stage requires --with_ctrlo')
        args.output_dir = str(Path(args.output_dir)/'baseline')
        if args.stage=='ctrlo':
            from pgot.eval.run_refcoco_eval import ctrlo
            ctrlo(args)
        else:
            from pgot.eval.run_ar_text_pair_eval import preflight, encode
            {'preflight':preflight,'encode':encode}[args.stage](args)
    else:
        globals()[args.stage](args)


if __name__ == '__main__':
    main()
