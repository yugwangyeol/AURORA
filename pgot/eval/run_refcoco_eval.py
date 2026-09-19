"""Frozen DeOR / CTRL-O, same original referring expression, one query per pass.

No generated text, GT category, point, box, mask, or GT-best slot is used in
inference. Target masks are opened only by the scoring stage. Images are not
cropped; ownership probabilities are resized to the original image dimensions.
"""

import argparse
import csv
import hashlib
import json
import pickle
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from pgot.eval.run_ar_text_pair_eval import (
    DEFAULT_MODEL, digest, embedding_path, encode, load_manifest, preflight, write_json,
)


VERSION = 1
THRESHOLDS = (.5, .6, .7, .8, .9)
RESERVED = ('<thing>', '<stuff>', '<ovt>', '<scene_end>', '<|im_start|>', '<|im_end|>')


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def referring_caption(text):
    """Fixed structural header; Object is NOT an annotation-derived class."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Empty referring expression; never silently skip it')
    if any(token in text for token in RESERVED):
        raise ValueError('Referring expression contains reserved model tokens')
    return f'<thing> Object 1: {text} <ovt>. <scene_end>'


def annotation_paths(args):
    folder = Path(args.data_root) / args.dataset
    return folder / f'refs({args.split_by}).p', folder / 'instances.json'


def resolve_image(root, image):
    root = Path(root)
    iid, name = int(image['id']), Path(image['file_name']).name
    if name != image['file_name']:
        raise ValueError('Unexpected image filename in annotation')
    candidates = [root / split / f'{iid:012d}.jpg' for split in ('train2017', 'val2017')]
    candidates += [root / split / name for split in ('train2014', 'val2014')]
    candidates += [root / name, root / 'images' / 'mscoco' / 'images' / name,
                   root / 'images' / 'train2014' / name]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f'Image {iid} is missing under {root}; cannot drop expressions')


def selected_references(refs, splits, max_refs=0):
    result = []
    for split in splits:
        selected = sorted((r for r in refs if r['split'] == split), key=lambda r: int(r['ref_id']))
        if not selected:
            raise ValueError(f'No references in requested split {split}')
        result.extend(selected[:max_refs] if max_refs else selected)
    return result


def make_queries(refs, max_samples=0):
    queries, seen = [], set()
    counts = Counter()
    for ref in refs:
        sentences = sorted(ref['sentences'], key=lambda s: int(s['sent_id']))
        if not sentences:
            raise ValueError(f'Reference {ref["ref_id"]} has no expressions')
        for sentence in sentences:
            if max_samples and counts[ref['split']] >= max_samples:
                break
            # raw retains original wording/capitalization/punctuation. Some
            # official versions provide only sent; neither form is rewritten.
            text = sentence.get('raw', sentence.get('sent'))
            caption = referring_caption(text)
            sid, rid = int(sentence['sent_id']), int(ref['ref_id'])
            if sid in seen:
                raise ValueError(f'Duplicate sent_id {sid}')
            seen.add(sid)
            queries.append(dict(dataset='refcoco', id=f'{ref["split"]}/{rid}/{sid}',
                split=ref['split'], ref_id=rid, sent_id=sid, image_id=int(ref['image_id']),
                annotation_id=int(ref['ann_id']), text=caption, shared_text=text,
                caption_sha256=digest(caption), query_sha256=digest(text),
                ovt_count=1, pairs=[dict(shared_text=text, slot_index=0, prediction_label=1)]))
            counts[ref['split']] += 1
    return queries


def prepare(args):
    refs_path, instances_path = annotation_paths(args)
    if not refs_path.is_file() or not instances_path.is_file():
        raise FileNotFoundError(f'Required official annotations: {refs_path}, {instances_path}')
    config_keys = ('model_path', 'data_root', 'image_root', 'dataset', 'split_by', 'splits',
                   'max_refs', 'max_samples', 'eval_size', 'dtype', 'seed', 'max_objects',
                   'max_caption_tokens', 'ctrlo_repo', 'checkpoint_dir', 'text_mode',
                   'llm2vec_base', 'llm2vec_adapter', 'embedding_max_length')
    if args.ours_only:
        config_keys = tuple(key for key in config_keys if key not in
                            {'ctrlo_repo','checkpoint_dir','llm2vec_base','llm2vec_adapter','embedding_max_length'})
    config = {key: getattr(args, key) for key in config_keys}
    config.update(version=VERSION, annotation_hashes={'refs': file_hash(refs_path),
                  'instances': file_hash(instances_path)})
    if args.ours_only:
        config['evaluation_models'] = ['deor']
    else:
        config.update(ctrlo_checkpoint_sha256=file_hash(Path(args.checkpoint_dir)/'pretrained_model.ckpt'),
                      ctrlo_config_sha256=file_hash(Path(args.checkpoint_dir)/'config.yaml'))
    weights = sorted(Path(args.model_path).glob('*.safetensors'))
    if not weights:
        raise FileNotFoundError('DeOR safetensors checkpoint is missing')
    config['deor_checkpoint_hashes'] = {p.name: file_hash(p) for p in weights}
    config['deor_config_sha256'] = file_hash(Path(args.model_path)/'config.json')
    fingerprint = digest(json.dumps(config, sort_keys=True))
    path = Path(args.output_dir)/'manifest.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['fingerprint'] != fingerprint:
            raise ValueError('Output has a different protocol/source; choose another OUTPUT_ROOT')
        for image_path, expected_hash in {(s['image_path'], s['image_sha256']) for s in previous['samples']}:
            if file_hash(image_path) != expected_hash:
                raise ValueError(f'Image changed since manifest preparation: {image_path}')
        print('Matching manifest exists; resume completed queries.', flush=True)
        return
    # Only author-distributed, locally trusted REFER pickle files are supported.
    with refs_path.open('rb') as stream:
        refs = pickle.load(stream, encoding='latin1')
    instances = json.loads(instances_path.read_text())
    images = {int(i['id']): i for i in instances['images']}
    annotations = {int(a['id']): a for a in instances['annotations']}
    selected = selected_references(refs, args.splits, args.max_refs)
    samples = make_queries(selected, args.max_samples)
    image_cache = {}
    for sample in samples:
        sample['dataset'] = args.dataset
        ann = annotations[sample['annotation_id']]
        if int(ann['image_id']) != sample['image_id']:
            raise ValueError('Reference/target image mismatch')
        iid = sample['image_id']
        if iid not in image_cache:
            info = images[iid]
            image_path = resolve_image(args.image_root, info)
            with Image.open(image_path) as image:
                if image.size != (int(info['width']), int(info['height'])):
                    raise ValueError(f'Image dimensions disagree with annotation: {image_path}')
                image.verify()
            image_cache[iid] = dict(image_path=str(image_path), width=int(info['width']),
                                   height=int(info['height']), image_sha256=file_hash(image_path))
        sample.update(image_cache[iid])
    protocol = dict(task='referring_expression_segmentation',
        query_source='original human referring expressions; every sentence, no rewriting',
        selection='full official split or deterministic ref-ID/sentence-ID prefix; no quality filtering',
        deor='one external expression in fixed Object 1 structural header, one OVT; NOT image-only AR',
        ctrlo='one expression-bound slot (slot 0), six unbound slots; no GT-best selection',
        image='entire image resized to 512 square, no crop or flip; native model processors afterward',
        mask='bilinear ownership probability resize to ORIGINAL H,W, then argmax against all valid competitors',
        target='the exact referred annotation; no class matching, Hungarian matching, or other-object repair',
        inference_gt_inputs='none: no target class, mask, box, coordinates, or other object descriptions',
        metrics='expression-weighted mIoU, pixel-aggregated oIoU, P@0.5/0.6/0.7/0.8/0.9; empty predictions retained',
        supervision='frozen checkpoints, no additional RefCOCO training',
        limitations='COCO-trained checkpoints may have seen these images during COCO training; NOT unseen-image zero-shot. Released CTRL-O VG+COCO checkpoint is not a reproduction of Table 3 VG checkpoints.',
        full_split_evaluated=not (args.max_refs or args.max_samples))
    if args.ours_only:
        protocol['evaluated_models'] = 'DeOR only; CTRL-O inference and LLM2Vec encoding skipped'
        protocol.pop('ctrlo')
    write_json(path, dict(fingerprint=fingerprint, config=config, protocol=protocol, samples=samples))
    print(f'Prepared {len(samples)} expressions / {len(image_cache)} images: {dict(Counter(s["split"] for s in samples))}', flush=True)


def inference_query(sample):
    """Explicit allowlist: GT annotation/ref identity is not passed to a model."""
    return {key: sample[key] for key in ('image_path', 'width', 'height', 'image_sha256',
                                        'shared_text', 'text', 'query_sha256', 'caption_sha256')}


def load_full_image(query, size):
    if file_hash(query['image_path']) != query['image_sha256']:
        raise ValueError('Image changed since manifest preparation')
    with Image.open(query['image_path']) as image:
        if image.size != (query['width'], query['height']):
            raise ValueError('Image changed since manifest preparation')
        return image.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)


def completed(folder, manifest, sample):
    path = folder/'metadata.json'
    if not path.exists():
        return False
    info = json.loads(path.read_text())
    for key, value in (('fingerprint', manifest['fingerprint']),
                       ('query_sha256', sample['query_sha256']), ('image_sha256', sample['image_sha256'])):
        if info.get(key) != value:
            raise ValueError(f'Prediction provenance mismatch: {folder}')
    return (folder/'prediction.npz').is_file()


def save_prediction(folder, mask, manifest, sample, **extra):
    if mask.shape != (sample['height'], sample['width']) or mask.dtype != np.bool_:
        raise ValueError('Prediction must be binary at original image resolution')
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder/'prediction.tmp.npz'
    np.savez_compressed(temporary, mask=mask)
    temporary.replace(folder/'prediction.npz')
    write_json(folder/'metadata.json', dict(fingerprint=manifest['fingerprint'],
        query_sha256=sample['query_sha256'], image_sha256=sample['image_sha256'],
        caption_sha256=sample['caption_sha256'], empty_mask=not mask.any(), **extra))


def deor_native_mask(object_probs, void_probs, height, width):
    """The native single valid OVT owner competes against summed void owners."""
    import torch
    import torch.nn.functional as F
    if object_probs.ndim != 3 or object_probs.shape[0] != 1 or object_probs.shape[1] < 1:
        raise ValueError('Invalid DeOR ownership shape')
    if void_probs is None or void_probs.ndim != 3 or void_probs.shape[1] < 1:
        raise ValueError('Void ownership probabilities are required')
    patches = object_probs.shape[-1]
    grid = round(patches ** .5)
    if grid * grid != patches or void_probs.shape[0] != 1 or void_probs.shape[-1] != patches:
        raise ValueError('Invalid ownership grid')
    # All padded OVTs are invalid. There is exactly one valid queried owner.
    maps = torch.cat((object_probs[:, :1].float(), void_probs.float().sum(1, keepdim=True)), dim=1)
    if not torch.isfinite(maps).all():
        raise ValueError('Nonfinite DeOR probabilities')
    up = F.interpolate(maps.reshape(1, 2, grid, grid), (height, width), mode='bilinear', align_corners=False)
    return (up[0].argmax(0) == 0).cpu().numpy()


def ours(args):
    import torch
    from tqdm import tqdm
    from pgot.eval.eval_recon_oracles import load_model_and_tokenizer
    from pgot.eval.pgot_inference import pgot_forward_eval
    torch.set_num_threads(3)
    torch.manual_seed(args.seed)
    manifest = load_manifest(args)
    root = Path(args.output_dir)/'ours'
    pending = [s for s in manifest['samples'] if not completed(root/s['id'], manifest, s)]
    if not pending:
        print('All DeOR expressions already completed.', flush=True)
        return
    model, tokenizer, device, dtype = load_model_and_tokenizer(args)
    towers = model.get_vision_tower_aux_list()
    write_json(Path(args.output_dir)/'ours_model_info.json', dict(checkpoint=args.model_path,
        caption_mode='external_original_referring_expression', structural_header='Object 1',
        object_memory=getattr(model.config, 'pgot_e11_object_memories_per_owner', None),
        register_memory=getattr(model.config, 'pgot_e11_register_memories_per_owner', None),
        writer_prior=getattr(model.config, 'pgot_one_shot_writer_owner_prior', None),
        gt_class_input=False, gt_mask_input=False, gt_point_input=False, dtype=args.dtype))
    for sample in tqdm(pending, desc='DeOR original referring expressions'):
        query = inference_query(sample)
        caption = torch.tensor(tokenizer.encode(query['text'], add_special_tokens=False))[None]
        if caption.shape[1] > args.max_caption_tokens:
            raise ValueError('Expression exceeds caption token limit; no truncation')
        positions = (caption[0] == tokenizer.convert_tokens_to_ids('<ovt>')).nonzero().flatten()
        if len(positions) != 1:
            raise ValueError('Exactly one query OVT is required')
        valid = torch.zeros((1, args.max_objects), dtype=torch.bool)
        pos = torch.zeros((1, args.max_objects), dtype=torch.long)
        valid[0, 0], pos[0, 0] = True, positions[0]
        image = load_full_image(query, args.eval_size)
        inputs = towers[0].image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        target = (towers[1] if len(towers)>1 else towers[0]).image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        with torch.inference_mode():
            result = pgot_forward_eval(model, images=inputs, target_images=target,
                caption_input_ids=caption, caption_attention_mask=torch.ones_like(caption, dtype=torch.bool),
                ovt_positions_in_caption=pos, ovt_valid_mask=valid)
            mask = deor_native_mask(result['ovt_object_probs'], result['ovt_void_probs'], query['height'], query['width'])
        del result
        save_prediction(root/sample['id'], mask, manifest, sample, query_ovt_index=0, native_readout=True)


def ctrlo(args):
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from tqdm import tqdm
    from omegaconf import OmegaConf
    from pgot.eval.figure_ctrlo_examples import load_ctrlo
    torch.set_num_threads(3)
    manifest = load_manifest(args)
    root = Path(args.output_dir)/'ctrlo'
    pending = [s for s in manifest['samples'] if not completed(root/s['id'], manifest, s)]
    if not pending:
        print('All CTRL-O expressions already completed.', flush=True)
        return
    model, info = load_ctrlo(args)
    cfg = OmegaConf.load(Path(args.checkpoint_dir)/'config.yaml')
    info.update(query_protocol='original human expression; one bound slot + six unbound slots',
        image_preprocessing='entire image bilinear 512 square resize, then 224 square; no crop',
        mask_readout='bilinear to original H,W; argmax over all seven slots; query slot 0 only',
        selection='fixed input-associated slot; never GT-best',
        released_training_shards=OmegaConf.to_container(cfg.dataset, resolve=False).get('train_shards'))
    write_json(Path(args.output_dir)/'ctrlo_model_info.json', info)
    transform = transforms.Compose([transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), transforms.Normalize([.485,.456,.406], [.229,.224,.225])])
    for sample in tqdm(pending, desc='CTRL-O original referring expressions'):
        query = inference_query(sample)
        seed = int.from_bytes(hashlib.sha256(f'{args.seed}/{query["image_sha256"]}/{query["query_sha256"]}'.encode()).digest()[:4], 'little')
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        names = np.zeros((7,4096), dtype=np.float32)
        rep = np.load(embedding_path(args, query['shared_text']))
        if rep.shape != (4096,) or not np.isfinite(rep).all():
            raise ValueError('Invalid official expression embedding')
        names[0] = rep
        inputs = dict(image=transform(load_full_image(query,args.eval_size))[None].cuda(), batch_size=1,
            name_embedding=torch.from_numpy(names)[None].cuda(),
            contrastive_loss_mask=torch.tensor([[1,0,0,0,0,0,0]], device='cuda'),
            bbox_centroids=torch.full((1,7,2), -1., device='cuda'), instance_bbox=torch.full((1,7,4), -1., device='cuda'))
        with torch.inference_mode():
            output = model(inputs={'input':inputs})
            probabilities = output['object_decoder'].masks_as_image
            if probabilities.shape != (1,7,224,224) or not torch.isfinite(probabilities).all():
                raise ValueError('Unexpected official CTRL-O masks')
            up = F.interpolate(probabilities, (query['height'],query['width']), mode='bilinear', align_corners=False)
            mask = (up[0].argmax(0)==0).cpu().numpy()
        del output
        save_prediction(root/sample['id'], mask, manifest, sample, query_slot=0, unbound_slots=list(range(1,7)), seed=seed)


def score_mask(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError('Prediction/GT dimensions differ')
    pred, gt = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if not gt.any():
        raise ValueError('Official referred GT mask is empty; investigate, never skip')
    intersection = int(np.count_nonzero(pred & gt))
    union = int(np.count_nonzero(pred | gt))
    return dict(intersection=intersection, union=union, iou=intersection/union, empty_mask=not pred.any())


def aggregate(rows, model):
    metrics = [r[model] for r in rows]
    if not metrics:
        raise ValueError('Cannot aggregate an empty split')
    result = dict(num_expressions=len(rows), num_images=len({r['image_id'] for r in rows}),
        num_references=len({r['ref_id'] for r in rows}),
        mIoU=sum(m['iou'] for m in metrics)/len(metrics),
        oIoU=sum(m['intersection'] for m in metrics)/sum(m['union'] for m in metrics),
        empty_masks=sum(m['empty_mask'] for m in metrics))
    result['precision_at_iou'] = {f'{t:.2f}':sum(m['iou'] >= t for m in metrics)/len(metrics) for t in THRESHOLDS}
    return result


def _score(args, model_folders):
    from pycocotools.coco import COCO
    from tqdm import tqdm
    manifest = load_manifest(args)
    root = Path(args.output_dir)
    coco = COCO(str(annotation_paths(args)[1]))

    @lru_cache(maxsize=128)
    def target_mask(aid):
        return coco.annToMask(coco.anns[aid]).astype(bool)

    rows = []
    temporary = root/'per_expression.tmp.jsonl'
    with temporary.open('w') as stream:
        for sample in tqdm(manifest['samples'], desc='RefCOCO exact referred-target scoring'):
            target = target_mask(sample['annotation_id'])
            if target.shape != (sample['height'],sample['width']):
                raise ValueError('Annotation dimensions do not match image')
            row = {key:sample[key] for key in ('split','ref_id','sent_id','image_id','annotation_id','shared_text')}
            for model, folder in model_folders:
                dest = root/folder/sample['id']
                if not completed(dest,manifest,sample):
                    raise RuntimeError(f'Missing prediction; cannot report a partial full split: {dest}')
                with np.load(dest/'prediction.npz') as data:
                    row[model] = score_mask(data['mask'],target)
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            rows.append(row)
    temporary.replace(root/'per_expression.jsonl')
    models = [model for model, _ in model_folders]
    result = dict(fingerprint=manifest['fingerprint'],config=manifest['config'],protocol=manifest['protocol'],
        model_info={name:json.loads((root/f'{folder}_model_info.json').read_text()) for name,folder in model_folders},
        by_split={split:{model:aggregate([r for r in rows if r['split']==split],model) for model in models} for split in args.splits})
    embedding_info = root/'embedding_info.json'
    if 'ctrlo' in models:
        result['embedding_info'] = json.loads(embedding_info.read_text())
    write_json(root/'summary.json',result)
    with (root/'comparison.csv').open('w',newline='') as stream:
        fields=['dataset','split','model','expressions','images','references','mIoU','oIoU','P@0.5','P@0.6','P@0.7','P@0.8','P@0.9','empty_masks']
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for split,methods in result['by_split'].items():
            for model,metrics in methods.items():
                writer.writerow(dict(dataset=args.dataset,split=split,model=model,
                    expressions=metrics['num_expressions'],images=metrics['num_images'],references=metrics['num_references'],
                    mIoU=metrics['mIoU'],oIoU=metrics['oIoU'],empty_masks=metrics['empty_masks'],
                    **{f'P@{t:.1f}':metrics['precision_at_iou'][f'{t:.2f}'] for t in THRESHOLDS}))
                print(split,model,metrics,flush=True)


def score(args):
    """Score the original paired DeOR/CTRL-O comparison."""
    _score(args, [('deor','ours'),('ctrlo','ctrlo')])


def score_ours(args):
    """Score DeOR only; no CTRL-O predictions or text embeddings are required."""
    _score(args, [('deor','ours')])


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['prepare','preflight','encode','ours','ctrlo','score','score_ours'])
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--model_path',default=DEFAULT_MODEL)
    parser.add_argument('--data_root',default='/home/jovyan/data/coco')
    parser.add_argument('--image_root',default='/home/jovyan/data/coco')
    parser.add_argument('--dataset',choices=['refcoco','refcoco+','refcocog'],default='refcoco')
    parser.add_argument('--split_by',choices=['unc','umd','google'])
    parser.add_argument('--splits',nargs='+')
    parser.add_argument('--max_refs',type=int,default=0,help='Per-split deterministic ref-ID prefix; 0=all')
    parser.add_argument('--max_samples',type=int,default=0,help='Per-split deterministic sentence prefix; 0=all')
    parser.add_argument('--ctrlo_repo',default='/home/jovyan/baselines/CTRL-O')
    parser.add_argument('--checkpoint_dir',default='/home/jovyan/baselines/ckpt/ctrlo')
    parser.add_argument('--eval_size',type=int,default=512)
    parser.add_argument('--dtype',choices=['fp32','bf16'],default='fp32')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--max_objects',type=int,default=50)
    parser.add_argument('--max_caption_tokens',type=int,default=4096)
    parser.add_argument('--llm2vec_base',default='McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp')
    parser.add_argument('--llm2vec_adapter',default='McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse')
    parser.add_argument('--embedding_batch_size',type=int,default=4)
    parser.add_argument('--embedding_max_length',type=int,default=1024)
    parser.add_argument('--ours_only',action='store_true',help='Prepare provenance for DeOR-only evaluation')
    args=parser.parse_args(argv)
    args.split_by=args.split_by or ('umd' if args.dataset=='refcocog' else 'unc')
    args.splits=args.splits or (['val','test'] if args.dataset=='refcocog' else ['val','testA','testB'])
    allowed={'train','val','test'} if args.dataset=='refcocog' else {'train','val','testA','testB'}
    if not set(args.splits)<=allowed or len(set(args.splits))!=len(args.splits):
        parser.error('Invalid/duplicate splits for this dataset')
    if args.dataset=='refcoco+' and args.split_by!='unc' or args.dataset=='refcocog' and args.split_by=='unc' or args.dataset=='refcoco' and args.split_by=='umd':
        parser.error('Invalid split convention for dataset')
    if min(args.max_refs,args.max_samples,args.seed)<0 or min(args.eval_size,args.max_objects,args.max_caption_tokens,args.embedding_batch_size,args.embedding_max_length)<1:
        parser.error('Invalid limits/resolution/seed')
    if args.max_refs and args.max_samples:
        parser.error('Choose at most one deterministic subset limit')
    args.text_mode='description'  # Exact original expression, never category-only.
    args.diffusion_inference_steps=10  # Loader compatibility only; no diffusion generation.
    return args


def main():
    args=parse_args()
    started=time.time()
    globals()[args.stage](args)
    print(f'{args.stage} completed in {time.time()-started:.1f}s',flush=True)


if __name__=='__main__':
    main()
