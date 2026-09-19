"""Same generated AR text, native DeOR masks versus official CTRL-O extraction.

Stages run in separate environments/processes so the PGOT and LLM2Vec models
need not coexist on GPU. Completed per-image results are resumable. No GT is
used in discovery, text embedding, conditioning or prediction selection.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
from PIL import Image

from pgot.eval.class_mask_pairs import ar_pairs, ctrlo_pair_masks, masks_for_pairs, rates, score_pairs

VERSION = 1
DEFAULT_MODEL = '/home/jovyan/PGOT/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000'
VOC_CANONICAL = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                 'dining table', 'dog', 'horse', 'motorcycle', 'person', 'potted plant', 'sheep', 'couch', 'train', 'tv']


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def center_crop(image, size=512, nearest=False):
    w, h = image.size
    factor = max(size/h, size/w)
    rw, rh = round(w*factor), round(h*factor)
    image = image.resize((rw, rh), Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR)
    left, top = max((rw-size)//2, 0), max((rh-size)//2, 0)
    return image.crop((left, top, left+size, top+size))


def load_image(args, sample):
    path = (Path(args.coco_root)/'val2017'/f'{int(sample["id"]):012d}.jpg' if sample['dataset']=='coco'
            else Path(args.voc_root)/'images'/f'{sample["id"]}.jpg')
    with Image.open(path) as image:
        return center_crop(image.convert('RGB'), args.eval_size)


def manifest_config(args):
    config = {key: getattr(args, key) for key in ['model_path', 'caption_root', 'coco_root', 'voc_root',
              'ctrlo_repo', 'checkpoint_dir', 'datasets', 'max_samples', 'image_ids', 'text_mode',
              'eval_size', 'dtype', 'seed', 'max_objects', 'queries_per_pass', 'thresholds',
              'ignore_overlaps', 'llm2vec_base', 'llm2vec_adapter', 'embedding_max_length']}
    config['version'] = VERSION
    config['source_hashes'] = {dataset: {name: file_digest(Path(args.caption_root)/f'{dataset}_ar'/name)
                for name in ['summary.json', 'generated_captions.jsonl']} for dataset in args.datasets}
    config['annotation_hashes'] = {'coco': file_digest(Path(args.coco_root)/'annotations/instances_val2017.json')} if 'coco' in args.datasets else {}
    if 'voc' in args.datasets:
        config['annotation_hashes']['voc_split'] = file_digest(Path(args.voc_root)/'sets/val.txt')
    config['ctrlo_checkpoint'] = file_digest(Path(args.checkpoint_dir)/'pretrained_model.ckpt')
    config['ctrlo_config'] = file_digest(Path(args.checkpoint_dir)/'config.yaml')
    return config


def prepare(args):
    out = Path(args.output_dir)
    config = manifest_config(args)
    fingerprint = digest(json.dumps(config, sort_keys=True))
    path = out/'manifest.json'
    if path.exists():
        if json.loads(path.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Existing output has a different protocol/source; choose a new OUTPUT_ROOT')
        print('Matching manifest exists; resume completed stages.', flush=True)
        return
    samples = []
    explicit = set(args.image_ids or [])
    seen = set()
    for dataset in args.datasets:
        source = Path(args.caption_root)/f'{dataset}_ar'
        summary = json.loads((source/'summary.json').read_text())
        if Path(summary['ckpt']).resolve() != Path(args.model_path).resolve() or summary['caption_mode'] != 'autoregressive':
            raise ValueError(f'{dataset}: cached source must be image-only AR from the requested checkpoint')
        records = [json.loads(line) for line in (source/'generated_captions.jsonl').read_text().splitlines() if line.strip()]
        captions = {str(r['image_id']): r for r in records}
        if len(captions) != len(records):
            raise ValueError('Duplicate caption image IDs')
        if dataset == 'coco':
            ann = json.loads((Path(args.coco_root)/'annotations/instances_val2017.json').read_text())
            ids = [str(i) for i in sorted(image['id'] for image in ann['images'])]
            expected_keys = set(ids)
            if len(ids) != 5000:
                raise ValueError('Expected all 5000 official COCO val2017 images')
        else:
            ids = (Path(args.voc_root)/'sets/val.txt').read_text().split()
            expected_keys = {str(i) for i in range(len(ids))}
        if set(captions) != expected_keys:
            raise ValueError(f'{dataset}: source captions do not cover the full split exactly')
        count = 0
        for index, iid in enumerate(ids):
            key = f'{dataset}:{iid}'
            if explicit and key not in explicit:
                continue
            record = captions[iid if dataset=='coco' else str(index)]
            text, ovts = record['text'], int(record['ovt_count'])
            if ovts > args.max_objects:
                raise ValueError('Caption exceeds model capacity; do not truncate')
            sample = dict(dataset=dataset, id=iid, source_index=index, text=text, ovt_count=ovts,
                          caption_sha256=digest(text), pairs=ar_pairs(text, ovts, args.text_mode))
            load_image(args, sample)  # Read-only validation, no GT inspection.
            samples.append(sample)
            seen.add(key)
            count += 1
            if args.max_samples and count >= args.max_samples:
                break
    if explicit-seen:
        raise ValueError(f'Requested image IDs not found: {sorted(explicit-seen)}')
    write_json(path, dict(fingerprint=fingerprint, config=config, samples=samples,
              protocol={'task': 'same_AR_text_class_mask_pair_evaluation',
              'source_selection': 'full split or explicit fixed subset; no GT-quality filtering',
              'deor': 'native cached image-only AR text replayed for original caption-associated ownership masks',
              'ctrlo': 'same category plus FULL description (not first sentence) when text_mode=description',
              'point_input': 'none; official -1 sentinels, no GT or predicted coordinates',
              'overflow': 'all generated pairs in original order; multiple CTRL-O passes with <=queries_per_pass bound slots and other padding to seven',
              'metric': 'maximum-cardinality class-compatible one-to-one mask IoU matching; all generated thing pairs incl. duplicates, unknown classes, empty masks and unbound trailing THING text',
              'recall_denominator': 'all visible noncrowd valid GT thing instances; missing AR objects are FN',
              'empty_scene': 'include images even with no GT or no generated pairs; no false positive perfect-score conventions',
              'matching_uses_gt': 'only scoring; never prediction routing or best-slot selection',
              'limitation': 'class/mask consistency, NOT correctness of every attribute/relation in descriptions; DeOR-style text, unequal conditioning architectures, multi-pass overflow must be disclosed',
              'ctrlo_published_Binding_Hit_comparison': False}))
    print(f'Prepared {len(samples)} images, {sum(len(s["pairs"]) for s in samples)} generated pairs; {args.text_mode}', flush=True)


def load_manifest(args):
    manifest = json.loads((Path(args.output_dir)/'manifest.json').read_text())
    # Each stage must use the manifest's actual immutable configuration.
    for key, value in manifest['config'].items():
        if hasattr(args, key) and getattr(args, key) != value:
            raise ValueError(f'CLI/manifest mismatch for {key}; use the same wrapper configuration')
    return manifest


def embedding_path(args, text):
    return Path(args.output_dir)/'embeddings'/f'{digest(text)}.npy'


def prediction_completed(folder, manifest, sample):
    path=folder/'metadata.json'
    if not path.exists():
        return False
    info=json.loads(path.read_text())
    if info['fingerprint']!=manifest['fingerprint'] or info['caption_sha256']!=sample['caption_sha256']:
        raise ValueError(f'Existing prediction cache provenance mismatch: {folder}')
    return (folder/'prediction.npz').exists()


def official_categories(args):
    path = Path(args.ctrlo_repo)/'scripts/datasets/data/category_name_to_llama3_emb.pkl'
    with path.open('rb') as stream:
        return pickle.load(stream)


def preflight(args):
    manifest = load_manifest(args)
    texts = {p['shared_text'] for s in manifest['samples'] for p in s['pairs']}
    if args.text_mode == 'category':
        known = official_categories(args)
        missing = texts-set(known)
        if missing:
            raise RuntimeError(f'Official category embeddings missing {len(missing)} generated names; no silent other substitution. Use full description mode with encoder access. Examples: {sorted(missing)[:10]}')
    elif texts and (not all(embedding_path(args, text).exists() for text in texts) or not (Path(args.output_dir)/'embedding_info.json').exists()):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError
        base = Path(args.llm2vec_base)
        try:
            if base.is_dir():
                if not (base/'config.json').exists():
                    raise ValueError('Local LLM2Vec base must have config.json')
            else:
                hf_hub_download(args.llm2vec_base, 'config.json')
                try:
                    adapter_path = hf_hub_download(args.llm2vec_base, 'adapter_config.json')
                except Exception as exc:
                    from huggingface_hub.errors import EntryNotFoundError
                    if not isinstance(exc, EntryNotFoundError):
                        raise
                else:
                    underlying = json.loads(Path(adapter_path).read_text())['base_model_name_or_path']
                    hf_hub_download(underlying, 'config.json')
            hf_hub_download(args.llm2vec_adapter, 'adapter_config.json') if not Path(args.llm2vec_adapter).is_dir() else None
        except GatedRepoError as exc:
            raise RuntimeError('Full-description evaluation requires authorized access to meta-llama/Meta-Llama-3-8B-Instruct. Obtain account access, then run bash scripts/login_ctrlo_hf.sh from PGOT. Never silently replace descriptions with category embeddings.') from exc
    print(f'Preflight OK: {args.text_mode}, {len(texts)} unique texts. No model loaded yet.', flush=True)


def ours(args):
    import torch
    from tqdm import tqdm
    from pgot.eval.eval_recon_oracles import load_model_and_tokenizer
    from pgot.eval.pgot_inference import pgot_forward_eval
    from pgot.model.pgot_utils import build_pred_mask_ovt_owner_eval
    torch.set_num_threads(3)
    torch.manual_seed(args.seed)
    manifest = load_manifest(args)
    root = Path(args.output_dir)/'ours'
    pending = [s for s in manifest['samples'] if not prediction_completed(root/s['dataset']/s['id'],manifest,s)]
    if not pending:
        print('All DeOR predictions already completed.', flush=True)
        return
    model, tokenizer, device, dtype = load_model_and_tokenizer(args)
    towers = model.get_vision_tower_aux_list()
    write_json(Path(args.output_dir)/'ours_model_info.json', dict(checkpoint=args.model_path,
               caption_mode='cached_image_only_AR', dtype=args.dtype, register_eval_route='unrestricted',
               object_memory=getattr(model.config,'pgot_e11_object_memories_per_owner',None),
               register_memory=getattr(model.config,'pgot_e11_register_memories_per_owner',None),
               gt_class_input=False, gt_mask_input=False, gt_count_input=False))
    for sample in tqdm(pending, desc='DeOR original AR ownership'):
        caption = torch.tensor(tokenizer.encode(sample['text'], add_special_tokens=False))[None]
        positions = (caption[0] == tokenizer.convert_tokens_to_ids('<ovt>')).nonzero().flatten()
        if len(positions) != sample['ovt_count']:
            raise ValueError('Tokenized/decode OVT mismatch')
        valid = torch.zeros((1,args.max_objects), dtype=torch.bool)
        pos = torch.zeros((1,args.max_objects),dtype=torch.long)
        pos[0,:len(positions)], valid[0,:len(positions)] = positions, True
        things = torch.zeros_like(valid)
        for pair in sample['pairs']:
            if pair['slot_index'] is not None:
                things[0,pair['slot_index']] = True
        image = load_image(args,sample)
        if len(positions):
            inputs = towers[0].image_processor.preprocess(image,return_tensors='pt')['pixel_values']
            target = (towers[1] if len(towers)>1 else towers[0]).image_processor.preprocess(image,return_tensors='pt')['pixel_values']
            with torch.inference_mode():
                result = pgot_forward_eval(model,images=inputs,target_images=target,caption_input_ids=caption,
                    caption_attention_mask=torch.ones_like(caption,dtype=torch.bool),ovt_positions_in_caption=pos,ovt_valid_mask=valid)
                pred = build_pred_mask_ovt_owner_eval(result['ovt_object_probs'],result['ovt_void_probs'],
                    valid.to(device),things.to(device),args.eval_size,1)[0].cpu().numpy()
            del result
        else:
            pred = np.zeros((args.eval_size,args.eval_size),np.uint16)
        dest=root/sample['dataset']/sample['id'];dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'prediction.npz',prediction=pred.astype(np.uint16))
        write_json(dest/'metadata.json',dict(fingerprint=manifest['fingerprint'],caption_sha256=sample['caption_sha256'],
                    pairs=sample['pairs'], empty_masks=sum(not m.any() for m in masks_for_pairs(pred,sample['pairs']))))


def encode(args):
    import torch
    from tqdm import tqdm
    manifest=load_manifest(args)
    texts=sorted({p['shared_text'] for s in manifest['samples'] for p in s['pairs']})
    pending=[t for t in texts if not embedding_path(args,t).exists()]
    root=Path(args.output_dir);(root/'embeddings').mkdir(exist_ok=True)
    if not texts:
        write_json(root/'embedding_info.json',dict(mode=args.text_mode,unique_texts=0,no_encoder_needed=True))
        return
    if not pending and not (root/'embedding_info.json').exists():
        # A crash after the last embedding but before metadata completion must
        # not be reported as a provenance-verified completed encoding stage.
        pending=texts
    if not pending:
        print('All shared-text embeddings already completed.',flush=True)
        return
    if args.text_mode=='category':
        known=official_categories(args)
        for text in pending:
            temporary=embedding_path(args,text).with_suffix('.tmp.npy')
            np.save(temporary,np.asarray(known[text],dtype=np.float32))
            temporary.replace(embedding_path(args,text))
        info=dict(mode='category',official_asset_sha256=file_digest(Path(args.ctrlo_repo)/'scripts/datasets/data/category_name_to_llama3_emb.pkl'),
                  full_description_used=False,unique_texts=len(texts))
    else:
        from llm2vec import LLM2Vec
        torch.set_num_threads(3)
        encoder=LLM2Vec.from_pretrained(args.llm2vec_base,peft_model_name_or_path=args.llm2vec_adapter,
                    device_map='cuda',torch_dtype=torch.bfloat16,max_length=args.embedding_max_length,
                    doc_max_length=args.embedding_max_length)
        over_limit=[t for t in texts if len(encoder.tokenizer.encode(encoder.prepare_for_tokenization('!@#$%^&*()'+t).replace('!@#$%^&*()','')))>args.embedding_max_length]
        if over_limit:
            raise RuntimeError(f'{len(over_limit)} descriptions exceed embedding max length; increase EMBEDDING_MAX_LENGTH. Full text must not be silently truncated.')
        for start in tqdm(range(0,len(pending),args.embedding_batch_size),desc='Official LLM2Vec shared full text'):
            batch=pending[start:start+args.embedding_batch_size]
            with torch.inference_mode():
                reps=encoder.encode(batch,batch_size=args.embedding_batch_size,show_progress_bar=False).float().cpu().numpy()
            for text,rep in zip(batch,reps):
                if rep.shape!=(4096,) or not np.isfinite(rep).all():
                    raise ValueError('Official encoder must produce finite 4096-D embeddings')
                temporary=embedding_path(args,text).with_suffix('.tmp.npy')
                np.save(temporary,rep)
                temporary.replace(embedding_path(args,text))
        info=dict(mode='description',base=args.llm2vec_base,adapter=args.llm2vec_adapter,
                  dtype='bfloat16',full_description_used=True,max_length=args.embedding_max_length,
                  unique_texts=len(texts),pooling_mode=encoder.pooling_mode,
                  source='official CTRL-O LLM2Vec; no substitute encoder or category-only fallback')
    write_json(root/'embedding_info.json',info)


def ctrlo(args):
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from tqdm import tqdm
    from pgot.eval.figure_ctrlo_examples import load_ctrlo
    manifest=load_manifest(args);root=Path(args.output_dir)/'ctrlo'
    pending=[s for s in manifest['samples'] if not prediction_completed(root/s['dataset']/s['id'],manifest,s)]
    if not pending:
        print('All CTRL-O predictions already completed.',flush=True)
        return
    torch.set_num_threads(3)
    model,info=load_ctrlo(args)
    info.update(query_protocol=f'same generated AR {args.text_mode}; NO GT text or coordinates',
                selection='full split or fixed subset; no GT-quality selection',
                queries_per_pass=args.queries_per_pass,all_queries_retained=True)
    from omegaconf import OmegaConf
    released=OmegaConf.load(Path(args.checkpoint_dir)/'config.yaml')
    info['released_training_shards']=OmegaConf.to_container(released.dataset,resolve=False).get('train_shards')
    write_json(Path(args.output_dir)/'ctrlo_model_info.json',info)
    transform=transforms.Compose([transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    for sample in tqdm(pending,desc='CTRL-O same generated AR text'):
        image=transform(load_image(args,sample))[None].cuda()
        count=len(sample['pairs']);maps=[];groups=[]
        for start in range(0,count,args.queries_per_pass):
            group=sample['pairs'][start:start+args.queries_per_pass]
            seed=int.from_bytes(hashlib.sha256(f'{args.seed}/{sample["dataset"]}/{sample["id"]}/{start}'.encode()).digest()[:4],'little')
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            names=np.zeros((7,4096),np.float32)
            for index,pair in enumerate(group):
                emb=np.load(embedding_path(args,pair['shared_text']))
                if emb.shape!=(4096,) or not np.isfinite(emb).all():
                    raise ValueError('Invalid shared-text embedding')
                names[index]=emb
            # Unbound other slots use the checkpoint's learned slot embeddings;
            # language projection is masked, so their text vector is irrelevant.
            inputs=dict(image=image,batch_size=1,name_embedding=torch.from_numpy(names)[None].cuda(),
                contrastive_loss_mask=torch.tensor([[1]*len(group)+[0]*(7-len(group))],device='cuda'),
                bbox_centroids=torch.full((1,7,2),-1.,device='cuda'),instance_bbox=torch.full((1,7,4),-1.,device='cuda'))
            with torch.inference_mode():
                output=model(inputs={'input':inputs})
                masks=output['object_decoder'].masks_as_image
                if masks.shape!=(1,7,224,224):
                    raise ValueError(f'Unexpected CTRL-O native masks {masks.shape}')
                pred=F.interpolate(masks,size=(args.eval_size,args.eval_size),mode='bilinear',align_corners=False)[0].argmax(0).cpu().numpy().astype(np.uint8)
            maps.append(pred)
            groups.append(dict(start=start,count=len(group),seed=seed,shared_texts=[p['shared_text'] for p in group],
                        query_slots=list(range(len(group))),padding_slots=list(range(len(group),7))))
            del output
        dest=root/sample['dataset']/sample['id'];dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'prediction.npz',prediction=np.stack(maps) if maps else np.zeros((0,args.eval_size,args.eval_size),np.uint8))
        write_json(dest/'metadata.json',dict(fingerprint=manifest['fingerprint'],caption_sha256=sample['caption_sha256'],groups=groups,
                   num_pairs=count,num_passes=len(groups),slot_selection='input-associated slot ONLY; never GT-best slot'))


class GroundTruth:
    def __init__(self,args):
        self.args=args
        self.coco=None
        if 'coco' in args.datasets:
            from pycocotools.coco import COCO
            self.coco=COCO(str(Path(args.coco_root)/'annotations/instances_val2017.json'))

    def __call__(self,sample):
        args=self.args;size=args.eval_size;masks=[];classes=[];identities=[]
        if sample['dataset']=='coco':
            ignore=np.zeros((size,size),bool)
            anns=sorted(self.coco.loadAnns(self.coco.getAnnIds(imgIds=[int(sample['id'])])),key=lambda a:a['id'])
            for ann in anns:
                if ann.get('area',0)<=0:
                    continue
                mask=np.asarray(center_crop(Image.fromarray(self.coco.annToMask(ann)),size,nearest=True))>0
                if ann.get('iscrowd') or ann.get('ignore'):
                    ignore|=mask
                elif mask.any():
                    masks.append(mask);classes.append(self.coco.cats[ann['category_id']]['name']);identities.append(ann['id'])
        else:
            root=Path(args.voc_root)
            with Image.open(root/'SegmentationObject'/f'{sample["id"]}.png') as im:
                inst=np.asarray(center_crop(im,size,nearest=True))
            with Image.open(root/'SegmentationClass'/f'{sample["id"]}.png') as im:
                sem=np.asarray(center_crop(im,size,nearest=True))
            ignore=(inst==255)|(sem==255)
            for label in np.unique(inst):
                if label in [0,255]:
                    continue
                mask=(inst==label)&~ignore
                semantic=sem[mask];semantic=semantic[(semantic>0)&(semantic<=20)]
                if mask.any():
                    if not len(semantic):
                        raise ValueError('VOC instance has no valid thing semantic category')
                    category=int(np.bincount(semantic,minlength=21).argmax())
                    masks.append(mask);classes.append(VOC_CANONICAL[category-1]);identities.append(int(label))
        targets=np.stack(masks) if masks else np.zeros((0,size,size),bool)
        if args.ignore_overlaps:
            ignore|=targets.sum(0)>1
        keep=[i for i,mask in enumerate(targets) if (mask&~ignore).any()]
        return targets[keep],[classes[i] for i in keep],[identities[i] for i in keep],~ignore


def aggregate(rows,thresholds):
    result={'num_images':len(rows),'gt_instances':sum(r['gt_objects'] for r in rows),
            'generated_pairs':sum(r['generated_pairs'] for r in rows),
            'zero_gt_images':sum(r['gt_objects']==0 for r in rows),
            'zero_generated_pair_images':sum(r['generated_pairs']==0 for r in rows)}
    for model in ['deor','ctrlo']:
        result[model]={}
        for threshold in thresholds:
            key=f'{threshold:.2f}';metrics=[r[model]['scores'][key] for r in rows]
            tp,fp,fn=[sum(m[k] for m in metrics) for k in ['tp','fp','fn']]
            result[model][key]=dict(rates(tp,fp,fn),matched_mask_iou_mean=sum(m['matched_iou_sum'] for m in metrics)/tp if tp else None)
        result[model]['empty_masks']=sum(r[model]['empty_masks'] for r in rows)
    pairs=result['generated_pairs']
    result['paired_mask_iou_mean']=sum(r['pair_iou_sum'] for r in rows)/pairs if pairs else None
    result['same_query_gt_alignment_iou50']={key:sum(r['same_query_gt_alignment_iou50'][key] for r in rows)
            for key in ['both_correct_same_instance','both_correct_different_instances','only_deor_correct','only_ctrlo_correct','neither_correct']}
    result['same_query_gt_alignment_iou50']['same_instance_fraction_all_generated_pairs']=(
        result['same_query_gt_alignment_iou50']['both_correct_same_instance']/pairs if pairs else None)
    return result


def score(args):
    from tqdm import tqdm
    manifest=load_manifest(args);root=Path(args.output_dir);gt=GroundTruth(args);rows=[]
    for sample in tqdm(manifest['samples'],desc='Class-mask pair scoring (GT here only)'):
        a=root/'ours'/sample['dataset']/sample['id'];b=root/'ctrlo'/sample['dataset']/sample['id']
        for dest in [a,b]:
            meta=json.loads((dest/'metadata.json').read_text())
            if meta['fingerprint']!=manifest['fingerprint'] or meta['caption_sha256']!=sample['caption_sha256']:
                raise ValueError('Prediction provenance mismatch')
        ours_masks=masks_for_pairs(np.load(a/'prediction.npz')['prediction'],sample['pairs'])
        ctrl_maps=np.load(b/'prediction.npz')['prediction']
        ctrl_masks=ctrlo_pair_masks(ctrl_maps,len(sample['pairs']),args.queries_per_pass)
        targets,classes,ids,valid=gt(sample)
        pred_classes=[p['normalized_category'] for p in sample['pairs']]
        row=dict(dataset=sample['dataset'],id=sample['id'],gt_objects=len(targets),generated_pairs=len(sample['pairs']),
                 gt_classes=classes,gt_instance_ids=ids,pairs=sample['pairs'],
                 malformed_pairs=sum(not p['parsed'] for p in sample['pairs']),
                 unbound_pairs=sum(p['unbound_text'] for p in sample['pairs']),
                 ctrlo_passes=len(ctrl_maps))
        for name,masks in [('deor',ours_masks),('ctrlo',ctrl_masks)]:
            row[name]=dict(scores=score_pairs(masks,pred_classes,targets,classes,valid,args.thresholds),
                           empty_masks=sum(not (mask&valid).any() for mask in masks))
        assignments={name:{m['pair_index']:m['gt_index'] for m in row[name]['scores']['0.50']['matches']}
                    for name in ['deor','ctrlo']}
        alignment=dict(both_correct_same_instance=0,both_correct_different_instances=0,
                       only_deor_correct=0,only_ctrlo_correct=0,neither_correct=0)
        for pair_index in range(len(sample['pairs'])):
            am,bm=assignments['deor'].get(pair_index),assignments['ctrlo'].get(pair_index)
            key=('both_correct_same_instance' if am==bm else 'both_correct_different_instances') if am is not None and bm is not None else (
                'only_deor_correct' if am is not None else 'only_ctrlo_correct' if bm is not None else 'neither_correct')
            alignment[key]+=1
        row['same_query_gt_alignment_iou50']=alignment
        pair_ious=[]
        for am,bm in zip(ours_masks,ctrl_masks):
            intersection=np.count_nonzero(am&bm&valid);union=np.count_nonzero((am|bm)&valid)
            pair_ious.append(intersection/union if union else 0.)
        row['pair_mask_ious']=pair_ious;row['pair_iou_sum']=sum(pair_ious)
        rows.append(row)
    # Rebuild small score reports from complete prediction caches on every run.
    with (root/'per_image.jsonl.tmp').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    (root/'per_image.jsonl.tmp').replace(root/'per_image.jsonl')
    summary=dict(fingerprint=manifest['fingerprint'],protocol=manifest['protocol'],config=manifest['config'],
                 primary_iou_threshold=.5,by_dataset={})
    summary['model_info']={name:json.loads((root/f'{name}_model_info.json').read_text()) for name in ['ours','ctrlo']}
    if (root/'embedding_info.json').exists():
        summary['embedding_info']=json.loads((root/'embedding_info.json').read_text())
    from pgot.eval.voc_object_counts import count_bucket
    for dataset in args.datasets:
        selected=[r for r in rows if r['dataset']==dataset]
        group=aggregate(selected,args.thresholds)
        group['by_object_count']={bucket:aggregate([r for r in selected if count_bucket(r['gt_objects'])==bucket],args.thresholds)
                         for bucket in ['0','1-3','4-6','7-10','11+'] if any(count_bucket(r['gt_objects'])==bucket for r in selected)}
        group['full_split_evaluated']=not args.max_samples and not args.image_ids
        group['total_ctrlo_passes']=sum(r['ctrlo_passes'] for r in selected)
        summary['by_dataset'][dataset]=group
    write_json(root/'summary.json',summary)
    with (root/'comparison.csv.tmp').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['dataset','model','iou_threshold','images','generated_pairs','gt_instances',
                    'tp','fp','fn','precision','recall','f1','matched_mask_iou_mean'])
        writer.writeheader()
        for dataset,result in summary['by_dataset'].items():
            for model in ['deor','ctrlo']:
                for threshold in args.thresholds:
                    writer.writerow(dict(dataset=dataset,model=model,iou_threshold=threshold,images=result['num_images'],
                        generated_pairs=result['generated_pairs'],gt_instances=result['gt_instances'],**result[model][f'{threshold:.2f}']))
    (root/'comparison.csv.tmp').replace(root/'comparison.csv')
    for dataset,result in summary['by_dataset'].items():
        print(dataset,{model:result[model]['0.50'] for model in ['deor','ctrlo']},flush=True)
    print(f'Saved {root/"summary.json"}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['prepare','preflight','ours','encode','ctrlo','score'])
    p.add_argument('--output_dir',required=True)
    p.add_argument('--model_path',default=DEFAULT_MODEL)
    p.add_argument('--caption_root',default='/home/jovyan/PGOT/outputs/segeval_pgot_memory8_full_coco_voc_ar')
    p.add_argument('--coco_root',default='/home/jovyan/data/coco')
    p.add_argument('--voc_root',default='/home/jovyan/data/voc')
    p.add_argument('--ctrlo_repo',default='/home/jovyan/baselines/CTRL-O')
    p.add_argument('--checkpoint_dir',default='/home/jovyan/baselines/ckpt/ctrlo')
    p.add_argument('--datasets',nargs='+',choices=['coco','voc'],default=['coco','voc'])
    p.add_argument('--max_samples',type=int,default=0,help='Per-dataset fixed prefix; 0 = full split')
    p.add_argument('--image_ids',nargs='+',help='Explicit dataset:id strings for smoke only')
    p.add_argument('--text_mode',choices=['description','category'],default='description')
    p.add_argument('--queries_per_pass',type=int,default=6,help='At least one official unbound slot; all overflow queries use more passes')
    p.add_argument('--eval_size',type=int,default=512)
    p.add_argument('--thresholds',nargs='+',type=float,default=[.25,.5,.75])
    p.add_argument('--ignore_overlaps',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--dtype',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max_objects',type=int,default=50)
    p.add_argument('--llm2vec_base',default='McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp')
    p.add_argument('--llm2vec_adapter',default='McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse')
    p.add_argument('--embedding_batch_size',type=int,default=4)
    p.add_argument('--embedding_max_length',type=int,default=1024)
    args=p.parse_args()
    if args.max_samples<0 or args.eval_size<1 or args.max_objects<1 or args.seed<0 or args.embedding_batch_size<1 or args.embedding_max_length<1:
        p.error('Invalid limits/resolution/capacity/seed')
    if not 1<=args.queries_per_pass<=6 or .5 not in args.thresholds or any(not 0<t<=1 for t in args.thresholds):
        p.error('Require 1..6 queries per pass, valid IoU thresholds including 0.5')
    if args.image_ids and args.max_samples or len(set(args.datasets))!=len(args.datasets):
        p.error('No mixed sample selectors or duplicate datasets')
    if args.image_ids and len(set(args.image_ids))!=len(args.image_ids):
        p.error('Duplicate explicit image IDs')
    if len({f'{t:.2f}' for t in args.thresholds})!=len(args.thresholds):
        p.error('Thresholds must remain distinct at two-decimal reporting precision')
    args.diffusion_inference_steps=10  # Loader compatibility only; no diffusion decoding.
    started=time.time()
    globals()[args.stage](args)
    print(f'{args.stage} completed in {time.time()-started:.1f}s',flush=True)


if __name__=='__main__':
    main()
