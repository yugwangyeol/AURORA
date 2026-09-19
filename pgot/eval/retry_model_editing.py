"""Bounded retry of Add/background editing, with separately labeled variants.

Native: unchanged Reader and learned memory bundles. Prediction-guided: same
weights, but predicted spatial ownership contributes a log-prior to Reader
owner logits; selected object Writer pooling also excludes other predicted
regions. This is an inference intervention, not a new trained editing model.
No GT mask, GT caption, external inpainting or RGB/latent paste is used.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from pgot.eval.ar_binding_router import parse_ar_slots
from pgot.eval.figure_model_editing import binary_iou
from pgot.eval.figure_segmentation_examples import write_json


def prepare(args):
    root, out = Path(args.source_dir), Path(args.output_dir)
    if (out/"tasks.json").exists():
        raise FileExistsError("Retry manifest exists; use render to resume")
    original = json.loads((root/"candidates.json").read_text())
    if original['checkpoint'] != args.model_path:
        raise ValueError("Checkpoint mismatch")
    scenes, pool = [], []
    for entry in original['entries']:
        if entry['dataset'] != 'coco':
            continue
        slots = parse_ar_slots(entry['ar_caption']['text'],entry['ar_caption']['ovt_count'])
        if not 1 <= len(slots) <= 5 or any(not s['is_thing'] or not s['parsed'] for s in slots):
            continue
        cache = root/'cache/coco'/entry['id']
        pred = np.load(cache/'ours.npz')['prediction']
        quality = json.loads((cache/'ours.json').read_text())['mIoU']
        if quality < .65:
            continue
        scene = dict(id=entry['id'],entry=entry,pred=pred,slots=slots,quality=quality,
                     categories={s['normalized_category'] for s in slots},bg=float((pred==0).mean()))
        scenes.append(scene)
        counts = Counter(s['normalized_category'] for s in slots)
        for slot in slots:
            mask = pred==slot['prediction_label']
            area = float(mask.mean())
            if counts[slot['normalized_category']]==1 and .10<=area<=.65:
                pool.append(dict(scene,slot=slot,mask=mask,area=area))
    tasks, entries = [], {}
    for mode in ['add','background']:
        if mode=='add':
            targets = [s for s in scenes if s['bg']>=.48 and len(s['slots'])<=3]
            targets.sort(key=lambda s:s['quality']+.25*s['bg'], reverse=True)
        else:
            targets = [s for s in scenes if .38<=s['bg']<=.86 and len(s['slots'])<=2]
            targets.sort(key=lambda s:s['quality']+.15*s['bg'], reverse=True)
        used, used_cats = set(),set()
        count = 0
        for diversity in [True,False]:
            for target in targets:
                if target['id'] in used:
                    continue
                primary = max(target['slots'], key=lambda s:np.count_nonzero(target['pred']==s['prediction_label']))
                category = primary['normalized_category']
                if diversity and category in used_cats:
                    continue
                if mode=='add':
                    donors = [d for d in pool if d['id']!=target['id'] and d['slot']['normalized_category'] not in target['categories']]
                    def donor_score(d):
                        overlap = np.count_nonzero(d['mask']&(target['pred']>0))/max(np.count_nonzero(d['mask']),1)
                        return d['quality']+.6*d['area']-2.5*overlap
                    donors = [d for d in donors if np.count_nonzero(d['mask']&(target['pred']>0))/np.count_nonzero(d['mask'])<=.18]
                else:
                    donors = [d for d in pool if d['id']!=target['id'] and d['slot']['normalized_category']==category
                              and .30<=d['bg']<=.88 and len(d['slots'])<=2]
                    if category!='person':
                        donors = [d for d in donors if 'person' not in d['categories']]
                    target_mask = target['pred']==primary['prediction_label']
                    def donor_score(d):
                        # Compatible pose/scale; no humans/electronics introduced as
                        # arbitrary "background" donors to an animal scene.
                        return 1.7*binary_iou(target_mask,d['mask'])+d['quality']-.4*abs(d['area']-float(target_mask.mean()))
                donors.sort(key=donor_score,reverse=True)
                unique = []
                for donor in donors:
                    if donor['id'] not in {d['id'] for d in unique}:
                        unique.append(donor)
                    if len(unique)==2:
                        break
                if not unique:
                    continue
                used.add(target['id']);used_cats.add(category)
                entries[target['id']]=target['entry']
                for pair, donor in enumerate(unique):
                    entries[donor['id']]=donor['entry']
                    for variant in ['native','predicted_guided']:
                        tasks.append(dict(task_id=f'{mode}_{variant}_{count:02}_{pair}_{target["id"]}',
                            mode=mode,variant=variant,target_id=target['id'],target_slot=primary['slot_index'],
                            target_category=category,target_prediction_label=primary['prediction_label'],
                            donor_id=donor['id'],donor_slot=donor['slot']['slot_index'],
                            donor_category=donor['slot']['normalized_category'],donor_prediction_label=donor['slot']['prediction_label'],
                            reader_prior_strength=0. if variant=='native' else 2.,
                            object_writer_cleanup=variant=='predicted_guided',
                            placement='donor native predicted location; no prompt-controlled placement'))
                count+=1
                if count>=args.candidates:
                    break
            if count>=args.candidates:
                break
    write_json(out/'tasks.json',dict(checkpoint=args.model_path,seed=args.seed,steps=args.diffusion_inference_steps,
        entries=entries,tasks=tasks,selection='GT-assisted source quality; prediction-only pairing, fixed seed, native vs guided separated'))
    print('Prepared',len(tasks),'tasks from',len(entries),'source images',flush=True)


def reader_with_prior(model, raw, objects, registers, obj_mem, reg_mem, prior, strength):
    """Replicate learned one-shot Reader, changing ONLY spatial owner logits.

    Returns condition and diagnostic native/guided attention. Strength zero is
    tested against the original Reader, never used as a new random projection.
    """
    import torch
    import torch.nn.functional as F
    from pgot.model.visual_memory import _build_memory_valid_mask
    reader=model.pgot_e8_reader
    units=list(objects)+list(registers)
    semantic=torch.stack([s for s,_ in units])[None]
    memory=torch.stack([m for _,m in units])[None]
    B,Q,D=raw.shape;S,J=memory.shape[1:3];H=reader.num_heads;Dh=D//H
    valid=torch.ones(B,S,dtype=torch.bool,device=raw.device)
    mv=_build_memory_valid_mask(slot_valid=valid,object_count=len(objects),
        object_memories_per_owner=obj_mem,register_memories_per_owner=reg_mem,max_memories_per_owner=J)
    if prior.shape!=(B,Q,S) or not torch.isfinite(prior).all() or (prior<0).any():
        raise ValueError('Invalid predicted spatial prior')
    dtype=reader.query.weight.dtype
    query=reader.query(reader.query_norm(raw.to(dtype))).reshape(B,Q,H,Dh)
    key=reader.key(reader.semantic_norm(semantic.to(dtype))).reshape(B,S,H,Dh)
    scale=math.sqrt(Dh)*max(reader.temperature,1e-6)
    logits=torch.einsum('bqhd,bshd->bhqs',query.float(),key.float())/scale
    owner_valid=valid&mv.any(-1)
    logits=logits.masked_fill(~owner_valid[:,None,None],-1e4)
    native=F.softmax(logits,-1)
    guided=F.softmax(logits+strength*prior.clamp_min(.025).log()[:,None],-1)
    guided=guided*owner_valid[:,None,None].float()
    guided=guided/guided.sum(-1,keepdim=True).clamp_min(1e-8)
    normalized=reader.memory_norm(memory.to(dtype))
    if reader.readout_mode=='memory_content':
        keys=reader.content_key(normalized).reshape(B,S,J,H,Dh)
    else:
        keys=reader.memory_key_embeddings[None,None].expand(B,S,J,D).reshape(B,S,J,H,Dh)
    inner=torch.einsum('bqhd,bsjhd->bhqsj',query.float(),keys.float())/scale
    beta=F.softmax(inner.masked_fill(~mv[:,None,None],-1e4),-1)*mv[:,None,None].float()
    beta=beta/beta.sum(-1,keepdim=True).clamp_min(1e-8)
    values=reader.value(normalized).reshape(B,S,J,H,Dh)
    context=torch.einsum('bhqsj,bsjhd->bqhd',(guided[...,None]*beta).to(values.dtype),values).reshape(B,Q,D)
    condition=reader.output_norm(reader.output(context)).to(raw.dtype)
    return condition,native.mean(1),guided.mean(1)


def guided_condition(model,a,b,objects,registers,task,obj_mem,reg_mem):
    import torch
    import torch.nn.functional as F
    from pgot.eval.probe_removal_and_instruction import rewrite_memories
    if task['object_writer_cleanup']:
        def clean(out,source_objects):
            winning=torch.cat([out['ovt_object_probs'],out['ovt_void_probs']],1).argmax(1)
            side=int(round(winning.shape[-1]**.5))
            cleaned=[]
            for k,(semantic,_) in enumerate(source_objects):
                owned=F.max_pool2d((winning==k).reshape(1,1,side,side).float(),3,1,1).reshape(1,-1)>0
                if not owned.any():
                    raise ValueError('Empty predicted object for Writer cleanup')
                memory=rewrite_memories(model,out,exclude=~owned)[0,k]
                cleaned.append((semantic,memory))
            return cleaned
        from pgot.eval.probe_object_editing import owner_units
        original,_=owner_units(a)
        objects=clean(a,original)
        if task['mode']=='add':
            donor_objects,_=owner_units(b)
            objects.append(clean(b,donor_objects)[task['donor_slot']])
    valid=a['ovt_object_valid'][0].bool()
    routing=torch.cat([a['ovt_object_probs'][:,valid],a['ovt_void_probs']],1).float()
    Q=a['raw_rae_hidden'].shape[1];side=int(round(Q**.5));p=int(round(routing.shape[-1]**.5))
    routing=F.interpolate(routing.reshape(1,-1,p,p),size=(side,side),mode='bilinear',align_corners=False).flatten(2).transpose(1,2)
    n=int(valid.sum())
    if task['mode']=='add':
        win=torch.cat([b['ovt_object_probs'],b['ovt_void_probs']],1).argmax(1)
        side_b=int(round(win.shape[-1]**.5))
        region=F.interpolate((win==task['donor_slot']).reshape(1,1,side_b,side_b).float(),size=(side,side),mode='nearest').flatten(2).transpose(1,2)
        probability=.90*region+.025*(1-region)
        prior=torch.cat([routing[:,:,:n]*(1-probability),probability,routing[:,:,n:]*(1-probability)],-1)
    else:
        bg=b['ovt_void_probs'].float()
        p=int(round(bg.shape[-1]**.5))
        bg=F.interpolate(bg.reshape(1,-1,p,p),size=(side,side),mode='bilinear',align_corners=False).flatten(2).transpose(1,2)
        bg=bg/bg.sum(-1,keepdim=True).clamp_min(1e-8)
        fg=routing[:,:,:n]
        prior=torch.cat([fg,bg*(1-fg.sum(-1,keepdim=True)).clamp_min(0)],-1)
        region=fg.sum(-1,keepdim=True)
    condition,native,guided=reader_with_prior(model,a['raw_rae_hidden'],objects,registers,obj_mem,reg_mem,prior,task['reader_prior_strength'])
    stats={'variant':'predicted_guided','reader_prior_strength':task['reader_prior_strength'],
        'object_writer_cleanup':task['object_writer_cleanup'],'gt_routing_used':False,
        'native_owner_attention_mean':native.mean(1)[0].cpu().tolist(),
        'guided_owner_attention_mean':guided.mean(1)[0].cpu().tolist()}
    if task['mode']=='add':
        mask=region[0,:,0]>.5
        stats['donor_attention_inside_native']=float(native[0,mask,n].mean())
        stats['donor_attention_inside_guided']=float(guided[0,mask,n].mean())
    return condition,stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','render','select'],required=True)
    p.add_argument('--model_path',default='/home/jovyan/PGOT/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000')
    p.add_argument('--source_dir',default='/home/jovyan/PGOT/outputs/figure_segmentation_over_under')
    p.add_argument('--output_dir',default='/home/jovyan/PGOT/outputs/figure_model_editing_retry_add_background')
    p.add_argument('--owner_cache_dir',default='/home/jovyan/PGOT/outputs/figure_model_editing_memory8_register32_noprior/owner_cache')
    p.add_argument('--candidates',type=int,default=8)
    p.add_argument('--examples',type=int,default=8)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--dtype',default='fp32')
    p.add_argument('--diffusion_inference_steps',type=int,default=20)
    p.add_argument('--guidance_scale',type=float,default=1.)
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--selection_subdir',default='review')
    p.add_argument('--task_filter',default='')
    args=p.parse_args()
    from pgot.eval.figure_model_editing import render,select
    {'prepare':prepare,'render':render,'select':select}[args.stage](args)


if __name__=='__main__':
    main()
