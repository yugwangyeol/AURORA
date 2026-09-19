import json
import pickle
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from pgot.eval.natural_object_selection import candidates_from_caption, selector_messages
from pgot.eval.run_natural_object_selection import write_json
from pgot.eval.run_refcoco_object_selection import (
    manifest, metrics, owner_prediction, parse_args, prepare, score, selected_mask,
)


def fixture(tmp_path):
    model=tmp_path/'model'; model.mkdir()
    (model/'config.json').write_text('{}')
    (model/'fake.safetensors').write_bytes(b'fixture')
    write_json(model/'model.safetensors.index.json',{'weight_map':{'weight':'fake.safetensors'}})
    root=tmp_path/'coco'; (root/'refcoco').mkdir(parents=True); (root/'train2017').mkdir()
    Image.new('RGB',(8,4),'white').save(root/'train2017/000000000042.jpg')
    refs=[dict(ref_id=1,split='val',image_id=42,ann_id=99,
               sentences=[dict(sent_id=10,raw='Bus on the LEFT!'),dict(sent_id=11,raw='the green bus')])]
    with (root/'refcoco/refs(unc).p').open('wb') as stream: pickle.dump(refs,stream)
    instances=dict(images=[dict(id=42,file_name='COCO_train2014_000000000042.jpg',width=8,height=4)],
        annotations=[dict(id=99,image_id=42,category_id=1,iscrowd=0,
                          area=32,bbox=[0,0,8,4],segmentation=[[0,0,8,0,8,4,0,4]])],
        categories=[dict(id=1,name='bus')])
    write_json(root/'refcoco/instances.json',instances)
    train=tmp_path/'train.jsonl'; train.write_text('{"image_id":42}\n')
    return parse_args(['--stage','prepare','--output_dir',str(tmp_path/'out'),
        '--model_path',str(model),'--data_root',str(root),'--image_root',str(root),
        '--train_annotations',str(train)])


def test_prepare_preserves_raw_queries_but_does_not_query_condition_extraction(tmp_path):
    args=fixture(tmp_path); prepare(args)
    root,m=manifest(args)
    assert [q['request'] for q in m['queries']]==['Bus on the LEFT!','the green bus']
    assert len(m['samples'])==1
    assert set(m['samples'][0])=={'image_id','image_path','width','height','image_sha256'}
    assert m['protocol']['train_image_overlap_count']==1
    assert m['protocol']['full_split_evaluated'] is True
    assert not (root/'gt').exists() # no GT mask decoding at preparation
    prepare(args)


def test_manifest_rejects_changed_source_or_settings(tmp_path):
    args=fixture(tmp_path); prepare(args)
    args.ar_max_new_tokens+=1
    with pytest.raises(ValueError,match='changed'): manifest(args)


def test_baseline_receives_same_exact_expressions_and_independent_manifest(tmp_path):
    args=fixture(tmp_path)
    ckpt=tmp_path/'ctrlo'; ckpt.mkdir()
    (ckpt/'pretrained_model.ckpt').write_bytes(b'fixture')
    (ckpt/'config.yaml').write_text('{}')
    args.checkpoint_dir=str(ckpt); args.with_ctrlo=True
    prepare(args)
    root,m=manifest(args)
    baseline=json.loads((root/'baseline/manifest.json').read_text())
    assert [s['shared_text'] for s in baseline['samples']]==[q['request'] for q in m['queries']]
    assert baseline['config']['text_mode']=='description'
    assert baseline['fingerprint']!=m['fingerprint']


def test_native_competition_matches_existing_readout_at_square_resolution():
    from pgot.model.pgot_utils import build_pred_mask_ovt_owner_eval
    object_probs=torch.tensor([[[.7,.1,.1,.7],[.1,.7,.1,.1],[.1,.1,.7,.1]]])
    void_probs=torch.full((1,2,4),.05)
    valid=torch.tensor([[True,True,False]])
    candidates=[dict(slot_index=0,prediction_label=1),dict(slot_index=1,prediction_label=2)]
    actual=owner_prediction(object_probs,void_probs,valid,candidates,6,6)
    expected=build_pred_mask_ovt_owner_eval(object_probs,void_probs,valid,valid,6,1)[0].numpy()
    np.testing.assert_array_equal(actual,expected)
    assert owner_prediction(object_probs,void_probs,valid,candidates,3,9).shape==(3,9)


def test_stuff_competes_without_returning_a_thing_region():
    obj=torch.tensor([[[.2,.2,.2,.2],[.7,.7,.7,.7]]])
    void=torch.full((1,1,4),.1)
    actual=owner_prediction(obj,void,torch.tensor([[True,True]]),
        [dict(slot_index=0,prediction_label=1)],4,8)
    assert not actual.any()


def test_nonfinite_owner_maps_rejected():
    with pytest.raises(ValueError,match='Nonfinite'):
        owner_prediction(torch.full((1,1,4),float('nan')),torch.zeros(1,1,4),torch.ones(1,1,dtype=torch.bool),[],4,4)


def test_unbound_invalid_and_null_selections_have_empty_masks():
    prediction=np.ones((4,8),np.uint16)
    candidates=[dict(id='good',prediction_label=1),dict(id='unbound',prediction_label=None)]
    assert selected_mask(prediction,candidates,'good').all()
    for selected in (None,'unknown','unbound'):
        assert not selected_mask(prediction,candidates,selected).any()


def test_selector_has_no_gt_image_geometry_or_slot_ids():
    candidates=candidates_from_caption('<thing> Bus 1: a green bus. <ovt>. <scene_end>',1,42)
    candidates[0].update(annotation_id=99,image_path='/secret',bbox=[0,0,8,4])
    for condition in ('category','description'):
        payload=json.loads(selector_messages('the green bus',candidates,condition)[1]['content'])
        assert set(payload['candidates'][0])==({'id','category','description'} if condition=='description' else {'id','category'})
        assert payload['request']=='the green bus'


def test_metrics_count_failures_and_do_not_gate_iou_by_correspondence():
    rows=[dict(image_id=42,iou=.8,intersection=8,union=10,empty_mask=False,
               target_present=False,selection_correct=False),
          dict(image_id=42,iou=0,intersection=0,union=10,empty_mask=True,
               target_present=False,selection_correct=False)]
    result=metrics(rows)
    assert result['mIoU']==.4 and result['oIoU']==.4
    assert result['success_50']==.5 and result['target_coverage']==0
    assert result['selection_accuracy_present'] is None


def test_ctrlo_has_no_candidate_id_selection_accuracy():
    stats=metrics([dict(condition='ctrlo',image_id=42,iou=1,intersection=10,union=10,
                        empty_mask=False,target_present=True,selection_correct=False)])
    assert stats['mIoU']==1
    assert stats['target_coverage'] is None
    assert stats['selection_accuracy_present'] is None


def test_cpu_score_actual_selected_region_not_gt_best_or_class_gate(tmp_path):
    args=fixture(tmp_path); prepare(args); root,m=manifest(args)
    sample=m['samples'][0]
    # Deliberately wrong generated category: class-based coverage is zero,
    # but actual region matches GT. Main IoU must NOT be zeroed by that fact.
    text='<thing> Dog 1: a green object. <ovt>. <scene_end>'
    candidates=candidates_from_caption(text,1,42)
    sample.update(caption=text,caption_sha256='fixture',ovt_count=1,candidates=candidates)
    write_json(root/'manifest.json',m)
    folder=root/'owners/42'; folder.mkdir(parents=True)
    np.savez_compressed(folder/'prediction.npz',prediction=np.ones((4,8),np.uint16))
    write_json(folder/'metadata.json',dict(fingerprint=m['fingerprint'],image_sha256=sample['image_sha256'],
                                         ar_record=dict(scene_end_generated=True)))
    for condition in ('category','description'):
        dest=root/'selections'/condition
        write_json(dest/'selector_info.json',dict(spec=dict(condition=condition,fingerprint=m['fingerprint'],model='fixture'),
                                                 selection_fingerprint=condition))
        for query in m['queries']:
            write_json(dest/'queries'/f'{query["id"]}.json',dict(query_id=query['id'],selection_fingerprint=condition,
                selected_id=candidates[0]['id'] if condition=='description' else None,status='ok'))
    args.save_examples=0
    score(args)
    result=json.loads((root/'summary.json').read_text())['by_split']['val']
    assert result['description']['mIoU']==1 and result['description']['target_coverage']==0
    assert result['category']['mIoU']==0 and result['category']['num_expressions']==2
    assert result['oracle_id']['mIoU']==0
    (root/'selections/category/queries'/f'{m["queries"][0]["id"]}.json').unlink()
    with pytest.raises(FileNotFoundError): score(args) # cannot report partial full split


@pytest.mark.parametrize('options',[
    ['--max_refs','-1'],['--max_refs','1','--max_samples','1'],
    ['--splits','train'],['--splits','val','val'],['--eval_size','224'],
])
def test_invalid_protocol_rejected(options):
    with pytest.raises(SystemExit): parse_args(['--stage','prepare','--output_dir','/unused']+options)


def test_launcher_covers_all_stages_and_separates_old_direct_protocol():
    text=(Path(__file__).parents[1]/'scripts/run_refcoco_object_selection.sh').read_text()
    for stage in ('prepare','extract','select','encode','ctrlo','score'):
        assert f'run_stage {stage}' in text
    assert 'run_refcoco_object_selection' in text
    assert 'GPU="${GPU:-1}"' in text
    assert 'run_refcoco_eval --stage ours' not in text


@pytest.mark.parametrize('failure',[False,True])
def test_real_shell_stage_order_gpu_isolation_and_failure_propagation(tmp_path,failure):
    runner=tmp_path/'fake_python'
    runner.write_text('''#!/usr/bin/env python3
import json,os,sys
a=sys.argv
stage=a[a.index('--stage')+1]
condition=a[a.index('--condition')+1] if '--condition' in a else None
print('EVENT:'+json.dumps(dict(stage=stage,condition=condition,gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                             pythonpath=os.environ['PYTHONPATH'])),flush=True)
if os.environ.get('FAIL_CATEGORY')=='1' and condition=='category': sys.exit(7)
''')
    runner.chmod(0o755)
    env=dict(os.environ,DEOR_PYTHON=str(runner),SELECTOR_PYTHON=str(runner),CTRLO_PYTHON=str(runner),
             GPU='1',OUTPUT_DIR=str(tmp_path/'out'),WITH_CTRLO='1',
             SELECTOR_SITE_PACKAGES=str(tmp_path/'selector_packages'),FAIL_CATEGORY='1' if failure else '0')
    path=Path(__file__).parents[1]/'scripts/run_refcoco_object_selection.sh'
    result=subprocess.run(['bash',str(path)],env=env,text=True,capture_output=True)
    events=[json.loads(line[6:]) for line in result.stdout.splitlines() if line.startswith('EVENT:')]
    assert all(e['gpu']=='1' for e in events)
    assert all((str(tmp_path/'selector_packages') in e['pythonpath']) == (e['stage']=='select') for e in events)
    if failure:
        assert result.returncode!=0
        assert not any(e['stage']=='score' for e in events)
    else:
        assert result.returncode==0
        assert [e['stage'] for e in events]==['prepare','preflight','extract','select','select','encode','ctrlo','score']
