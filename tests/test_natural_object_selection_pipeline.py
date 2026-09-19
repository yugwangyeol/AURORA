import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from pgot.eval.run_natural_object_selection import prepare, score, write_json


def test_cpu_prepare_score_and_provenance(tmp_path):
    model=tmp_path/'model';model.mkdir()
    (model/'config.json').write_text('{}')
    (model/'fake.safetensors').write_bytes(b'fixture-not-real-weights')
    write_json(model/'model.safetensors.index.json',{'weight_map':{'weight':'fake.safetensors'}})
    source=tmp_path/'annotations.jsonl'
    mask=np.zeros((4,4,3),np.uint8)
    mask[:,:2,0]=1;mask[:,2:,0]=2
    Image.fromarray(mask).save(tmp_path/'gt.png')
    Image.new('RGB',(4,4),'white').save(tmp_path/'image.png')
    record=dict(image_id=42,image_path=str(tmp_path/'image.png'),panoptic_mask_path=str(tmp_path/'gt.png'),
                segments=[dict(segment_id=1,category='bus',description='a green bus',is_thing=True),
                          dict(segment_id=2,category='bus',description='a red bus',is_thing=True)])
    source.write_text(json.dumps(record)+'\n')
    train=tmp_path/'train.jsonl';train.write_text('{"image_id":999}\n')
    captions=tmp_path/'captions.jsonl'
    captions.write_text(json.dumps(dict(image_id=42,text='<thing> Bus 1: green bus. <ovt>. <scene_end>',ovt_count=1))+'\n')
    caption_summary=tmp_path/'caption_summary.json'
    write_json(caption_summary,dict(ckpt=str(model),caption_mode='autoregressive'))
    root=tmp_path/'output'
    args=SimpleNamespace(output_dir=str(root),model_path=str(model),annotations=str(source),
          train_annotations=str(train),captions=str(captions),caption_summary=str(caption_summary),
          queries=None,eval_size=4,seed=0,image_ids='',max_images=None,presence_iou=.1,
          dtype='fp32',max_objects=50,max_caption_tokens=4096,cache_bundles=False,save_examples=2)
    prepare(args)
    prepared=json.loads((root/'manifest.json').read_text())
    prepare(args)  # identical configuration resumes without rewriting predictions
    assert prepared['protocol']['train_image_overlap_count']==0
    sample=prepared['samples'][0];candidate=sample['candidates'][0]['id']
    folder=root/'owners/42';folder.mkdir(parents=True)
    Image.new('RGB',(4,4),'white').save(folder/'image.png')
    prediction=np.zeros((4,4),np.uint16);prediction[:,:2]=1
    np.savez_compressed(folder/'prediction.npz',prediction=prediction)
    write_json(folder/'metadata.json',dict(fingerprint=prepared['fingerprint'],caption_sha256=sample['caption_sha256']))
    common=dict(fingerprint=prepared['fingerprint'],selector_model='fake-selector',model_config_sha256='a',
        weights_index_sha256='b',tokenizer_config_sha256='c',system_sha256='d',max_new_tokens=64,
        max_input_tokens=8192,do_sample=False,seed=0,dtype='bfloat16')
    for condition in ('category','description'):
        dest=root/'selections'/condition
        write_json(dest/'selector_info.json',dict(spec=dict(common,condition=condition),selection_fingerprint=condition))
        for query in prepared['queries']:
            selected=None if condition=='category' else candidate
            write_json(dest/'queries'/f"{query['id']}.json",dict(query_id=query['id'],selection_fingerprint=condition,
                       selected_id=selected,status='abstain' if selected is None else 'ok'))
    score(args)
    result=json.loads((root/'summary.json').read_text())
    assert result['conditions']['description']['overall']['mean_iou']==.5
    assert result['conditions']['oracle_id']['overall']['target_coverage']==.5
    assert result['conditions']['category']['overall']['grounding_success_50']==0
    assert len(list((root/'examples').glob('*/selection_overlay.png')))==2
    assert (root/'comparison.csv').is_file() and (root/'results.md').is_file()
    (model/'fake.safetensors').write_bytes(b'changed-weights')
    with pytest.raises(ValueError,match='changed after preparation'):
        score(args)
