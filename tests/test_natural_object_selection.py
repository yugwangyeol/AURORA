import json

import numpy as np
import pytest

from pgot.eval.natural_object_selection import (
    aggregate, annotated_queries, candidates_from_caption, fixed_correspondence,
    parse_selection, score_query, selector_messages,
)


def scene():
    prediction=np.array([[1,1,2,2],[1,1,2,2]])
    text='<thing> Bus 1: a green bus. <ovt>. <thing> Bus 2: a red bus. <ovt>. <scene_end>'
    candidates=candidates_from_caption(text,2,42)
    gt=[dict(segment_id=9,category='bus',description='green bus'),
        dict(segment_id=8,category='bus',description='red bus'),
        dict(segment_id=7,category='dog',description='a dog')]
    targets=np.stack([prediction==1,prediction==2,np.ones(prediction.shape,bool)])
    return prediction,candidates,gt,targets


def test_selector_payload_whitelist_and_description_ablation():
    _,candidates,_,_=scene()
    candidates[0]['gt_mask']='SECRET_GT'
    candidates[0]['coordinates']=[.1,.2]
    for condition in ('category','description'):
        messages=selector_messages('green bus',candidates,condition)
        payload=json.loads(messages[1]['content'])
        assert 'SECRET_GT' not in str(messages)
        for item in payload['candidates']:
            assert set(item)==({'id','category'} if condition=='category' else {'id','category','description'})
    assert selector_messages('q',candidates,'category')[0]==selector_messages('q',candidates,'description')[0]


@pytest.mark.parametrize('raw,status', [('not JSON','invalid_json'),('[]','invalid_schema'),
    ('{"object_id": 0}','invalid_id'),('{"object_id":"fake"}','invalid_id'),
    ('{"object_id":null}','abstain'),('{"object_id":null,"extra":0}','invalid_schema')])
def test_invalid_completion_never_falls_back_to_first_object(raw,status):
    _,candidates,_,_=scene()
    assert parse_selection(raw,candidates)==(None,status)


def test_fixed_mapping_missing_target_and_no_best_slot_repair():
    prediction,candidates,gt,targets=scene()
    mapping,matrix=fixed_correspondence(prediction,candidates,targets,gt)
    assert set(mapping)=={'9','8'}
    queries=annotated_queries(42,gt)
    wrong=score_query(queries[0],mapping['8'],candidates,gt,matrix,mapping)
    assert wrong['iou']==0 and not wrong['selection_correct']
    correct=score_query(queries[0],mapping['9'],candidates,gt,matrix,mapping)
    assert correct['iou']==1 and correct['selection_correct']
    missing=score_query(queries[2],None,candidates,gt,matrix,mapping)
    stats=aggregate([correct,missing])['overall']
    assert stats['target_coverage']==.5
    assert stats['selection_accuracy_present']==1
    assert stats['grounding_success_50']==.5 and stats['mean_iou']==.5


def test_ambiguous_annotations_retained_and_reported():
    gt=[dict(segment_id=i,category='chair',description='a wooden chair') for i in [1,2]]
    queries=annotated_queries(42,gt)
    assert len(queries)==2
    assert all(q['ambiguous_annotation'] and q['same_category_multiple'] for q in queries)
    assert queries[0]['id']!=queries[1]['id']


def test_stuff_label_compaction_and_unbound_tail():
    text='<stuff> Wall 1: a wall. <ovt>. <thing> Bus 1: red bus. <ovt>. <thing> Dog 1: brown dog.'
    candidates=candidates_from_caption(text,2,42)
    bus=next(c for c in candidates if c['category']=='Bus')
    dog=next(c for c in candidates if c['category']=='Dog')
    assert bus['slot_index']==1 and bus['prediction_label']==1
    assert dog['slot_index'] is None and dog['prediction_label'] is None
    assert candidates==candidates_from_caption(text,2,42)
