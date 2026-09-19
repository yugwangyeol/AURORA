import pytest

from pgot.eval.ar_binding_router import parse_ar_slots
from pgot.eval.object_representation_transfer import replace_description, transfer_objects


def test_description_replacement_preserves_slot_identity_and_other_chunks():
    text='<thing> Dog 1: brown dog. <ovt>. <thing> Dog 2: white dog. <ovt>. <scene_end>'
    edited=replace_description(text,1,'striped dog.')
    assert text.split('<ovt>')[0]==edited.split('<ovt>')[0]
    assert text.split('<ovt>')[2]==edited.split('<ovt>')[2]
    slots=parse_ar_slots(edited,2)
    assert slots[1]['description']=='striped dog.'
    assert slots[1]['category']=='Dog'
    assert slots[0]['description']=='brown dog.'


@pytest.mark.parametrize('description',['','bad <ovt>','bad <thing>','bad <scene_end>'])
def test_invalid_description_rejected(description):
    with pytest.raises(ValueError):replace_description('<thing> Dog 1: dog. <ovt>',0,description)


@pytest.mark.parametrize('mode,expected',[
    ('reconstruction',('sa','ma')),('text_only',('st','ma')),
    ('semantic_only',('sb','ma')),('visual_only',('sa','mb')),('bundle_swap',('sb','mb'))])
def test_transfer_changes_only_requested_components(mode,expected):
    untouched=(object(),object())
    objects=[untouched,('sa','ma')]
    result=transfer_objects(objects,1,[('sb','mb')],0,mode,'st')
    assert result[0] is untouched
    assert result[1]==expected
    assert objects[1]==('sa','ma')


def test_text_only_cannot_silently_use_donor_semantic_state():
    with pytest.raises(ValueError):transfer_objects([('a','a')],0,[('b','b')],0,'text_only')


def test_bad_slots_rejected():
    with pytest.raises(ValueError):transfer_objects([('a','a')],3,[('b','b')],0,'bundle_swap')


def test_outline_does_not_cover_object_appearance():
    import numpy as np
    from PIL import Image
    from pgot.eval.run_object_representation_transfer import outline
    mask=np.zeros((12,12),bool);mask[1:11,1:11]=True
    original=np.full((12,12,3),[220,140,30],np.uint8)
    rendered=np.asarray(outline(Image.fromarray(original),mask))
    np.testing.assert_array_equal(rendered[4:8,4:8],original[4:8,4:8])
    np.testing.assert_array_equal(rendered[~mask],original[~mask])
    assert not np.array_equal(rendered[mask],original[mask])


def test_cross_category_text_control_changes_name_not_owner_identity():
    text='<thing> Person 1: sitting person. <ovt>. <thing> Snowboard 4: white board. <ovt>. <scene_end>'
    edited=replace_description(text,1,'yellow board with a star.','Surfboard')
    assert '<thing> Surfboard 4:' in edited
    assert edited.split('<ovt>')[0]==text.split('<ovt>')[0]
    assert edited.split('<ovt>')[2]==text.split('<ovt>')[2]
    slots=parse_ar_slots(edited,2)
    assert slots[1]['category']=='Surfboard'
    assert slots[1]['slot_index']==1 and slots[1]['description']=='yellow board with a star.'


@pytest.mark.parametrize('category',['','bad: name','<ovt>','bad\nname'])
def test_cross_category_cannot_inject_new_schema_markers(category):
    with pytest.raises(ValueError):replace_description('<thing> Dog 1: brown dog. <ovt>',0,'board.',category)


def test_prepare_cross_pair_uses_generated_category_and_description(tmp_path):
    import json
    from types import SimpleNamespace
    from PIL import Image
    from pgot.eval.run_object_representation_transfer import prepare
    from pgot.eval.run_natural_object_selection import write_json
    model=tmp_path/'model';model.mkdir()
    (model/'config.json').write_text('{}')
    (model/'fake.safetensors').write_bytes(b'fixture')
    write_json(model/'model.safetensors.index.json',{'weight_map':{'weight':'fake.safetensors'}})
    images=[]
    for iid in (1,2):
        path=tmp_path/f'{iid}.png';Image.new('RGB',(4,4),'white').save(path)
        images.append(dict(image_id=iid,image_path=str(path),segments=[{'description':'SECRET_GT'}]))
    annotations=tmp_path/'annotations.jsonl';annotations.write_text('\n'.join(json.dumps(r) for r in images))
    captions=tmp_path/'captions.jsonl'
    captions.write_text('\n'.join(json.dumps(r) for r in [
        dict(image_id=1,text='<thing> Snowboard 1: white snowboard. <ovt>. <scene_end>',ovt_count=1),
        dict(image_id=2,text='<thing> Surfboard 1: yellow star surfboard. <ovt>. <scene_end>',ovt_count=1)]))
    summary=tmp_path/'caption_summary.json';write_json(summary,dict(ckpt=str(model),caption_mode='autoregressive'))
    pairs=tmp_path/'pairs.jsonl';pairs.write_text(json.dumps(dict(pair_id='board',target_image_id=1,
        target_slot=0,donor_image_id=2,donor_slot=0,target_category='snowboard',donor_category='surfboard',
        instruction='replace the board')))
    args=SimpleNamespace(output_dir=str(tmp_path/'out'),model_path=str(model),pairs=str(pairs),
        annotations=str(annotations),captions=str(captions),caption_summary=str(summary),seed=1234,
        diffusion_inference_steps=10,guidance_scale=1.,max_caption_tokens=4096)
    prepare(args)
    result=json.loads((tmp_path/'out/manifest.json').read_text())
    pair=result['pairs'][0]
    assert pair['cross_category'] is True
    assert pair['target_prediction_label']==1 and pair['donor_prediction_label']==1
    assert '<thing> Surfboard 1: yellow star surfboard.' in pair['text_only_caption']
    assert 'SECRET_GT' not in json.dumps(result)
    prepare(args)  # immutable manifest resumes unchanged
