import json
from pathlib import Path

from PIL import Image
import pytest

from pgot.eval.object_representation_gallery import ASSETS,COMPARABLE,assemble
from pgot.eval.run_natural_object_selection import write_json


def fixture(tmp_path):
    sources=[]
    for index in range(2):
        root=tmp_path/f'source{index}';folder=root/'pairs'/f'pair{index}';folder.mkdir(parents=True)
        conf={k:'same' for k in COMPARABLE}
        conf.update(writer_rewrite=False,reader_prior=False,pixel_paste=False)
        write_json(root/'manifest.json',dict(config=conf,fingerprint=f'fp{index}'))
        for name in ASSETS:
            if name.endswith('.png'):Image.new('RGB',(32,32),(20+index,80,150)).save(folder/name)
        write_json(folder/'metadata.json',dict(pair_id=f'pair{index}',fingerprint=f'fp{index}'))
        sources.append(dict(pair_id=f'pair{index}',source_root=str(root),label='test',note='not benchmark'))
    selection=tmp_path/'selection.json';write_json(selection,dict(scope='test',examples=sources))
    return selection,sources


def test_gallery_copies_native_outputs_exactly_and_resumes(tmp_path):
    selection,sources=fixture(tmp_path);out=tmp_path/'delivery'
    before=Path(sources[0]['source_root'])/'pairs/pair0/bundle_swap_native.png'
    original=before.read_bytes()
    assemble(selection,out);assemble(selection,out)
    assert (out/'01_pair0/bundle_swap_native.png').read_bytes()==original
    assert before.read_bytes()==original
    assert (out/'overview.jpg').is_file() and (out/'review_overview.jpg').is_file()
    report=json.loads((out/'selection.json').read_text())
    assert report['selection_is_qualitative'] is True
    assert report['generated_outputs_copied_without_pixel_changes'] is True


def test_gallery_rejects_mixed_protocols_before_writing(tmp_path):
    selection,sources=fixture(tmp_path);source=Path(sources[1]['source_root'])/'manifest.json'
    value=json.loads(source.read_text());value['config']['steps']=99;write_json(source,value)
    with pytest.raises(ValueError,match='different inference protocols'):assemble(selection,tmp_path/'delivery')
    assert not (tmp_path/'delivery').exists()


def test_gallery_does_not_hide_writer_rewriting(tmp_path):
    selection,sources=fixture(tmp_path);source=Path(sources[1]['source_root'])/'manifest.json'
    value=json.loads(source.read_text());value['config']['writer_rewrite']=True;write_json(source,value)
    with pytest.raises(ValueError,match='native-Reader'):assemble(selection,tmp_path/'delivery')
