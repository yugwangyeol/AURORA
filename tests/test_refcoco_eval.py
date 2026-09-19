import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from pgot.eval.run_refcoco_eval import (
    aggregate, completed, deor_native_mask, file_hash, inference_query, load_full_image,
    make_queries, parse_args, referring_caption, resolve_image, save_prediction,
    score_mask, selected_references,
)


def reference(rid=1, split='val', sentences=None):
    return dict(ref_id=rid, split=split, image_id=42, ann_id=999,
                sentences=sentences or [dict(sent_id=rid*10, raw='Person on the LEFT!')])


class RefCocoEvalTests(unittest.TestCase):
    def test_caption_keeps_entire_original_expression_without_gt_category(self):
        text='Person on the LEFT! Beside the blue bag.'
        caption=referring_caption(text)
        self.assertEqual(caption, f'<thing> Object 1: {text} <ovt>. <scene_end>')
        self.assertEqual(caption.count('<ovt>'),1)

    def test_empty_and_reserved_text_are_not_skipped(self):
        for text in ('', ' ', '<ovt> dog', '<|im_start|>user'):
            with self.assertRaises(ValueError):
                referring_caption(text)

    def test_every_sentence_keeps_own_target_identity_and_original_raw(self):
        refs=[reference(sentences=[dict(sent_id=11,raw='Same person.'),dict(sent_id=10,raw='same person')])]
        queries=make_queries(refs)
        self.assertEqual([q['sent_id'] for q in queries],[10,11])
        self.assertEqual([q['shared_text'] for q in queries],['same person','Same person.'])
        self.assertTrue(all(q['annotation_id']==999 for q in queries))
        self.assertTrue(all(q['ovt_count']==1 for q in queries))
        self.assertEqual(queries[0]['pairs'][0]['shared_text'],'same person')

    def test_duplicate_sentence_id_fails(self):
        with self.assertRaises(ValueError):
            make_queries([reference(),reference()])

    def test_sent_fallback_and_no_text_rewriting(self):
        queries=make_queries([reference(sentences=[dict(sent_id=1,sent='the top-right one')])])
        self.assertEqual(queries[0]['shared_text'],'the top-right one')

    def test_deterministic_prefix_and_split_limits(self):
        refs=[reference(3),reference(1),reference(2),reference(4,'testA')]
        selected=selected_references(refs,['val','testA'],max_refs=2)
        self.assertEqual([r['ref_id'] for r in selected],[1,2,4])
        queries=make_queries(selected,max_samples=1)
        self.assertEqual([q['ref_id'] for q in queries],[1,4])
        with self.assertRaises(ValueError):
            selected_references(refs,['testB'])

    def test_gt_metadata_not_in_inference_allowlist(self):
        query=make_queries([reference()])[0]
        query.update(image_path='/tmp/example.jpg',width=10,height=20,image_sha256='hash')
        allowed=inference_query(query)
        for field in ('annotation_id','image_id','ref_id','sent_id','split','category_id','pairs'):
            self.assertNotIn(field,allowed)
        self.assertEqual(allowed['shared_text'],'Person on the LEFT!')

    def test_train2014_id_maps_to_existing_train2017(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'train2017').mkdir()
            image=root/'train2017'/'000000000042.jpg'
            image.touch()
            self.assertEqual(resolve_image(root,dict(id=42,file_name='COCO_train2014_000000000042.jpg')),image)
            with self.assertRaises(FileNotFoundError):
                resolve_image(root,dict(id=43,file_name='COCO_train2014_000000000043.jpg'))

    def test_full_frame_resize_preserves_both_edges_without_crop(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'image.png'
            pixels=np.zeros((3,9,3),np.uint8)
            pixels[:,:3,0]=255
            pixels[:,-3:,2]=255
            Image.fromarray(pixels).save(p)
            image=load_full_image(dict(image_path=str(p),width=9,height=3,image_sha256=file_hash(p)),6)
            self.assertEqual(image.size,(6,6))
            array=np.asarray(image)
            self.assertEqual(array[3,0,0],255)
            self.assertEqual(array[3,-1,2],255)

    def test_exact_referred_target_not_best_other_object(self):
        target=np.array([[1,0],[0,0]],bool)
        another=np.array([[0,1],[0,0]],bool)
        score=score_mask(another,target)
        self.assertEqual(score['iou'],0)
        self.assertEqual(score['union'],2)

    def test_empty_prediction_zero_iou_and_invalid_gt_fails(self):
        score=score_mask(np.zeros((2,2),bool),np.ones((2,2),bool))
        self.assertEqual(score['iou'],0)
        self.assertTrue(score['empty_mask'])
        with self.assertRaises(ValueError):
            score_mask(np.ones((2,2),bool),np.zeros((2,2),bool))
        with self.assertRaises(ValueError):
            score_mask(np.ones((1,2),bool),np.ones((2,2),bool))

    def test_expression_weighted_miou_pixel_weighted_oiou_and_threshold_boundary(self):
        rows=[dict(image_id=1,ref_id=1,deor=dict(iou=.5,intersection=1,union=2,empty_mask=False)),
              dict(image_id=1,ref_id=1,deor=dict(iou=0,intersection=0,union=8,empty_mask=True))]
        result=aggregate(rows,'deor')
        self.assertEqual(result['mIoU'],.25)
        self.assertEqual(result['oIoU'],.1)
        self.assertEqual(result['precision_at_iou']['0.50'],.5)
        self.assertEqual(result['precision_at_iou']['0.60'],0)
        self.assertEqual(result['num_expressions'],2)
        self.assertEqual(result['num_references'],1)
        self.assertEqual(result['empty_masks'],1)

    def test_single_owner_void_competition_and_native_non_square_output(self):
        import torch
        objects=torch.tensor([[[.9,.1,.9,.1],[100.,100.,100.,100.]]])
        void=torch.tensor([[[.1,.9,.1,.9]]])
        mask=deor_native_mask(objects,void,4,8)
        self.assertEqual(mask.shape,(4,8))
        self.assertTrue(mask[:,0].all())
        self.assertFalse(mask[:,-1].any())
        self.assertEqual(mask.dtype,np.bool_)

    def test_readout_matches_established_single_valid_owner_readout(self):
        import torch
        from pgot.model.pgot_utils import build_pred_mask_ovt_owner_eval
        torch.manual_seed(42)
        objects=torch.rand(1,3,16)
        void=torch.rand(1,2,16)
        valid=torch.tensor([[True,False,False]])
        expected=build_pred_mask_ovt_owner_eval(objects,void,valid,valid,8,1,patch_grid=4)[0].numpy()==1
        np.testing.assert_array_equal(deor_native_mask(objects,void,8,8),expected)

    def test_prediction_resume_checks_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/'ours'
            sample=dict(width=2,height=1,query_sha256='query',image_sha256='image',caption_sha256='caption')
            manifest=dict(fingerprint='version')
            self.assertFalse(completed(folder,manifest,sample))
            save_prediction(folder,np.array([[True,False]],bool),manifest,sample)
            self.assertTrue(completed(folder,manifest,sample))
            with self.assertRaises(ValueError):
                completed(folder,dict(fingerprint='different'),sample)
            with self.assertRaises(ValueError):
                save_prediction(folder,np.ones((2,2),bool),manifest,sample)

    def test_cli_defaults_full_refcoco_and_optional_refcocog_umd(self):
        args=parse_args(['--stage','prepare','--output_dir','/tmp/test'])
        self.assertEqual(args.splits,['val','testA','testB'])
        self.assertEqual(args.split_by,'unc')
        self.assertEqual(args.text_mode,'description')
        self.assertEqual(args.max_refs,0)
        args=parse_args(['--stage','prepare','--output_dir','/tmp/test','--dataset','refcocog'])
        self.assertEqual(args.splits,['val','test'])
        self.assertEqual(args.split_by,'umd')


if __name__=='__main__':
    unittest.main()
