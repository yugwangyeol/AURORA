import unittest
import numpy as np
from pgot.eval.figure_segmentation_examples import diagnose, contingency, color_map, score


class DiagnosticsTests(unittest.TestCase):
    def test_proper_split_merge_and_background(self):
        gt = np.array([[1,1,2,2],[1,1,2,2],[0,0,0,0]])
        ignore = np.zeros_like(gt,dtype=bool)
        proper = diagnose(gt,gt,ignore,min_area=2)
        self.assertEqual((proper['split_count'],proper['merge_count']),(0,0))
        split = gt.copy();split[0,:2]=3
        self.assertEqual(diagnose(gt,split,ignore,min_area=2)['split_count'],1)
        merge = gt.copy();merge[merge==2]=1
        self.assertEqual(diagnose(gt,merge,ignore,min_area=2)['merge_count'],1)
        background = gt.copy();background[2]=[3,4,5,6]
        self.assertEqual(diagnose(gt,background,ignore,min_area=2)['split_count'],0)

    def test_ignored_pixels_and_sparse_labels(self):
        gt=np.array([[1,1,4,4],[1,1,4,4]])
        pred=gt.copy();pred[0,:2]=7
        ignore=np.zeros_like(gt,dtype=bool);ignore[0,:2]=True
        self.assertEqual(diagnose(gt,pred,ignore,min_area=2)['split_count'],0)
        inter,iou=contingency(gt,pred,ignore)
        self.assertEqual(inter.shape,(5,9))
        self.assertTrue(np.isfinite(iou).all())

    def test_display_preserves_predictions(self):
        gt=np.array([[0,1],[2,2]])
        pred=np.array([[3,2],[1,1]])
        before=pred.copy()
        rgb=color_map(pred,gt,np.zeros_like(gt,dtype=bool))
        np.testing.assert_array_equal(pred,before)
        self.assertEqual(rgb.shape,(2,2,3))

    def test_scores_match_existing_metric_functions(self):
        import torch
        from pgot.eval.pgot_metrics import fari_metric, mbo_metric, miou_metric
        gt=np.array([[0,1,1],[2,2,0],[4,4,0]],dtype=np.int64)
        pred=np.array([[0,2,2],[1,1,0],[3,4,0]],dtype=np.int64)
        ignore=np.zeros_like(gt,dtype=bool);ignore[2,1]=True
        result=score(gt,pred,ignore)
        tensors=[torch.from_numpy(x)[None] for x in (gt,pred,ignore)]
        for key,fn in [('fARI',fari_metric),('mBO',mbo_metric),('mIoU',miou_metric)]:
            self.assertAlmostEqual(result[key],fn(*tensors),places=6)


if __name__=='__main__': unittest.main()
