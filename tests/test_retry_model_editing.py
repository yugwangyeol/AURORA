import unittest
from types import SimpleNamespace

import torch

from pgot.eval.retry_model_editing import reader_with_prior
from pgot.eval.probe_object_editing import condition_from_units
from pgot.model.visual_memory import PGOTOneShotMemoryReader


class EditingRetryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.raw=torch.randn(1,4,8)
        self.objects=[(torch.randn(8),torch.randn(3,8)) for _ in range(2)]
        self.registers=[(torch.randn(8),torch.randn(3,8))]
        self.prior=torch.full((1,4,3),1/3)

    def test_zero_prior_strength_matches_original_learned_reader(self):
        for mode in ['memory_content','memory_id']:
            reader=PGOTOneShotMemoryReader(dim=8,num_heads=2,memories_per_owner=3,readout_mode=mode)
            model=SimpleNamespace(pgot_e8_reader=reader)
            original=condition_from_units(model,self.raw,self.objects,self.registers,2,3)
            got,native,guided=reader_with_prior(model,self.raw,self.objects,self.registers,2,3,self.prior,0.)
            torch.testing.assert_close(got,original,rtol=1e-6,atol=1e-6)
            torch.testing.assert_close(native,guided)

    def test_predicted_prior_changes_attention_without_updating_weights(self):
        reader=PGOTOneShotMemoryReader(dim=8,num_heads=2,memories_per_owner=3)
        model=SimpleNamespace(pgot_e8_reader=reader)
        weights={k:v.clone() for k,v in reader.state_dict().items()}
        prior=self.prior.clone();prior[:,:,1]=.99;prior[:,:,0]=.005;prior[:,:,2]=.005
        _,native,guided=reader_with_prior(model,self.raw,self.objects,self.registers,2,3,prior,2.)
        self.assertGreater(float(guided[:,:,1].mean()),float(native[:,:,1].mean()))
        for key,value in reader.state_dict().items():
            torch.testing.assert_close(value,weights[key])

    def test_invalid_prior_rejected(self):
        reader=PGOTOneShotMemoryReader(dim=8,num_heads=2,memories_per_owner=3)
        model=SimpleNamespace(pgot_e8_reader=reader)
        with self.assertRaises(ValueError):
            reader_with_prior(model,self.raw,self.objects,self.registers,2,3,self.prior[:,:,:2],2.)


if __name__=='__main__':
    unittest.main()
