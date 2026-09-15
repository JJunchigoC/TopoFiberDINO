import unittest
from unittest.mock import patch
import torch
from spinning.mrf import prepare_mrf,mrf_energy,settings
from spinning.triangle import soft_triangle
from spinning.train import batch_loss
from main import compatible_run
from spinning.comparison import comparable_reference


class MRFTests(unittest.TestCase):
    def energy(self,logits,image,valid):
        data=prepare_mrf(image,valid,.1)
        return mrf_energy(logits,data['mrf_weights'],data['mrf_normalizer'])

    def test_expected_binary_potts_energy_and_contrast(self):
        logits=torch.tensor([[[[-20.,20.]]]])
        image=torch.zeros(1,3,1,2); valid=torch.ones(1,1,1,2)
        energy=self.energy(logits,image,valid)
        self.assertAlmostEqual(energy.item(),1/(4*2**.5),places=6)
        image[:,:,:,1]=1
        self.assertLess(self.energy(logits,image,valid).item(),energy.item()/1000)
        self.assertLess(self.energy(torch.full_like(logits,20),torch.zeros_like(image),valid).item(),1e-6)

    def test_padding_has_no_edges_and_does_not_dilute_loss(self):
        logits=torch.tensor([[[[-20.,20.]]]])
        image=torch.zeros(1,3,1,2); valid=torch.ones(1,1,1,2)
        original=self.energy(logits,image,valid)
        pad=torch.nn.functional.pad
        padded=self.energy(pad(logits,(2,2,2,2),value=10),pad(image,(2,2,2,2)),pad(valid,(2,2,2,2)))
        self.assertAlmostEqual(original.item(),padded.item(),places=6)
        empty=self.energy(logits.requires_grad_(),image,valid*0)
        empty.backward()
        self.assertEqual(empty.item(),0)
        self.assertTrue(torch.equal(logits.grad,torch.zeros_like(logits)))

    def test_diagonal_edges_and_triangle_coordinate_gradients(self):
        image=torch.zeros(1,3,32,64); valid=torch.ones(1,1,32,64)
        data=prepare_mrf(image,valid,.1)
        self.assertAlmostEqual(data['mrf_weights'][0,2,10,10].item(),2**-.5,places=6)
        points=torch.tensor([[[.2,.5],[.7,.3],[.72,.75]]],requires_grad=True)
        output=soft_triangle(points,(32,64))
        self.energy(output['mask'],image,valid).backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertGreater(points.grad.abs().sum().item(),0)

    def test_mrf_enters_only_ours_training_objective(self):
        image=torch.zeros(1,6,16,32); valid=torch.ones(1,1,16,32)
        points=torch.tensor([[[.2,.5],[.7,.3],[.72,.75]]],requires_grad=True)
        output=soft_triangle(points,(16,32))
        data=dict(images=image,features=torch.zeros(1,1),target=output['mask'].sigmoid().detach(),
                  confidence=valid,valid=valid,points=points.detach(),has_vertices=torch.ones(1,1,1,1))
        data.update(prepare_mrf(image,valid,.1))
        config=dict(coordinate_weight=5,mrf=dict(weight=.05))
        model=lambda *args: output
        with patch.dict('spinning.train.MODEL_INFO',{'full':dict(auxiliary=False,temporal=False)}):
            base,_=batch_loss(model,data,torch.tensor([0]),config,'baseline')
            full,parts=batch_loss(model,data,torch.tensor([0]),config,'full')
        self.assertGreater(parts['mrf_pairwise'],0)
        self.assertAlmostEqual((full-base).item(),.05*parts['mrf_pairwise'],places=6)

    def test_reuse_controls_but_reject_old_ours_and_changed_data(self):
        old=dict(config=dict(variants=['full'],epochs=1000,name='old'),dataset_sha256='a')
        new=dict(config=dict(variants=['full','baseline'],epochs=1000,name='MRF',mrf=dict(weight=.05)),dataset_sha256='a')
        self.assertTrue(compatible_run(old,new,'baseline'))
        self.assertFalse(compatible_run(old,new,'full'))
        self.assertTrue(comparable_reference(old,new))
        new['dataset_sha256']='b'
        self.assertFalse(comparable_reference(old,new))
        self.assertFalse(compatible_run(old,new,'baseline'))

    def test_settings_validation(self):
        self.assertEqual(settings({})['weight'],0)
        for value in [dict(weight=-1),dict(weight=float('nan')),dict(color_sigma=0),dict(kind='fake')]:
            with self.subTest(value=value),self.assertRaises(ValueError): settings(dict(mrf=value))


if __name__=='__main__': unittest.main()
