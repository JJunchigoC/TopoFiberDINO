import unittest
import torch
from spinning.triangle import scene_points,scene_logits_from_points,CalibratedSpatialHead
from spinning.train import coordinate_distribution_loss,forward_model,optimizer_settings
from spinning.model import build_model


class CoordinateDistributionTests(unittest.TestCase):
    def test_distribution_cannot_be_enabled_on_a_derived_heatmap_head(self):
        with self.assertRaises(ValueError):
            optimizer_settings(dict(learning_rate=.001,coordinate_head='spatial_integral',distribution_weight=.2))

    def test_scene_inverse_preserves_all_six_coordinates(self):
        torch.manual_seed(9)
        q=(.15+.7*torch.rand(100,6)).requires_grad_()
        points=scene_points(q)
        actual=scene_points(scene_logits_from_points(points).sigmoid())
        torch.testing.assert_close(actual,points,atol=2e-6,rtol=2e-6)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_spatial_c_is_actual_height_not_relative_height(self):
        latent=torch.full((1,3,50,112),-20.)
        for k,y,x in [(0,17,22),(1,20,65),(2,31,65)]: latent[0,k,y,x]=20
        output=CalibratedSpatialHead()(latent)
        actual=scene_points(output['logits'].sigmoid())
        target=torch.tensor([[[22/111,17/49],[65/111,20/49],[65/111,31/49]]])
        torch.testing.assert_close(actual,target,atol=1e-5,rtol=1e-5)

    def test_distribution_supervision_has_finite_independent_gradients(self):
        latent=torch.zeros(2,3,50,112,requires_grad=True)
        output=CalibratedSpatialHead()(latent)
        target=torch.tensor([[[.2,.34],[.6,.4],[.6,.65]]]).repeat(2,1,1)
        loss=coordinate_distribution_loss({'coordinate_log_probability':output['log_probability']},target,torch.tensor([1.,0.]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertGreater(float(latent.grad[0].abs().sum()),0)
        self.assertEqual(float(latent.grad[1].abs().sum()),0)

    def test_new_head_inference_cannot_read_pseudo_targets(self):
        torch.set_num_threads(2)
        config=dict(architecture_revision=5,coordinate_head='calibrated_spatial',topology=dict(enabled=False))
        model=build_model('full',config).eval()
        data=dict(features=torch.rand(1,384,4,8),images=torch.rand(1,6,32,64),points=torch.rand(1,3,2))
        with torch.no_grad():
            a=forward_model(model,data,torch.tensor([0]))['points']
            data['points'].fill_(1000)
            b=forward_model(model,data,torch.tensor([0]))['points']
        torch.testing.assert_close(a,b,atol=0,rtol=0)


if __name__=='__main__': unittest.main()
