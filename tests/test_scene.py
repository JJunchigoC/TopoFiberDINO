import unittest
import torch
from spinning.triangle import scene_points,triangle_mask
from spinning.train import translated_batch,optimizer_settings,bounded_shift
from spinning.model import build_model


class SceneTests(unittest.TestCase):
    def test_apex_can_be_above_base_and_geometry_is_always_ordered(self):
        q=torch.rand(10000,6)
        q=torch.cat([q,torch.zeros(1,6),torch.ones(1,6)])
        points=scene_points(q)
        a,b,c=points.unbind(1)
        cross=(b[:,0]-a[:,0])*(c[:,1]-a[:,1])-(b[:,1]-a[:,1])*(c[:,0]-a[:,0])
        self.assertTrue((cross>0).all())
        self.assertTrue((a[:,0]<torch.minimum(b[:,0],c[:,0])).all())
        self.assertTrue((b[:,1]<c[:,1]).all())
        self.assertTrue(((points>=0)&(points<=1)).all())
        self.assertTrue((a[:,1]<b[:,1]).any())
        for p in points[:50]: triangle_mask(p.numpy()*[447,199],(200,448))

    def test_formerly_unrepresentable_target_and_all_coordinate_gradients(self):
        # A above B and base x below .45 were both excluded by the old head.
        target=torch.tensor([[[.24,.34],[.42,.38],[.42,.60]]])
        q=torch.tensor([[(.24-.10)/.25,(.42-.37)/.41,.5,(.38-.24)/.34,(.60-.38-.06)/(.84-.38-.06),(.34-.26)/.34]],requires_grad=True)
        torch.testing.assert_close(scene_points(q),target)
        (scene_points(q)*torch.tensor([[[1.,2.],[3.,4.],[5.,6.]]])).sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue((q.grad.abs()>0).all())

    def test_translation_moves_pixels_points_and_excludes_padding(self):
        x=torch.zeros(1,1,11,21); x[0,0,5,10]=1
        data=dict(features=x,images=x,valid=torch.ones_like(x),target=x,confidence=torch.ones_like(x),points=torch.tensor([[[.5,.5],[.7,.4],[.7,.7]]]),has_vertices=torch.ones(1,1,1,1))
        result=translated_batch(data,torch.tensor([0]),torch.tensor([[.1,-.1]]))
        self.assertEqual(float(result['target'][0,0,4,12]),1)
        torch.testing.assert_close(result['points'],data['points']+torch.tensor([.1,-.1]))
        self.assertEqual(float(result['valid'][:,:,:,-1].sum()),10)
        self.assertEqual(float(result['confidence'][:,:,:,:2].sum()),0)
        self.assertEqual(float(data['target'][0,0,5,10]),1)

    def test_translation_does_not_reintroduce_unrepresentable_targets(self):
        points=torch.tensor([[[.3333,.3],[.7702,.29],[.7702,.77]]])
        shift=bounded_shift(points,torch.tensor([[.02,-.025]]))
        moved=points+shift[:,None]
        self.assertLess(float(moved[0,1,0]),.78)
        self.assertLess(float(moved[0,0,0]),.35)
        self.assertGreaterEqual(float(moved[0,1,1]),.24)
        safe=bounded_shift(points,torch.tensor([[-.01,.01]]))
        torch.testing.assert_close(safe,torch.tensor([[-.01,.01]]))

    def test_zero_topology_residual_preserves_local_prediction_and_learns(self):
        torch.set_num_threads(2)
        config=dict(architecture_revision=5,feature_dropout=0,topology=dict(enabled=True,embedding_dim=16,grid_size=[8,4]))
        model=build_model('full',config).eval()
        f=torch.randn(2,384,4,8); x=torch.rand(2,6,32,64); valid=torch.ones(2,1,32,64)
        model.topology.centers.copy_(torch.nn.functional.normalize(torch.randn(4,384),dim=-1))
        expected=scene_points(model.coordinate_head(model.encoder(f,x)).sigmoid())
        output=model(f,x,valid)
        torch.testing.assert_close(output['points'],expected)
        output['points'].square().sum().backward()
        self.assertGreater(float(model.topology_residual.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is not None for p in model.coordinate_head.parameters()))

    def test_reject_invalid_regularization_settings(self):
        for cfg in [dict(feature_dropout=1),dict(translation_weight=-1),dict(translation_range=[.2,.01]),dict(topology_residual_scale=float('nan'))]:
            with self.subTest(cfg=cfg),self.assertRaises(ValueError): optimizer_settings(dict(learning_rate=.001,**cfg))

    def test_spatial_head_uses_image_evidence_and_preserves_three_points(self):
        from spinning.triangle import SpatialCoordinateHead
        head=SpatialCoordinateHead()
        uniform=torch.zeros(1,3,50,112)
        peak=uniform.clone(); peak[0,0,18,16]=15; peak.requires_grad_(True)
        a=scene_points(head(uniform).sigmoid())
        b=scene_points(head(peak).sigmoid())
        self.assertLess(float(b[0,0,0]),float(a[0,0,0]))
        self.assertLess(float(b[0,0,1]),float(a[0,0,1]))
        b.sum().backward()
        self.assertGreater(float(peak.grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(peak.grad).all())
        triangle_mask(b.detach().numpy()[0]*[447,199],(200,448))
        config=dict(architecture_revision=5,coordinate_head='spatial_integral',topology=dict(enabled=False))
        self.assertEqual(build_model('full',config).encoder.refine[-1].out_channels,3)
        self.assertEqual(build_model('baseline',config).encoder.refine[-1].out_channels,1)

    def test_inference_never_substitutes_automatic_target_coordinates(self):
        from spinning.train import forward_model
        config=dict(architecture_revision=5,coordinate_head='spatial_integral',topology=dict(enabled=False))
        model=build_model('full',config).eval()
        data=dict(features=torch.randn(1,384,4,8),images=torch.rand(1,6,32,64),valid=torch.ones(1,1,32,64),points=torch.zeros(1,3,2),target=torch.zeros(1,1,32,64))
        with torch.no_grad():
            before=forward_model(model,data,torch.tensor([0]))['points']
            data['points'].fill_(999); data['target'].fill_(1)
            after=forward_model(model,data,torch.tensor([0]))['points']
        torch.testing.assert_close(before,after,rtol=0,atol=0)


if __name__=='__main__': unittest.main()
