import unittest
import numpy as np
import torch
from spinning.prepare import apportioned_counts
from spinning.geometry import transform,measure
from spinning.train import segmentation_loss
from spinning.model import FiberAdapter
from spinning.pseudo import temporal_pairs


class PipelineTests(unittest.TestCase):
    def test_split_integer_allocation(self):
        self.assertEqual(apportioned_counts(192),[134,39,19])
        for n in [1,9,94,98,192,201]:
            self.assertEqual(sum(apportioned_counts(n)),n)

    def test_geometry_and_units(self):
        vertices=np.array([[0.,0.],[4.,-3.],[4.,3.]])
        result=measure(vertices,vertices)
        self.assertAlmostEqual(result['triangle_area'],12)
        self.assertAlmostEqual(result['base_width'],6)
        scaled=transform(vertices,[[.1,0,2],[0,.1,3],[0,0,1]])
        result=measure(scaled,scaled)
        self.assertAlmostEqual(result['triangle_area'],.12)

    def test_projection_roundtrip(self):
        matrix=np.array([[1.1,.1,34],[-.1,1.1,-450],[0,0,1]])
        points=np.array([[30,500],[170,550],[350,600]])
        np.testing.assert_allclose(transform(transform(points,matrix),np.linalg.inv(matrix)),points,atol=1e-10)

    def test_empty_mask_not_a_triangle(self):
        with self.assertRaises(ValueError):
            measure(np.zeros((3,2)))

    def test_invalid_pixels_cannot_change_training_loss(self):
        logits=torch.zeros(2,1,8,8)
        target=torch.zeros_like(logits)
        valid=torch.ones_like(logits)
        valid[:,:,:,4:]=0
        a=segmentation_loss(logits,target,valid)
        logits[:,:,:,4:]=100
        target[:,:,:,4:]=1
        self.assertAlmostEqual(float(a),float(segmentation_loss(logits,target,valid)))

    def test_decoder_heads_and_backward(self):
        torch.set_num_threads(2)
        model=FiberAdapter()
        features=torch.randn(2,384,3,6)
        image=torch.rand(2,6,24,48)
        output=model(features,image)
        self.assertEqual(output.shape,(2,1,24,48))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_temporal_pairs_do_not_cross_splits(self):
        rows=[dict(sequence='V01',source_frame_number=i,time_seconds=i/30,split='train' if i<2 else 'val') for i in range(4)]
        neighbors,_,_=temporal_pairs(rows,torch.rand(4,6,24,32),torch.ones(4,1,24,32))
        self.assertEqual(neighbors,[1,0,3,2])


if __name__=='__main__':
    unittest.main()
