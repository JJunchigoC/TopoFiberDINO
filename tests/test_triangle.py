import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from spinning.triangle import triangle_mask,soft_triangle,annotation
from spinning.model import build_model,MODEL_INFO
from spinning.geometry import measure,transform
from spinning.io import ROOT


class TriangleTests(unittest.TestCase):
    def test_exact_three_points_and_area(self):
        vertices=np.array([[10.,30.],[90.,10.],[90.,50.]])
        mask=triangle_mask(vertices,(64,100))
        self.assertEqual(mask[30,10],1)
        self.assertEqual(mask[30,90],1)
        values=measure(vertices)
        self.assertAlmostEqual(values['triangle_area'],1600)
        self.assertNotIn('envelope_area',values)
        self.assertEqual(len(vertices),3)

    def test_reject_invalid_not_clip(self):
        vertices=np.array([[10.,30.],[90.,10.],[90.,50.]])
        valid=np.ones((64,100),bool); valid[:,85:]=False
        with self.assertRaises(ValueError): triangle_mask(vertices,valid.shape,valid)
        with self.assertRaises(ValueError): triangle_mask(vertices[[0,2,1]],valid.shape)
        with self.assertRaises(ValueError): triangle_mask(np.zeros((4,2)),valid.shape)

    def test_soft_render_and_coordinate_gradients(self):
        points=torch.tensor([[[.1,.5],[.9,.2],[.9,.8]]],requires_grad=True)
        result=soft_triangle(points,(64,100))
        self.assertGreater(float(result['mask'][0,0,32,60].detach()),0)
        self.assertLess(float(result['mask'][0,0,2,2].detach()),0)
        result['mask'].sigmoid().mean().backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertGreater(float(points.grad.abs().sum()),0)

    def test_all_six_models_output_valid_ordered_triangles(self):
        torch.set_num_threads(2)
        for name in MODEL_INFO:
            with self.subTest(model=name):
                model=build_model(name)
                result=model(torch.randn(2,384,4,8),torch.rand(2,6,32,64))
                self.assertEqual(result['points'].shape,(2,3,2))
                for points in result['points'].detach().numpy(): triangle_mask(points*[63,31],(32,64))
                result['mask'].square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.coordinate_head.parameters()))

    def test_annotation_json_and_native_coordinate_scaling(self):
        with patch('spinning.triangle.load',return_value={'vertices_roi':[[10,30],[90,10],[90,50]]}),patch.object(Path,'exists',return_value=True):
            points=[[10,30],[90,10],[90,50]]
            mask,actual,_=annotation(ROOT/'tests','frame',(64,100),np.ones((64,100),bool))
            np.testing.assert_equal(actual,points)
            self.assertGreater(mask.sum(),0)
            original=transform(actual,[[2,0,3],[0,2,7]])
            self.assertAlmostEqual(measure(original)['triangle_area'],4*measure(actual)['triangle_area'])


if __name__=='__main__': unittest.main()
