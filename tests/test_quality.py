import unittest
import numpy as np
import torch
from spinning.model import MODEL_INFO,build_model
from spinning.quality import mask_metrics,review_priority
from main import compatible_run


class QualityTests(unittest.TestCase):
    def test_six_architectures_propagate_gradients(self):
        torch.set_num_threads(2)
        for name in MODEL_INFO:
            with self.subTest(model=name):
                model=build_model(name)
                features=torch.randn(2,384,4,8,requires_grad=True)
                image=torch.randn(2,6,32,64)
                output=model(features,image)
                self.assertEqual(output['mask'].shape,(2,1,32,64))
                output['mask'].square().mean().backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                if name=='unet_small': self.assertIsNone(features.grad)
                else: self.assertIsNotNone(features.grad)

    def test_known_overlap_and_padding(self):
        a=np.zeros((20,20),bool); b=a.copy(); valid=np.ones_like(a)
        a[5:10,5:10]=True; b[5:10,6:11]=True
        result=mask_metrics(a,b,valid)
        self.assertAlmostEqual(result['iou'],20/30)
        self.assertAlmostEqual(result['dice'],.8)
        self.assertEqual(result['boundary_f1'],1.)
        self.assertEqual(result['hd95_roi_px'],1.)
        valid[:3]=False; a[:3]=True
        self.assertEqual(result,mask_metrics(a,b,valid))

    def test_empty_metrics_do_not_invent_distances(self):
        a=np.zeros((12,12),bool); valid=np.ones_like(a)
        both=mask_metrics(a,a,valid)
        self.assertEqual(both['iou'],1.)
        self.assertIsNone(both['assd_roi_px'])
        b=a.copy(); b[3:6,3:6]=True
        one=mask_metrics(a,b,valid)
        self.assertEqual(one['dice'],0.)
        self.assertIsNone(one['precision'])
        self.assertIsNone(one['hd95_roi_px'])
        self.assertEqual(one['boundary_f1'],0.)

    def test_priority_handles_missing_and_empty(self):
        self.assertEqual(review_priority(1,0,0,0),0)
        self.assertEqual(review_priority(1,None,None,0),0)
        self.assertEqual(review_priority(1,0,0,0,True),100)
        self.assertGreater(review_priority(.5,.5,.1,.2),review_priority(.9,.1,.01,0))

    def test_adding_models_preserves_training_compatibility(self):
        old=dict(config=dict(variants=['full'],epochs=80),dataset_sha256='x')
        new=dict(config=dict(variants=['full','unet_small'],epochs=80),dataset_sha256='x',annotation_hashes={})
        self.assertTrue(compatible_run(old,new))
        new['config']['epochs']=81
        self.assertFalse(compatible_run(old,new))


if __name__=='__main__': unittest.main()
