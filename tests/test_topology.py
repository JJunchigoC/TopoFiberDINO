import unittest
from uuid import uuid4
import numpy as np
import torch
from spinning.io import ROOT,load,dump
from spinning.output import remove_managed,prepare_output
from spinning.topology import persistence_histogram,fit_clusters,pooled_inputs,prepare_topology,topology_loss,settings
from spinning.model import build_model
from spinning.triangle import triangle_mask


class TopologyTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2)

    def test_persistent_hole_in_ring_and_empty_cluster(self):
        y,x=np.mgrid[:21,:21]; xy=np.stack([x/20,y/20],-1)
        valid=np.ones((21,21),bool)
        ring=(np.maximum(abs(x-10),abs(y-10))==6)
        bins=[0,.02,.05,.1,.2,.4,1.5]
        hist=persistence_histogram(ring,valid,xy,1,bins)
        self.assertGreater(hist[1].sum(),0)
        self.assertGreater(hist[1,3:].sum(),0)
        self.assertEqual(persistence_histogram(ring*False,valid,xy,1,bins).sum(),0)
        self.assertEqual(persistence_histogram(valid,valid,xy,1,bins)[1].sum(),0)

    def test_clustering_does_not_fit_validation_or_padding(self):
        features=torch.randn(3,384,4,8); valid=torch.ones(3,1,8,16)
        valid[:,:,:,12:]=0
        values,keep=pooled_inputs(features,valid,[8,4])
        a=fit_clusters(values,keep,[0],3,123)
        values[1:]=1000*torch.randn_like(values[1:]); values[0][~keep[0]]=999
        b=fit_clusters(values,keep,[0],3,123)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_topology_branch_shares_coordinates_and_receives_auxiliary_gradients(self):
        config=load(ROOT/'configs/ours.json')
        config['topology'].update(grid_size=[8,4],embedding_dim=16)
        model=build_model('full',config)
        model.topology.centers.copy_(torch.nn.functional.normalize(torch.randn(4,384),dim=-1))
        features=torch.randn(2,384,4,8); images=torch.rand(2,6,32,64); valid=torch.ones(2,1,32,64)
        output=model(features,images,valid=valid)
        for points in output['points'].detach().numpy(): triangle_mask(points*[63,31],(32,64))
        output['points'].sum().backward(retain_graph=True)
        if hasattr(model,'topology_residual'):
            self.assertGreater(model.topology_residual.weight.grad.abs().sum().item(),0)
            # Zero-init deliberately blocks coordinate gradients into the global
            # encoder on step zero; they enter after the residual starts learning.
            with torch.no_grad(): model.topology_residual.weight.add_(.01*torch.randn_like(model.topology_residual.weight))
            model.zero_grad(); output=model(features,images,valid=valid)
            output['points'].sum().backward(retain_graph=True)
        self.assertGreater(model.topology.project.weight.grad.abs().sum().item(),0)
        model.zero_grad()
        data=dict(topology_hist=torch.zeros(2,4,2,8),topology_proportions=torch.full((2,4),.25))
        loss,_=topology_loss(output,data,torch.arange(2),config)
        loss.backward()
        self.assertGreater(model.topology.project.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_fitted_topology_is_reused_and_normalization_is_train_only(self):
        out=ROOT/'outputs'/('test_topology_'+uuid4().hex); out.mkdir()
        dump(out/'.managed.json',dict(owner='fiberdino-triangle'))
        pending=out.with_name('.'+out.name+'.pending')
        try:
            cfg=load(ROOT/'configs/ours.json'); cfg['topology'].update(grid_size=[8,4],clusters=2)
            rows=[dict(id=str(i),split=s) for i,s in enumerate(['train','val','test'])]
            data=dict(features=torch.randn(3,384,4,8),valid=torch.ones(3,1,8,16))
            a=prepare_topology(data,rows,cfg,out)
            state=torch.load(out/'topology/fitted.pt',weights_only=True)
            raw=np.array([r['histogram'] for r in load(out/'topology/targets.json')['frames']])
            np.testing.assert_equal(state['histogram_scale'],np.maximum(raw[0].max(0),1))
            # Exercise the same copying path used by --predict-only.
            (out/'full').mkdir()
            for f in ['best.pt','last.pt','history.json','training_summary.json']: (out/'full'/f).write_bytes(b'test')
            prepare_output(out,True,['full'])
            b=prepare_topology(data,rows,cfg,pending,reuse=True)
            for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        finally:
            remove_managed(out); remove_managed(pending)

    def test_context_ignores_invalid_tokens_and_requires_valid_mask(self):
        cfg=load(ROOT/'configs/ours.json'); cfg['topology'].update(grid_size=[8,4],embedding_dim=16)
        model=build_model('full',cfg).eval()
        f=torch.randn(1,384,4,8); valid=torch.ones(1,1,4,8); valid[:,:,:,-2:]=0
        with torch.no_grad():
            a=model.topology(f,valid)[0]
            f[:,:,:,-2:]=9999
            b=model.topology(f,valid)[0]
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        with self.assertRaises(ValueError): model(f,torch.zeros(1,6,32,64))

    def test_settings_reject_invalid_topology(self):
        for v in [dict(grid_size=[1,2]),dict(clusters=0),dict(bins=[0,1,.5]),dict(weight=-1),dict(embedding_dim=7)]:
            with self.subTest(v=v),self.assertRaises(ValueError): settings(dict(topology=v))


if __name__=='__main__': unittest.main()
