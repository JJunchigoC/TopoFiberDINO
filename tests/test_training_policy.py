import unittest
from unittest.mock import patch
from uuid import uuid4
import torch
from spinning.io import ROOT,dump,load
from spinning.output import output_path,prepare_output,commit_output,remove_managed,output_session
from spinning.train import train_variant,optimizer_settings


class TrainingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.out=ROOT/'outputs'/('test_policy_'+uuid4().hex)
        self.out.mkdir(parents=True)
        dump(self.out/'.managed.json',dict(owner='fiberdino-triangle'))

    def tearDown(self):
        for path in [self.out,self.out.with_name('.'+self.out.name+'.pending'),self.out.with_name('.'+self.out.name+'.previous')]:
            remove_managed(path)
        lock=self.out.with_name('.'+self.out.name+'.lock')
        if lock.exists(): lock.unlink()

    def test_full_epoch_budget_even_when_validation_worsens(self):
        config=dict(epochs=24,patience=2,log_interval=10,seed=1,learning_rate=.001,weight_decay=.002,batch_size=1,shots=0,representation='triangle_vertices_v2',architecture_revision=3)
        calls=[0]
        def evaluate(*args,**kwargs):
            calls[0]+=1
            return dict(pseudo_supervision_loss=float(calls[0]),pseudo_mask_iou=.4,pseudo_vertex_error_normalized=.1),None,None
        def loss(model,*args,**kwargs):
            value=model.weight.square().mean()
            return value,dict(segmentation=float(value.detach()))
        model=torch.nn.Linear(1,1)
        rows=[dict(split=s) for s in ['train','val','test']]
        (self.out/'baseline').mkdir()
        (self.out/'baseline/best.pt').write_bytes(b'obsolete weights must not skip fresh training')
        with patch('spinning.train.build_model',return_value=model),patch('spinning.train.evaluate',side_effect=evaluate),patch('spinning.train.batch_loss',side_effect=loss),patch('spinning.train.torch.optim.AdamW',wraps=torch.optim.AdamW) as optimizer:
            train_variant('baseline',{},rows,config,self.out,torch.device('cpu'))
        self.assertEqual(optimizer.call_args.kwargs,dict(lr=.001,weight_decay=.002))
        folder=self.out/'baseline'
        summary=load(folder/'training_summary.json')
        self.assertEqual(summary['epochs_run'],24)
        self.assertEqual(summary['best_epoch'],1)
        self.assertFalse(summary['early_stopping'])
        self.assertEqual(len(load(folder/'history.json')),24)
        self.assertEqual(torch.load(folder/'last.pt',weights_only=True)['epoch'],24)

    def test_replace_output_removes_stale_files_and_old_models(self):
        (self.out/'stale.txt').write_text('old')
        (self.out/'old_model').mkdir()
        (self.out/'old_model/unused.pt').write_text('old')
        with output_session(self.out) as out:
            pending=prepare_output(out,False,['full'])
            self.assertTrue((out/'stale.txt').exists())
            (pending/'new.txt').write_text('new')
            dump(pending/'verification.json',dict(artifact_integrity='PASS'))
            commit_output(out,pending)
        self.assertTrue((self.out/'new.txt').exists())
        self.assertFalse((self.out/'stale.txt').exists())
        self.assertFalse((self.out/'old_model').exists())

    def test_validation_trial_never_scores_test_or_collects_predictions(self):
        model=torch.nn.Linear(1,1)
        cfg=dict(epochs=3,seed=1,learning_rate=.001,batch_size=1,shots=0,representation='triangle_vertices_v2',architecture_revision=5)
        rows=[dict(split=s) for s in ['train','val','test']]
        seen=[]
        def evaluate(model,data,ids,*args,**kwargs):
            seen.append(ids.tolist())
            self.assertFalse(kwargs.get('collect',False))
            return dict(pseudo_supervision_loss=.5,pseudo_mask_iou=.4),None,None
        def loss(model,*args,**kwargs): return model.weight.square().mean(),{}
        with patch('spinning.train.build_model',return_value=model),patch('spinning.train.evaluate',side_effect=evaluate),patch('spinning.train.batch_loss',side_effect=loss):
            metrics,probability,points=train_variant('baseline',{},rows,cfg,self.out,torch.device('cpu'),validation_only=True)
        self.assertEqual(seen,[[1]]*4)
        self.assertIsNone(probability); self.assertIsNone(points)
        self.assertTrue((self.out/'baseline/validation_metrics.json').exists())
        self.assertFalse((self.out/'baseline/metrics.json').exists())

    def test_vertex_selection_overrides_conflicting_region_ranking(self):
        cfg=dict(epochs=3,seed=1,learning_rate=.001,batch_size=1,shots=0,representation='triangle_vertices_v2',architecture_revision=5,checkpoint_selection='vertex_error')
        model=torch.nn.Linear(1,1); calls=[0]
        def evaluate(*args,**kwargs):
            i=min(calls[0],2); calls[0]+=1
            return dict(pseudo_supervision_loss=[.1,.2,.3][i],pseudo_vertex_error_normalized=[.3,.1,.2][i],pseudo_mask_iou=.5),None,None
        def loss(model,*args,**kwargs): return model.weight.square().mean(),{}
        with patch('spinning.train.build_model',return_value=model),patch('spinning.train.evaluate',side_effect=evaluate),patch('spinning.train.batch_loss',side_effect=loss):
            train_variant('baseline',{},[dict(split=s) for s in ['train','val','test']],cfg,self.out,torch.device('cpu'),validation_only=True)
        summary=load(self.out/'baseline/training_summary.json')
        state=torch.load(self.out/'baseline/best.pt',weights_only=True)
        self.assertEqual(summary['best_epoch'],2)
        self.assertEqual(summary['epochs_run'],3)
        self.assertEqual(state['selection_metric'],'vertex_error')
        self.assertEqual(state['validation_pseudo_loss'],.2)
        self.assertEqual(state['validation_selection_score'],.1)

    def test_predict_only_preserves_only_requested_checkpoints(self):
        for name in ['full','baseline']:
            (self.out/name).mkdir()
            for filename in ['best.pt','last.pt','history.json','training_summary.json','stale_overlay.jpg']:
                (self.out/name/filename).write_text('saved')
        pending=prepare_output(self.out,True,['full'])
        self.assertEqual((pending/'full/best.pt').read_text(),'saved')
        self.assertFalse((pending/'full/stale_overlay.jpg').exists())
        self.assertFalse((pending/'baseline').exists())

    def test_failed_verification_does_not_replace_previous_results(self):
        (self.out/'keep.txt').write_text('previous successful run')
        pending=prepare_output(self.out,False,['full'])
        dump(pending/'verification.json',dict(artifact_integrity='FAIL'))
        with self.assertRaises(ValueError): commit_output(self.out,pending)
        self.assertTrue((self.out/'keep.txt').exists())

    def test_output_rejects_dataset_root_and_nested_paths(self):
        for path in [ROOT,ROOT/'dataset',ROOT/'outputs',ROOT/'outputs/outer/nested']:
            with self.subTest(path=path),self.assertRaises(ValueError): output_path(path)

    def test_default_is_fixed_500_epochs(self):
        config=load(ROOT/'configs/default.json')
        self.assertEqual(config['epochs'],500)
        self.assertNotIn('patience',config)

    def test_optimizer_legacy_default_and_invalid_settings(self):
        self.assertEqual(optimizer_settings(dict(learning_rate=.001)),dict(lr=.001,weight_decay=.0001))
        for values in [dict(learning_rate=0),dict(learning_rate=float('nan')),dict(learning_rate=.001,weight_decay=-1),dict(learning_rate=.001,weight_decay=float('inf'))]:
            with self.subTest(values=values),self.assertRaises(ValueError): optimizer_settings(values)

    def test_ours_profile_preserves_budget_but_requires_fresh_training(self):
        from main import compatible_run
        ours=load(ROOT/'configs/ours.json')
        old={**ours,'epochs':1000}
        self.assertEqual(ours['variants'],['full'])
        self.assertEqual(ours['epochs'],500)
        self.assertFalse(compatible_run(dict(config=old),dict(config=ours),'full'))
        optimizer=torch.optim.AdamW(torch.nn.Linear(1,1).parameters(),**optimizer_settings(ours))
        self.assertEqual(optimizer.param_groups[0]['lr'],.0003)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'],.001)


if __name__=='__main__': unittest.main()
