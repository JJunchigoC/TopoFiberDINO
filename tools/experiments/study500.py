"""Validation-only selection and locked 500-epoch six-model comparison.

The test set is not passed to trial trainers. The selected 500-epoch ours
checkpoint is reused for final evaluation, not retrained with a larger budget.
"""
import os
import sys
import shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import cv2
from spinning.io import ROOT,load,dump,sha256
from spinning.prepare import prepare_dataset
from spinning.backbone import extract_features
from spinning.pseudo import prepare_training_tensors
from spinning.mrf import prepare_mrf
from spinning.topology import prepare_topology,export_figures
from spinning.train import train_variant,selection_score
from spinning.output import output_session,prepare_output,commit_output
from spinning.figures import export_results
from spinning.quality import export_quality
from spinning.verify import verify_run

CACHE=ROOT/'.cache/study500'
NAMES=['baseline','directional','full','unet_small','dino_linear','dino_pyramid']

def environment():
    os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    torch.set_num_threads(4); cv2.setNumThreads(4)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare(config,meta,device,reuse=False):
    folder=CACHE/'preprocessing'; folder.mkdir(parents=True,exist_ok=True)
    features=extract_features(meta,config,ROOT/'.cache/dino_840x378',device)
    data=prepare_training_tensors(meta,features,config,folder)
    data.update(prepare_mrf(data['images'],data['valid'],config['mrf']['color_sigma']))
    data.update(prepare_topology(data,meta['frames'],config,folder,reuse=reuse))
    return data

def select(config,meta,cpu,device):
    keep=torch.tensor([i for i,r in enumerate(meta['frames']) if r['split']!='test'])
    mapping={old:new for new,old in enumerate(keep.tolist())}
    data={k:(v if k=='topology_centers' else v[keep]).to(device) for k,v in cpu.items()}
    data['neighbors']=torch.tensor([mapping[int(cpu['neighbors'][i])] for i in keep],device=device)
    rows=[meta['frames'][i] for i in keep]
    assert len(rows)==173 and all(r['split']!='test' for r in rows)
    specs=[('old_spatial','spatial_integral',0.),('corrected','calibrated_spatial',0.),
           ('distribution015','calibrated_spatial',.15),('distribution060','calibrated_spatial',.6)]
    results={}
    for name,head,weight in specs:
        cfg={**config,'variants':['full'],'coordinate_head':head,'distribution_weight':weight}
        folder=CACHE/'trials'/name; folder.mkdir(parents=True,exist_ok=True)
        dump(folder/'config.json',cfg)
        print(f'VALIDATION ONLY {name}: 500 epochs, 134 train + 39 val; 19 test excluded',flush=True)
        metrics,_,_=train_variant('full',data,rows,cfg,folder,device,validation_only=True)
        results[name]=dict(config=cfg,metrics=metrics,summary=load(folder/'full/training_summary.json'),checkpoint_sha256=sha256(folder/'full/best.pt'))
        dump(CACHE/'validation_trials.json',dict(test_evaluated=False,trials=results))
    chosen=min(results,key=lambda n:selection_score(results[n]['metrics'],results[n]['config']))
    selection=dict(selected=chosen,rule='Minimum validation normalized vertex error. Same 500-epoch budget per trial. No test scores used.',
                   trial_train_ids=[r['id'] for r in rows if r['split']=='train'],trial_validation_ids=[r['id'] for r in rows if r['split']=='val'],
                   test_evaluated_before_selection=False,historical_test_seen=True,
                   tuning_budget=dict(ours_trials=4,epochs_per_trial=500,baseline_hyperparameter_trials=0),
                   dataset_sha256=sha256(ROOT/'dataset/metadata.json'),trials=results)
    dump(CACHE/'selection.json',selection)
    dump(ROOT/'docs/records/study500_selection.json',selection)
    print('LOCKED VALIDATION SELECTION: '+chosen,flush=True)
    return selection

def publish(selection,meta,cpu,device):
    selected=selection['selected']; config={**selection['trials'][selected]['config'],'variants':NAMES}
    assert selection['dataset_sha256']==sha256(ROOT/'dataset/metadata.json')
    assert selected==min(selection['trials'],key=lambda n:selection_score(selection['trials'][n]['metrics'],selection['trials'][n]['config']))
    assert sha256(CACHE/'trials'/selected/'full/best.pt')==selection['trials'][selected]['checkpoint_sha256']
    assert len(load(CACHE/'trials'/selected/'full/history.json'))==config['epochs']==500
    config['name']='TopoFiberDINO Coordinate (ours)'
    # Persist the decision and settings before any final test evaluation.
    dump(ROOT/'configs/default.json',config)
    dump(ROOT/'configs/ours.json',{**config,'variants':['full']})
    dump(ROOT/'configs/ours_ablation.json',{**config,'variants':['full'],'name':'TopoFiberDINO Coordinate without topology loss (ours ablation)',
         'topology':{**config['topology'],'weight':0.,'proportion_weight':0.}})
    with output_session(ROOT/'outputs/triangle_ours') as out:
        pending=prepare_output(out,False,NAMES)
        for item in (CACHE/'preprocessing').iterdir():
            if item.is_dir(): shutil.copytree(item,pending/item.name,dirs_exist_ok=True)
            else: shutil.copy2(item,pending/item.name)
        shutil.copytree(CACHE/'trials'/selected/'full',pending/'full')
        dump(pending/'run.json',dict(config=config,dataset_sha256=sha256(ROOT/'dataset/metadata.json'),annotation_hashes={}))
        dump(pending/'selection_protocol.json',selection)
        dump(pending/'execution.json',dict(mode='locked_500_epoch_comparison',retrained_models=[n for n in NAMES if n!='full'],reused_models=['full'],
             epochs_per_model=500,early_stopping=False,replaces_previous_output=True,ours_checkpoint_source='selected validation-only 500-epoch trial'))
        data={k:v.to(device) for k,v in cpu.items()}; variants={}
        for name in NAMES:
            print('FINAL MODEL '+name+' 500 epochs'+(' (reuse locked 500-epoch validation trial)' if name=='full' else ''),flush=True)
            metrics,probability,points=train_variant(name,data,meta['frames'],config,pending,device,predict_only=name=='full')
            variants[name]=dict(metrics=metrics,probability=probability,points=points)
        export_results(meta,config,cpu,variants,pending)
        export_quality(meta,cpu,variants,pending)
        export_figures(pending)
        from spinning.study import export_study
        export_study(pending)
        verify_run(pending)
        commit_output(out,pending)
    print('COMPLETE: six models at 500 epochs each; output replaced atomically.',flush=True)

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--stage',choices=['select','publish','all'],default='all')
    args=parser.parse_args(); config=load(ROOT/'configs/ours.json'); config['epochs']=500
    CACHE.mkdir(parents=True,exist_ok=True); device=environment(); meta=prepare_dataset(); cpu=prepare(config,meta,device,reuse=args.stage=='publish')
    selection=select(config,meta,cpu,device) if args.stage!='publish' else load(CACHE/'selection.json')
    if args.stage!='select': publish(selection,meta,cpu,device)

if __name__=='__main__': main()
