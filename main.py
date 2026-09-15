"""One command: train selected models for a fixed epoch budget and replace outputs."""
import argparse
import os
from pathlib import Path
import random
import numpy as np
import torch
from spinning.io import ROOT,load,dump,sha256
from spinning.prepare import prepare_dataset
from spinning.model import MODEL_INFO
from spinning.output import output_session,prepare_output,commit_output


def compatible_run(previous,current,variant=None):
    """Requested model subset can change; training settings and data must match."""
    a={**previous,'config':dict(previous['config'])}
    b={**current,'config':dict(current['config'])}
    a.setdefault('annotation_hashes',{})
    b.setdefault('annotation_hashes',{})
    a['config'].pop('variants',None)
    b['config'].pop('variants',None)
    # Display names do not affect training. MRF is used only by full.
    for item in [a,b]:
        item['config'].pop('name',None)
        if variant is not None and variant!='full': item['config'].pop('mrf',None)
    return a==b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/default.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/triangle_ours')
    parser.add_argument('--epochs',type=int,help='Exact epochs per model; default 500, no early stopping')
    parser.add_argument('--shots',type=int,help='Genuine training annotations; 0 uses automatic targets')
    parser.add_argument('--predict-only',action='store_true',help='Keep compatible weights and replace generated results')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--retrain',action='store_true',help='Alias for default fresh training; replaces the same output directory')
    parser.add_argument('--models',nargs='+',choices=list(MODEL_INFO))
    parser.add_argument('--retrain-models',nargs='+',choices=list(MODEL_INFO),help='Freshly train these models, reuse compatible controls, and replace all selected outputs')
    args=parser.parse_args()
    config=load(args.config)
    from spinning.topology import settings as topology_settings,prepare_topology,export_figures as export_topology_figures,compare_ablation
    topo=topology_settings(config)
    if config.get('representation')!='triangle_vertices_v2' or config.get('architecture_revision') not in [3,4,5]:
        parser.error('Use triangle_vertices_v2 with architecture_revision 3, 4 or 5.')
    if topo['enabled'] and config['architecture_revision'] not in [4,5]: parser.error('Topology requires architecture_revision=4 or 5')
    MODEL_INFO['full']['label']=config.get('name','FiberDINO Triangle-MRF (ours)')
    if args.models: config['variants']=list(dict.fromkeys(args.models))
    if args.epochs is not None: config['epochs']=args.epochs
    if args.shots is not None: config['shots']=args.shots
    if config['epochs']<1 or config['shots']<0 or config.get('log_interval',25)<1:
        parser.error('epochs/log_interval must be positive; shots must be nonnegative')
    if not config['variants'] or any(name not in MODEL_INFO for name in config['variants']):
        parser.error('Invalid model selection in configuration')
    if args.predict_only and args.retrain: parser.error('--predict-only and --retrain are mutually exclusive')
    if args.retrain_models and (args.predict_only or args.retrain): parser.error('--retrain-models cannot be combined with --predict-only or --retrain')
    if args.retrain_models and not set(args.retrain_models)<=set(config['variants']): parser.error('--retrain-models must be included in --models/config variants')
    from spinning.mrf import settings as mrf_settings,prepare_mrf
    mrf=mrf_settings(config)
    from spinning.train import optimizer_settings
    optimizer_settings(config)  # Validate before touching any generated results.
    metadata=prepare_dataset()
    print(f'Dataset: {metadata["counts"]}; no ROI pixels changed.',flush=True)
    if args.prepare_only: return
    os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
    os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
    torch.set_num_threads(4)
    torch.manual_seed(config['seed']); random.seed(config['seed']); np.random.seed(config['seed'])
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import cv2
    cv2.setNumThreads(4)
    hashes={p.relative_to(ROOT/'dataset').as_posix():sha256(p) for p in sorted((ROOT/'dataset/annotations/train').iterdir()) if p.suffix in ['.png','.json']} if config['shots'] else {}
    signature=dict(config=config,dataset_sha256=sha256(ROOT/'dataset/metadata.json'),annotation_hashes=hashes)
    from spinning.backbone import extract_features
    from spinning.pseudo import prepare_training_tensors
    from spinning.train import train_variant
    from spinning.figures import export_results
    from spinning.quality import export_quality
    from spinning.verify import verify_run
    from spinning.comparison import preserve_reference,export_mrf_comparison
    with output_session(args.output) as out:
        reuse=config['variants'] if args.predict_only else [n for n in config['variants'] if args.retrain_models and n not in args.retrain_models]
        if reuse:
            if not (out/'run.json').exists(): raise ValueError('No trained run exists for reusing controls.')
            # Check all weights before creating/replacing any pending results.
            for name in reuse:
                if not compatible_run(load(out/'run.json'),signature,name):
                    raise ValueError(f'No compatible trained {name}; include it in --retrain-models or run fresh training.')
                for filename in ['best.pt','last.pt','history.json','training_summary.json']:
                    if not (out/name/filename).is_file(): raise FileNotFoundError(out/name/filename)
        pending=prepare_output(out,bool(reuse),reuse)
        preserve_reference(out,pending,signature)
        dump(pending/'run.json',signature)
        dump(pending/'execution.json',dict(mode='predict_only' if args.predict_only else 'selective_retraining' if args.retrain_models else 'fresh_training',retrained_models=[n for n in config['variants'] if n not in reuse],reused_models=reuse,epochs_per_model=config['epochs'],early_stopping=False,replaces_previous_output=True))
        size='x'.join(map(str,config['backbone_size']))
        features=extract_features(metadata,config,ROOT/'.cache'/('dino_'+size),device)
        print('Preparing three-point targets and within-split optical flow...',flush=True)
        cpu_data=prepare_training_tensors(metadata,features,config,pending)
        if 'full' in config['variants'] and mrf['weight']>0:
            cpu_data.update(prepare_mrf(cpu_data['images'],cpu_data['valid'],mrf['color_sigma']))
        if 'full' in config['variants'] and topo['enabled']:
            print('Preparing train-fitted clusters and DTM persistent homology...',flush=True)
            cpu_data.update(prepare_topology(cpu_data,metadata['frames'],config,pending,reuse='full' in reuse))
        data={key:value.to(device) for key,value in cpu_data.items()}
        variants={}
        for name in config['variants']:
            print(f'Running {MODEL_INFO[name]["label"]}: '+('reusing compatible weights' if name in reuse else f'{config["epochs"]} epochs, early stopping OFF'),flush=True)
            metrics,probability,points=train_variant(name,data,metadata['frames'],config,pending,device,name in reuse)
            variants[name]=dict(metrics=metrics,probability=probability,points=points)
        export_results(metadata,config,cpu_data,variants,pending)
        print('Exporting diagnostics, comparisons and review videos...',flush=True)
        export_quality(metadata,cpu_data,variants,pending)
        export_topology_figures(pending)
        compare_ablation(pending,ROOT/'outputs/triangle_ours_ablation')
        export_mrf_comparison(pending)
        from spinning.study import export_study
        export_study(pending)
        verify_run(pending)
        commit_output(out,pending)
        print(f'Complete. Replaced {out}. Review {out/"review.html"}.',flush=True)


if __name__=='__main__':
    main()
