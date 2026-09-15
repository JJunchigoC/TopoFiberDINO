"""Fair within-run comparison, preserving adverse metrics and all frames."""
import numpy as np
from .io import ROOT,load,dump,sha256


def export_study(out):
    run=load(out/'run.json'); config=run['config']; names=config['variants']
    if config['epochs']!=500: return
    metrics=load(out/'metrics_comparison.json')
    points={n:{r['id']:r['vertices_roi'] for r in load(out/n/'predictions.json')} for n in names}
    pseudo={r['id']:r['vertices_normalized'] for r in load(out/'pseudo_vertices.json')}
    rows=load(ROOT/'dataset/metadata.json')['frames']
    comparisons={}; frames=[]
    for n in names:
        summary=load(out/n/'training_summary.json')
        comparisons[n]=dict(label=metrics[n]['model_label'],train=metrics[n]['train'],val=metrics[n]['val'],test=metrics[n]['test'],
            epochs=summary['epochs_run'],best_epoch=summary['best_epoch'],trainable_parameters=summary['trainable_parameters'],
            training_seconds=summary['elapsed_seconds'],checkpoint_sha256=sha256(out/n/'best.pt'))
    for r in rows:
        target=pseudo[r['id']]; record=dict(id=r['id'],split=r['split'],sequence=r['sequence'],models={})
        for n in names:
            prediction=points[n][r['id']]
            error=np.linalg.norm(np.asarray(prediction)/(np.asarray(r['image_size'])-1)-target,axis=1) if prediction is not None and target is not None else None
            record['models'][n]=dict(vertex_error=None if error is None else float(error.mean()),per_vertex_error=None if error is None else error.tolist())
        frames.append(record)
    rankings={}
    for split in ['val','test']:
        for metric,descending in [('pseudo_mask_iou',True),('pseudo_vertex_error_normalized',False),('flow_warp_mae',False)]:
            rankings[f'{split}_{metric}']=sorted(names,key=lambda n:comparisons[n][split][metric],reverse=descending)
    ours_best={k:v[0]=='full' for k,v in rankings.items()} if 'full' in names else {}
    assessment=dict(ours_first_by_metric=ours_best,ours_best_on_all_reported_quality_metrics=bool(ours_best) and all(ours_best.values()),
                    human_accuracy_verified=False,test_is_independent_blind_test=False,
                    note='Ranks are within this run; pseudo-target agreement and temporal consistency are not physical accuracy. Equal final epoch budgets do not equalize architecture-search budgets.')
    destination=out/'comparison'; destination.mkdir(exist_ok=True)
    dump(destination/'study500.json',dict(config=config,dataset_sha256=run['dataset_sha256'],models=comparisons,rankings=rankings,assessment=assessment))
    dump(destination/'study500_frames.json',dict(frames=frames,note='All frames retained, including failures. No manual point adjustment.'))
    lines=['\n## 500 epoch comparison\n','Same split, pseudo targets, epoch budget and validation vertex-error selection. No human accuracy established.\n',
           '| Model | Best epoch | Val pseudo IoU | Test pseudo IoU | Test vertex error | Test flow MAE |',
           '|---|---:|---:|---:|---:|---:|']
    for n,m in comparisons.items():
        lines.append(f'| {m["label"]} | {m["best_epoch"]} | {m["val"]["pseudo_mask_iou"]:.4f} | {m["test"]["pseudo_mask_iou"]:.4f} | {m["test"]["pseudo_vertex_error_normalized"]:.4f} | {m["test"]["flow_warp_mae"]:.4f} |')
    lines+=['','Ours ranks first on all reported quality metrics: '+str(assessment['ours_best_on_all_reported_quality_metrics']),
            '[All metrics and ranks](comparison/study500.json) | [Every frame](comparison/study500_frames.json)']
    with (out/'REPORT.md').open('a',encoding='utf-8') as stream: stream.write('\n'.join(lines)+'\n')
