"""Compact, provenance-checked before/after MRF ablation inside fixed outputs."""
import shutil
from .io import load,dump,sha256,ROOT,read
from .mrf import settings


def comparable_reference(previous,current):
    # The sole training change may be MRF; names and model subsets are metadata.
    a={**previous,'config':dict(previous['config'])}
    b={**current,'config':dict(current['config'])}
    for item in [a,b]:
        item.setdefault('annotation_hashes',{})
        for key in ['variants','name','mrf']: item['config'].pop(key,None)
    return a==b and settings(previous['config'])['weight']==0


def preserve_reference(old,pending,current):
    if 'full' not in current['config']['variants'] or settings(current['config'])['weight']==0:
        return
    reference=old/'comparison/no_mrf'
    target=pending/'comparison/no_mrf'
    if (reference/'provenance.json').is_file():
        if comparable_reference(load(reference/'provenance.json')['run'],current):
            shutil.copytree(reference,target)
        return
    if not (old/'run.json').is_file() or not (old/'full/best.pt').is_file(): return
    previous=load(old/'run.json')
    if not comparable_reference(previous,current): return
    target.mkdir(parents=True)
    for name in ['best.pt','history.json','training_summary.json','predictions.json']:
        shutil.copy2(old/'full'/name,target/name)
    fixed=load(old/'figures/selection.json')
    dump(target/'selection.json',fixed)
    dump(target/'metrics.json',load(old/'metrics_comparison.json')['full'])
    dump(target/'provenance.json',dict(run=previous,best_sha256=sha256(target/'best.pt'),
         role=f'Prior no-MRF ablation, not the active model; same seed/splits/{previous["config"]["epochs"]} epochs.',
         history_sha256=sha256(target/'history.json')))


def export_mrf_comparison(out):
    reference=out/'comparison/no_mrf'
    if not (reference/'provenance.json').exists() or not (out/'full/metrics.json').exists(): return
    import matplotlib.pyplot as plt
    from .figures import overlay
    before=load(reference/'metrics.json')
    after=load(out/'metrics_comparison.json')['full']
    delta={s:{k:after[s][k]-before[s][k] for k in ['pseudo_supervision_loss','pseudo_mask_iou','pseudo_vertex_error_normalized','flow_warp_mae']} for s in ['train','val','test']}
    provenance=load(reference/'provenance.json')
    assert sha256(reference/'best.pt')==provenance['best_sha256']
    assert sha256(reference/'history.json')==provenance['history_sha256']
    dump(out/'comparison/mrf_ablation.json',dict(before=before,after=after,delta_after_minus_before=delta,
         before_training=load(reference/'training_summary.json'),after_training=load(out/'full/training_summary.json'),
         mrf=settings(load(out/'run.json')['config']),reference_checkpoint_sha256=provenance['best_sha256'],
         current_checkpoint_sha256=sha256(out/'full/best.pt'),
         protocol='Same data, seed, optimizer and epoch budget; only MRF loss added. Best selected using validation region loss. One seed; test frames previously examined; no ground truth.',
         true_accuracy_improvement=None))
    old={r['id']:r for r in load(reference/'predictions.json')}
    new={r['id']:r for r in load(out/'full/predictions.json')}
    rows=load(ROOT/'dataset/metadata.json')['frames']
    # Fixed evenly spaced test frames, shared with the main comparison figure.
    selected=load(out/'figures/selection.json')['ids']
    selected_rows=[r for r in rows if r['id'] in selected]
    fig,axes=plt.subplots(len(selected_rows),3,figsize=(12,2.3*len(selected_rows)),squeeze=False,layout='constrained')
    for rr,row in enumerate(selected_rows):
        image=read(ROOT/'dataset'/row['image'])
        panels=[image,overlay(image,None,old[row['id']]['vertices_roi']),overlay(image,None,new[row['id']]['vertices_roi'])]
        for cc,panel in enumerate(panels):
            axes[rr,cc].imshow(panel[:,:,::-1]); axes[rr,cc].set_xticks([]); axes[rr,cc].set_yticks([])
            if rr==0: axes[rr,cc].set_title(['Input','Ours before MRF','Triangle-MRF (ours)'][cc])
            if cc==0: axes[rr,cc].set_ylabel(row['id'],fontsize=8)
    epochs=load(out/'run.json')['config']['epochs']
    fig.suptitle(f'Fixed test frames | same {epochs}-epoch budget | no ground-truth vertices')
    for ext in ['png','svg']: fig.savefig(out/f'figures/mrf_comparison.{ext}',dpi=300)
    fig.savefig(out/'figures/mrf_comparison_preview.png',dpi=110); plt.close(fig)
    histories=[load(reference/'history.json'),load(out/'full/history.json')]
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    for label,history in zip(['Before MRF','Triangle-MRF (ours)'],histories):
        for ax,end in zip(axes[:2],[50,len(history)]):
            ax.plot([r['epoch'] for r in history[:end]],[r['pseudo_supervision_loss'] for r in history[:end]],label=label)
    axes[0].set(title='Validation: first 50 epochs',xlabel='Epoch',ylabel='Same region loss')
    axes[1].set(title='Validation: full budget',xlabel='Epoch',ylabel='Same region loss')
    history=histories[1]
    weight=settings(load(out/'run.json')['config'])['weight']
    axes[2].plot([r['epoch'] for r in history],[weight*r['components']['mrf_pairwise'] for r in history],label='Weighted MRF term')
    axes[2].set(title='MRF contribution during training',xlabel='Epoch',ylabel='Weight × expected Potts energy')
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2)
    for ext in ['png','svg']: fig.savefig(out/f'figures/mrf_learning_curves.{ext}',dpi=300)
    fig.savefig(out/'figures/mrf_learning_curves_preview.png',dpi=110); plt.close(fig)
    config=load(out/'run.json')['config']; mrf=settings(config)
    lines=['','## MRF 同预算消融','',
           f'加入前后 full 均完成 {config["epochs"]} 轮。MRF 权重 {mrf["weight"]}、颜色 σ={mrf["color_sigma"]}；本对比为同种子的一组结果，未按测试指标搜索参数。其他模型的训练/复用模式见 execution.json。',
           '', '| 指标 | 加入前 | 加入后 | 后 − 前 |','|---|---:|---:|---:|']
    for split,key,label in [('val','pseudo_supervision_loss','验证区域损失 ↓'),('test','pseudo_mask_iou','测试伪标签 IoU ↑'),('test','pseudo_vertex_error_normalized','测试归一化伪顶点误差 ↓'),('test','flow_warp_mae','测试光流对齐 MAE ↓')]:
        lines.append(f'| {label} | {before[split][key]:.6f} | {after[split][key]:.6f} | {delta[split][key]:+.6f} |')
    lines+=['','这些指标均不是真实测量精度。MRF 在训练阶段作用于软三角形，输出仍为 A/B/C 三点；更低的平滑能量可能伴随收缩，必须核对端点位置。',
            '', '[固定帧前后对照](figures/mrf_comparison.png) · [前后验证曲线](figures/mrf_learning_curves.png) · [指标与权重来源](comparison/mrf_ablation.json)']
    with (out/'REPORT.md').open('a',encoding='utf-8') as stream: stream.write('\n'.join(lines)+'\n')
