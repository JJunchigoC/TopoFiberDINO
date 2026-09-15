"""Auditable diagnostics, optional real-label evaluation, and offline review."""
import csv
import json
import cv2
import numpy as np
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion
from .io import ROOT, read, save, dump, load, sha256
from .figures import entropy
from .model import MODEL_INFO,model_label
from .triangle import annotation


def overlap(a,b,valid):
    a=np.asarray(a,bool)&valid; b=np.asarray(b,bool)&valid
    tp=int((a&b).sum()); fp=int((a&~b).sum()); fn=int((~a&b).sum())
    return dict(iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None)


def mask_metrics(pred,gt,valid,tolerance=2):
    result=overlap(pred,gt,valid)
    pred=np.asarray(pred,bool)&valid; gt=np.asarray(gt,bool)&valid
    # Ignore the one-pixel rim of invalid padding: it is not an object boundary.
    interior=binary_erosion(valid,border_value=1)
    a=(pred&~binary_erosion(pred))&interior
    b=(gt&~binary_erosion(gt))&interior
    result.update(boundary_f1=None,assd_roi_px=None,hd95_roi_px=None)
    if a.any() and b.any():
        ab=distance_transform_edt(~b)[a]; ba=distance_transform_edt(~a)[b]
        p=float((ab<=tolerance).mean()); r=float((ba<=tolerance).mean())
        result.update(boundary_f1=2*p*r/(p+r) if p+r else 0.,
                      assd_roi_px=float((ab.sum()+ba.sum())/(len(ab)+len(ba))),
                      hd95_roi_px=float(np.percentile(np.concatenate([ab,ba]),95)))
    elif a.any() or b.any():
        result['boundary_f1']=0.
    return result


def review_priority(stability,disagreement,flow,discarded,empty=False):
    if empty:
        return 100.
    terms=[(.45,1-stability),(.10,discarded)]
    if disagreement is not None:
        terms.append((.25,disagreement))
    if flow is not None:
        terms.append((.20,min(flow/.1,1.)))
    return round(100*sum(w*v for w,v in terms)/sum(w for w,_ in terms),3)


def write_csv(path,rows):
    if not rows:
        return
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def heatmap(values,valid):
    view=cv2.applyColorMap(np.uint8(np.clip(values,0,1)*255),cv2.COLORMAP_INFERNO)
    view[~valid]=32
    return view


def export_quality(metadata,data,variants,out):
    folder=out/'quality'; folder.mkdir(exist_ok=True)
    rows=metadata['frames']; names=list(variants)
    predictions={name:load(out/name/'predictions.json') for name in names}
    probs=np.stack([variants[name]['probability'][:,0].numpy() for name in names])
    valid=data['valid'][:,0].numpy()>0
    votes=(probs>.5).mean(axis=0)
    disagreement=4*votes*(1-votes)
    records=[]; ground_truth=[]; annotation_hashes={}
    temporal={}
    for name,values in variants.items():
        p=values['probability']
        warped=F.grid_sample(p[data['neighbors']],data['grids'],align_corners=True)
        numerator=((p-warped).abs()*data['visibility']).flatten(1).sum(1)
        denominator=data['visibility'].flatten(1).sum(1)
        temporal[name]=[float(n/d) if d>0 else None for n,d in zip(numerator,denominator)]
    for i,row in enumerate(rows):
        save(folder/'disagreement'/f'{row["id"]}.jpg',heatmap(disagreement[i],valid[i]))
        union=(probs[:,i]>.5).any(axis=0)&valid[i]
        disagree=float(disagreement[i][union].mean()) if union.any() and len(names)>1 else None
        native_valid=read(ROOT/'dataset'/row['valid_mask'],True)>0
        truth=annotation(ROOT/'dataset/annotations'/row['split'],row['id'],native_valid.shape,native_valid) if row['split'] in ['val','test'] else None
        gt,gt_points,gt_path=truth if truth else (None,None,None)
        if truth:
            annotation_hashes[gt_path.relative_to(ROOT).as_posix()]=sha256(gt_path)
        for k,name in enumerate(names):
            p=probs[k,i]; v=valid[i]; binary=(p>.5)&v
            stability=overlap(p>.4,p>.6,v)['iou']
            count,_,stats,_=cv2.connectedComponentsWithStats(binary.astype(np.uint8),8)
            discarded=1-float(stats[1:,cv2.CC_STAT_AREA].max())/binary.sum() if binary.any() else 0.
            boundary=cv2.morphologyEx(binary.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8))>0
            region=(binary|boundary)&v
            pseudo=overlap(binary,data['target'][i,0].numpy()>.5,v)
            result=predictions[name][i]
            dimensions=result['metrics_raw_px'] or {}
            flags=[]
            if result['vertices_roi'] is None: flags.append('invalid_triangle')
            if not binary.any(): flags.append('empty_prediction')
            if stability<.8: flags.append('threshold_sensitive')
            if disagree is not None and disagree>.35: flags.append('models_disagree')
            flow=temporal[name][i]
            if flow is not None and flow>.05: flags.append('temporal_change')
            if discarded>.1: flags.append('fragmented_raw_mask')
            rec=dict(id=row['id'],model=name,model_label=model_label(name),split=row['split'],sequence=row['sequence'],time_seconds=row['time_seconds'],
                     review_priority=review_priority(stability,disagree,flow,discarded,not binary.any() or result['vertices_roi'] is None),
                     threshold_stability_iou=stability,model_disagreement=disagree,
                     foreground_boundary_entropy=float(entropy(p)[region].mean()) if region.any() else None,
                     flow_warp_mae=flow,raw_components=count-1,discarded_fraction=discarded,
                     foreground_fraction=float(binary.sum()/v.sum()),pseudo_iou=pseudo['iou'],pseudo_dice=pseudo['dice'],
                     base_width_raw_px=dimensions.get('base_width'),height_raw_px=dimensions.get('perpendicular_height'),
                     triangle_area_raw_px2=dimensions.get('triangle_area'),flags=';'.join(flags),
                     original='../../dataset/'+row['image'],overlay=f'{name}/'+result['overlay'],
                     entropy=f'quality/{name}/entropy/{row["id"]}.jpg',
                     disagreement=f'quality/disagreement/{row["id"]}.jpg',true_iou=None,true_dice=None,
                     true_vertex_error_roi_px=None,vertices_roi=result['vertices_roi'],triangle_svg=f'{name}/'+result['triangle_svg'])
            save(folder/name/'entropy'/f'{row["id"]}.jpg',heatmap(entropy(p),v))
            quantized=np.uint16(np.clip(p,0,1)*65535); quantized[~v]=0
            save(folder/name/'probability'/f'{row["id"]}.png',quantized)
            if gt is not None:
                predicted=read(out/name/result['mask'],True)>0
                metrics=mask_metrics(predicted,gt>0,native_valid)
                metrics['vertex_error_roi_px']=float(np.linalg.norm(np.asarray(result['vertices_roi'])-gt_points,axis=1).mean()) if result['vertices_roi'] is not None else None
                ground_truth.append(dict(id=row['id'],model=name,split=row['split'],**metrics))
                rec.update(true_iou=metrics['iou'],true_dice=metrics['dice'],true_vertex_error_roi_px=metrics['vertex_error_roi_px'])
                error=np.zeros((*gt.shape,3),np.uint8)
                error[predicted&(gt>0)&native_valid]=(0,180,0)
                error[predicted&(gt==0)&native_valid]=(0,0,255)
                error[~predicted&(gt>0)&native_valid]=(255,100,0)
                save(folder/name/'ground_truth_errors'/f'{row["id"]}.png',error)
            records.append(rec)
    ranked=sorted(records,key=lambda r:r['review_priority'],reverse=True)
    dump(folder/'frame_diagnostics.json',ranked); write_csv(folder/'frame_diagnostics.csv',ranked)
    metric_keys=['iou','dice','precision','recall','boundary_f1','assd_roi_px','hd95_roi_px','vertex_error_roi_px']
    true_summary={}
    for name in names:
        true_summary[name]={}
        for split in ['val','test']:
            subset=[r for r in ground_truth if r['model']==name and r['split']==split]
            true_summary[name][split]={'count':len(subset)}
            for key in metric_keys:
                values=[r[key] for r in subset if r[key] is not None]
                true_summary[name][split][key]=float(np.mean(values)) if values else None
    dump(folder/'ground_truth_metrics.json',dict(status='available' if ground_truth else 'no_ground_truth',
         boundary_tolerance_roi_px=2,aggregation='Macro mean over annotated frames with defined metric',
         annotation_hashes=annotation_hashes,summary=true_summary,frames=ground_truth))
    if ground_truth: write_csv(folder/'ground_truth_metrics.csv',ground_truth)
    # Validation only; test diagnostics are never used for this view default.
    from .train import selection_score
    selection_config=load(out/'run.json')['config']
    suggested=min(names,key=lambda n:selection_score(variants[n]['metrics']['val'],selection_config))
    summary=[]
    for name in names:
        subset=[r for r in records if r['model']==name and r['split']=='test']
        training=load(out/name/'training_summary.json')
        summary.append(dict(model=name,label=MODEL_INFO[name]['label'],
            epochs_run=training['epochs_run'],best_epoch=training['best_epoch'],last_epoch=training['last_epoch'],
            validation_pseudo_loss=variants[name]['metrics']['val']['pseudo_supervision_loss'],
            validation_vertex_error_normalized=variants[name]['metrics']['val']['pseudo_vertex_error_normalized'],
            checkpoint_selection=selection_config.get('checkpoint_selection','region_loss'),
            test_pseudo_iou=variants[name]['metrics']['test']['pseudo_mask_iou'],
            test_pseudo_vertex_error_normalized=variants[name]['metrics']['test']['pseudo_vertex_error_normalized'],
            test_threshold_stability=float(np.mean([r['threshold_stability_iou'] for r in subset])),
            test_flagged_frames=sum(bool(r['flags']) for r in subset),
            true_test_iou=true_summary[name]['test']['iou'],true_test_dice=true_summary[name]['test']['dice']))
    dump(folder/'model_summary.json',dict(models=summary,default_review_model=suggested,
         selection=selection_config.get('checkpoint_selection','region_loss')+' on validation; NOT verified physical accuracy',
         diagnostic_resolution=list(probs.shape[-2:]),probability_encoding='uint16 / 65535; invalid pixels zero; use dataset valid mask resized nearest',
         priority_formula='100 * weighted mean: .45*(1-stability), .25*disagreement, .20*min(flow/.1,1), .10*discarded. Missing terms omitted. Empty mask=100.',
         priority_is_accuracy=False))
    write_csv(folder/'model_summary.csv',summary)
    quality_figures(metadata,records,predictions,names,out)
    export_videos(metadata,predictions,suggested,out)
    write_viewer(records,names,suggested,out)
    (folder/'README.md').write_text(QUALITY_GUIDE.replace('{epochs}',str(load(out/'run.json')['config']['epochs'])),encoding='utf-8')
    with (out/'REPORT.md').open('a',encoding='utf-8') as stream:
        stream.write(f'\n新增质量输出：{len(names)} 个模型 × {len(rows)} 帧，共 {len(records)} 条逐帧诊断。真实标注评估条数：{len(ground_truth)}。\n\n')
        stream.write('[全部模型对比图](figures/all_models_comparison.png) · [优先复核帧](figures/review_priority_examples.png) · [尺寸时间曲线](figures/measurement_timeseries.png) · [逐帧CSV](quality/frame_diagnostics.csv) · [模型汇总](quality/model_summary.csv)\n\n')
        stream.write(f'复核页初始模型为 `{suggested}`，按配置指定的验证指标选择，不代表真实准确率最高。尺寸仍为像素，复核优先级不是准确率。\n')
    print(f'Quality outputs: {len(records)} frame/model records; {len(ground_truth)} real-label evaluations.',flush=True)


def quality_figures(metadata,records,predictions,names,out):
    import matplotlib.pyplot as plt
    rows=metadata['frames']; folder=out/'figures'
    ids=load(folder/'selection.json')['ids']
    lookup={r['id']:i for i,r in enumerate(rows)}
    fig,axes=plt.subplots(len(ids),len(names)+1,figsize=(3*(len(names)+1),2*len(ids)),layout='constrained',squeeze=False)
    for j,id_ in enumerate(ids):
        i=lookup[id_]; row=rows[i]
        panels=[read(ROOT/'dataset'/row['image'])]+[read(out/name/predictions[name][i]['overlay']) for name in names]
        for k,panel in enumerate(panels):
            axes[j,k].imshow(panel[:,:,::-1]); axes[j,k].set_xticks([]); axes[j,k].set_yticks([])
            if j==0: axes[j,k].set_title((['Input']+[model_label(n) for n in names])[k],fontsize=10)
            if k==0: axes[j,k].set_ylabel(id_)
    fig.suptitle('Fixed test frames | actual model outputs | no ground-truth accuracy established')
    save_figure(fig,folder/'all_models_comparison')
    # Each model-frame is ranked; choose distinct frames to avoid repeated views.
    selected=[]; seen=set()
    for rec in sorted(records,key=lambda r:r['review_priority'],reverse=True):
        if rec['split']=='test' and rec['id'] not in seen:
            selected.append(rec); seen.add(rec['id'])
        if len(selected)==8: break
    fig,axes=plt.subplots(4,2,figsize=(13,9),layout='constrained')
    for ax,rec in zip(axes.flat,selected):
        ax.imshow(read(out/rec['overlay'])[:,:,::-1]); ax.axis('off')
        ax.set_title(f'{rec["id"]} | {model_label(rec["model"])} | priority {rec["review_priority"]:.1f}',fontsize=9)
    fig.suptitle('Highest review priority: selected by diagnostics, not proof of errors')
    save_figure(fig,folder/'review_priority_examples')
    dump(folder/'review_priority_selection.json',dict(rule='Top diagnostic priority, unique test frames, across requested models',items=[dict(id=r['id'],model=r['model'],priority=r['review_priority']) for r in selected]))
    sequences=sorted({r['sequence'] for r in rows})
    fig,axes=plt.subplots(len(sequences),3,figsize=(15,3.7*len(sequences)),layout='constrained',squeeze=False)
    for j,seq in enumerate(sequences):
        for k,(key,label) in enumerate([('base_width_raw_px','Base width [raw px]'),('height_raw_px','Height [raw px]'),('triangle_area_raw_px2','Triangle area [raw px2]')]):
            ax=axes[j,k]
            for name in names:
                subset=sorted([r for r in records if r['sequence']==seq and r['model']==name],key=lambda r:r['time_seconds'])
                ax.plot([r['time_seconds'] for r in subset],[r[key] if r[key] is not None else np.nan for r in subset],label=model_label(name),linewidth=1)
            seqrows=[r for r in rows if r['sequence']==seq]
            for split in ['val','test']:
                starts=[r['time_seconds'] for r in seqrows if r['split']==split]
                if starts: ax.axvline(min(starts),color='gray',linestyle=':',alpha=.6)
            ax.set(title=f'{seq}: {label}',xlabel='Original video time [s]',ylabel=label); ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('Unsmoothed candidate measurements | dotted lines: val/test starts | no physical calibration')
    save_figure(fig,folder/'measurement_timeseries')


def save_figure(fig,path):
    import matplotlib.pyplot as plt
    for extension in ['png','svg']: fig.savefig(path.with_suffix('.'+extension),dpi=300)
    fig.savefig(path.with_name(path.name+'_preview.png'),dpi=100)
    plt.close(fig)


def export_videos(metadata,predictions,model,out):
    folder=out/'quality/videos'; folder.mkdir(exist_ok=True)
    index=[]
    for sequence in sorted({r['sequence'] for r in metadata['frames']}):
        pairs=[(r,p) for r,p in zip(metadata['frames'],predictions[model]) if r['sequence']==sequence]
        pairs.sort(key=lambda rp:rp[0]['time_seconds'])
        lookup={r['source_frame_number']:(r,p) for r,p in pairs}
        steps=np.diff([r['source_frame_number'] for r,p in pairs]); elapsed=np.diff([r['time_seconds'] for r,p in pairs])
        fps=float(np.median(steps/elapsed))
        path=folder/f'{sequence}_{model}.mp4'
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,352))
        if not writer.isOpened(): raise RuntimeError(f'Video writer failed: {path}')
        missing=[]
        try:
            for number in range(min(lookup),max(lookup)+1):
                canvas=np.full((352,1280,3),25,np.uint8)
                if number in lookup:
                    row,pred=lookup[number]
                    for x,im in [(0,read(ROOT/'dataset'/row['image'])),(640,read(out/model/pred['overlay']))]:
                        scale=min(640/im.shape[1],304/im.shape[0]); resized=cv2.resize(im,(round(im.shape[1]*scale),round(im.shape[0]*scale)))
                        canvas[40:40+resized.shape[0],x:x+resized.shape[1]]=resized
                    text=f'{row["id"]} | t={row["time_seconds"]:.3f}s | {row["split"]} | {model_label(model)} | UNVALIDATED'
                else:
                    missing.append(number); text=f'{sequence} source frame {number}: NO RETAINED FRAME'
                cv2.putText(canvas,text,(12,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),1,cv2.LINE_AA)
                writer.write(canvas)
        finally: writer.release()
        capture=cv2.VideoCapture(str(path)); count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)); capture.release()
        if count!=max(lookup)-min(lookup)+1: raise RuntimeError(f'Incomplete review video: {path}')
        index.append(dict(sequence=sequence,model=model,file=path.relative_to(out).as_posix(),fps=fps,frames=count,missing_source_frames=missing,
                          note='Starts at first retained frame, original frame intervals; missing frames shown as blanks'))
    dump(folder/'index.json',index)


def write_viewer(records,names,suggested,out):
    payload=json.dumps(dict(records=records,models=names,labels={n:model_label(n) for n in names},default_model=suggested,videos=load(out/'quality/videos/index.json')),ensure_ascii=False).replace('<','\\u003c')
    template=(ROOT/'spinning/review_template.html').read_text(encoding='utf-8')
    links='<p><a href="figures/mrf_comparison.png">MRF 加入前后固定帧对照</a> · <a href="figures/mrf_learning_curves.png">MRF 训练曲线</a> · <a href="comparison/mrf_ablation.json">MRF 前后指标</a></p>' if (out/'comparison/no_mrf/provenance.json').exists() else ''
    if (out/'full/topology_predictions.json').exists():
        links+='<p><a href="figures/topology_diagnostics.png">拓扑聚类与持续同调诊断</a> · <a href="topology/metrics.json">拓扑重建指标</a> · <a href="../../docs/TOPOLOGY.md">TopoSlide 借鉴与差异</a></p>'
    if load(out/'run.json')['config']['epochs']==500:
        links+='<p><a href="comparison/study500.json">500 轮完整指标与排名</a> · <a href="comparison/study500_frames.json">全部帧的三点误差</a> · <a href="../../docs/STUDY500.md">本次实验协议</a></p>'
    (out/'review.html').write_text(template.replace('__REVIEW_DATA__',payload).replace('__MRF_LINKS__',links),encoding='utf-8')


QUALITY_GUIDE='''# 三点三角形质量说明

本次所选模型统一训练{epochs}个epoch，关闭早停。full为ours，具体版本以run.json和model_summary为准。model_summary中epochs_run/last_epoch是实际轮数，best_epoch是配置所指定验证指标的最优轮次；默认图像用best.pt，第{epochs}轮权重另存last.pt。两者不必是同一轮。

full 训练增加 8 邻域对比敏感 Potts MRF 正则，最终仍只有三个顶点。MRF 能量低不代表三点真实位置更准，平滑项还可能使三角形收缩。有兼容的旧无 MRF 参考时，comparison/mrf_ablation.json 和 figures/mrf_comparison.png 提供同预算前后比较；没有真实三点时不能据此声称实际测量精度提高。

输出契约为 A/B/C 三个坐标及 AB/BC/CA 三条直边。最终 SVG 和叠加图只画三点三边，二值 mask 是同一三角形的栅格化，尺寸只有三角形面积与周长。

打开 [三点复核页](../review.html)，对照原图检查 A 的汇聚位置、B/C 的端点定义和三条边的方向。切换模型保持当前帧，随后用连续视频和尺寸曲线查看时间变化。B/C 自动目标仍是最大展开截面代理，尚未验证为物理钳口。

## 指标如何理解

- pseudo_iou / pseudo_dice：三角形与自动三点目标的一致性，不是真实准确率。
- pseudo_vertex_error_normalized：模型三点与自动目标三点的平均欧氏距离，x/y 各自除以 ROI 宽/高后计算；不是像素或毫米误差。
- threshold_stability_iou：软三角形在阈值0.4/0.6下的重合程度。三角形结构已被强制约束，高稳定性不代表端点准确。
- foreground_boundary_entropy：软栅格化在前景及5×5边缘带中的二元熵。它主要反映边缘过渡，不能当作定位置信度。
- model_disagreement：4×前景投票率×(1−投票率)，在所有模型前景并集内平均；模型共享自动目标，错误可能相关。
- flow_warp_mae：同视频、同分组邻帧经光流对齐后的软三角形差异，无有效门控像素时为空。
- discarded_fraction / raw_components：检查阈值区域与有效掩膜相交后的连通性；新版不执行丢弃分量后处理。
- triangle_area_raw_px2：由原视频坐标三点计算的三角形面积。

复核优先级是排序启发式，不是合格分或准确率。公式：100×加权平均，0.45×(1−阈值稳定性)、0.25×模型分歧、0.20×min(时序MAE/0.1,1)、0.10×区域碎片比例。缺失项去除后归一；空预测或无效三角形为100。提示阈值分别是稳定性<0.8、分歧>0.35、MAE>0.05、碎片比例>0.1。这些权重尚未按人工误差校准。

## 真正的三点评估

在 dataset/annotations/val 或 test 放置同名 JSON：
{"vertices_roi":[[Ax,Ay],[Bx,By],[Cx,Cy]]}
使用原 ROI 像素坐标，顺序固定 A左 / B右上 / C右下。运行 python main.py --predict-only，无需重新训练即可计算三点平均欧氏误差、三角形 IoU/Dice、精确率/召回率、边界F1、ASSD和HD95。人工三角形PNG只用于兼容，必须确实是单一三角形；JSON优先。没有标注时真实指标为null。

边界容差2个原 ROI 像素，距离指标同样使用 ROI 像素。区域距离由导出的硬三角形计算；各指标按有定义的帧取宏平均，缺失值不当作0分。只有一方为空时IoU/Dice为0；两者为空为1；无边界可比较时距离为null。错误图绿色TP、红色FP、蓝色FN。

## 文件

frame_diagnostics.csv/json 为全部模型逐帧诊断；model_summary.csv/json 为模型汇总；ground_truth_metrics.json 为独立人工评估。各模型 probability/ 是工作尺寸200×448的uint16 PNG，除以65535还原；无效像素置0。entropy/ 和 disagreement/ 采用固定0–1色阶，黑紫低、橙黄高，无效像素灰色。

figures/all_models_comparison.* 为固定等间距4张测试帧；review_priority_examples.* 为诊断排序选取的8张不同测试帧，各自有选图记录，后者不是随机示例。measurement_timeseries.* 显示三角形宽度、高度和面积，分组边界用虚线。videos/ 按原30fps时间间隔输出，V01第8帧缺失以空白显示，视频从首个保留帧开始。

7:2:1来自同两段视频的时间分块，不能代表独立工况泛化。真实物理尺寸还需要标尺；当前毫米字段为空。
'''
