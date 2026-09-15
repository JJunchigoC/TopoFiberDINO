"""Publication-style figures of actual predictions, with no invented ground truth."""
import csv
import cv2
import numpy as np
from torch.nn import functional as F
from .io import ROOT, read, save, dump, load
from .geometry import transform, measure
from .triangle import triangle_mask,annotation
from .model import model_label




def overlay(image,mask,vertices=None):
    output=image.copy()
    if vertices is not None:
        cv2.polylines(output,[np.rint(vertices).astype(np.int32)],True,(230,200,20),2,cv2.LINE_AA)
        for name,point in zip('ABC',vertices):
            point=tuple(np.rint(point).astype(int))
            cv2.circle(output,point,5,(40,80,255),-1)
            cv2.putText(output,name,(point[0]+7,point[1]-8),cv2.FONT_HERSHEY_SIMPLEX,.6,(30,60,255),2,cv2.LINE_AA)
    return output


def entropy(probability):
    p=np.clip(probability,1e-6,1-1e-6)
    return -(p*np.log2(p)+(1-p)*np.log2(1-p))


def export_results(metadata,config,data,variants,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
    rows=metadata['frames']
    primary='full' if 'full' in variants else next(iter(variants))
    reference='baseline' if 'baseline' in variants else next(iter(variants))
    figures=out/'figures'
    figures.mkdir(exist_ok=True)
    results,measurements,summary={},[],{}
    for variant,values in variants.items():
        probability=values['probability']
        neighbor=probability[data['neighbors']]
        warped=F.grid_sample(neighbor,data['grids'],align_corners=True)
        for split in ['train','val','test']:
            ids=[i for i,r in enumerate(rows) if r['split']==split]
            gate=data['visibility'][ids]
            temporal=float(((probability[ids]-warped[ids]).abs()*gate).sum()/gate.sum().clamp_min(1))
            summary.setdefault(variant,{})[split]=dict(**values['metrics'][split],flow_warp_mae=temporal,
                                                       foreground_fraction=float(((probability[ids]>.5)*data['valid'][ids]).sum()/data['valid'][ids].sum()))
        summary[variant]['model_label']=model_label(variant)
        results[variant]=[]
        for i,row in enumerate(rows):
            image=read(ROOT/'dataset'/row['image'])
            valid=read(ROOT/'dataset'/row['valid_mask'],True)>0
            h,w=image.shape[:2]
            p=cv2.resize(probability[i,0].numpy(),(w,h),interpolation=cv2.INTER_LINEAR)
            vertices=values['points'][i].numpy()*[w-1,h-1]
            try:
                mask=triangle_mask(vertices,(h,w),valid)
                geometry=dict(vertices=vertices,contour=vertices.copy())
            except ValueError:
                mask=np.zeros((h,w),np.uint8); geometry=None
            record=dict(id=row['id'],split=row['split'],sequence=row['sequence'],time_seconds=row['time_seconds'],
                        status='unvalidated_triangle' if geometry else 'invalid_triangle',
                        representation='triangle_vertices_v2',model=variant,model_label=model_label(variant),vertex_order=['A','B','C'],edges=[[0,1],[1,2],[2,0]],
                        vertices_roi=None,contour_roi=None,vertices_raw=None,contour_raw=None,
                        metrics_raw_px=None,metrics_mm=None,measurement_approved=False,
                        predicted_entropy_mean=float(entropy(p)[valid].mean()),
                        flags=['no_true_accuracy_evaluation','uncalibrated_mm','base_is_max_spread_proxy_not_nip'],
                        mask=f'masks/{row["id"]}.png',overlay=f'overlays/{row["id"]}.jpg',
                        triangle_svg=f'triangles/{row["id"]}.svg')
            if geometry:
                inverse=np.linalg.inv(np.vstack([row['raw_to_roi'],[0,0,1]]))
                raw_vertices=transform(geometry['vertices'],inverse)
                raw_contour=transform(geometry['contour'],inverse)
                record.update(vertices_roi=geometry['vertices'].tolist(),contour_roi=geometry['contour'].tolist(),
                              vertices_raw=raw_vertices.tolist(),contour_raw=raw_contour.tolist(),
                              metrics_raw_px=measure(raw_vertices,raw_contour))
            save(out/variant/record['mask'],mask*255)
            view=overlay(image,mask,None if not geometry else geometry['vertices'])
            header=np.full((32,w,3),245,np.uint8)
            cv2.putText(header,f'{row["id"]} | {row["split"]} | {model_label(variant)} | UNVALIDATED',
                        (8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1,cv2.LINE_AA)
            save(out/variant/record['overlay'],np.vstack([header,view]))
            svg=out/variant/record['triangle_svg']; svg.parent.mkdir(parents=True,exist_ok=True)
            point_string=' '.join(f'{x:.6f},{y:.6f}' for x,y in vertices) if geometry else ''
            polygon=f'<polygon points="{point_string}" fill="none" stroke="#16a6b6" stroke-width="2"/>' if geometry else ''
            dots=''.join(f'<circle cx="{x:.6f}" cy="{y:.6f}" r="4" fill="#ed543a"/>' for x,y in vertices) if geometry else ''
            svg.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">{polygon}{dots}</svg>',encoding='utf-8')
            results[variant].append(record)
        dump(out/variant/'predictions.json',results[variant])
    # Optional genuine annotations are evaluated separately and never fabricated.
    for variant in variants:
        for split in ['val','test']:
            intersection,union,annotated=0,0,0
            for row,record in zip(rows,results[variant]):
                if row['split']!=split:
                    continue
                valid=read(ROOT/'dataset'/row['valid_mask'],True)>0
                truth=annotation(ROOT/'dataset/annotations'/split,row['id'],valid.shape,valid)
                if truth is None: continue
                gt,_,_=truth
                predicted=read(out/variant/record['mask'],True)>0
                intersection+=int((predicted&(gt>0)&valid).sum())
                union+=int(((predicted|(gt>0))&valid).sum())
                annotated+=1
            summary[variant][split]['true_annotation_count']=annotated
            summary[variant][split]['true_mask_iou']=(intersection/union if union else 1.) if annotated else None
    dump(out/'metrics_comparison.json',summary)
    for record in results[primary]:
        values=record['metrics_raw_px'] or {}
        flat=dict(id=record['id'],split=record['split'],time_seconds=record['time_seconds'],status=record['status'])
        for key in ['base_width','perpendicular_height','upper_side_length','lower_side_length','triangle_area','triangle_perimeter']:
            flat[key+('_raw_px2' if 'area' in key else '_raw_px')]=values.get(key)
            flat[key+('_mm2' if 'area' in key else '_mm')]=None
        flat['apex_angle_deg']=values.get('apex_angle_deg')
        flat['model']=primary
        flat['model_label']=model_label(primary)
        for space in ['roi','raw']:
            for k,name in enumerate('ABC'):
                for axis,dim in enumerate('xy'):
                    flat[f'{name}_{dim}_{space}_px']=record[f'vertices_{space}'][k][axis] if record[f'vertices_{space}'] is not None else None
        flat['measurement_approved']=False
        measurements.append(flat)
    with (out/'measurements.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(measurements[0]))
        writer.writeheader()
        writer.writerows(measurements)
    # A deterministic spread of held-out frames, not selected by apparent quality.
    test_ids=[i for i,r in enumerate(rows) if r['split']=='test']
    selected=np.array(test_ids)[np.linspace(0,len(test_ids)-1,4).round().astype(int)].tolist()
    dump(figures/'selection.json',dict(ids=[rows[i]['id'] for i in selected],rule='Four evenly spaced entries of the predefined test list; no quality selection.'))
    shown=list(dict.fromkeys([reference,primary]))
    columns=4+len(shown)
    fig,axes=plt.subplots(4,columns,figsize=(3*columns,6.5),layout='constrained')
    mappables={}
    titles=['Input ROI','DINO prototype response','Automatic 3-point target']+[model_label(n) for n in shown]+['Soft triangle entropy']
    for rr,i in enumerate(selected):
        row=rows[i]
        image=read(ROOT/'dataset'/row['image'])
        h,w=image.shape[:2]
        valid=read(ROOT/'dataset'/row['valid_mask'],True)>0
        prototype_map=cv2.resize(data['prototype_probability'][i,0].numpy(),(w,h))
        prototype_map[~valid]=np.nan
        panels=[image[:,:,::-1],prototype_map,
                overlay(image,None,data['points'][i].numpy()*[w-1,h-1] if data['has_vertices'][i].item() else None)[:,:,::-1]]
        for variant in shown:
            record=results[variant][i]
            mask=read(out/variant/record['mask'],True)>0
            panels.append(overlay(image,mask.astype(np.uint8),record['vertices_roi'])[:,:,::-1])
        entropy_map=cv2.resize(entropy(variants[primary]['probability'][i,0].numpy()),(w,h))
        entropy_map[~valid]=np.nan
        panels.append(entropy_map)
        for cc,panel in enumerate(panels):
            mappables[cc]=axes[rr,cc].imshow(panel,cmap='magma' if cc==columns-1 else 'viridis',vmin=0 if cc in [1,columns-1] else None,vmax=1 if cc in [1,columns-1] else None)
            axes[rr,cc].set_xticks([])
            axes[rr,cc].set_yticks([])
            if rr==0:
                axes[rr,cc].set_title(titles[cc],fontsize=10)
            if cc==0:
                axes[rr,cc].set_ylabel(row['id'],fontsize=9)
    for cc,label in [(1,'Prototype probability'),(columns-1,'Entropy; invalid pixels excluded')]:
        fig.colorbar(mappables[cc],ax=list(axes[:,cc]),location='bottom',shrink=.85,fraction=.035,pad=.025,label=label)
    fig.suptitle('Held-out frame comparison | no ground-truth masks supplied | entropy is not calibrated error',fontsize=12)
    for extension in ['png','svg']:
        fig.savefig(figures/f'qualitative_comparison.{extension}',dpi=300)
    fig.savefig(figures/'qualitative_comparison_preview.png',dpi=100)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,3.7),layout='constrained')
    for variant in variants:
        history=load(out/variant/'history.json')
        axes[0].plot([v['epoch'] for v in history],[v['train_loss'] for v in history],label=model_label(variant))
        axes[1].plot([v['epoch'] for v in history],[v['pseudo_supervision_loss'] for v in history],label=model_label(variant))
        axes[2].plot([v['epoch'] for v in history],[v['pseudo_vertex_error_normalized'] for v in history],label=model_label(variant))
    axes[0].set(title='Training objective (terms differ)',xlabel='Epoch',ylabel='Loss')
    axes[1].set(title='Validation pseudo-supervision loss',xlabel='Epoch',ylabel='Same mask loss for all variants')
    axes[2].set(title='Validation three-point error',xlabel='Epoch',ylabel='Mean normalized vertex distance')
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=.15)
    for extension in ['png','svg']:
        fig.savefig(figures/f'learning_curves.{extension}',dpi=300)
    fig.savefig(figures/'learning_curves_preview.png',dpi=120)
    plt.close(fig)
    names=list(variants)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    axes[0].bar([model_label(n) for n in names],[summary[v]['test']['pseudo_mask_iou'] for v in names],color=['#a3b5c5','#67a9a1','#217d88'])
    axes[0].set(ylim=(0,1),title='Test agreement with automatic masks',ylabel='Pseudo IoU (NOT ground-truth IoU)')
    axes[1].bar([model_label(n) for n in names],[summary[v]['test']['flow_warp_mae'] for v in names],color=['#a3b5c5','#67a9a1','#217d88'])
    axes[1].set(title='Test temporal consistency diagnostic',ylabel='Flow-warped probability MAE')
    for ax in axes:
        ax.tick_params(axis='x',rotation=25)
    for extension in ['png','svg']:
        fig.savefig(figures/f'ablation_diagnostics.{extension}',dpi=300)
    fig.savefig(figures/'ablation_diagnostics_preview.png',dpi=120)
    plt.close(fig)
    architecture_figure(figures,config)
    counts=metadata['counts']
    lines=[f'# {config["name"]} 实验结果','',f'数据集：训练 {counts["train"]}，验证 {counts["val"]}，测试 {counts["test"]}。',
           '',f'所有模型完整训练 {config["epochs"]} 个 epoch；关闭早停。best.pt 为验证最优轮次，last.pt 为最后一轮；最佳轮次不是停止轮次。',
           '', 'DINO 系列使用官方 DINOv2 ViT-S/14 Registers 冻结骨干与轻量解码器；另有从零训练的小型 RGB U-Net 对照。无人工标签时为伪标签自训练，不是标准的有标签 few-shot 基准。',
           '', '**结果尚未经真实标注验证。** 伪标签 IoU 表示与自动候选的一致性；不能当作识别准确率或与顶会论文的真实 IoU 数值比较。毫米字段为空。',
           '', '| 模型 | 验证伪标签 IoU | 测试伪标签 IoU | 测试时序 MAE |', '|---|---:|---:|---:|']
    for variant in names:
        lines.append(f'| {model_label(variant)} | {summary[variant]["val"]["pseudo_mask_iou"]:.4f} | {summary[variant]["test"]["pseudo_mask_iou"]:.4f} | {summary[variant]["test"]["flow_warp_mae"]:.4f} |')
    lines.extend(['','本地结构对照和消融实验，不是对原论文的完整复现。更多模型及逐帧诊断见 [交互式复核页](review.html) 和 [质量说明](quality/README.md)。',
                  '', '[论文格式对比图](figures/qualitative_comparison.png) · [训练曲线](figures/learning_curves.png) · [消融诊断](figures/ablation_diagnostics.png) · [模型结构](figures/architecture.png)',
                  '', '所选模型直接回归 A/B/C 三点坐标，再连接 AB、BC、CA 三条直边。叠加图不再显示弯曲轮廓或区域填色。`triangles/` 提供纯三角形 SVG，`masks/` 仅为同一三角形的栅格化结果。',
                  '', '训练使用三点伪标签、坐标损失和可微三角形栅格化损失。B/C 的自动目标仍来自最大可见展开截面，尚非已确认的物理钳口；三角形形状正确不代表三点位置准确。无效区外预测明确标记失败，不裁切成额外顶点。',
                  '', '7:2:1 为同视频内时间分块；相邻边界帧仍有相关性，且这些图片在以前开发中已被查看，不能宣称独立盲测。'])
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def architecture_figure(folder,config):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig,ax=plt.subplots(figsize=(12,4.3),layout='constrained')
    ax.set(xlim=(0,12),ylim=(0,4))
    ax.axis('off')
    topo=config.get('topology',{}).get('enabled',False)
    scene=config.get('architecture_revision')==5
    boxes=[(.2,2.3,2.0,1.0,'ROI + valid pixels\n134 / 39 / 19'),(2.8,2.3,2.4,1.0,'Frozen DINOv2-S/14\n4 register tokens'),
           (5.8,2.3,2.3,1.0,('Spatial coordinates + KL' if config.get('distribution_weight',0)>0 else 'Mapped spatial coordinates') if config.get('coordinate_head')=='calibrated_spatial' else 'Spatial evidence + integral' if config.get('coordinate_head')=='spatial_integral' else 'Local feature adapter'),(8.8,2.3,2.8,1.0,'Free apex + ordered base\nDifferentiable triangle' if scene else 'Six coordinate outputs\nDifferentiable triangle'),
           (2.8,.3,2.4,1.0,'Conditional PH context\nBounded zero-init residual' if scene and topo else 'Spatial global context\nConditional PH histograms' if topo else 'Fiber direction branch\nSpatial operators'),(5.8,.3,2.3,1.0,'Train: pseudo + translation\nEMA / weak Potts MRF' if scene else 'EMA + optical flow\n8-neighbor Potts MRF'),
           (8.8,.3,2.8,1.0,'A-B-C: three straight sides\nPixel length / triangle area')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor='#edf5f7',edgecolor='#317b88',linewidth=1.3))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
    for start,end in [((2.2,2.8),(2.8,2.8)),((5.2,2.8),(5.8,2.8)),((8.1,2.8),(8.8,2.8)),((5.2,.8),(8.8,2.55)),((8.1,.8),(8.8,2.5)),((10.2,2.3),(10.2,1.3))]:
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#317b88',lw=1.5))
    ax.set_title(config['name']+' | output remains three vertices',fontsize=12)
    for extension in ['png','svg']:
        fig.savefig(folder/f'architecture.{extension}',dpi=300)
    fig.savefig(folder/'architecture_preview.png',dpi=120)
    plt.close(fig)
