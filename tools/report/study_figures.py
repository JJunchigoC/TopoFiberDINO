"""Six-model, 500-epoch figures from completed runs only."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parents[2]
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
import numpy as np
from PIL import Image

load=lambda p:json.loads((ROOT/p).read_text(encoding='utf-8'))
out=ROOT/'reports/assets/study500'; out.mkdir(parents=True,exist_ok=True)
study=load('outputs/triangle_ours/comparison/study500.json')
selection=load('outputs/triangle_ours/selection_protocol.json')
models=study['models']; names=list(models)
assert set(names)=={'baseline','directional','full','unet_small','dino_linear','dino_pyramid'}
assert all(m['epochs']==500 for m in models.values())
labels={'baseline':'DINO base','directional':'DINO direction','full':'TopoFiberDINO (ours)','unet_small':'RGB U-Net','dino_linear':'DINO linear','dino_pyramid':'DINO pyramid'}
colors=['#168782' if n=='full' else '#6c859b' for n in names]
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,'axes.unicode_minus':False,'savefig.facecolor':'white'})
def save(fig,name):
    for ext in ['png','svg']: fig.savefig(out/f'{name}.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,key,title in zip(axes,['pseudo_mask_iou','pseudo_vertex_error_normalized'],['测试伪 IoU  越高越好','测试伪顶点误差  越低越好']):
    values=[models[n]['test'][key] for n in names]
    b=ax.barh([labels[n] for n in names],values,color=colors,height=.6)
    ax.bar_label(b,fmt='%.4f',padding=3,fontsize=10)
    ax.set_xlim(0,max(values)*1.27); ax.invert_yaxis(); ax.set_title(title)
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
save(fig,'metrics')

fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for name,trial in selection['trials'].items():
    hist=load(f'.cache/study500/trials/{name}/full/history.json')
    axes[0].plot([r['epoch'] for r in hist],[r['pseudo_vertex_error_normalized'] for r in hist],lw=1,label=name)
axes[0].set(xlabel='Epoch',ylabel='验证伪顶点误差',title='仅按验证三点误差选择方案'); axes[0].legend(fontsize=8,frameon=False)
v=[models['full'][s]['pseudo_mask_iou'] for s in ['train','val','test']]
b=axes[1].bar(['训练 134','验证 39','测试 19'],v,color=['#168782','#6c859b','#abb7c2'],width=.55)
axes[1].bar_label(b,fmt='%.4f',padding=4)
axes[1].set(ylim=(0,1.05),ylabel='伪 IoU',title='最终 ours 的分组表现')
for ax in axes: ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
save(fig,'selection')

fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for n in names:
    hist=load(f'outputs/triangle_ours/{n}/history.json')
    axes[0].plot([r['epoch'] for r in hist],[r['pseudo_vertex_error_normalized'] for r in hist],lw=1.4 if n=='full' else .8,label=labels[n])
    axes[1].plot([r['epoch'] for r in hist],[r['pseudo_mask_iou'] for r in hist],lw=1.4 if n=='full' else .8,label=labels[n])
for ax,title in zip(axes,['验证三点误差  越低越好','验证伪 IoU  越高越好']):
    ax.set(xlabel='Epoch',title=title); ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.15)
axes[0].legend(fontsize=8,ncol=2,frameon=False)
save(fig,'learning')

fig,ax=plt.subplots(figsize=(10,4.8)); ax.set(xlim=(0,10),ylim=(0,4.8)); ax.axis('off')
def box(x,y,w,h,text):
    ax.add_patch(Rectangle((x,y),w,h,facecolor='#eef4f5',edgecolor='#536a7c'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
def arrow(a,b,dashed=False): ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#536a7c',linestyle='--' if dashed else '-'))
box(.1,3.45,1.4,.8,'ROI\n有效区域')
box(2,3.45,2,.8,'冻结 DINOv2\nRegisters')
box(4.6,3.45,2.3,.8,'方向适配器\n空间坐标证据')
box(7.5,3.45,2.4,.8,'场景几何映射\nA B C 与三直边')
for a,b in [((1.5,3.85),(2,3.85)),((4,3.85),(4.6,3.85)),((6.9,3.85),(7.5,3.85))]: arrow(a,b)
box(2,1.75,2,1,'训练集簇与 DTM\n拓扑监督目标')
box(4.6,1.75,2.3,1,'32 维全局分支\n有界坐标残差')
arrow((3,3.45),(3,2.75)); arrow((4,3.45),(5.1,2.75)); arrow((4,2.25),(4.6,2.25),True); arrow((6.9,2.25),(8.2,3.45))
box(.1,.1,9.8,.85,'验证候选  原空间头  坐标映射修正  坐标分布 KL 0.15 / 0.60\n共同训练项  三点与区域目标  同步平移  弱 MRF  EMA 光流')
ax.text(5,4.6,'最终头与分布权重按验证集选择  不依赖测试标签',ha='center',fontsize=13)
save(fig,'pipeline')

meta={r['id']:r for r in load('dataset/metadata.json')['frames']}
predictions={n:{p['id']:p['vertices_roi'] for p in load(f'outputs/triangle_ours/{n}/predictions.json')} for n in names}
ids=load('outputs/triangle_ours/figures/selection.json')['ids']
for k,id_ in enumerate(ids):
    fig,axes=plt.subplots(2,3,figsize=(10,4.7),layout='constrained')
    im=np.asarray(Image.open(ROOT/'dataset'/meta[id_]['image']).convert('RGB'))
    for ax,n in zip(axes.flat,names):
        ax.imshow(im); ax.axis('off'); pts=predictions[n][id_]
        if pts is not None:
            p=np.array(pts); closed=np.vstack([p,p[0]])
            ax.plot(closed[:,0],closed[:,1],color='#00d9e5',lw=1.8); ax.scatter(p[:,0],p[:,1],s=18,color='#ff6839')
            for label,(x,y) in zip('ABC',p): ax.text(x+6,y-6,label,color='#ff6839',fontsize=13,weight='bold',bbox=dict(facecolor='black',alpha=.35,pad=.5,edgecolor='none'))
        else: ax.text(.5,.5,'INVALID TRIANGLE',transform=ax.transAxes,ha='center',color='red')
        ax.set_title(labels[n],fontsize=12)
    fig.suptitle(id_,fontsize=12)
    save(fig,f'frame_{k+1}')

snapshot=dict(study=study,selection=selection,selected_ids=ids,verification=load('outputs/triangle_ours/verification.json'))
(out/'source_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
print('500-epoch report figures prepared from completed six-model outputs.')
