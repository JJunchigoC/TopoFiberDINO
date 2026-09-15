"""TopoSlide-inspired cluster-conditional persistent-homology supervision.

Independent scene adaptation: DINO clusters, DTM cubical H0/H1 histograms,
and a compact shared spatial encoder. No pathology code/weights are imported.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .io import dump,load


def settings(config):
    cfg=dict(enabled=False,clusters=4,grid_size=[28,12],embedding_dim=64,
             dtm_neighbors=3,bins=[0,.01,.02,.04,.08,.12,.2,.35,1.2],
             weight=.1,cdf_weight=.5,proportion_weight=.05,warmup_epochs=25)
    provided=config.get('topology',{})
    if set(provided)-set(cfg): raise ValueError('Unknown topology setting')
    cfg.update(provided)
    if not isinstance(cfg['enabled'],bool): raise ValueError('topology.enabled must be boolean')
    for key in ['clusters','embedding_dim','dtm_neighbors','warmup_epochs']:
        if not isinstance(cfg[key],int) or cfg[key]<1: raise ValueError(f'Invalid topology.{key}')
    if cfg['embedding_dim']%4: raise ValueError('topology.embedding_dim must be divisible by 4')
    if len(cfg['grid_size'])!=2 or any(not isinstance(v,int) or v<2 for v in cfg['grid_size']):
        raise ValueError('topology.grid_size requires two integers >=2 (width,height)')
    if len(cfg['bins'])<3 or cfg['bins'][0]!=0 or not all(math.isfinite(v) for v in cfg['bins']) or any(b<=a for a,b in zip(cfg['bins'],cfg['bins'][1:])):
        raise ValueError('topology.bins must increase strictly from zero')
    for key in ['weight','cdf_weight','proportion_weight']:
        if not math.isfinite(cfg[key]) or cfg[key]<0: raise ValueError(f'Invalid topology.{key}')
    return cfg


def pooled_inputs(features,valid,grid_size):
    size=tuple(reversed(grid_size))
    support=F.adaptive_avg_pool2d(valid.float(),size)
    # Exclude cells overlapping invalid padding, rather than assigning it a cluster.
    keep=support[:,0]>.99999
    values=F.adaptive_avg_pool2d(features.float(),size).permute(0,2,3,1)
    return values,keep


@torch.no_grad()
def fit_clusters(values,keep,train_indices,count,seed):
    x=F.normalize(values[train_indices][keep[train_indices]],dim=-1)
    if len(x)<count: raise ValueError('Too few valid training patches for topology clusters')
    generator=torch.Generator().manual_seed(seed)
    if len(x)>16000: x=x[torch.randperm(len(x),generator=generator)[:16000]]
    centers=[x[torch.randint(len(x),(1,),generator=generator).item()]]
    for _ in range(1,count):
        similarity=x@torch.stack(centers).T
        centers.append(x[similarity.max(1).values.argmin()])
    centers=torch.stack(centers)
    for _ in range(25):
        assignment=(x@centers.T).argmax(1)
        updated=torch.stack([x[assignment==k].mean(0) if (assignment==k).any() else centers[k] for k in range(count)])
        centers=F.normalize(updated,dim=-1)
    return centers


def persistence_histogram(cluster,valid,xy,neighbors,bins):
    """Finite intervals of the DTM sublevel cubical complex, dimensions 0/1."""
    import gudhi
    from scipy.spatial import cKDTree
    result=np.zeros((2,len(bins)-1),np.float32)
    points=xy[cluster&valid]
    if not len(points): return result
    distances=cKDTree(points).query(xy.reshape(-1,2),k=min(neighbors,len(points)))[0]
    field=(distances.mean(1) if distances.ndim==2 else distances).reshape(valid.shape)
    field[~valid]=np.inf
    complex_=gudhi.CubicalComplex(top_dimensional_cells=field)
    complex_.compute_persistence(homology_coeff_field=2,min_persistence=0)
    for dimension in [0,1]:
        intervals=complex_.persistence_intervals_in_dimension(dimension)
        intervals=intervals[np.isfinite(intervals).all(1)]
        lifetime=intervals[:,1]-intervals[:,0]
        # Last bin includes overflow; no persistent feature is silently dropped.
        result[dimension]=np.histogram(np.minimum(lifetime,np.nextafter(float(bins[-1]),0)),bins=bins)[0]
    return result


@torch.no_grad()
def prepare_topology(data,rows,config,out,reuse=False):
    cfg=settings(config); folder=out/'topology'; folder.mkdir(exist_ok=True)
    values,keep=pooled_inputs(data['features'],data['valid'],cfg['grid_size'])
    train=[i for i,r in enumerate(rows) if r['split']=='train']
    if reuse:
        fitted=torch.load(folder/'fitted.pt',map_location='cpu',weights_only=True)
        if fitted['settings']!=cfg or fitted['training_ids']!=[rows[i]['id'] for i in train]:
            raise ValueError('Incompatible topology preprocessing; retrain ours')
        centers=fitted['centers']
    else:
        centers=fit_clusters(values,keep,train,cfg['clusters'],config['seed'])
    assignments=(F.normalize(values,dim=-1)@centers.T).argmax(-1).numpy()
    valid=keep.numpy(); h,w=valid.shape[-2:]
    # Physical aspect of the work image; distance units are fractions of its width.
    yy,xx=np.mgrid[:h,:w]
    xy=np.stack([(xx+.5)/w,(yy+.5)/h*config['work_size'][1]/config['work_size'][0]],-1)
    hist=[]; proportions=[]
    for i in range(len(rows)):
        if not valid[i].any(): raise ValueError(f'No valid topology tokens: {rows[i]["id"]}')
        hist.append([persistence_histogram(assignments[i]==k,valid[i],xy,cfg['dtm_neighbors'],cfg['bins']) for k in range(cfg['clusters'])])
        proportions.append([float(((assignments[i]==k)&valid[i]).sum()/valid[i].sum()) for k in range(cfg['clusters'])])
    raw=torch.tensor(np.asarray(hist))
    scale=fitted['histogram_scale'] if reuse else raw[train].amax(dim=(0,1)).clamp_min(1)
    if not reuse:
        fitted=dict(centers=centers,histogram_scale=scale,settings=cfg,training_ids=[rows[i]['id'] for i in train])
        torch.save(fitted,folder/'fitted.pt')
    assignments[~valid]=-1
    dump(folder/'targets.json',dict(settings=cfg,homology='Cubical DTM sublevel H0/H1, finite intervals only; essential intervals omitted',
         distance_units='work image width',histogram_scale=scale.tolist(),training_ids=fitted['training_ids'],
         fit_scope='Only training patches fit clusters and histogram scales; val/test transformed independently.',
         frames=[dict(id=r['id'],split=r['split'],histogram=raw[i].tolist(),proportions=proportions[i],cluster_map=assignments[i].tolist()) for i,r in enumerate(rows)]))
    return dict(topology_hist=raw/scale,topology_proportions=torch.tensor(proportions),topology_centers=centers)


class TopologyContext(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        d=cfg['embedding_dim']; bins=len(cfg['bins'])-1
        self.register_buffer('centers',torch.zeros(cfg['clusters'],384))
        self.project=nn.Linear(384,d)
        self.position=nn.Linear(2,d,bias=False)
        self.attention=nn.TransformerEncoderLayer(d,4,dim_feedforward=d*2,dropout=0,batch_first=True,norm_first=True)
        self.norm=nn.LayerNorm(d)
        self.condition=nn.Linear(384,d)
        self.gate=nn.Sequential(nn.Linear(2*d,d),nn.Sigmoid())
        self.hist_head=nn.Linear(d,2*bins)
        self.proportion_head=nn.Linear(d,1)

    def forward(self,features,valid):
        values,keep=pooled_inputs(features,valid,self.cfg['grid_size'])
        n,h,w,_=values.shape
        values=values.reshape(n,h*w,384); keep=keep.flatten(1)
        if not keep.any(1).all(): raise ValueError('Topology context requires valid patches')
        # Invalid inputs cannot contribute to keys/values or pooled context.
        values=values.masked_fill(~keep[:,:,None],0)
        y,x=torch.meshgrid(torch.linspace(0,1,h,device=values.device),torch.linspace(0,1,w,device=values.device),indexing='ij')
        xy=torch.stack([x,y],-1).reshape(1,h*w,2)
        tokens=self.project(values)+self.position(xy)
        encoded=self.attention(tokens,src_key_padding_mask=~keep)
        context=self.norm((encoded*keep[:,:,None]).sum(1)/keep.sum(1,keepdim=True))
        labels=(F.normalize(values,dim=-1)@self.centers.T).argmax(-1)
        membership=F.one_hot(labels,self.cfg['clusters']).float()*keep[:,:,None]
        count=membership.sum(1)
        queries=torch.einsum('ntk,ntd->nkd',membership,values)/count[:,:,None].clamp_min(1)
        queries=torch.where(count[:,:,None]>0,queries,self.centers[None])
        query=self.condition(queries)
        global_=context[:,None].expand_as(query)
        conditional=self.gate(torch.cat([global_,query],-1))*global_
        histogram=F.softplus(self.hist_head(conditional)).reshape(n,self.cfg['clusters'],2,-1)
        proportion=self.proportion_head(conditional).squeeze(-1).sigmoid()
        return context,histogram,proportion


def topology_loss(output,data,ids,config):
    cfg=settings(config)
    prediction=output['topology_hist']; target=data['topology_hist'][ids]
    direct=F.smooth_l1_loss(prediction,target,beta=.1)
    # Average cumulative counts keeps magnitude independent of histogram length.
    cumulative=F.smooth_l1_loss(prediction.cumsum(-1)/prediction.shape[-1],target.cumsum(-1)/target.shape[-1],beta=.1)
    proportions=F.smooth_l1_loss(output['topology_proportions'],data['topology_proportions'][ids],beta=.1)
    total=cfg['weight']*(direct+cfg['cdf_weight']*cumulative)+cfg['proportion_weight']*proportions
    return total,dict(topology_hist=float(direct.detach()),topology_cdf=float(cumulative.detach()),topology_proportion=float(proportions.detach()))


@torch.no_grad()
def export_predictions(model,data,rows,config,out):
    records=[]; model.eval()
    for start in range(0,len(rows),config['batch_size']):
        ids=torch.arange(start,min(start+config['batch_size'],len(rows)),device=data['images'].device)
        output=model(data['features'][ids],data['images'][ids],valid=data['valid'][ids])
        for j,i in enumerate(ids.tolist()):
            records.append(dict(id=rows[i]['id'],split=rows[i]['split'],predicted_histogram=output['topology_hist'][j].cpu().tolist(),
                 target_histogram=data['topology_hist'][i].cpu().tolist(),predicted_proportions=output['topology_proportions'][j].cpu().tolist(),
                 target_proportions=data['topology_proportions'][i].cpu().tolist()))
    dump(out/'topology_predictions.json',dict(note='Self-supervised descriptor reconstruction, not vertex accuracy.',frames=records))


def export_figures(out):
    """Show real cluster maps and PH reconstruction on the fixed test frames."""
    if not (out/'full/topology_predictions.json').exists(): return
    import matplotlib.pyplot as plt
    from .io import ROOT,read
    target=load(out/'topology/targets.json')
    records=load(out/'full/topology_predictions.json')['frames']
    selected=load(out/'figures/selection.json')['ids']
    lookup={r['id']:r for r in load(ROOT/'dataset/metadata.json')['frames']}
    targets={r['id']:r for r in target['frames']}; predicted={r['id']:r for r in records}
    fig,axes=plt.subplots(len(selected),4,figsize=(14,2.5*len(selected)),squeeze=False,layout='constrained')
    for i,id_ in enumerate(selected):
        t=targets[id_]; p=predicted[id_]
        axes[i,0].imshow(read(ROOT/'dataset'/lookup[id_]['image'])[:,:,::-1]); axes[i,0].set_ylabel(id_,fontsize=8)
        clusters=np.ma.masked_less(np.array(t['cluster_map']),0)
        cmap=plt.get_cmap('tab10').copy(); cmap.set_bad('#444444')
        axes[i,1].imshow(clusters,cmap=cmap,vmin=0,vmax=9,interpolation='nearest')
        for dim in [0,1]:
            ax=axes[i,dim+2]
            a=np.array(p['target_histogram'])[:,dim].sum(0)
            b=np.array(p['predicted_histogram'])[:,dim].sum(0)
            ax.plot(a,'o-',label='PH target',ms=3); ax.plot(b,'x--',label='Predicted',ms=3)
            ax.set(xlabel='Persistence bin',ylabel='Sum of normalized counts'); ax.grid(alpha=.2)
            if i==0: ax.legend(fontsize=8)
        for ax in axes[i,:2]: ax.set_xticks([]); ax.set_yticks([])
        if i==0:
            for ax,title in zip(axes[i],['Input ROI','DINO cluster map; gray=invalid','H0: finite components','H1: finite holes']): ax.set_title(title,fontsize=10)
    fig.suptitle('TopoSlide-inspired self-supervision | cluster IDs are not physical labels | not vertex accuracy')
    for ext in ['png','svg']: fig.savefig(out/f'figures/topology_diagnostics.{ext}',dpi=300)
    fig.savefig(out/'figures/topology_diagnostics_preview.png',dpi=100); plt.close(fig)
    summary={}
    for split in ['train','val','test']:
        subset=[r for r in records if r['split']==split]
        summary[split]=dict(frames=len(subset),histogram_mae=float(np.mean([np.abs(np.array(r['predicted_histogram'])-r['target_histogram']).mean() for r in subset])),
                            proportion_mae=float(np.mean([np.abs(np.array(r['predicted_proportions'])-r['target_proportions']).mean() for r in subset])))
    dump(out/'topology/metrics.json',dict(metrics=summary,meaning='Self-supervised topology reconstruction; not location accuracy.'))
    with (out/'REPORT.md').open('a',encoding='utf-8') as stream:
        stream.write('\n## TopoSlide-inspired adaptation\n\nTrain-only DINO clusters, DTM cubical H0/H1 histograms, conditional histogram/CDF/proportion supervision and a shared global context branch. MRF and three-point output are retained. This is an independent ROI adaptation, not the published pathology model.\n\n[Topology diagnostics](figures/topology_diagnostics.png) · [Descriptor metrics](topology/metrics.json) · [Method and differences](../../docs/TOPOLOGY.md)\n')


def compare_ablation(out,control):
    """Compare only matched finished runs differing in the topology loss weights."""
    from .io import sha256,ROOT,read
    import copy
    if not (out/'full/topology_predictions.json').exists() or not (control/'verification.json').exists(): return
    a=load(out/'run.json'); b=load(control/'run.json')
    if not settings(a['config'])['enabled'] or settings(a['config'])['weight']<=0: return
    if not settings(b['config'])['enabled'] or any(settings(b['config'])[k]!=0 for k in ['weight','proportion_weight']): return
    aa=copy.deepcopy(a); bb=copy.deepcopy(b)
    for signature in [aa,bb]:
        for key in ['name','note']: signature['config'].pop(key,None)
        for key in ['weight','proportion_weight']: signature['config']['topology'].pop(key,None)
    if aa!=bb or load(control/'verification.json')['artifact_integrity']!='PASS': return
    ours_fit=torch.load(out/'topology/fitted.pt',weights_only=True)
    control_fit=torch.load(control/'topology/fitted.pt',weights_only=True)
    for key in ['centers','histogram_scale']: torch.testing.assert_close(ours_fit[key],control_fit[key],rtol=0,atol=0)
    ours=load(out/'metrics_comparison.json')['full']; baseline=load(control/'metrics_comparison.json')['full']
    keys=['pseudo_supervision_loss','pseudo_mask_iou','pseudo_vertex_error_normalized','flow_warp_mae']
    delta={s:{k:ours[s][k]-baseline[s][k] for k in keys} for s in ['train','val','test']}
    dump(out/'topology/ablation.json',dict(ours=ours,control=baseline,delta_ours_minus_control=delta,
         ours_config=a['config'],control_config=b['config'],ours_training=load(out/'full/training_summary.json'),
         control_training=load(control/'full/training_summary.json'),
         ours_checkpoint_sha256=sha256(out/'full/best.pt'),control_checkpoint_sha256=sha256(control/'full/best.pt'),
         topology_fit_identical=True,protocol='Same architecture/seed/optimizer/data/epochs; only topology auxiliary loss weights differ. One seed, no ground truth.'))
    import matplotlib.pyplot as plt
    from .figures import overlay
    predictions=[{r['id']:r for r in load(path/'full/predictions.json')} for path in [control,out]]
    rows={r['id']:r for r in load(ROOT/'dataset/metadata.json')['frames']}
    ids=load(out/'figures/selection.json')['ids']
    fig,axes=plt.subplots(len(ids),3,figsize=(12,2.3*len(ids)),squeeze=False,layout='constrained')
    for i,id_ in enumerate(ids):
        im=read(ROOT/'dataset'/rows[id_]['image'])
        panels=[im]+[overlay(im,None,p[id_]['vertices_roi']) for p in predictions]
        for j,panel in enumerate(panels):
            axes[i,j].imshow(panel[:,:,::-1]); axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
            if i==0: axes[i,j].set_title(['Input','Same network, topology loss OFF','TopoFiberDINO (ours)'][j],fontsize=10)
            if j==0: axes[i,j].set_ylabel(id_,fontsize=8)
    fig.suptitle('Matched topology-loss ablation | fixed test frames | no ground-truth vertices')
    for ext in ['png','svg']: fig.savefig(out/f'figures/topology_ablation.{ext}',dpi=300)
    fig.savefig(out/'figures/topology_ablation_preview.png',dpi=100); plt.close(fig)
