import copy
import math
import time
import torch
from torch.nn import functional as F
from .io import dump
from .model import build_model,MODEL_INFO
from .mrf import settings as mrf_settings,mrf_energy
from .topology import settings as topology_settings,topology_loss,export_predictions


def forward_model(model,data,ids,image=None):
    image=data['images'][ids] if image is None else image
    if getattr(model,'uses_topology',False):
        return model(data['features'][ids],image,valid=data['valid'][ids])
    return model(data['features'][ids],image)


def optimizer_settings(config):
    lr=float(config['learning_rate'])
    decay=float(config.get('weight_decay',.0001))
    if config.get('checkpoint_selection','region_loss') not in ['region_loss','vertex_error']:
        raise ValueError('checkpoint_selection must be region_loss or vertex_error')
    if config.get('coordinate_head','pooled') not in ['pooled','spatial_integral','calibrated_spatial']:
        raise ValueError('Unknown coordinate_head')
    if not isinstance(config.get('bounded_translation',False),bool): raise ValueError('bounded_translation must be boolean')
    if not math.isfinite(lr) or lr<=0 or not math.isfinite(decay) or decay<0:
        raise ValueError('learning_rate must be finite/positive; weight_decay finite/nonnegative')
    for key,default in [('feature_dropout',0),('translation_weight',0),('topology_residual_scale',.15),('distribution_weight',0)]:
        value=float(config.get(key,default))
        if not math.isfinite(value) or value<0 or (key=='feature_dropout' and value>=1):
            raise ValueError(f'Invalid {key}')
    if config.get('distribution_weight',0)>0 and config.get('coordinate_head')!='calibrated_spatial':
        raise ValueError('distribution_weight requires calibrated_spatial independent evidence maps')
    shift=config.get('translation_range',[.02,.025])
    if len(shift)!=2 or any(not math.isfinite(v) or v<0 or v>.05 for v in shift):
        raise ValueError('translation_range requires two values in [0, .05]')
    return dict(lr=lr,weight_decay=decay)


def coordinate_distribution_loss(output,points,support):
    """Axis-marginal KL on independent spatial evidence, never rendered vertices.

    Targets are truncated to the same fixed support as the coordinate head.
    This remains pseudo supervision, not human landmark ground truth.
    """
    lp=output['coordinate_log_probability']; h,w=lp.shape[-2:]
    y,x=torch.meshgrid(torch.linspace(0,1,h,device=lp.device),torch.linspace(0,1,w,device=lp.device),indexing='ij')
    dist=((x[None,None]-points[:,:,0,None,None])/.025).square()+((y[None,None]-points[:,:,1,None,None])/.025).square()
    target=(-.5*dist).masked_fill(lp.detach() < -1000,-1e4).flatten(2).softmax(-1).reshape_as(lp)
    loss=lp.new_zeros(lp.shape[:2])
    for dim in [-1,-2]:
        marginal=target.sum(dim); predicted=torch.logsumexp(lp,dim=dim)
        loss+=(marginal*(marginal.clamp_min(1e-12).log()-predicted)).sum(-1)
    return weighted_mean(loss,support.reshape(-1,1))


def weighted_mean(value, weight):
    return (value*weight).sum()/weight.expand_as(value).sum().clamp_min(1)


def selection_score(metrics,config):
    key='pseudo_vertex_error_normalized' if config.get('checkpoint_selection','region_loss')=='vertex_error' else 'pseudo_supervision_loss'
    value=metrics.get(key)
    if value is None or not math.isfinite(value): raise ValueError(f'No finite validation selection metric: {key}')
    return value


def segmentation_loss(logits, target, confidence):
    bce=weighted_mean(F.binary_cross_entropy_with_logits(logits,target,reduction='none'),confidence)
    prediction=logits.sigmoid()
    intersection=(prediction*target*confidence).sum((1,2,3))
    denominator=((prediction+target)*confidence).sum((1,2,3))
    dice=1-((2*intersection+1)/(denominator+1)).mean()
    return bce+dice


def translated_batch(data,ids,shift):
    """Translate inputs AND supervision by normalized [dx,dy], train only.

    Auxiliary losses and the optical-flow teacher stay on the original batch;
    their coordinate systems and topology targets are never silently mixed.
    """
    theta=torch.eye(2,3,device=shift.device,dtype=shift.dtype)[None].repeat(len(ids),1,1)
    theta[:,:,2]=-2*shift
    result={}
    for key in ['features','images','valid','target','confidence']:
        value=data[key][ids]
        grid=F.affine_grid(theta,value.shape,align_corners=True)
        mode='nearest' if key=='target' else 'bilinear'
        result[key]=F.grid_sample(value,grid,mode=mode,padding_mode='zeros',align_corners=True)
    result['valid']=(result['valid']>.99999).float()
    result['confidence']=result['confidence']*result['valid']
    result['target']=result['target']*result['valid']
    result['points']=data['points'][ids]+shift[:,None,:]
    result['has_vertices']=data['has_vertices'][ids]
    return result


def bounded_shift(points,shift):
    """Keep translated targets inside the scene head's representable range.

    Used for the vertical-base automatic targets. A target outside these priors
    receives no translation; the original supervision is left unchanged.
    """
    a,b,c=points.unbind(1); base=(b[:,0]+c[:,0])/2
    lo=torch.stack([torch.maximum(.1001-a[:,0],.3701-base),torch.maximum(.2601-a[:,1],.2401-b[:,1])],1)
    hi=torch.stack([torch.minimum(.3499-a[:,0],.7799-base),torch.minimum(torch.minimum(.5999-a[:,1],.5799-b[:,1]),.8399-c[:,1])],1)
    possible=((lo<=0)&(hi>=0)).all(1)
    limited=torch.minimum(torch.maximum(shift,lo),hi)
    return torch.where(possible[:,None],limited,torch.zeros_like(shift))


def batch_loss(model,data,ids,config,variant,teacher=None):
    image=data['images'][ids].clone()
    image[:,:3]=(image[:,:3]*(.9+.2*torch.rand((len(ids),1,1,1),device=image.device))).clamp(0,1)
    output=forward_model(model,data,ids,image)
    loss=segmentation_loss(output['mask'],data['target'][ids],data['confidence'][ids])
    parts={'segmentation':float(loss.detach())}
    coordinate=weighted_mean(F.smooth_l1_loss(output['points'],data['points'][ids],reduction='none',beta=.05),data['has_vertices'][ids].reshape(-1,1,1))
    loss=loss+config['coordinate_weight']*coordinate
    parts['coordinates']=float(coordinate.detach())
    distribution_weight=config.get('distribution_weight',0) if variant=='full' else 0
    if distribution_weight>0:
        distribution=coordinate_distribution_loss(output,data['points'][ids],data['has_vertices'][ids])
        loss=loss+distribution_weight*distribution
        parts['coordinate_distribution']=float(distribution.detach())
    if MODEL_INFO[variant]['auxiliary'] and config['boundary_weight']>0:
        boundary_weight=data['valid'][ids]*(1+6*data['boundary'][ids])
        boundary=weighted_mean(F.binary_cross_entropy_with_logits(output['boundary'],data['boundary'][ids],reduction='none'),boundary_weight)
        loss=loss+config['boundary_weight']*boundary
        parts['boundary']=float(boundary.detach())
    if MODEL_INFO[variant]['auxiliary'] and config['vertex_weight']>0:
        heat=data['heatmaps'][ids]
        vertex_weight=data['valid'][ids]*data['has_vertices'][ids]*(1+30*heat)
        vertex=weighted_mean(F.binary_cross_entropy_with_logits(output['vertices'],heat,reduction='none'),vertex_weight)
        loss=loss+config['vertex_weight']*vertex
        parts['vertices']=float(vertex.detach())
    if MODEL_INFO[variant]['temporal']:
        neighbors=data['neighbors'][ids]
        with torch.no_grad():
            neighbor=forward_model(teacher,data,neighbors)['mask'].sigmoid()
            warped=F.grid_sample(neighbor,data['grids'][ids],align_corners=True,padding_mode='zeros')
        gate=data['visibility'][ids]*data['confidence'][ids]
        temporal=weighted_mean((output['mask'].sigmoid()-warped).abs(),gate)
        loss=loss+config['temporal_weight']*temporal
        parts['temporal']=float(temporal.detach())
    mrf=mrf_settings(config)
    if variant=='full' and mrf['weight']>0:
        pairwise=mrf_energy(output['mask'],data['mrf_weights'][ids],data['mrf_normalizer'][ids])
        loss=loss+mrf['weight']*pairwise
        parts['mrf_pairwise']=float(pairwise.detach())
    if getattr(model,'uses_topology',False):
        topological,topo_parts=topology_loss(output,data,ids,config)
        ramp=min(config.get('_epoch',1)/topology_settings(config)['warmup_epochs'],1.)
        loss=loss+ramp*topological
        parts.update(topo_parts,topology_weighted=float((ramp*topological).detach()))
    if config.get('translation_weight',0)>0:
        shift=(2*torch.rand((len(ids),2),device=image.device)-1)*image.new_tensor(config.get('translation_range',[.02,.025]))
        if config.get('bounded_translation',False): shift=bounded_shift(data['points'][ids],shift)
        augmented=translated_batch(data,ids,shift)
        shifted=forward_model(model,augmented,torch.arange(len(ids),device=image.device))
        spatial=segmentation_loss(shifted['mask'],augmented['target'],augmented['confidence'])
        spatial=spatial+config['coordinate_weight']*weighted_mean(F.smooth_l1_loss(shifted['points'],augmented['points'],reduction='none',beta=.05),augmented['has_vertices'].reshape(-1,1,1))
        if distribution_weight>0:
            spatial=spatial+distribution_weight*coordinate_distribution_loss(shifted,augmented['points'],augmented['has_vertices'])
        loss=loss+config['translation_weight']*spatial
        parts['translation_weighted']=float((config['translation_weight']*spatial).detach())
    return loss,parts


@torch.no_grad()
def evaluate(model,data,indices,batch_size=8,collect=False):
    model.eval()
    losses,intersection,union,count=[],0.,0.,0
    probabilities,points=[],[]
    vertex_distance,vertex_count=0.,0
    for ids in indices.split(batch_size):
        output=forward_model(model,data,ids)
        losses.append(float(segmentation_loss(output['mask'],data['target'][ids],data['confidence'][ids]))*len(ids))
        mask=(output['mask'].sigmoid()>.5)*data['valid'][ids].bool()
        target=data['target'][ids].bool()
        intersection+=float((mask&target).sum())
        union+=float((mask|target).sum())
        count+=len(ids)
        if collect:
            probabilities.append(output['mask'].sigmoid().cpu())
            points.append(output['points'].cpu())
        support=data['has_vertices'][ids].reshape(-1)
        distances=(output['points']-data['points'][ids]).norm(dim=-1).mean(1)
        vertex_distance+=float((distances*support).sum()); vertex_count+=float(support.sum())
    return dict(pseudo_supervision_loss=sum(losses)/count,
                pseudo_mask_iou=intersection/max(union,1),true_mask_iou=None,
                pseudo_vertex_error_normalized=vertex_distance/vertex_count if vertex_count else None,
                note='Triangle pseudo-label agreement; not true location accuracy.'),torch.cat(probabilities) if collect else None,torch.cat(points) if collect else None


def train_variant(variant,data,rows,config,out,device,predict_only=False,validation_only=False):
    if variant=='full': MODEL_INFO['full']['label']=config.get('name',MODEL_INFO['full']['label'])
    folder=out/variant
    folder.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(config['seed'])
    model=build_model(variant,config).to(device)
    if getattr(model,'uses_topology',False): model.topology.centers.copy_(data['topology_centers'])
    train_ids=torch.tensor([i for i,r in enumerate(rows) if r['split']=='train'],device=device)
    val_ids=torch.tensor([i for i,r in enumerate(rows) if r['split']=='val'],device=device)
    test_ids=torch.tensor([i for i,r in enumerate(rows) if r['split']=='test'],device=device)
    checkpoint=folder/'best.pt'
    if not predict_only:
        optimizer=torch.optim.AdamW(model.parameters(),**optimizer_settings(config))
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'])
        teacher=copy.deepcopy(model).eval().requires_grad_(False) if MODEL_INFO[variant]['temporal'] else None
        best,best_epoch,history=float('inf'),0,[]
        started=time.perf_counter()
        for epoch in range(1,config['epochs']+1):
            model.train()
            order=train_ids[torch.randperm(len(train_ids),device=device)]
            losses=[]
            components={}
            for ids in order.split(config['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                loss,parts=batch_loss(model,data,ids,{**config,'_epoch':epoch},variant,teacher)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                optimizer.step()
                if teacher is not None:
                    with torch.no_grad():
                        for target,source in zip(teacher.parameters(),model.parameters()):
                            target.mul_(.99).add_(source,alpha=.01)
                losses.append(float(loss.detach())*len(ids))
                for key,value in parts.items():
                    components[key]=components.get(key,0)+value*len(ids)
            metrics,_,_=evaluate(model,data,val_ids,config['batch_size'])
            history.append(dict(epoch=epoch,train_loss=sum(losses)/len(train_ids),
                                **metrics,components={k:v/len(train_ids) for k,v in components.items()}))
            score=selection_score(metrics,config)
            if score<best:
                best=score
                best_epoch=epoch
                torch.save(dict(state_dict=model.state_dict(),epoch=epoch,variant=variant,
                                config=config,model_name=MODEL_INFO[variant]['label'],
                                validation_pseudo_loss=metrics['pseudo_supervision_loss'],
                                selection_metric=config.get('checkpoint_selection','region_loss'),validation_selection_score=best),checkpoint)
            if epoch==1 or epoch%config.get('log_interval',25)==0 or epoch==config['epochs']:
                dump(folder/'history.json',history)
                print(f'{variant} {epoch}/{config["epochs"]}: train {history[-1]["train_loss"]:.4f}; val region {metrics["pseudo_supervision_loss"]:.4f}; best {config.get("checkpoint_selection","region_loss")} {best:.4f} @ epoch {best_epoch}; pseudo IoU {metrics["pseudo_mask_iou"]:.3f}',flush=True)
            schedule.step()
        torch.save(dict(state_dict=model.state_dict(),epoch=epoch,variant=variant,config=config,model_name=MODEL_INFO[variant]['label']),folder/'last.pt')
        dump(folder/'training_summary.json',dict(epochs_run=epoch,epochs_requested=config['epochs'],early_stopping=False,best_epoch=best_epoch,last_epoch=epoch,inference_checkpoint='best.pt',stop_reason='completed_requested_epochs',elapsed_seconds=time.perf_counter()-started,
                                                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                                                 temporal_loss_enabled=MODEL_INFO[variant]['temporal'],ground_truth_used=config['shots']>0,
                                                 mrf=mrf_settings(config) if variant=='full' else None,
                                                 topology=topology_settings(config) if variant=='full' else None,
                                                 checkpoint_selection=config.get('checkpoint_selection','region_loss'),validation_selection_score=best,
                                                 model_info=MODEL_INFO[variant]))
    state=torch.load(checkpoint,map_location=device,weights_only=True)
    if state['config'].get('checkpoint_selection','region_loss')!=config.get('checkpoint_selection','region_loss'):
        raise ValueError('Checkpoint was selected with a different validation metric; retrain with the requested selection policy.')
    if state['config'].get('representation')!='triangle_vertices_v2' or state['config'].get('architecture_revision')!=config['architecture_revision']:
        raise ValueError('Checkpoint architecture differs; run default training to replace it.')
    if variant=='full' and state['config'].get('coordinate_head','pooled')!=config.get('coordinate_head','pooled'):
        raise ValueError('Coordinate decoding differs from the trained checkpoint; retrain full.')
    if variant=='full' and mrf_settings(state['config'])!=mrf_settings(config):
        raise ValueError('Ours checkpoint was trained with different MRF settings; retrain full.')
    if variant=='full' and topology_settings(state['config'])!=topology_settings(config):
        raise ValueError('Ours checkpoint has different topology settings; retrain full.')
    model.load_state_dict(state['state_dict'])
    if validation_only:
        # Development trials never evaluate test or generate test predictions.
        metrics,_,_=evaluate(model,data,val_ids,config['batch_size'])
        dump(folder/'validation_metrics.json',metrics)
        return metrics,None,None
    metrics={}
    # Test is evaluated only after checkpoint selection. No test loss enters training.
    for split,ids in [('train',train_ids),('val',val_ids),('test',test_ids)]:
        metrics[split],_,_=evaluate(model,data,ids,config['batch_size'])
    _,probabilities,points=evaluate(model,data,torch.arange(len(rows),device=device),config['batch_size'],collect=True)
    # Dense maps are exported as masks/figures; do not retain gigabytes of
    # redundant intermediate tensors that can be regenerated from best.pt.
    dump(folder/'metrics.json',metrics)
    if getattr(model,'uses_topology',False): export_predictions(model,data,rows,config,folder)
    return metrics,probabilities,points
