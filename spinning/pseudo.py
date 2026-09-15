"""Train-only prototypes plus independent image seeds; no old-run predictions used."""
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from .io import ROOT, read, dump, save
from .geometry import ridge_features, candidate_triangle
from .triangle import triangle_mask,annotation


def prepare_training_tensors(metadata, features, config, out):
    rows=metadata['frames']
    w,h=config['work_size']
    images,valids,seeds,confidence=[],[],[],[]
    vertex_targets=[]; vertex_available=[]
    for row in rows:
        native=read(ROOT/'dataset'/row['image'])
        native_valid=read(ROOT/'dataset'/row['valid_mask'],True)>0
        descriptors,residual=ridge_features(native)
        score=residual*(.3+.7*descriptors[:,:,5])
        proposal=candidate_triangle(score,native_valid)
        vertex_targets.append(np.zeros((3,2),np.float32) if proposal is None else (proposal['vertices']/np.array([native.shape[1]-1,native.shape[0]-1])).astype(np.float32))
        vertex_available.append(proposal is not None)
        seed=np.zeros(native.shape[:2],np.uint8) if proposal is None else proposal['mask']
        im=cv2.resize(native,(w,h),interpolation=cv2.INTER_AREA)
        desc,_=ridge_features(im)
        image=np.dstack([im[:,:,::-1].astype(np.float32)/255,desc[:,:,3:6]])
        valid=cv2.resize(native_valid.astype(np.float32),(w,h),interpolation=cv2.INTER_AREA)>.99999
        valid=cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        images.append(image.transpose(2,0,1))
        valids.append(valid.astype(np.float32))
        seeds.append((cv2.resize(seed,(w,h),interpolation=cv2.INTER_NEAREST)>0).astype(np.float32))
    images=torch.tensor(np.stack(images))
    valid=torch.tensor(np.stack(valids))[:,None]
    seeds=torch.tensor(np.stack(seeds))[:,None]
    train_indices=[i for i,r in enumerate(rows) if r['split']=='train']
    feat=F.normalize(features.float(),dim=1)
    down=F.interpolate(seeds,size=feat.shape[-2:],mode='area')
    valid_small=F.interpolate(valid,size=feat.shape[-2:],mode='area')>.999
    fg=(down>.85)&valid_small
    bg=(down<.02)&valid_small
    prototypes=[]
    for support in [bg,fg]:
        selection=support[train_indices].float()
        if selection.sum()<10:
            raise RuntimeError('Insufficient automatic seed support for prototypes')
        prototypes.append((feat[train_indices]*selection).sum((0,2,3))/selection.sum())
    prototypes=F.normalize(torch.stack(prototypes),dim=1)
    probability=torch.einsum('nchw,kc->nkhw',feat,prototypes).mul(12).softmax(dim=1)[:,1:2]
    probability=F.interpolate(probability,size=(h,w),mode='bilinear',align_corners=False)
    # Exact three-point targets; prototypes weight supervision but never bend edges.
    targets=[]
    for i in range(len(rows)):
        proposal=triangle_mask(vertex_targets[i]*[w-1,h-1],(h,w)) if vertex_available[i] else np.zeros((h,w),np.uint8)
        targets.append(proposal.astype(np.float32)*valids[i])
    targets=torch.tensor(np.stack(targets))[:,None]
    agreement=1-(targets-probability).abs()
    # High-intensity, predominantly vertical responses are down-weighted as a
    # reflection heuristic. This does not identify every transparent-guard reflection.
    reflection=((images[:,:3].mean(1,keepdim=True)>.8)&(images[:,5:6]<.4)).float()
    confidence=(.25+.75*agreement)*(1-.7*reflection)*valid
    supervised_ids=[]
    available=[i for i in train_indices if any((ROOT/'dataset/annotations/train'/f'{rows[i]["id"]}{suffix}').exists() for suffix in ['.png','.json'])]
    shots=config.get('shots',0)
    if shots:
        if len(available)<shots:
            raise ValueError(f'--shots {shots} needs {shots} genuine train masks; found {len(available)}. Pseudo labels are not counted as shots.')
        selected=np.array(available)[np.linspace(0,len(available)-1,shots).round().astype(int)].tolist()
        for i in selected:
            native_valid=read(ROOT/'dataset'/rows[i]['valid_mask'],True)>0
            mask,points,path=annotation(ROOT/'dataset/annotations/train',rows[i]['id'],native_valid.shape,native_valid)
            vertex_targets[i]=(points/np.array([native_valid.shape[1]-1,native_valid.shape[0]-1])).astype(np.float32)
            vertex_available[i]=True
            targets[i,0]=torch.from_numpy(triangle_mask(vertex_targets[i]*[w-1,h-1],(h,w)).astype(np.float32))*valid[i,0]
            confidence[i]=valid[i]
            supervised_ids.append(rows[i]['id'])
    boundaries,heatmaps,has_vertices=[],[],[]
    for i,row in enumerate(rows):
        mask=targets[i,0].numpy().astype(np.uint8)
        boundary=cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8))
        boundaries.append(boundary.astype(np.float32))
        points=vertex_targets[i]*[w-1,h-1] if vertex_available[i] else None
        heat=np.zeros((3,h,w),np.float32)
        if points is not None:
            yy,xx=np.mgrid[:h,:w]
            for k,(x,y) in enumerate(points):
                heat[k]=np.exp(-((xx-x)**2+(yy-y)**2)/(2*4**2))
        heatmaps.append(heat)
        has_vertices.append(float(points is not None))
        save(out/'pseudo_labels'/row['split']/(row['id']+'.png'),mask*255)
    torch.save(dict(prototypes=prototypes,training_ids=[rows[i]['id'] for i in train_indices]),out/'prototypes.pt')
    dump(out/'pseudo_label_info.json',dict(source='Exactly three-point triangle seeds; DINO train-only prototypes weight confidence',
                                          representation='triangle_vertices_v2',vertex_order=['A_left_apex','B_upper_right','C_lower_right'],
                                          human_training_ids=supervised_ids,human_training_count=len(supervised_ids),
                                          ground_truth=False,old_run_reused=False))
    neighbors,grids,visibility=temporal_pairs(rows,images,valid)
    dump(out/'temporal_pairs.json',[dict(id=row['id'],neighbor_id=rows[j]['id'],split=row['split'],
                                       delta_seconds=abs(row['time_seconds']-rows[j]['time_seconds']))
                                  for row,j in zip(rows,neighbors)])
    dump(out/'pseudo_vertices.json',[dict(id=r['id'],split=r['split'],vertices_normalized=vertex_targets[i].tolist() if vertex_available[i] else None) for i,r in enumerate(rows)])
    return dict(features=features.float(),images=images,valid=valid,target=targets,confidence=confidence,
                points=torch.tensor(np.stack(vertex_targets)),
                boundary=torch.tensor(np.stack(boundaries))[:,None],heatmaps=torch.tensor(np.stack(heatmaps)),
                has_vertices=torch.tensor(has_vertices)[:,None,None,None],prototype_probability=probability,
                neighbors=torch.tensor(neighbors),grids=torch.tensor(np.stack(grids)),visibility=torch.tensor(np.stack(visibility))[:,None])


def temporal_pairs(rows,images,valid):
    n,_,h,w=images.shape
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    neighbors,grids,visibility=[],[],[]
    for i,row in enumerate(rows):
        possible=[j for j,r in enumerate(rows) if r['sequence']==row['sequence'] and r['split']==row['split'] and
                  0<abs(r['source_frame_number']-row['source_frame_number'])<=2]
        j=min(possible,key=lambda j:abs(rows[j]['time_seconds']-row['time_seconds'])) if possible else i
        current=np.uint8(images[i,:3].mean(0).numpy()*255)
        following=np.uint8(images[j,:3].mean(0).numpy()*255)
        forward=cv2.calcOpticalFlowFarneback(current,following,None,.5,3,15,3,5,1.2,0)
        backward=cv2.calcOpticalFlowFarneback(following,current,None,.5,3,15,3,5,1.2,0)
        mx,my=xx+forward[:,:,0],yy+forward[:,:,1]
        reverse=cv2.remap(backward,mx,my,cv2.INTER_LINEAR)
        warped=cv2.remap(following,mx,my,cv2.INTER_LINEAR)
        support=cv2.remap(valid[j,0].numpy(),mx,my,cv2.INTER_LINEAR)>.999
        fb=np.linalg.norm(forward+reverse,axis=2)<1.5
        photometric=np.abs(current.astype(np.float32)-warped.astype(np.float32))<25
        gate=support*fb*photometric*valid[i,0].numpy()
        gate*=np.exp(-abs(rows[j]['time_seconds']-row['time_seconds'])/.0667) if i!=j else 0
        grid=np.stack([2*mx/(w-1)-1,2*my/(h-1)-1],axis=-1).astype(np.float32)
        neighbors.append(j)
        grids.append(grid)
        visibility.append(gate.astype(np.float32))
    return neighbors,grids,visibility
