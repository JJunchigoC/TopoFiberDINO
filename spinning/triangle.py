"""Three ordered vertices are the sole geometric output contract."""
import cv2
import numpy as np
import torch
from torch import nn
from .io import load,read

def triangle_mask(vertices,shape,valid=None):
    points=np.asarray(vertices,np.float64)
    h,w=shape
    if points.shape!=(3,2) or not np.isfinite(points).all():
        raise ValueError('Exactly three finite [x,y] vertices required')
    if np.any(points<0) or np.any(points[:,0]>w-1) or np.any(points[:,1]>h-1):
        raise ValueError('Triangle vertex outside ROI')
    cross=np.linalg.det(np.stack([points[1]-points[0],points[2]-points[0]]))
    if cross<=1 or points[0,0]>=min(points[1:,0]) or points[1,1]>=points[2,1]:
        raise ValueError('Need nondegenerate ordered vertices: A left apex, B upper right, C lower right')
    mask=np.zeros((h,w),np.uint8)
    cv2.fillConvexPoly(mask,np.rint(points).astype(np.int32),1)
    if valid is not None and np.any(mask.astype(bool)&~valid):
        raise ValueError('Triangle crosses invalid pixels; do not clip into another polygon')
    return mask


def annotation(folder,id_,shape,valid):
    """Prefer explicit 3-point JSON; legacy PNG must itself depict a triangle."""
    path=folder/(id_+'.json')
    if path.exists():
        points=np.asarray(load(path)['vertices_roi'],np.float64)
        return triangle_mask(points,shape,valid),points,path
    path=folder/(id_+'.png')
    if not path.exists(): return None
    mask=read(path,True)
    if mask.shape!=tuple(shape) or not set(np.unique(mask)).issubset({0,255}):
        raise ValueError(f'Expected native-size binary triangle annotation: {path}')
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)!=1: raise ValueError(f'Expected one annotated triangle: {path}')
    contour=contours[0]
    points=cv2.approxPolyDP(contour,.015*cv2.arcLength(contour,True),True)[:,0].astype(float)
    if len(points)!=3: raise ValueError(f'Use three-point JSON; this annotation is not triangular: {path}')
    a=np.argmin(points[:,0]); rest=np.delete(points,a,axis=0)
    points=np.vstack([points[a],rest[np.argsort(rest[:,1])]])
    rendered=triangle_mask(points,shape,valid)
    union=((mask>0)|(rendered>0)).sum()
    if ((mask>0)&(rendered>0)).sum()/max(union,1)<.97:
        raise ValueError(f'Annotation differs from a straight-sided triangle: {path}')
    return rendered,points,path


def soft_triangle(points,size,auxiliaries=True):
    """Differentiable signed-distance rasterizer; normalized coordinates in [0,1]."""
    h,w=size
    scale=points.new_tensor([w-1,h-1])
    pixels=points*scale
    y,x=torch.meshgrid(torch.arange(h,device=points.device),torch.arange(w,device=points.device),indexing='ij')
    grid=torch.stack([x,y],-1).to(points.dtype)
    edges=pixels.roll(-1,dims=1)-pixels
    offset=grid[None,None]-pixels[:,:,None,None]
    signed=(edges[:,:,None,None,0]*offset[...,1]-edges[:,:,None,None,1]*offset[...,0])/edges.norm(dim=-1)[:,:,None,None].clamp_min(1e-6)
    logits=signed.min(dim=1,keepdim=True).values/1.25
    if not auxiliaries: return dict(mask=logits,points=points)
    distance=(grid[None,None]-pixels[:,:,None,None]).square().sum(-1)
    heat=torch.exp(-distance/(2*4**2)).clamp(1e-5,1-1e-5)
    boundary=torch.exp(-logits.abs()).clamp(1e-5,1-1e-5)
    return dict(mask=logits,boundary=torch.logit(boundary),vertices=torch.logit(heat),points=points)


class TriangleModel(nn.Module):
    """Shared coordinate head on each encoder/decoder, trained end-to-end.

    Six parameters locate A, B and C. Left apex and ordered right-side base are
    explicit priors for these registered ROIs, not a general triangle detector.
    """
    def __init__(self,encoder):
        super().__init__()
        self.encoder=encoder
        self.coordinate_head=nn.Sequential(nn.AdaptiveAvgPool2d((4,8)),nn.Flatten(),nn.Linear(32,64),nn.GELU(),nn.Linear(64,6))

    def forward(self,features,image_features):
        latent=self.encoder(features,image_features)
        q=self.coordinate_head(latent).sigmoid()
        return coordinate_triangle(q,image_features.shape[-2:])


def coordinate_triangle(q,size):
    ax=.10+.30*q[:,0]; bx=.45+.40*q[:,1]; cx=.45+.40*q[:,2]
    by=.22+.38*q[:,3]; cy=by+.035+(.86-by-.035)*q[:,4]
    ay=by+(cy-by)*q[:,5]
    points=torch.stack([torch.stack([ax,ay],-1),torch.stack([bx,by],-1),torch.stack([cx,cy],-1)],1)
    return soft_triangle(points,size)


class TopologyTriangleModel(TriangleModel):
    uses_topology=True

    def __init__(self,encoder,cfg):
        super().__init__(encoder)
        from .topology import TopologyContext
        self.topology=TopologyContext(cfg)
        self.coordinate_head=nn.Sequential(nn.Linear(32+cfg['embedding_dim'],64),nn.GELU(),nn.Linear(64,6))

    def forward(self,features,image_features,valid=None):
        if valid is None: raise ValueError('Topology model requires an explicit valid-pixel mask')
        latent=self.encoder(features,image_features)
        local=torch.nn.functional.adaptive_avg_pool2d(latent,(4,8)).flatten(1)
        context,histogram,proportions=self.topology(features,valid)
        q=self.coordinate_head(torch.cat([local,context],1)).sigmoid()
        output=coordinate_triangle(q,image_features.shape[-2:])
        output.update(topology_hist=histogram,topology_proportions=proportions)
        return output


def scene_points(q):
    """Free apex height and a bounded base tilt; positive area for every q.

    The old head incorrectly required By <= Ay <= Cy. Here the base midpoint
    stays to the right of A and the tilt bound guarantees a positive cross
    product even when the apex lies above or below the base endpoints.
    """
    ax=.10+.25*q[:,0]; base=.37+.41*q[:,1]
    by=.24+.34*q[:,3]; cy=by+.06+(.84-by-.06)*q[:,4]
    ay=.26+.34*q[:,5]
    gap=base-ax; height=cy-by
    bound=(.5*gap*height/((by+cy-2*ay).abs()+height)).clamp(max=.055)
    tilt=(2*q[:,2]-1)*bound
    return torch.stack([torch.stack([ax,ay],-1),torch.stack([base+tilt,by],-1),torch.stack([base-tilt,cy],-1)],1)


class SceneTriangleModel(TriangleModel):
    """Small local predictor with optional bounded topology residual.

    Both controls and ours use the same corrected geometry and feature dropout.
    The topology branch starts at zero contribution, rather than replacing the
    local head with an unconstrained concatenation of random global features.
    """
    def __init__(self,encoder,config,topo=None):
        super().__init__(encoder)
        if config.get('coordinate_head','pooled')=='spatial_integral':
            self.coordinate_head=SpatialCoordinateHead()
        elif config.get('coordinate_head')=='calibrated_spatial':
            self.coordinate_head=CalibratedSpatialHead()
        self.feature_dropout=float(config.get('feature_dropout',0))
        self.auxiliary_maps=any(config.get(k,0)>0 for k in ['boundary_weight','vertex_weight'])
        self.uses_topology=topo is not None
        if self.uses_topology:
            from .topology import TopologyContext
            self.topology=TopologyContext(topo)
            self.topology_residual=nn.Linear(topo['embedding_dim'],6)
            nn.init.zeros_(self.topology_residual.weight)
            nn.init.zeros_(self.topology_residual.bias)
            self.topology_residual_scale=float(config.get('topology_residual_scale',.15))

    def forward(self,features,image_features,valid=None):
        features=torch.nn.functional.dropout2d(features,p=self.feature_dropout,training=self.training)
        head=self.coordinate_head(self.encoder(features,image_features))
        logits=head['logits'] if isinstance(head,dict) else head
        if self.uses_topology:
            if valid is None: raise ValueError('Topology model requires an explicit valid-pixel mask')
            context,histogram,proportions=self.topology(features,valid)
            logits=logits+self.topology_residual_scale*self.topology_residual(context).tanh()
        output=soft_triangle(scene_points(logits.sigmoid()),image_features.shape[-2:],self.auxiliary_maps)
        if isinstance(head,dict): output['coordinate_log_probability']=head['log_probability']
        if self.uses_topology: output.update(topology_hist=histogram,topology_proportions=proportions)
        return output


class SpatialCoordinateHead(nn.Module):
    """Spatial expectation within scene supports, instead of a flattened MLP.

    Inspired by integral coordinate regression (ECCV 2018). These are spatial
    evidence distributions feeding the ordered scene parameters, not calibrated
    landmark probabilities. No target or candidate coordinates enter inference.
    """
    def forward(self,latent):
        logits=torch.nn.functional.adaptive_avg_pool2d(latent,(50,112))
        h,w=logits.shape[-2:]
        y,x=torch.meshgrid(torch.linspace(0,1,h,device=logits.device),torch.linspace(0,1,w,device=logits.device),indexing='ij')
        support=torch.stack([(x>=.10)&(x<=.35)&(y>=.26)&(y<=.60),
                             (x>=.37)&(x<=.78)&(y>=.24)&(y<=.58),
                             (x>=.37)&(x<=.78)&(y>=.50)&(y<=.84)])
        probability=logits.masked_fill(~support[None],-torch.inf).flatten(2).softmax(-1).reshape_as(logits)
        px=(probability*x).sum((2,3)); py=(probability*y).sum((2,3))
        q=torch.stack([(px[:,0]-.10)/.25,((px[:,1]+px[:,2])/2-.37)/.41,
                       .5+(px[:,1]-px[:,2])/.22,(py[:,1]-.24)/.34,
                       (py[:,2]-.50)/.34,(py[:,0]-.26)/.34],1)
        return torch.logit(q.clamp(1e-4,1-1e-4))


def scene_logits_from_points(points):
    """Invert the scene mapping for feasible coordinates, retaining gradients.

    Clamp only incompatible proposals to the shared geometry. Unlike the old
    head, the C heatmap's spatial y is decoded as C's actual y, not as a second
    relative-height parameter. No targets are consulted during inference.
    """
    a,b,c=points.unbind(1)
    ax=a[:,0]; ay=a[:,1]; base=(b[:,0]+c[:,0])/2
    by=b[:,1]; cy=torch.maximum(c[:,1],by+.0601).clamp(max=.8399)
    gap=base-ax; height=cy-by
    bound=(.5*gap*height/((by+cy-2*ay).abs()+height)).clamp(max=.055)
    q=torch.stack([(ax-.10)/.25,(base-.37)/.41,.5+(b[:,0]-c[:,0])/(4*bound.clamp_min(1e-6)),
                   (by-.24)/.34,(cy-by-.06)/(.84-by-.06),(ay-.26)/.34],1)
    return torch.logit(q.clamp(1e-4,1-1e-4))


class CalibratedSpatialHead(nn.Module):
    """Spatial expectations with an exact scene-coordinate conversion.

    Optional train-time axis distribution supervision is inspired by SimCC
    (ECCV 2022); this joint-map marginal adaptation is not official SimCC.
    'Calibrated' refers only to coordinate mapping, not probability calibration.
    """
    def forward(self,latent):
        logits=torch.nn.functional.adaptive_avg_pool2d(latent,(50,112))
        h,w=logits.shape[-2:]
        y,x=torch.meshgrid(torch.linspace(0,1,h,device=logits.device),torch.linspace(0,1,w,device=logits.device),indexing='ij')
        support=torch.stack([(x>=.10)&(x<=.35)&(y>=.26)&(y<=.60),
                             (x>=.37)&(x<=.78)&(y>=.24)&(y<=.58),
                             (x>=.37)&(x<=.78)&(y>=.50)&(y<=.84)])
        log_probability=logits.masked_fill(~support[None],-1e4).flatten(2).log_softmax(-1).reshape_as(logits)
        probability=log_probability.exp()
        px=(probability*x).sum((2,3)); py=(probability*y).sum((2,3))
        points=torch.stack([px,py],-1)
        return dict(logits=scene_logits_from_points(points),log_probability=log_probability)
