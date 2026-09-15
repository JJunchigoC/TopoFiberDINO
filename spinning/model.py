import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, a, b, stride=1):
        super().__init__(nn.Conv2d(a,b,3,stride,1),nn.GroupNorm(4,b),nn.GELU())


class FiberAdapter(nn.Module):
    """A small trainable decoder on a frozen recent-conference backbone.

    Directional operators and multi-task heads are a scene adaptation, not a
    claim of a newly published or proven state-of-the-art method.
    """
    def __init__(self, directional=True):
        super().__init__()
        self.directional=directional
        self.semantic=nn.Sequential(nn.Conv2d(384,48,1),nn.GroupNorm(4,48),nn.GELU())
        self.texture=nn.Sequential(ConvBlock(6,24,2),ConvBlock(24,32,2))
        self.fusion=ConvBlock(80,48)
        if directional:
            self.horizontal=nn.Conv2d(48,48,(1,9),padding=(0,4),groups=48)
            self.vertical=nn.Conv2d(48,48,(9,1),padding=(4,0),groups=48)
        self.refine=nn.Sequential(ConvBlock(48,32),nn.Conv2d(32,1,1))

    def forward(self, features, image_features):
        texture=self.texture(image_features)
        semantic=F.interpolate(self.semantic(features),size=texture.shape[-2:],mode='bilinear',align_corners=False)
        fused=self.fusion(torch.cat([texture,semantic],dim=1))
        if self.directional:
            orientation=F.interpolate(image_features[:,5:6],size=fused.shape[-2:],mode='bilinear',align_corners=False)
            fused=fused+.25*(orientation*self.horizontal(fused)+(1-orientation)*self.vertical(fused))
        output=F.interpolate(self.refine(fused),size=image_features.shape[-2:],mode='bilinear',align_corners=False)
        return output


class SmallUNet(nn.Module):
    """Compact U-shaped RGB control, trained from scratch; not the original U-Net."""
    def __init__(self):
        super().__init__()
        self.e1=ConvBlock(3,16,2)
        self.e2=ConvBlock(16,32,2)
        self.e3=ConvBlock(32,64,2)
        self.d2=ConvBlock(96,32)
        self.d1=ConvBlock(48,16)
        self.mask=nn.Conv2d(16,1,1)

    def forward(self,features,image_features):
        a=self.e1(image_features[:,:3]); b=self.e2(a); c=self.e3(b)
        d=self.d2(torch.cat([b,F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False)],1))
        d=self.d1(torch.cat([a,F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False)],1))
        return F.interpolate(self.mask(d),size=image_features.shape[-2:],mode='bilinear',align_corners=False)


class DinoLinear(nn.Module):
    """Frozen DINO features with a trained linear pixel classifier."""
    def __init__(self):
        super().__init__()
        self.classifier=nn.Conv2d(384,1,1)

    def forward(self,features,image_features):
        return F.interpolate(self.classifier(features),size=image_features.shape[-2:],mode='bilinear',align_corners=False)


class DinoPyramid(nn.Module):
    """Local multiscale decoder adaptation, not an official published architecture."""
    def __init__(self):
        super().__init__()
        self.project=nn.Sequential(nn.Conv2d(384,48,1),nn.GroupNorm(4,48),nn.GELU())
        self.scales=nn.ModuleList([nn.Sequential(nn.Conv2d(48,24,3,padding=d,dilation=d),nn.GroupNorm(4,24),nn.GELU()) for d in [1,2,4]])
        self.global_context=nn.Conv2d(48,24,1)
        self.texture=nn.Sequential(ConvBlock(6,16,2),ConvBlock(16,24,2))
        self.fuse=nn.Sequential(ConvBlock(120,48),ConvBlock(48,24),nn.Conv2d(24,1,1))

    def forward(self,features,image_features):
        x=self.project(features)
        context=self.global_context(F.adaptive_avg_pool2d(x,1)).expand(-1,-1,*x.shape[-2:])
        scales=torch.cat([layer(x) for layer in self.scales]+[context],1)
        texture=self.texture(image_features)
        scales=F.interpolate(scales,size=texture.shape[-2:],mode='bilinear',align_corners=False)
        logits=self.fuse(torch.cat([scales,texture],1))
        return F.interpolate(logits,size=image_features.shape[-2:],mode='bilinear',align_corners=False)


MODEL_INFO={
    'baseline':dict(label='DINO base adapter',backbone='DINOv2-S/14 Registers',auxiliary=False,temporal=False),
    'directional':dict(label='DINO direction + auxiliary',backbone='DINOv2-S/14 Registers',auxiliary=True,temporal=False),
    'full':dict(label='FiberDINO Triangle-MRF (ours)',backbone='DINOv2-S/14 Registers',auxiliary=True,temporal=True),
    'unet_small':dict(label='Small RGB U-Net',backbone='None; RGB encoder trained from scratch',auxiliary=False,temporal=False),
    'dino_linear':dict(label='DINO linear probe',backbone='DINOv2-S/14 Registers',auxiliary=False,temporal=False),
    'dino_pyramid':dict(label='DINO multiscale decoder',backbone='DINOv2-S/14 Registers',auxiliary=False,temporal=False),
}


def model_label(name):
    return MODEL_INFO[name]['label'] if name=='full' else name


def build_model(name,config=None):
    if name not in MODEL_INFO:
        raise ValueError(f'Unknown model {name}; choose from {list(MODEL_INFO)}')
    from .triangle import TriangleModel,TopologyTriangleModel,SceneTriangleModel
    from .topology import settings
    encoder=FiberAdapter(directional=name!='baseline') if name in ['baseline','directional','full'] else {'unet_small':SmallUNet,'dino_linear':DinoLinear,'dino_pyramid':DinoPyramid}[name]()
    cfg=settings(config or {})
    if (config or {}).get('architecture_revision')==5:
        head=config.get('coordinate_head','pooled') if name=='full' else 'pooled'
        if head in ['spatial_integral','calibrated_spatial']:
            encoder.refine[-1]=nn.Conv2d(32,3,1)
        return SceneTriangleModel(encoder,{**config,'coordinate_head':head},cfg if name=='full' and cfg['enabled'] else None)
    return TopologyTriangleModel(encoder,cfg) if name=='full' and cfg['enabled'] else TriangleModel(encoder)
