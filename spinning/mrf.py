"""Contrast-sensitive 8-neighbor Potts MRF regularization of a soft triangle.

This optimizes the expected pairwise energy through the three coordinates.
It is not graph-cut inference and never changes the output into a free mask.
"""
import math
import torch


OFFSETS=((0,1),(1,0),(1,1),(1,-1))  # Each undirected edge occurs once.


def settings(config):
    value=config.get('mrf',{})
    weight=float(value.get('weight',0.))
    sigma=float(value.get('color_sigma',.1))
    if not math.isfinite(weight) or weight<0 or not math.isfinite(sigma) or sigma<=0:
        raise ValueError('MRF weight must be finite/nonnegative; color_sigma finite/positive')
    if value.get('kind','potts_8')!='potts_8':
        raise ValueError('Only the potts_8 MRF is supported')
    return dict(kind='potts_8',weight=weight,color_sigma=sigma)


def edge_slices(height,width,dy,dx):
    a=(slice(0,height-dy),slice(max(0,-dx),min(width,width-dx)))
    b=(slice(dy,height),slice(max(0,dx),min(width,width+dx)))
    return a,b


@torch.no_grad()
def prepare_mrf(images,valid,sigma):
    """Fixed affinities use unaugmented RGB [0,1], excluding both padding ends."""
    n,_,h,w=images.shape
    weights=images.new_zeros(n,4,h,w)
    for k,(dy,dx) in enumerate(OFFSETS):
        a,b=edge_slices(h,w,dy,dx)
        difference=(images[:,:3,*a]-images[:,:3,*b]).square().mean(1)
        support=valid[:,0,*a]*valid[:,0,*b]
        weights[:,k,*a]=support*torch.exp(-difference/(2*sigma*sigma))/math.hypot(dy,dx)
    # Per-image perimeter-scale normalization. No batch/padding-area dilution.
    normalizer=4*valid.flatten(1).sum(1).sqrt().clamp_min(1)
    return dict(mrf_weights=weights,mrf_normalizer=normalizer)


def mrf_energy(logits,weights,normalizer):
    p=logits.sigmoid()[:,0]
    h,w=p.shape[-2:]
    energy=p.flatten(1).sum(1)*0
    for k,(dy,dx) in enumerate(OFFSETS):
        a,b=edge_slices(h,w,dy,dx)
        pa,pb=p[:,*a],p[:,*b]
        # E[1(label_i != label_j)] under independent Bernoulli marginals.
        disagreement=pa*(1-pb)+(1-pa)*pb
        energy+=(weights[:,k,*a]*disagreement).flatten(1).sum(1)
    return (energy/normalizer).mean()
