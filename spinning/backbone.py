import sys
import subprocess
import warnings
import numpy as np
import torch
from .io import ROOT, read, load, dump, sha256


def assets():
    if not (ROOT/'weights/provenance.json').exists():
        subprocess.run([sys.executable, str(ROOT/'tools/download_backbone.py')], check=True)
    provenance = load(ROOT/'weights/provenance.json')
    if sha256(ROOT/'weights/dinov2_vits14_reg4_pretrain.pth') != provenance['weights_sha256']:
        raise RuntimeError('Backbone weight checksum differs from provenance')
    for relative, digest in provenance['code_sha256'].items():
        if sha256(ROOT/'third_party/dinov2'/relative) != digest:
            raise RuntimeError(f'Backbone source changed: {relative}')
    return provenance


def extract_features(metadata, config, cache, device):
    import cv2
    provenance = assets()
    cache.mkdir(parents=True, exist_ok=True)
    signature = dict(dataset_sha256=sha256(ROOT/'dataset/metadata.json'),
                     weights_sha256=provenance['weights_sha256'], size=config['backbone_size'])
    if (cache/'features.pt').exists() and (cache/'features_signature.json').exists():
        if load(cache/'features_signature.json') == signature:
            return torch.load(cache/'features.pt', map_location='cpu', weights_only=True)
        print('Dataset/backbone signature changed: rebuilding shared feature cache.',flush=True)
    sys.path.insert(0, str(ROOT/'third_party/dinov2'))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='xFormers is not available.*')
        from dinov2.hub.backbones import dinov2_vits14_reg
        model = dinov2_vits14_reg(pretrained=False)
    model.load_state_dict(torch.load(ROOT/'weights/dinov2_vits14_reg4_pretrain.pth', map_location='cpu', weights_only=True))
    model = model.eval().to(device)
    model.requires_grad_(False)
    print(f'Frozen official backbone: {sum(p.numel() for p in model.parameters()):,} parameters', flush=True)
    w, h = config['backbone_size']
    mean = torch.tensor([.485,.456,.406], device=device)[None,:,None,None]
    std = torch.tensor([.229,.224,.225], device=device)[None,:,None,None]
    results=[]
    with torch.inference_mode():
        for start in range(0,len(metadata['frames']),4):
            batch=[]
            for row in metadata['frames'][start:start+4]:
                im=cv2.resize(read(ROOT/'dataset'/row['image']), (w,h), interpolation=cv2.INTER_AREA)
                batch.append(im[:,:,::-1].copy().transpose(2,0,1))
            x=torch.tensor(np.stack(batch), device=device, dtype=torch.float32)/255
            with torch.autocast(device_type=device.type, enabled=device.type=='cuda', dtype=torch.float16):
                features=model.forward_features((x-mean)/std)['x_norm_patchtokens']
            results.append(features.reshape(len(batch),h//14,w//14,384).permute(0,3,1,2).cpu().half())
            if start%32==0:
                print(f'DINO features: {min(start+4,len(metadata["frames"]))}/{len(metadata["frames"])}', flush=True)
    features=torch.cat(results)
    torch.save(features, cache/'features.pt')
    dump(cache/'features_signature.json', signature)
    dump(cache/'backbone.json',dict(model='dinov2_vits14_reg',frozen_parameters=sum(p.numel() for p in model.parameters()),
                                   feature_shape=list(features.shape),provenance=provenance))
    del model
    if device.type=='cuda':
        torch.cuda.empty_cache()
    return features
