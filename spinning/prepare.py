"""Validate the already unified dataset; preprocessing is not repeated."""
import numpy as np
from .io import ROOT,load,sha256


def apportioned_counts(n,ratios=(.7,.2,.1)):
    exact=np.asarray(ratios)*n
    counts=np.floor(exact).astype(int)
    for i in np.argsort(-(exact-counts),kind='stable')[:n-int(counts.sum())]:
        counts[i]+=1
    return counts.tolist()


def prepare_dataset(destination=None):
    destination=destination or ROOT/'dataset'
    metadata=destination/'metadata.json'
    if not metadata.exists():
        raise FileNotFoundError('Restore dataset/metadata.json and its images; this project uses the existing unified dataset.')
    result=load(metadata)
    rows=result['frames']
    splits=load(destination/'splits.json')
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('Duplicate dataset IDs')
    expected=apportioned_counts(len(rows))
    for split,count in zip(['train','val','test'],expected):
        ids=[r['id'] for r in rows if r['split']==split]
        if len(ids)!=count or set(ids)!=set(splits[split]):
            raise ValueError(f'Invalid 7:2:1 split: {split}')
        if result['counts'][split]!=count: raise ValueError('Metadata counts differ')
    for row in rows:
        for name in ['image','valid_mask']:
            if sha256(destination/row[name])!=row[name+'_sha256']:
                raise ValueError(f'Dataset file changed: {row[name]}')
    return result
