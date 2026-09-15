import numpy as np
import torch
from .io import ROOT, load, dump, read, sha256
from .geometry import transform, measure
from .triangle import triangle_mask
from .mrf import settings as mrf_settings
from .topology import settings as topology_settings


def verify_run(out):
    metadata=load(ROOT/'dataset/metadata.json')
    rows=metadata['frames']
    splits=load(ROOT/'dataset/splits.json')
    assert metadata['counts']=={'train':134,'val':39,'test':19}
    assert not set(splits['train'])&set(splits['val']) and not set(splits['train'])&set(splits['test']) and not set(splits['val'])&set(splits['test'])
    assert len(set(sum(splits.values(),[])))==192
    prototypes=torch.load(out/'prototypes.pt',map_location='cpu',weights_only=True)
    assert set(prototypes['training_ids'])==set(splits['train'])
    lookup={r['id']:r for r in rows}
    for pair in load(out/'temporal_pairs.json'):
        a,b=lookup[pair['id']],lookup[pair['neighbor_id']]
        assert a['split']==b['split'] and a['sequence']==b['sequence']
        assert abs(a['source_frame_number']-b['source_frame_number'])<=2
    for row in rows:
        assert sha256(ROOT/'dataset'/row['image'])==row['image_sha256']
        assert sha256(ROOT/'dataset'/row['valid_mask'])==row['valid_mask_sha256']
        assert row['id'] in splits[row['split']]
    max_error=0
    valid_triangles=0
    variants=load(out/'run.json')['config']['variants']
    for variant in variants:
        predictions=load(out/variant/'predictions.json')
        assert len(predictions)==192
        for row,result in zip(rows,predictions):
            assert row['id']==result['id'] and row['time_seconds']==result['time_seconds']
            valid=read(ROOT/'dataset'/row['valid_mask'],True)>0
            mask=read(out/variant/result['mask'],True)
            assert mask.shape==valid.shape and not np.any((mask>0)&~valid)
            assert result['metrics_mm'] is None and result['measurement_approved'] is False
            if result['vertices_roi'] is not None:
                valid_triangles+=1
                assert result['representation']=='triangle_vertices_v2'
                assert np.asarray(result['vertices_roi']).shape==(3,2)
                assert result['edges']==[[0,1],[1,2],[2,0]]
                np.testing.assert_allclose(result['contour_roi'],result['vertices_roi'])
                np.testing.assert_array_equal(mask,triangle_mask(result['vertices_roi'],valid.shape,valid)*255)
                assert (out/variant/result['triangle_svg']).is_file()
                error=float(np.max(np.abs(transform(result['vertices_raw'],row['raw_to_roi'])-result['vertices_roi'])))
                max_error=max(error,max_error)
                assert error<1e-8
                expected=measure(result['vertices_raw'],result['contour_raw'])
                assert all(np.isclose(expected[k],v) for k,v in result['metrics_raw_px'].items())
        assert (out/variant/'best.pt').is_file()
        history=load(out/variant/'history.json')
        assert history and all(np.isfinite(r['train_loss']) for r in history)
        summary=load(out/variant/'training_summary.json')
        epochs=load(out/'run.json')['config']['epochs']
        assert len(history)==summary['epochs_run']==summary['last_epoch']==epochs
        assert [r['epoch'] for r in history]==list(range(1,epochs+1))
        assert summary['early_stopping'] is False
        assert summary['stop_reason']=='completed_requested_epochs'
        last=torch.load(out/variant/'last.pt',map_location='cpu',weights_only=True)
        best=torch.load(out/variant/'best.pt',map_location='cpu',weights_only=True)
        assert last['epoch']==epochs and best['epoch']==summary['best_epoch']
        config=load(out/'run.json')['config']
        if config.get('checkpoint_selection','region_loss')=='vertex_error':
            selected=min(history,key=lambda r:r['pseudo_vertex_error_normalized'])
            assert selected['epoch']==summary['best_epoch']==best['epoch']
            assert summary['checkpoint_selection']=='vertex_error'
        if config.get('architecture_revision')==5 and config.get('translation_weight',0)>0:
            assert all(np.isfinite(r['components']['translation_weighted']) and r['components']['translation_weighted']>=0 for r in history)
        mrf=mrf_settings(load(out/'run.json')['config'])
        if variant=='full' and mrf['weight']>0:
            assert summary['mrf']==mrf_settings(best['config'])==mrf_settings(last['config'])==mrf
            assert all(np.isfinite(r['components']['mrf_pairwise']) and r['components']['mrf_pairwise']>=0 for r in history)
    result=dict(artifact_integrity='PASS',frames=192,counts=metadata['counts'],variants=variants,
                original_roi_hashes_preserved=True,invalid_padding_excluded=True,
                maximum_coordinate_roundtrip_error=max_error,mm_fields_empty=True,
                independent_video_test=False,scientific_accuracy_verified=False)
    diagnostics=load(out/'quality/frame_diagnostics.json')
    assert len(diagnostics)==len(rows)*len(variants)
    assert len({(r['model'],r['id']) for r in diagnostics})==len(diagnostics)
    for record in diagnostics:
        assert 0<=record['review_priority']<=100
        assert record['split']==lookup[record['id']]['split']
        for key in ['original','overlay','entropy','disagreement']:
            assert (out/record[key]).is_file()
    for video in load(out/'quality/videos/index.json'):
        assert (out/video['file']).is_file()
    assert (out/'review.html').is_file()
    result.update(quality_records=len(diagnostics),review_assets_verified=True,representation='triangle_vertices_v2',exactly_three_vertices=True,straight_triangle_masks_verified=True,valid_triangles=valid_triangles,invalid_triangles=len(diagnostics)-valid_triangles,epochs_per_model=epochs,all_models_completed_requested_epochs=True)
    result['mrf_training_verified']='full' in variants and mrf['weight']>0
    topo=topology_settings(load(out/'run.json')['config'])
    if 'full' in variants and topo['enabled']:
        fitted=torch.load(out/'topology/fitted.pt',weights_only=True)
        assert set(fitted['training_ids'])==set(splits['train'])
        assert fitted['settings']==topo
        assert list(fitted['centers'].shape)==[topo['clusters'],384]
        assert torch.isfinite(fitted['centers']).all() and (fitted['histogram_scale']>=1).all()
        history=load(out/'full/history.json')
        assert all(np.isfinite(r['components']['topology_weighted']) and r['components']['topology_weighted']>=0 for r in history)
        for name in ['best.pt','last.pt']:
            state=torch.load(out/'full'/name,weights_only=True,map_location='cpu')
            assert topology_settings(state['config'])==topo
            torch.testing.assert_close(state['state_dict']['topology.centers'],fitted['centers'])
        records=load(out/'full/topology_predictions.json')['frames']
        assert [r['id'] for r in records]==[r['id'] for r in rows]
        assert all(np.isfinite(r['predicted_histogram']).all() and np.isfinite(r['target_histogram']).all() for r in records)
        assert (out/'figures/topology_diagnostics.png').is_file()
    result['topology_pipeline_verified']='full' in variants and topo['enabled']
    result['topology_training_verified']=result['topology_pipeline_verified'] and (topo['weight']>0 or topo['proportion_weight']>0)
    dump(out/'verification.json',result)
    print('Artifact verification PASS; scientific accuracy remains unverified.',flush=True)
    return result
