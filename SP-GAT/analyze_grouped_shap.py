"""Read-only numerical audit of completed SHAP; writes only review products."""
import hashlib
import json
import argparse
from pathlib import Path
from release_paths import resolve_path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'nested_holdout_outputs/grouped_shap_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=OUT)
    parser.add_argument('--pfi-root', type=Path, default=ROOT/'nested_holdout_outputs/unified_pfi_v1')
    args = parser.parse_args()
    OUT = resolve_path(args.source)
    pfi_root = resolve_path(args.pfi_root)
    manifest = json.loads((OUT / 'experiment.json').read_text())
    status = json.loads((OUT / 'summary_outer/status.json').read_text())
    assert status['complete_runs'] == 45 and not status['missing']
    for name, expected in manifest['config']['sources'].items():
        assert sha(ROOT / name) == expected
    rows, quality, mc = [], [], []
    models = ['PLSR','RF','SVR','XGBoost','MLP','ResNet','Transformer','GAT','PP-SGAT']
    seeds = [42,123,456,789,2026]
    for name in models:
        for seed in seeds:
            folder = OUT / name / f'seed{seed}/outer'
            marker = json.loads((folder / 'complete.json').read_text())
            assert marker['stamp']['signature'] == manifest['signature']
            for file, expected in marker['files'].items():
                assert sha(folder/file) == expected
            with np.load(folder/'grouped_shap.npz') as z:
                v,pairs,pred,base,ids,bg = (z[k].copy() for k in ['values','pair_values','predictions','expected_value','sample_ids','background_ids'])
                residual = v.sum(1)-(pred-base)
                np.testing.assert_allclose(residual,z['residual'],atol=1e-12)
            assert v.shape==(16,45) and pairs.shape==(16,16,45)
            assert np.isfinite(v).all() and np.isfinite(pairs).all()
            np.testing.assert_allclose(v,pairs.mean(1),atol=1e-10)
            assert np.all(np.abs(residual)<=.001+1e-5*np.abs(pred-base))
            with np.load(OUT/'shared_design'/f'seed{seed}_outer.npz') as shared:
                np.testing.assert_array_equal(shared['background_ids'],bg)
                np.testing.assert_array_equal(shared['sample_ids'],ids)
                for i,sid in enumerate(ids):
                    with np.load(folder/'samples'/f'sample{sid}.npz') as sample:
                        np.testing.assert_array_equal(sample['orders'],shared['orders'][i])
                        np.testing.assert_allclose(sample['values'],v[i],atol=1e-10)
            with np.load(pfi_root/name/f'seed{seed}/predictions.npz') as reference:
                np.testing.assert_array_equal(reference['outer_ids'],ids)
                np.testing.assert_allclose(pred,reference['baseline_prediction'],atol=1e-5,rtol=1e-6)
            table=pd.read_csv(folder/'group_importance.csv').sort_values('group')
            np.testing.assert_allclose(table.mean_abs_shap,np.abs(v).mean(0),atol=1e-10)
            for g,value in enumerate(np.abs(v).mean(0)):
                rows.append(dict(model=name,seed=seed,group=g,importance=value))
            q=pd.read_csv(folder/'sample_quality.csv')
            # Cancellation is descriptive: large opposing allocations may cancel.
            cancellation=np.abs(v).sum(1)/np.maximum(np.abs(pred-base),1.)
            quality.append(dict(model=name,seed=seed,max_residual=float(np.abs(residual).max()),
                expected_value=float(base), mean_abs_sum=float(np.abs(v).sum(1).mean()),
                mean_abs_prediction_difference=float(np.abs(pred-base).mean()),
                cancellation_median=float(np.median(cancellation)), hybrid_min=float(q.hybrid_min.min()),
                hybrid_max=float(q.hybrid_max.max())))
            first,second=pairs[:,:8].mean(1),pairs[:,8:].mean(1)
            rho=float(spearmanr(np.abs(first).mean(0),np.abs(second).mean(0)).statistic)
            mc.append(dict(model=name,seed=seed,spearman=rho,
                relative_half_difference=float(np.abs(first-second).mean()/np.abs((first+second)/2).mean())))
    frame=pd.DataFrame(rows)
    summary=pd.read_csv(OUT/'summary_outer/importance_summary.csv').sort_values(['model','group'])
    calc=frame.groupby(['model','group']).importance.agg(['mean','std']).reset_index().sort_values(['model','group'])
    np.testing.assert_allclose(summary.mean_abs_shap_mean,calc['mean'],atol=1e-10)
    np.testing.assert_allclose(summary.mean_abs_shap_std,calc['std'],atol=1e-10)
    stable=pd.read_csv(OUT/'summary_outer/stability.csv')
    globals=stable[stable.scope=='global'].groupby('model')[['spearman','top20_overlap']].mean()
    local=stable[stable.scope=='shared_sample'].groupby(['model','seed_a','seed_b'])[['spearman','top20_overlap']].mean().groupby('model').mean()
    diagnostics=pd.read_csv(OUT/'summary_outer/sampling_diagnostics.csv')
    overview=diagnostics.groupby('model')[['spearman','top20_overlap','relative_half_difference']].mean().add_prefix('sampling_')
    overview=overview.join(globals.add_prefix('cross_seed_')).join(local.add_prefix('shared_sample_'))
    top=[]
    for name in models:
        part=summary[summary.model==name].nlargest(5,'mean_abs_shap_mean')
        for rank,r in enumerate(part.itertuples(),1):
            group=int(r.group)
            hit=sum(group in set(frame[(frame.model==name)&(frame.seed==seed)].nlargest(9,'importance').group) for seed in seeds)
            top.append(dict(model=name,rank=rank,group=group,range_nm=f'{r.wavelength_min:.1f}-{r.wavelength_max:.1f}',
                mean=r.mean_abs_shap_mean,std=r.mean_abs_shap_std,top9_seeds=hit))
    review=OUT/'review';review.mkdir(exist_ok=True)
    for name,table in [('overview',overview.reset_index()),('top_groups',pd.DataFrame(top)),('quality',pd.DataFrame(quality))]:
        table.to_csv(review/(name+'.csv'),index=False,encoding='utf-8-sig')
    audit=dict(runs=45,samples=720,attributions=32400,all_finite=True,additivity_passed=True,
        max_residual=max(r['max_residual'] for r in quality),mean_absolute_aggregation_verified=True,
        hashes_verified=True,shared_orders_verified=True,baseline_predictions_verified=True,
        note='Saved arrays audited; no formal inference rerun.')
    (review/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit))
    print(overview.to_string())
    print(pd.DataFrame(top).to_string(index=False))
    print(pd.DataFrame(quality).groupby('model')[['mean_abs_sum','mean_abs_prediction_difference','cancellation_median','hybrid_min','hybrid_max']].agg(['mean','min','max']).to_string())


if __name__=='__main__':
    main()
