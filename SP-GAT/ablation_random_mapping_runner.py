"""Add a node-to-wavelength random mapping control without changing old runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import controlled_graph_experiments as cg
from release_paths import resolve_path, path_reference, require_data


def random_mapping(ei, ea, n, seed):
    permutation = np.random.default_rng(seed).permutation(n)
    mapped = torch.as_tensor(permutation, dtype=torch.long)[ei]
    inverse = torch.as_tensor(np.argsort(permutation), dtype=torch.long)
    assert torch.equal(inverse[mapped], ei)
    old = {tuple(sorted(p)) for p in ei.t().tolist()}
    new = {tuple(sorted(p)) for p in mapped.t().tolist()}
    info = dict(graph_seed=int(seed), old_to_new=permutation.tolist(),
        pair_overlap_fraction=len(old & new) / max(1, len(old)),
        fixed_nodes=int((permutation == np.arange(n)).sum()),
        adjacent_pairs=sum(b - a == 1 for a, b in new),
        n_directed_edges=int(mapped.shape[1]))
    return mapped, ea.clone(), info


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=cg.ROOT/'nested_holdout_outputs/ablation_controlled_v2')
    p.add_argument('--output', type=Path, default=cg.ROOT/'nested_holdout_outputs/ablation_random_mapping_v1')
    p.add_argument('--data', type=Path, default=cg.ROOT/'data/AGB.csv')
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args(argv)
    args.source, args.output = resolve_path(args.source), resolve_path(args.output)
    args.data = resolve_path(args.data)
    if args.source == args.output or args.source in args.output.parents or args.output in args.source.parents:
        p.error('Source and output must be separate directories.')
    if args.plan_only:
        print(f'Source: {args.source}')
        print(f'Data required for training: {args.data}')
        print(f'Add graph_repeats random_mapping runs per source seed; {cg.EPOCHS} epochs per run.')
        print(f'Output: {args.output}; no data reads, training or writes.')
        return
    require_data(args.data)
    if not (args.source/'manifest.json').is_file():
        raise FileNotFoundError('Missing base ablation manifest. Run controlled_graph_experiments.py ablation first, '
                                'or use ablation_vi_runner.py --train for the full sequence.')
    old = json.loads((args.source/'manifest.json').read_text(encoding='utf-8'))
    actual = hashlib.sha256(b''.join((cg.ROOT/f).read_bytes() for f in cg.SOURCE_FILES)).hexdigest()
    if old['source_hash'] != actual or old['data_hash'] != sha(args.data):
        raise ValueError('Old implementation/data changed; cannot pair silently.')
    if old['kind'] != 'ablation' or old['epochs'] != cg.EPOCHS or old['torch_version'] != torch.__version__:
        raise ValueError('Source experiment/version mismatch.')
    cg.check_main_architecture()
    args.device = old['device']
    frame = pd.read_csv(args.data)
    x, y = frame.iloc[:, :-1].to_numpy(np.float32), frame.iloc[:, -1].to_numpy(np.float32)
    wl = np.asarray(frame.columns[:-1], float)
    ei, ea = cg.prior_graph(wl, 'all')
    source_runs = pd.read_csv(args.source/'runs.csv')
    expected = {(m, s, r) for s in old['seeds'] for m in cg.MODES
                for r in range(old['graph_repeats'] if m == 'random_vi' else 1)}
    actual_runs = list(zip(source_runs['mode'], source_runs.seed, source_runs['repeat']))
    if set(actual_runs) != expected or len(actual_runs) != len(expected):
        raise ValueError('Require complete original nine-mode results without duplicates.')
    paths = {f'{m}_seed{s}_graph{r}': args.source/'checkpoints'/f'{m}_seed{s}_graph{r}.pt'
             for m, s, r in expected}
    if any(not v.exists() for v in paths.values()):
        raise FileNotFoundError('Missing source checkpoint.')
    print(f'Add {len(old["seeds"])*old["graph_repeats"]} random_mapping runs; '
          f'{cg.EPOCHS} epochs. Keep original {len(expected)} runs. Output: {args.output}', flush=True)
    # Bind every source checkpoint and table, rather than bypassing old signatures.
    args.manifest = dict(kind='random_mapping_addition', source_manifest=old,
        source=path_reference(args.source, relative_to=args.output),
        source_manifest_sha256=sha(args.source/'manifest.json'),
        source_runs_sha256=sha(args.source/'runs.csv'),
        source_checkpoints={k: sha(v) for k, v in sorted(paths.items())},
        implementation_sha256=sha(Path(__file__)), rng_stream=91729,
        mapping='Uniform all-node permutation of edge endpoints only; attributes travel with edges; inputs fixed')
    manifest_path = args.output/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != args.manifest:
            raise ValueError('Changed configuration/source: use a new output directory.')
    elif args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Nonempty output without manifest.')
    cg.save_json(manifest_path, args.manifest)
    rows = []
    for seed in old['seeds']:
        fit, inner, outer = cg.nested_holdout_split(len(x), seed)
        base_id = f'all_seed{seed}_graph0'
        baseline = torch.load(paths[base_id], map_location='cpu', weights_only=False)
        for name, ids in [('fit_idx', fit), ('select_idx', inner), ('eval_idx', outer)]:
            np.testing.assert_array_equal(baseline[name], ids)
        assert torch.equal(baseline['edge_index'], ei) and torch.equal(baseline['edge_attr'], ea)
        # Audit all old records against their signed configurations and predictions.
        for row in source_runs[source_runs.seed == seed].to_dict('records'):
            run_id = f'{row["mode"]}_seed{seed}_graph{row["repeat"]}'
            cp = torch.load(paths[run_id], map_location='cpu', weights_only=False)
            gh = hashlib.sha256(cp['edge_index'].numpy().tobytes()+cp['edge_attr'].numpy().tobytes()).hexdigest()
            config = dict(run_id=run_id, seed=seed, graph_hash=gh, experiment=old)
            if cp['config'] != config or cp['signature'] != cg.digest(config):
                raise ValueError(f'Invalid source signature: {run_id}')
            cg.enforce_pairing(cp, baseline)
            np.testing.assert_array_equal(cp['eval_idx'], outer)
            np.testing.assert_array_equal(cp['targets'], y[outer])
            for metric, value in cg.metrics(cp['targets'], cp['predictions']).items():
                np.testing.assert_allclose(row[metric], value, rtol=1e-6, atol=1e-6)
        cg.save_split_manifest(args.output, seed, fit, inner, outer, args.data)
        for rep in range(old['graph_repeats']):
            graph_seed = int(np.random.SeedSequence([seed, 91729, rep]).generate_state(1)[0])
            mapped, attrs, info = random_mapping(ei, ea, len(wl), graph_seed)
            run_id = f'random_mapping_seed{seed}_graph{rep}'
            cg.save_json(args.output/'mapping_diagnostics'/(run_id+'.json'), info)
            cp = cg.train_or_load(args, run_id, seed, mapped, attrs, x, y, wl)
            cg.enforce_pairing(cp, baseline)
            row = dict(mode='random_mapping', seed=seed, repeat=rep,
                n_directed_edges=mapped.shape[1], **cp['metrics'])
            row.update({'delta_'+m: cp['metrics'][m]-baseline['metrics'][m] for m in ['r2','rmse','mae']})
            rows.append(row)
            cg.save_csv(args.output/'new_runs.csv', rows)
    combined = pd.concat([source_runs, pd.DataFrame(rows)], ignore_index=True)
    cg.save_csv(args.output/'runs.csv', combined)
    keys = ['r2','rmse','mae','delta_r2','delta_rmse','delta_mae']
    per = combined.groupby(['mode','seed'])[keys].mean().reset_index()
    cg.save_csv(args.output/'per_seed.csv', per)
    summary = per.groupby('mode')[keys].agg(['mean','std','count'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    cg.save_csv(args.output/'summary.csv', summary.reset_index())
    spread = combined[combined['mode'].isin(['random_vi','random_mapping'])].groupby(['mode','seed'])[keys].agg(['mean','std','count'])
    spread.columns = ['_'.join(c) for c in spread.columns]
    cg.save_csv(args.output/'random_within_seed.csv', spread.reset_index())
    paired = []
    for mode in ['random_vi', 'random_mapping']:
        for seed in old['seeds']:
            a = per[(per['mode']=='all') & (per.seed==seed)].iloc[0]
            b = per[(per['mode']==mode) & (per.seed==seed)].iloc[0]
            paired.append(dict(control=mode, seed=seed, **{'all_minus_control_'+m: a[m]-b[m] for m in ['r2','rmse','mae']}))
    cg.save_csv(args.output/'paired_differences.csv', paired)
    cg.save_json(args.output/'status.json', dict(complete=True, original_runs=len(source_runs), new_runs=len(rows),
        note='Random repeats averaged within seed; overlapping holdouts; descriptive comparisons only.'))
    print('Complete: original nine modes plus random_mapping; source outputs unchanged.')


if __name__ == '__main__':
    main()
