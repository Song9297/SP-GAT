"""Fixed-weight, within-prior-type edge deletion; never trains a model.

Run: python typed_edge_deletion_runner.py
Uses the complete-model checkpoints underlying the ablation attention figures.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from controlled_graph_experiments import predictor_from, digest, state_digest
from evaluation_protocol import nested_holdout_split
from explainability_core import attention_pair_scores, pair_records, prior_graph, metrics
from release_paths import resolve_path, require_data

ROOT = Path(__file__).resolve().parent
TYPES = ['Red-NIR', 'Green-NIR', 'Red edge-NIR']
SEEDS = [42, 123, 456, 789, 2026]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def deletion_cases(keys, slots, scores, n_edges, seed, ratios, repeats):
    """Stable ties by wavelength indices; nested sets within each strategy/repeat."""
    types = np.array([key[2] for key in keys])
    for typ in range(3):
        ids = np.flatnonzero(types == typ)
        # Sorting tied scores by node indices is deterministic for both rankings.
        high = sorted(ids, key=lambda i: (-scores[i], keys[i][0], keys[i][1]))
        low = sorted(ids, key=lambda i: (scores[i], keys[i][0], keys[i][1]))
        for strategy in ['top', 'random', 'low']:
            for rep in range(repeats if strategy == 'random' else 1):
                rng = np.random.default_rng(np.random.SeedSequence([seed, 92501, typ, rep]))
                order = rng.permutation(ids) if strategy == 'random' else high if strategy == 'top' else low
                for ratio in ratios:
                    count = int(math.ceil(len(ids) * ratio))
                    selected = np.asarray(order[:count], dtype=int)
                    mask = torch.ones(n_edges, dtype=torch.bool)
                    for i in selected:
                        mask[slots[i]] = False
                    assert int((~mask).sum()) == 2 * count
                    assert all(keys[i][2] == typ for i in selected)
                    yield dict(edge_type=TYPES[typ], type_id=typ, strategy=strategy,
                               repeat=rep, nominal_fraction=ratio,
                               actual_fraction=count / len(ids), n_type_pairs=len(ids),
                               n_deleted_pairs=count), selected, mask


def summarize(out, seeds):
    runs = pd.concat([pd.read_csv(out / f'seed{s}/runs.csv') for s in seeds], ignore_index=True)
    runs.to_csv(out / 'runs.csv', index=False)
    columns = ['rmse', 'delta_rmse', 'mae', 'delta_mae', 'r2', 'delta_r2']
    groups = ['edge_type', 'strategy', 'nominal_fraction', 'actual_fraction',
              'n_type_pairs', 'n_deleted_pairs']
    per_seed = runs.groupby(groups + ['seed'], as_index=False)[columns].mean()
    per_seed.to_csv(out / 'per_seed.csv', index=False)
    summary = per_seed.groupby(groups)[columns].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / 'summary.csv', index=False)
    spread = runs[runs.strategy == 'random'].groupby(groups + ['seed'])[columns].agg(['mean', 'std', 'count'])
    spread.columns = ['_'.join(c) for c in spread.columns]
    spread.to_csv(out / 'random_within_seed.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.family': 'DejaVu Serif', 'font.size': 12,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'xtick.direction': 'in', 'ytick.direction': 'in'}):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
        fig.subplots_adjust(left=.075, right=.985, bottom=.25, top=.87, wspace=.3)
        for i, (ax, name) in enumerate(zip(axes, TYPES)):
            for strategy, color in [('top', '#1f77b4'), ('random', '#2ca02c'), ('low', '#ff7f0e')]:
                p = summary[(summary.edge_type == name) & (summary.strategy == strategy)].sort_values('nominal_fraction')
                ax.errorbar(np.r_[0, p.nominal_fraction * 100], np.r_[0, p.delta_rmse_mean],
                            yerr=np.r_[0, p.delta_rmse_std.fillna(0)], color=color,
                            marker='o', ms=4, lw=1.2, capsize=3, label=strategy.capitalize())
            ax.axhline(0, color='gray', lw=.6)
            ax.set_title(f'({chr(97+i)}) {name}', loc='left', fontsize=12)
            ax.set_xticks([0, 10, 20, 30])
        fig.supxlabel('Deleted connections within prior type (%)', y=.12, fontsize=12)
        fig.supylabel('ΔRMSE (g/m²)', x=.01, fontsize=12)
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False)
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(out / f'typed_edge_deletion.{ext}', dpi=600)
        fig.savefig(out / 'preview.png', dpi=110)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'nested_holdout_outputs/ablation_controlled_v2')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/AGB.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'nested_holdout_outputs/typed_edge_deletion_v1')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--ratios', nargs='+', type=float, default=[.1, .2, .3])
    parser.add_argument('--random-repeats', type=int, default=20)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--plan-only', action='store_true', help='Print the checkpoint and evaluation plan without data or writes.')
    args = parser.parse_args()
    for name in ['source', 'data', 'output']:
        setattr(args, name, resolve_path(getattr(args, name)))
    if args.source == args.output or args.source in args.output.parents or args.output in args.source.parents:
        parser.error('Source and output must be separate, non-nested directories.')
    if len(set(args.seeds)) != len(args.seeds) or any(s not in SEEDS for s in args.seeds):
        parser.error('Use unique seeds from the five saved models.')
    if args.random_repeats < 2 or len(set(args.ratios)) != len(args.ratios) or any(not 0 < r < 1 for r in args.ratios):
        parser.error('Require >=2 random repeats and unique fractions between zero and one.')
    if args.threads < 1:
        parser.error('Threads must be positive.')
    args.ratios.sort()
    paths = {s: args.source / 'checkpoints' / f'all_seed{s}_graph0.pt' for s in args.seeds}
    if args.plan_only:
        print(f'Fixed-weight deletion: {len(args.seeds)} base checkpoints; '
              f'{len(args.seeds) * 3 * len(args.ratios) * (args.random_repeats + 2)} evaluations.')
        print(f'Source: {args.source}; output: {args.output}')
        print('First generate complete-model checkpoints with controlled_graph_experiments.py ablation.')
        print('Plan only: no data reads, model loading or writes.')
        return
    require_data(args.data)
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing base checkpoints: ' + ', '.join(missing) +
                                '. Run controlled_graph_experiments.py ablation first.')
    torch.set_num_threads(args.threads)
    frame = pd.read_csv(args.data)
    x, y = frame.iloc[:, :-1].to_numpy(np.float32), frame.iloc[:, -1].to_numpy(np.float32)
    wl = np.asarray(frame.columns[:-1], float)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    data_hash = sha(args.data)
    config = dict(data_hash=data_hash, checkpoints={str(s): sha(p) for s, p in paths.items()},
                  seeds=args.seeds, ratios=args.ratios, random_repeats=args.random_repeats,
                  device=args.device, torch_version=torch.__version__, script_hash=sha(__file__),
                  dependencies={n: sha(ROOT/n) for n in ['explainability_core.py', 'explainability_models.py',
                                'controlled_graph_experiments.py', 'evaluation_protocol.py']},
                  rank_source='inner validation only; last-layer reciprocal mean attention',
                  deletion='ceil(fraction * pairs within type); both directions; other types intact',
                  self_loops='retained by model; attention recomputed and renormalized in forward pass',
                  summary='random repeats averaged within seed; then equal seed mean and sample SD',
                  training=False)
    manifest = args.output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != config:
        raise ValueError('Output configuration differs; use a new output directory.')
    if args.output.exists() and any(args.output.iterdir()) and not manifest.is_file():
        raise ValueError('Nonempty output has no manifest; use a new output directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(manifest, config)
    write_json(args.output / 'status.json', {'complete': False})
    for seed, path in paths.items():
        out = args.output / f'seed{seed}'
        if (out / 'complete.json').exists():
            done = json.loads((out / 'complete.json').read_text())
            assert all(sha(out / name) == h for name, h in done['hashes'].items())
            print(f'Reuse seed {seed}', flush=True)
            continue
        out.mkdir(exist_ok=True)
        cp = torch.load(path, map_location='cpu', weights_only=False)
        assert cp['signature'] == digest(cp['config'])
        assert cp['config']['experiment']['data_hash'] == data_hash
        assert cp['config']['seed'] == seed
        fit, inner, outer = nested_holdout_split(len(x), seed)
        for key, ids in [('fit_idx', fit), ('select_idx', inner), ('eval_idx', outer)]:
            np.testing.assert_array_equal(cp[key], ids)
        np.testing.assert_array_equal(cp['targets'], y[outer])
        np.testing.assert_allclose(cp['stats']['y_mean'], y[fit].mean())
        ei, ea = prior_graph(wl, 'all')
        assert torch.equal(cp['edge_index'], ei) and torch.equal(cp['edge_attr'], ea)
        pred = predictor_from(cp, wl, args.device)
        frozen = state_digest(pred.model.state_dict())
        baseline = pred.predict(x[outer])
        np.testing.assert_allclose(baseline, cp['predictions'], rtol=1e-5, atol=1e-3)
        base_metrics = metrics(y[outer], baseline)
        keys, slots, scores_by_sample = attention_pair_scores(pred, x[inner])
        assert np.isfinite(scores_by_sample).all()
        scores = scores_by_sample.mean(0)
        pd.DataFrame([dict(pair_index=i, source_idx=a, target_idx=b, edge_type=TYPES[t],
                           source_nm=wl[a], target_nm=wl[b], inner_attention=scores[i])
                      for i, (a, b, t) in enumerate(keys)]).to_csv(out / 'ranking.csv', index=False)
        np.savez_compressed(out / 'inner_attention.npz', sample_ids=inner, edge_keys=keys, attention=scores_by_sample)
        pd.DataFrame(dict(sample_id=outer, actual=y[outer], predicted=baseline)).to_csv(out / 'baseline_predictions.csv', index=False)
        rows, predictions, deletions = [], [], []
        for info, selected, mask in deletion_cases(keys, slots, scores, ei.shape[1], seed, args.ratios, args.random_repeats):
            kept_ei, kept_ea = ei[:, mask], ea[mask]
            # Surviving other-type pairs must be identical, including edge attributes.
            remaining, _ = pair_records(kept_ei, kept_ea)
            assert set(remaining) == set(keys) - {keys[i] for i in selected}
            values = pred.predict(x[outer], kept_ei.to(args.device), kept_ea.to(args.device))
            assert np.isfinite(values).all()
            measures = metrics(y[outer], values)
            rows.append(dict(seed=seed, **info, **measures,
                             **{'delta_' + k: v - base_metrics[k] for k, v in measures.items()}))
            for idx, actual, value in zip(outer, y[outer], values):
                predictions.append(dict(seed=seed, **info, sample_id=int(idx), actual=float(actual), predicted=float(value)))
            deletions.append(dict(seed=seed, **info, deleted_pairs=[list(keys[i]) for i in selected]))
        assert state_digest(pred.model.state_dict()) == frozen
        assert sha(path) == config['checkpoints'][str(seed)]
        assert len(rows) == 3 * len(args.ratios) * (args.random_repeats + 2)
        pd.DataFrame(rows).to_csv(out / 'runs.csv', index=False)
        pd.DataFrame(predictions).to_csv(out / 'predictions.csv', index=False)
        write_json(out / 'deletions.json', deletions)
        write_json(out / 'baseline.json', dict(seed=seed, metrics=base_metrics, fit=fit.tolist(), inner=inner.tolist(), outer=outer.tolist()))
        files = [p for p in out.iterdir() if p.is_file() and p.name != 'complete.json']
        write_json(out / 'complete.json', {'hashes': {p.name: sha(p) for p in files}})
        print(f'Seed {seed}: {len(rows)} deletions evaluated; baseline RMSE={base_metrics["rmse"]:.4f}', flush=True)
    summarize(args.output, args.seeds)
    assert sha(args.data) == data_hash
    write_json(args.output / 'status.json', {'complete': True, 'seeds': args.seeds,
                'evaluations': len(args.seeds) * 3 * len(args.ratios) * (args.random_repeats + 2)})
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
