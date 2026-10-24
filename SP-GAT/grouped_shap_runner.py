"""Unified grouped permutation SHAP for nine fixed models; never trains.

Reuse checkpoint provenance from the completed unified PFI experiment.
See GROUPED_SHAP_EXPERIMENT.md for estimand, cost, limitations and commands.
"""
import argparse
import itertools
from pathlib import Path
from release_paths import resolve_path, require_data

import numpy as np
import pandas as pd
import torch

import unified_pfi_runner as pfi
from unified_perturbation_runner import (load_exact_model, mark_complete, completed,
                                        save_npz, verify_marker)
from tabular_band_core import fp32_mode, groups_for
from tabular_band_runner import csv_write, hash_file, json_write, package_version
from evaluation_protocol import nested_holdout_split
from grouped_shap_core import (explain_sample, finite_predict, permutation_pairs,
                               sampling_diagnostics, ranking_similarity)

ROOT = Path(__file__).resolve().parent


def explain_run(args, name, seed, predictor, provenance, x, ids, fit, groups, signature):
    folder = args.output / name / f'seed{seed}' / args.split
    stamp = dict(signature=signature, checkpoint_sha256=provenance['checkpoint_sha256'],
                 background_ids=fit.tolist(), explained_ids=ids.tolist())
    if completed(folder, 'complete.json', stamp):
        print(f'Reuse SHAP: {name}/seed{seed}/{args.split}', flush=True)
        return
    background = x[fit]
    expected = float(finite_predict(predictor, background).mean())
    values, pairs, rows = [], [], []
    sample_files = []
    for sid in ids:
        path = folder / 'samples' / f'sample{sid}.npz'
        orders = permutation_pairs(seed, int(sid), len(groups), args.pairs)
        sample_stamp = dict(stamp, sample_id=int(sid))
        sample_marker = path.with_suffix('.json')
        if sample_marker.exists():
            saved_marker = pfi.read_json(sample_marker)
            if saved_marker['stamp'] != sample_stamp or hash_file(path) != saved_marker['sha256']:
                raise ValueError(f'Changed partial SHAP sample: {path}')
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved['orders'], orders)
                value, pair = saved['values'].copy(), saved['pair_values'].copy()
            quality = saved_marker['quality']
        else:
            print(f'  Explain {name}/seed{seed}: sample {int(sid)}, {args.pairs} forward/reverse pairs', flush=True)
            result = explain_sample(predictor, x[sid], background, groups, orders,
                coalition_batch=args.coalition_batch, expected_value=expected, atol=args.atol, rtol=args.rtol)
            value, pair = result.pop('values'), result.pop('pair_values')
            quality = dict(model=name, seed=seed, sample_id=int(sid), **result)
            save_npz(path, values=value, pair_values=pair, orders=orders)
            json_write(sample_marker, dict(stamp=sample_stamp, sha256=hash_file(path), quality=quality))
        values.append(value)
        pairs.append(pair)
        rows.append(quality)
        sample_files.extend([path.relative_to(folder).as_posix(), sample_marker.relative_to(folder).as_posix()])
    values, pairs = np.asarray(values), np.asarray(pairs)
    prediction = finite_predict(predictor, x[ids])
    residual = values.sum(axis=1) - (prediction - expected)
    if np.any(np.abs(residual) > args.atol + args.rtol * np.abs(prediction - expected)):
        raise ValueError('Final/resumed attribution additivity check failed.')
    save_npz(folder / 'grouped_shap.npz', values=values, pair_values=pairs,
             sample_ids=ids, background_ids=fit, expected_value=np.array(expected),
             predictions=prediction, residual=residual)
    csv_write(folder / 'sample_quality.csv', rows)
    csv_write(folder / 'group_importance.csv', [dict(model=name, seed=seed, split=args.split, group=g,
        mean_abs_shap=float(np.abs(values[:, g]).mean()), mean_signed_shap=float(values[:, g].mean()),
        n_samples=len(ids), positive_samples=int((values[:, g] > 0).sum())) for g in range(len(groups))])
    csv_write(folder / 'signed_shap_samples.csv', [dict(model=name, seed=seed, split=args.split,
        sample_id=int(sid), group=g, shap=float(values[i, g])) for i, sid in enumerate(ids) for g in range(len(groups))])
    json_write(folder / 'sampling_diagnostics.json', sampling_diagnostics(pairs))
    json_write(folder / 'provenance.json', pfi.provenance_for_folder(provenance, folder))
    mark_complete(folder, 'complete.json', stamp, sample_files + ['grouped_shap.npz', 'sample_quality.csv',
        'group_importance.csv', 'signed_shap_samples.csv', 'sampling_diagnostics.json', 'provenance.json'])


def summarize(args, groups, wl, signature):
    frames, diagnostics, stability, missing, samples = [], [], [], [], {}
    for name in args.models:
        for seed in args.seeds:
            folder = args.output / name / f'seed{seed}' / args.split
            if not (folder / 'complete.json').exists():
                missing.append(dict(model=name, seed=seed))
                continue
            marker = verify_marker(folder)
            if marker['stamp']['signature'] != signature:
                raise ValueError(f'Mismatched SHAP result: {folder}')
            frames.append(pd.read_csv(folder / 'group_importance.csv'))
            diagnostics.append(dict(model=name, seed=seed, **pfi.read_json(folder / 'sampling_diagnostics.json')))
            with np.load(folder / 'grouped_shap.npz') as saved:
                samples[name, seed] = (saved['sample_ids'].copy(), saved['values'].copy())
    scope = pfi.digest(dict(models=sorted(args.models), seeds=sorted(args.seeds)))[:12]
    full = set(args.models) == set(pfi.MODELS) and set(args.seeds) == set(pfi.SEEDS)
    out = args.output / (f'summary_{args.split}' if full else f'summary_{args.split}_subset_{scope}')
    if frames:
        frame = pd.concat(frames, ignore_index=True)
        csv_write(out / 'importance_per_seed.csv', frame)
        summary = frame.groupby(['model', 'group'])[['mean_abs_shap', 'mean_signed_shap']].agg(['mean', 'std', 'count'])
        summary.columns = ['_'.join(c) for c in summary.columns]
        summary = summary.reset_index()
        summary['wavelength_min'] = summary.group.map(lambda g: float(wl[groups[g]].min()))
        summary['wavelength_max'] = summary.group.map(lambda g: float(wl[groups[g]].max()))
        summary['complete_requested_seeds'] = summary.mean_abs_shap_count == len(args.seeds)
        csv_write(out / 'importance_summary.csv', summary)
        csv_write(out / 'sampling_diagnostics.csv', diagnostics)
        for name in args.models:
            available = [s for s in args.seeds if (name, s) in samples]
            for left, right in itertools.combinations(available, 2):
                li, lv = samples[name, left]
                ri, rv = samples[name, right]
                stability.append(dict(model=name, seed_a=left, seed_b=right, scope='global',
                    n_shared=len(np.intersect1d(li, ri)),
                    **ranking_similarity(np.abs(lv).mean(0), np.abs(rv).mean(0))))
                for sid in np.intersect1d(li, ri):
                    stability.append(dict(model=name, seed_a=left, seed_b=right, scope='shared_sample',
                        sample_id=int(sid), n_shared=1,
                        **ranking_similarity(np.abs(lv[np.flatnonzero(li == sid)[0]]),
                                             np.abs(rv[np.flatnonzero(ri == sid)[0]]))))
        csv_write(out / 'stability.csv', stability)
        if not missing:
            plot_top10(summary, out, args)
    json_write(out / 'status.json', dict(signature=signature, missing=missing,
        complete_runs=len(frames), expected_runs=len(args.models) * len(args.seeds),
        note='Cross-seed/global and shared-sample comparisons have dependent pairs and different fit backgrounds; descriptive only.'))


def plot_top10(summary, output, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name in args.models:
        part = summary[summary.model == name].sort_values(['mean_abs_shap_mean', 'group'], ascending=[False, True]).head(10)
        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        fig.subplots_adjust(left=.23, right=.96, top=.85, bottom=.22)
        y = np.arange(len(part))
        ax.barh(y, part.mean_abs_shap_mean, xerr=part.mean_abs_shap_std.fillna(0),
                color='#347A9C', capsize=3, height=.62)
        ax.set_yticks(y)
        ax.set_yticklabels([f'{r.wavelength_min:.1f}–{r.wavelength_max:.1f}' for r in part.itertuples()])
        ax.invert_yaxis()
        ax.set_xlabel('Mean absolute grouped SHAP value (g/m²)')
        ax.set_ylabel('Wavelength group (nm)')
        ax.set_title(f'{name} | Top 10 grouped SHAP importance', pad=18)
        ax.axvline(0, color='gray', lw=.7)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        fig.text(.23, .045, f'{len(args.seeds)} seeds; error bars: sample SD of seed means.\n'
                 'Empirical interventional background; model-specific axis scale.', fontsize=9)
        for ext in ('png', 'pdf'):
            fig.savefig(output / f'{name}_top10_grouped_shap.{ext}', dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'AGB.csv')
    parser.add_argument('--pfi-root', type=Path, default=ROOT / 'nested_holdout_outputs/unified_pfi_v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'nested_holdout_outputs/grouped_shap_v1')
    parser.add_argument('--models', nargs='+', choices=pfi.MODELS, default=list(pfi.MODELS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(pfi.SEEDS))
    parser.add_argument('--split', choices=['outer', 'inner'], default='outer')
    parser.add_argument('--group-width', type=float, default=10.)
    parser.add_argument('--pairs', type=int, default=16, help='Random forward/reverse pairs (32 permutations by default).')
    parser.add_argument('--coalition-batch', type=int, default=4)
    parser.add_argument('--atol', type=float, default=1e-3)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads', type=int, default=4)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--plan-only', action='store_true')
    modes.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.pairs < 4 or args.pairs % 2 or args.coalition_batch < 1 or args.threads < 1:
        parser.error('Require an even number of pairs >=4, positive batch/threads.')
    if not np.isfinite(args.group_width) or args.group_width <= 0 or any(not np.isfinite(v) or v < 0 for v in (args.atol, args.rtol)):
        parser.error('Invalid width or additivity tolerance.')
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Duplicate models/seeds.')
    if any(s < 0 or s > 2**32 - 10001 for s in args.seeds):
        parser.error('Invalid seed.')
    args.data = resolve_path(args.data)
    args.pfi_root, args.output = resolve_path(args.pfi_root), resolve_path(args.output)
    if args.output == args.pfi_root or args.pfi_root in args.output.parents or args.output in args.pfi_root.parents:
        parser.error('SHAP output must be separate from the source experiment.')
    args.data = require_data(args.data)
    data = pd.read_csv(args.data)
    x, y = data.iloc[:, :-1].to_numpy(float), data.iloc[:, -1].to_numpy(float)
    wl = np.asarray(data.columns[:-1], float)
    if len(x) < 15 or len(wl) < 2 or not all(np.isfinite(a).all() for a in (x, y, wl)) or np.any(np.diff(wl) <= 0):
        parser.error('Invalid numeric spectral data.')
    groups = groups_for(wl, args.group_width)
    source = pfi.read_json(args.pfi_root / 'experiment.json')
    if source['signature'] != pfi.digest(source['config']) or source['config']['data_sha256'] != hash_file(args.data):
        raise ValueError('Source manifest/data mismatch.')
    if source['config']['protocol'] != 'nested-holdout':
        raise ValueError('Expected nested holdout checkpoints.')
    for filename, expected in source['config']['sources'].items():
        if not pfi.compatible_source(filename, expected):
            raise ValueError(f'Source implementation changed: {filename}')
    modules = {'scikit-learn': 'sklearn', 'torch-geometric': 'torch_geometric'}
    versions = {n: package_version(n, modules.get(n, n)) for n in source['config']['versions']}
    if versions != source['config']['versions']:
        raise ValueError('Source dependency versions changed.')
    config = dict(schema=1, source_signature=source['signature'],
        sources={n: hash_file(ROOT / n) for n in ['grouped_shap_core.py', 'grouped_shap_runner.py', 'unified_perturbation_runner.py']},
        versions=versions, width=args.group_width, pairs=args.pairs, coalition_batch=args.coalition_batch,
        atol=args.atol, rtol=args.rtol, device=args.device, threads=args.threads,
        game='Groups are players; v(S)=mean fit-background f(x_S,b_notS); identity link',
        estimator='Custom antithetic permutation Shapley; exact empirical background average',
        background='All fit rows only; missing groups share the same background row',
        global_importance='Mean abs signed local SHAP within seed, then equal-weight seeds',
        precision='Neural FP32, TF32 disabled; float64 attribution accumulation')
    signature = pfi.digest(config)
    upper = 0
    for seed in args.seeds:
        fit, inner, outer = nested_holdout_split(len(x), seed)
        ids = outer if args.split == 'outer' else inner
        upper += len(args.models) * len(ids) * 2 * args.pairs * (len(groups) - 1) * len(fit)
    print(f'{len(args.models)} models x {len(args.seeds)} seeds; {len(groups)} groups; {args.pairs} pairs.', flush=True)
    print(f'Hybrid spectral rows before coalition-cache savings: <= {upper:,}; no model training.', flush=True)
    if args.plan_only:
        missing = []
        for name in args.models:
            for seed in args.seeds:
                folder = args.pfi_root / name / f'seed{seed}'
                if not (folder / 'complete.json').exists():
                    missing.append(f'{name}/seed{seed}')
        print(f'Missing source runs: {missing}\nOutput: {args.output}', flush=True)
        return
    manifest = args.output / 'experiment.json'
    if manifest.exists():
        if pfi.read_json(manifest) != dict(signature=signature, config=config):
            raise ValueError('SHAP config/source changed; use a new output directory.')
    elif args.summarize_only:
        raise FileNotFoundError('No SHAP experiment to summarize.')
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError('Nonempty output without SHAP manifest.')
        json_write(manifest, dict(signature=signature, config=config))
    if args.summarize_only:
        summarize(args, groups, wl, signature)
        return
    torch.set_num_threads(args.threads)
    json_write(args.output / 'groups.json', [dict(group=i, bands=g.tolist(), wavelength_min=float(wl[g].min()),
        wavelength_max=float(wl[g].max())) for i, g in enumerate(groups)])
    for seed in args.seeds:
        fit, inner, outer = nested_holdout_split(len(x), seed)
        ids = outer if args.split == 'outer' else inner
        shared = args.output / 'shared_design' / f'seed{seed}_{args.split}.npz'
        orders = np.stack([permutation_pairs(seed, int(s), len(groups), args.pairs) for s in ids])
        if shared.exists():
            with np.load(shared) as saved:
                np.testing.assert_array_equal(saved['orders'], orders)
                np.testing.assert_array_equal(saved['background_ids'], fit)
                np.testing.assert_array_equal(saved['sample_ids'], ids)
        else:
            save_npz(shared, orders=orders, background_ids=fit, sample_ids=ids)
        for name in args.models:
            fp32_mode(seed)
            predictor, provenance = load_exact_model(args, name, seed, x, y, wl,
                                                     source['signature'], source['config']['data_sha256'])
            explain_run(args, name, seed, predictor, provenance, x, ids, fit, groups, signature)
            del predictor
    summarize(args, groups, wl, signature)


if __name__ == '__main__':
    main()
