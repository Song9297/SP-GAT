"""Five-model spectral reliance experiments. See TABULAR_BAND_EXPERIMENTS.md.

    python tabular_band_runner.py --dry-run
    python tabular_band_runner.py --stage all
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import itertools
import json
import platform
from pathlib import Path
from release_paths import resolve_path, require_data

import joblib
import numpy as np
import pandas as pd
import torch

from evaluation_protocol import nested_holdout_split, save_split_manifest
from tabular_band_core import (MODELS, FittedModel, donor_importance, fit_model,
                               fixed_perturbation, fp32_mode, groups_for, metrics,
                               removed_columns, similarity)

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 123, 456, 789, 2026]


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    temp = path.with_suffix('.csv.tmp')
    frame.to_csv(temp, index=False, encoding='utf-8-sig')
    temp.replace(path)


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_version(distribution, module=None):
    # The loaded implementation is authoritative when metadata are stale or absent.
    modules = {'scikit-learn': 'sklearn', 'torch-geometric': 'torch_geometric'}
    loaded = importlib.import_module(module or modules.get(distribution, distribution))
    version = getattr(loaded, '__version__', None)
    return str(version) if version is not None else importlib.metadata.version(distribution)


def cached_fit(folder, tag, signature, name, seed, x, y, fit, inner, columns, args):
    path = folder / 'checkpoints' / (tag + '.joblib')
    request = dict(signature=signature, model=name, seed=int(seed), columns=columns.tolist())
    if path.exists():
        saved = joblib.load(path)  # local runner-produced checkpoints only
        if saved['request'] != request:
            raise ValueError(f'Checkpoint mismatch: {path}; use a new output directory.')
        return FittedModel(saved['bundle'], args.device)
    print(f'  Fit {name} seed={seed} {tag}: {len(columns)} bands', flush=True)
    bundle = fit_model(name, seed, x, y, fit, inner, columns, args.epochs, args.device, args.jobs)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.joblib.tmp')
    joblib.dump(dict(request=request, bundle=bundle), temp)
    temp.replace(path)
    audit = dict(columns=columns.tolist(), y_mean=bundle['y_mean'], y_std=bundle['y_std'],
                 x_mean=bundle['scaler'].mean_.tolist(), x_scale=bundle['scaler'].scale_.tolist(),
                 best_epoch=bundle['best_epoch'], estimator_params=bundle.get('estimator_params'),
                 initial_state_sha256=bundle.get('initial_state_sha256'),
                 batch_order_sha256=bundle.get('batch_order_sha256'),
                 history=bundle['history'], signature=signature)
    json_write(folder / 'training_audit' / (tag + '.json'), audit)
    return FittedModel(bundle, args.device)


def run_explanation(folder, model, name, seed, x, y, fit, inner, outer, groups, args):
    cache = folder / 'importance.npz'
    if cache.exists():
        with np.load(cache) as saved:
            return saved['inner_scores'].copy(), saved['outer_scores'].copy()
    inner_scores, inner_repeats, inner_donors = donor_importance(
        model, x[inner], x[fit], groups, seed, args.importance_repeats)
    outer_scores, outer_repeats, outer_donors = donor_importance(
        model, x[outer], x[fit], groups, seed, args.importance_repeats)
    ranking_rows = []
    for split, scores in [('inner', inner_scores), ('outer', outer_scores)]:
        mean = scores.mean(0)
        for j, value in enumerate(mean):
            ranking_rows.append(dict(model=name, seed=seed, split=split, group=j,
                importance=float(value), relative_importance=float(value / mean.sum()) if mean.sum() else 0.,
                n_samples=len(scores)))
    csv_write(folder / 'group_importance.csv', ranking_rows)
    # Monte Carlo half-split checks quantify estimator noise separately from seed stability.
    mc = []
    for split, repeated in [('inner', inner_repeats), ('outer', outer_repeats)]:
        first, second = repeated[::2].mean(0), repeated[1::2].mean(0)
        mc.append(dict(model=name, seed=seed, split=split, scope='global',
                       **similarity(first.mean(0), second.mean(0))))
    csv_write(folder / 'importance_mc_stability.csv', mc)
    rows, samples = fixed_perturbation(model, x[outer], y[outer], outer, x[fit].mean(0),
        groups, outer_scores, seed, args.ratios, args.random_repeats)
    for items in (rows, samples):
        for row in items:
            row.update(model=name, seed=seed)
    csv_write(folder / 'band_perturbation.csv', rows)
    csv_write(folder / 'band_perturbation_samples.csv', samples)
    # Write completion marker LAST so interrupted fixed perturbations are rerun.
    temporary = folder / 'importance.tmp.npz'
    np.savez_compressed(temporary, inner_scores=inner_scores, outer_scores=outer_scores,
        inner_repeats=inner_repeats, outer_repeats=outer_repeats,
        inner_ids=inner, outer_ids=outer, fit_ids=fit,
        inner_donor_ids=fit[inner_donors], outer_donor_ids=fit[outer_donors])
    temporary.replace(cache)
    return inner_scores, outer_scores


def run_retraining(folder, baseline, name, seed, x, y, fit, inner, outer, groups,
                   inner_scores, args, signature):
    reference_prediction = baseline.predict(x[outer])
    reference = metrics(y[outer], reference_prediction)
    rows = [dict(model=name, seed=seed, strategy='baseline', repeat=0, ratio=0.,
                 n_groups=0, n_bands_removed=0, actual_fraction=0., **reference,
                 delta_r2=0., delta_rmse=0., delta_mae=0.)]
    samples = [dict(model=name, seed=seed, strategy='baseline', repeat=0, ratio=0.,
                    sample_id=int(i), target=float(y[i]), prediction=float(p))
               for i, p in zip(outer, reference_prediction)]
    selection_rows = []
    for strategy in ('top', 'random'):
        for repeat in range(args.retrain_repeats if strategy == 'random' else 1):
            if strategy == 'top':
                order = np.argsort(-inner_scores.mean(0), kind='stable')
            else:
                rng = np.random.default_rng(np.random.SeedSequence([seed, 62000, repeat]))
                order = rng.permutation(len(groups))
            for ratio in args.ratios:
                removed = removed_columns(order, groups, ratio)
                columns = np.setdiff1d(np.arange(x.shape[1]), removed)
                tag = f'{strategy}_repeat{repeat}_ratio{ratio:g}'
                model = cached_fit(folder, tag, signature, name, seed, x, y, fit, inner, columns, args)
                if name == 'MLP' and model.bundle['batch_order_sha256'] != baseline.bundle['batch_order_sha256']:
                    raise AssertionError('MLP batch orders differ across paired runs.')
                prediction = model.predict(x[outer])
                m = metrics(y[outer], prediction)
                common = dict(model=name, seed=seed, strategy=strategy, repeat=repeat, ratio=ratio)
                k = int(np.ceil(ratio * len(groups)))
                rows.append(dict(**common, n_groups=k, n_bands_removed=len(removed),
                    actual_fraction=k / len(groups), **m, delta_r2=reference['r2'] - m['r2'],
                    delta_rmse=m['rmse'] - reference['rmse'], delta_mae=m['mae'] - reference['mae']))
                samples.extend(dict(**common, sample_id=int(i), target=float(y[i]), prediction=float(p))
                               for i, p in zip(outer, prediction))
                selection_rows.append(dict(**common, ranking_split='inner',
                    removed_groups=order[:k].tolist(), removed_bands=removed.tolist(),
                    retained_bands=columns.tolist()))
                csv_write(folder / 'retraining.csv', rows)
                csv_write(folder / 'retraining_predictions.csv', samples)
                json_write(folder / 'retraining_selections.json', selection_rows)
    json_write(folder / 'retraining_complete.json', dict(signature=signature, runs=len(rows)))


def summarize(root, args, groups, wavelengths):
    def collect(filename):
        parts = [pd.read_csv(root / name / f'seed{seed}' / filename)
                 for name in args.models for seed in args.seeds
                 if (root / name / f'seed{seed}' / filename).exists()]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    output = root / 'summary'
    for experiment, filename in [('fixed', 'band_perturbation.csv'), ('retraining', 'retraining.csv')]:
        data = collect(filename)
        if data.empty:
            continue
        csv_write(output / f'{experiment}_runs.csv', data)
        values = ['r2', 'rmse', 'mae', 'delta_r2', 'delta_rmse', 'delta_mae']
        if 'prediction_change' in data:
            values.append('prediction_change')
        keys = ['model', 'seed', 'strategy', 'ratio']
        per_seed = data.groupby(keys)[values].mean().reset_index()
        csv_write(output / f'{experiment}_per_seed.csv', per_seed)
        aggregate = per_seed.groupby(['model', 'strategy', 'ratio'])[values].agg(['mean', 'std', 'count'])
        aggregate.columns = ['_'.join(c) for c in aggregate.columns]
        csv_write(output / f'{experiment}_summary.csv', aggregate.reset_index())
        control = per_seed[per_seed.strategy == 'random'].drop(columns='strategy')
        top = per_seed[per_seed.strategy == 'top'].drop(columns='strategy')
        paired = top.merge(control, on=['model', 'seed', 'ratio'], suffixes=('_top', '_random'))
        for metric in ('delta_r2', 'delta_rmse', 'delta_mae'):
            paired[f'{metric}_top_minus_random'] = paired[f'{metric}_top'] - paired[f'{metric}_random']
        csv_write(output / f'{experiment}_paired.csv', paired)
        random = data[data.strategy == 'random']
        spread = random.groupby(['model', 'seed', 'ratio'])[values].agg(['mean', 'std', 'min', 'max'])
        spread.columns = ['_'.join(c) for c in spread.columns]
        csv_write(output / f'{experiment}_random_within_seed.csv', spread.reset_index())
        plot_curves(aggregate.reset_index(), output / f'{experiment}_delta_rmse.png')

    importance = collect('group_importance.csv')
    if not importance.empty:
        csv_write(output / 'group_importance_per_seed.csv', importance)
        aggregate = importance.groupby(['model', 'split', 'group'])[['importance', 'relative_importance']].agg(['mean', 'std', 'count'])
        aggregate.columns = ['_'.join(c) for c in aggregate.columns]
        csv_write(output / 'group_importance_summary.csv', aggregate.reset_index())
        plot_importance(aggregate.reset_index(), groups, wavelengths, output / 'group_importance.png')
    mc = collect('importance_mc_stability.csv')
    if not mc.empty:
        csv_write(output / 'importance_mc_stability.csv', mc)
    pairs, shared_samples = [], []
    for name in args.models:
        loaded = {}
        for seed in args.seeds:
            path = root / name / f'seed{seed}' / 'importance.npz'
            if path.exists():
                with np.load(path) as z:
                    loaded[seed] = (z['outer_ids'].copy(), z['outer_scores'].copy())
        for a, b in itertools.combinations(sorted(loaded), 2):
            ids_a, scores_a = loaded[a]
            ids_b, scores_b = loaded[b]
            common = np.intersect1d(ids_a, ids_b)
            pairs.append(dict(model=name, seed_a=a, seed_b=b, scope='global_outer',
                n_shared=len(common), **similarity(scores_a.mean(0), scores_b.mean(0))))
            within = []
            for sid in common:
                m = similarity(scores_a[np.flatnonzero(ids_a == sid)[0]], scores_b[np.flatnonzero(ids_b == sid)[0]])
                within.append(m)
                shared_samples.append(dict(model=name, seed_a=a, seed_b=b, sample_id=int(sid), **m))
            if within:
                means = pd.DataFrame(within)[['spearman', 'top_overlap']].mean()
                pairs.append(dict(model=name, seed_a=a, seed_b=b, scope='shared_outer_samples',
                    n_shared=len(common), spearman=means['spearman'], top_overlap=means['top_overlap'],
                    valid=any(v['valid'] for v in within)))
    if pairs:
        csv_write(output / 'stability_seed_pairs.csv', pairs)
        stability = pd.DataFrame(pairs).groupby(['model', 'scope'])[['spearman', 'top_overlap']].agg(['mean', 'std', 'count'])
        stability.columns = ['_'.join(c) for c in stability.columns]
        csv_write(output / 'stability_summary.csv', stability.reset_index())
    if shared_samples:
        csv_write(output / 'stability_shared_samples.csv', shared_samples)
    performance = collect('baseline_performance.csv')
    if not performance.empty:
        csv_write(output / 'baseline_performance.csv', performance)
    json_write(output / 'status.json', dict(expected_model_seeds=len(args.models) * len(args.seeds),
        explanations_complete=sum((root / n / f'seed{s}' / 'importance.npz').exists() for n in args.models for s in args.seeds),
        retraining_complete=sum((root / n / f'seed{s}' / 'retraining_complete.json').exists() for n in args.models for s in args.seeds),
        note='Sample SD (ddof=1); repeated controls averaged within seed; seed pairs are dependent.'))


def plot_curves(frame, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = frame.model.unique()
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.5), squeeze=False)
    for ax, name in zip(axes[0], names):
        for strategy, part in frame[frame.model == name].groupby('strategy'):
            if strategy == 'baseline':
                continue
            part = part.sort_values('ratio')
            ax.errorbar(part.ratio * 100, part.delta_rmse_mean,
                yerr=part.delta_rmse_std.fillna(0), marker='o', capsize=3, label=strategy)
        ax.axhline(0, color='gray', lw=.6)
        ax.set(title=name, xlabel='Wavelength groups removed (%)', ylabel='Delta RMSE')
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_importance(frame, groups, wavelengths, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    centers = np.array([wavelengths[g].mean() for g in groups])
    for name, part in frame[frame.split == 'outer'].groupby('model'):
        part = part.sort_values('group')
        ax.plot(centers[part.group.to_numpy(int)], part.relative_importance_mean, label=name)
    ax.set(xlabel='Wavelength (nm)', ylabel='Within-model normalized reliance')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data' / 'AGB.csv')
    p.add_argument('--output', type=Path, default=ROOT / 'nested_holdout_outputs' / 'tabular_band_v1')
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    p.add_argument('--stage', choices=['all', 'explain', 'retrain', 'summarize'], default='all')
    p.add_argument('--group-width', type=float, default=10.)
    p.add_argument('--ratios', nargs='+', type=float, default=[.05, .1, .2, .3])
    p.add_argument('--importance-repeats', type=int, default=20)
    p.add_argument('--random-repeats', type=int, default=20)
    p.add_argument('--retrain-repeats', type=int, default=5)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--jobs', type=int, default=-1)
    p.add_argument('--dry-run', action='store_true', help='Validate and show plan; no training or file writes.')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.data = resolve_path(args.data)
    args.output = resolve_path(args.output)
    if (not np.isfinite(args.group_width) or args.group_width <= 0 or args.importance_repeats < 2
            or min(args.random_repeats, args.retrain_repeats, args.epochs) < 1):
        raise ValueError('Positive width/epochs/repeats required; importance repeats >= 2.')
    if any(not 0 < r < 1 for r in args.ratios) or len(set(args.ratios)) != len(args.ratios):
        raise ValueError('Ratios must be unique and strictly between 0 and 1.')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.models)) != len(args.models):
        raise ValueError('Duplicate models or seeds are not allowed.')
    args.ratios = sorted(args.ratios)
    args.data = require_data(args.data)
    data = pd.read_csv(args.data)
    x, y = data.iloc[:, :-1].to_numpy(float), data.iloc[:, -1].to_numpy(float)
    wl = np.asarray(data.columns[:-1], dtype=float)
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(wl).all()):
        raise ValueError('Data and wavelengths must be finite numeric values.')
    if len(x) < 15 or len(wl) < 2 or np.any(np.diff(wl) <= 0):
        raise ValueError('Need >=15 samples and strictly increasing unique wavelengths.')
    groups = groups_for(wl, args.group_width)
    for ratio in args.ratios:
        removed_columns(np.arange(len(groups)), groups, ratio)
    versions = {p: package_version(p, 'sklearn' if p == 'scikit-learn' else p)
                for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'joblib']}
    if 'XGBoost' in args.models:
        versions['xgboost'] = package_version('xgboost')
    # Models/seeds/stage excluded so one run can be resumed with subsets.
    config = dict(schema=1, data_sha256=hash_file(args.data),
        sources={p: hash_file(ROOT / p) for p in ['tabular_band_core.py', 'tabular_band_runner.py', 'evaluation_protocol.py']},
        versions=versions, python=platform.python_version(), device=args.device, jobs=args.jobs,
        group_width=args.group_width, ratios=args.ratios, importance_repeats=args.importance_repeats,
        random_repeats=args.random_repeats, retrain_repeats=args.retrain_repeats, epochs=args.epochs,
        protocol='nested-holdout', precision='MLP FP32, TF32 disabled; estimators use native precision',
        importance='mean absolute prediction change with joint fit-row donors',
        fixed_perturbation='fit-mean replacement, per-outer-sample ranking',
        retraining='physical column deletion; ranking from inner only; fresh fit',
        mlp_pairing='shared full-width initial weights sliced to retained columns; shared batches/dropout RNG')
    # Record XGBoost version even when resuming a subset of an existing full run.
    if 'xgboost' not in versions:
        try:
            versions['xgboost'] = package_version('xgboost')
        except (ImportError, importlib.metadata.PackageNotFoundError):
            pass
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    per_model = len(args.seeds) * (1 + len(args.ratios) * (1 + args.retrain_repeats))
    print(f'{len(x)} samples, {len(wl)} bands, {len(groups)} groups; models={args.models}', flush=True)
    print(f'Full protocol: {per_model} fits/model; {per_model * len(args.models)} total including baselines.', flush=True)
    print(f'MLP: {args.epochs} epochs, inner-selected checkpoint, no early stop. Output: {args.output}', flush=True)
    if args.dry_run:
        for seed in args.seeds:
            parts = nested_holdout_split(len(x), seed)
            print(f'  seed {seed}: fit/inner/outer = {tuple(map(len, parts))}')
        return
    manifest = args.output / 'experiment.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        if previous['signature'] != signature:
            raise ValueError('Experiment config/source/data mismatch. Use a new --output; previous results are untouched.')
    elif args.stage == 'summarize':
        raise ValueError('No experiment manifest found to summarize.')
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError('Nonempty output without experiment manifest; choose a new directory.')
        json_write(manifest, dict(signature=signature, config=config))
    json_write(args.output / 'groups.json', [dict(group=j, bands=g.tolist(),
        wavelength_min=float(wl[g].min()), wavelength_max=float(wl[g].max())) for j, g in enumerate(groups)])
    if args.stage != 'summarize':
        for name in args.models:
            for seed in args.seeds:
                print(f'[{name} seed={seed}]', flush=True)
                fp32_mode(seed)
                fit, inner, outer = nested_holdout_split(len(x), seed)
                folder = args.output / name / f'seed{seed}'
                save_split_manifest(folder, seed, fit, inner, outer, args.data, selects_weights=name == 'MLP')
                model = cached_fit(folder, 'baseline', signature, name, seed, x, y, fit, inner,
                                   np.arange(x.shape[1]), args)
                prediction = model.predict(x[outer])
                csv_write(folder / 'baseline_performance.csv', [dict(model=name, seed=seed, **metrics(y[outer], prediction))])
                csv_write(folder / 'baseline_predictions.csv', [dict(sample_id=int(i), target=float(y[i]), prediction=float(p))
                                                              for i, p in zip(outer, prediction)])
                inner_scores, _ = run_explanation(folder, model, name, seed, x, y, fit, inner, outer, groups, args)
                if args.stage in ('all', 'retrain'):
                    if not (folder / 'retraining_complete.json').exists():
                        run_retraining(folder, model, name, seed, x, y, fit, inner, outer,
                                       groups, inner_scores, args, signature)
    summarize(args.output, args, groups, wl)
    print('Completed requested stage. See summary/status.json for coverage.', flush=True)


if __name__ == '__main__':
    main()
