"""Nine-model fixed-weight masking, ranked by INNER-validation grouped PFI.

Requires completed unified_pfi_runner.py runs and their exact checkpoints.
Never fits models. See UNIFIED_PERTURBATION_EXPERIMENT.md.
"""
import argparse
import hashlib
import math
from pathlib import Path
from release_paths import path_reference, resolve_reference, resolve_path, require_data

import joblib
import numpy as np
import pandas as pd
import torch

import unified_pfi_runner as pfi
from evaluation_protocol import nested_holdout_split, save_split_manifest
from tabular_band_core import FittedModel, fp32_mode, groups_for, metrics
from tabular_band_runner import csv_write, hash_file, json_write, package_version

ROOT = Path(__file__).resolve().parent


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def verify_marker(folder, filename='complete.json'):
    marker = pfi.read_json(folder / filename)
    for name, expected in marker['files'].items():
        path = folder / name
        if not path.exists() or hash_file(path) != expected:
            raise ValueError(f'Missing/modified completed result: {path}')
    return marker


def mark_complete(folder, filename, stamp, files):
    json_write(folder / filename, dict(stamp=stamp, files={f: hash_file(folder / f) for f in files}))


def completed(folder, filename, stamp):
    marker = folder / filename
    if not marker.exists():
        return False
    old = pfi.read_json(marker)
    if old['stamp'] != stamp:
        raise ValueError(f'Provenance changed: {marker}. Use a new output directory.')
    # Corrupt/missing artifacts must be visible rather than silently aggregated.
    verify_marker(folder, filename)
    return True


def shared_design(seed, n_groups, n_inner, pfi_repeats, random_repeats):
    # Independent of outer PFI random stream (71000), and independent of model.
    permutations = np.empty((n_groups, pfi_repeats, n_inner), dtype=np.int64)
    for group in range(n_groups):
        for repeat in range(pfi_repeats):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 72000, group, repeat]))
            permutations[group, repeat] = rng.permutation(n_inner)
    random_orders = np.stack([np.random.default_rng(np.random.SeedSequence(
        [seed, 73000, repeat])).permutation(n_groups) for repeat in range(random_repeats)])
    tie_order = np.random.default_rng(np.random.SeedSequence([seed, 74000])).permutation(n_groups)
    pfi.check_permutations(permutations, n_groups, pfi_repeats, n_inner)
    if not np.all(np.sort(random_orders, axis=1) == np.arange(n_groups)):
        raise ValueError('Invalid random group orders.')
    return permutations, random_orders, tie_order


def ranking_orders(scores, tie_order):
    """Signed PFI ordering; reproducible model-independent random tie breaking.

    Low is the reverse of top, so all-tied scores do not select the same prefix
    for both controls. Ties still provide no evidence of importance differences.
    """
    tie_rank = np.empty(len(scores), dtype=int)
    tie_rank[tie_order] = np.arange(len(scores))
    top = np.lexsort((tie_rank, -np.asarray(scores)))
    return top, top[::-1].copy()


def source_run(args, name, seed, source_signature):
    folder = args.pfi_root / name / f'seed{seed}'
    marker = verify_marker(folder)
    if marker['stamp']['signature'] != source_signature:
        raise ValueError(f'PFI run/manifest signature mismatch: {folder}')
    provenance = pfi.read_json(folder / 'provenance.json')
    checkpoint = resolve_reference(provenance['checkpoint'], relative_to=folder)
    provenance = dict(provenance, checkpoint=path_reference(checkpoint))
    checkpoint_hash = hash_file(checkpoint)
    if (checkpoint_hash != provenance['checkpoint_sha256']
            or checkpoint_hash != marker['stamp']['checkpoint_sha256']):
        raise ValueError(f'Checkpoint differs from the outer PFI model: {checkpoint}')
    return folder, marker, provenance, checkpoint


def load_exact_model(args, name, seed, x, y, wl, source_signature, data_hash):
    folder, marker, provenance, checkpoint = source_run(args, name, seed, source_signature)
    fit, inner, outer = nested_holdout_split(len(x), seed)
    np.testing.assert_array_equal(marker['stamp']['outer_ids'], outer)
    if name in pfi.DEEP_MODELS:
        predictor, _ = pfi.load_deep(checkpoint, name, seed, x, y, wl, data_hash, args.device)
    else:
        # Hash-verified checkpoint recorded by the completed outer-PFI run.
        saved = joblib.load(checkpoint)
        if saved['request']['model'] != name or saved['request']['seed'] != seed:
            raise ValueError('Tabular model/seed mismatch.')
        if provenance['kind'] == 'pfi_baseline' and saved['request']['signature'] != source_signature:
            raise ValueError('PFI baseline signature mismatch.')
        pfi.validate_tabular(saved['bundle'], name, seed, x, y, fit)
        predictor = FittedModel(saved['bundle'], args.device)
    with np.load(folder / 'predictions.npz') as saved:
        np.testing.assert_array_equal(saved['outer_ids'], outer)
        np.testing.assert_array_equal(saved['targets'], y[outer])
        baseline = saved['baseline_prediction'].copy()
    current = np.asarray(predictor.predict(x[outer]), dtype=np.float64)
    # Same weights/preprocessing/precision must reproduce the PFI baseline.
    np.testing.assert_allclose(current, baseline, rtol=1e-6, atol=1e-5,
                               err_msg='Baseline predictions differ from the outer PFI run.')
    provenance = dict(provenance, outer_pfi_directory=path_reference(folder),
        outer_pfi_complete_sha256=hash_file(folder / 'complete.json'),
        max_baseline_prediction_difference=float(np.max(np.abs(current - baseline))))
    return predictor, provenance


def inner_ranking(folder, predictor, x_inner, y_inner, inner_ids, groups, group_info,
                  permutations, tie_order, stamp, name, seed):
    ranking_stamp = dict(stamp, inner_permutation_sha256=array_hash(permutations),
                         tie_order_sha256=array_hash(tie_order), inner_ids=inner_ids.tolist())
    marker = 'inner_complete.json'
    if completed(folder, marker, ranking_stamp):
        with np.load(folder / 'inner_pfi.npz') as saved:
            return saved['top_order'].copy(), saved['low_order'].copy()
    # This function never receives outer reflectance or outer labels.
    base, predictions, mse, permuted_mse, scores = pfi.grouped_pfi(
        predictor, x_inner, y_inner, groups, permutations)
    mean_scores = scores.mean(axis=1)
    if not np.isfinite(mean_scores).all():
        raise ValueError('Nonfinite inner PFI scores.')
    top, low = ranking_orders(mean_scores, tie_order)
    rows = [dict(model=name, seed=seed, group=g, repeat=r, **group_info[g],
                 baseline_mse=mse, permuted_mse=float(permuted_mse[g, r]), pfi=float(scores[g, r]))
            for g in range(len(groups)) for r in range(scores.shape[1])]
    csv_write(folder / 'inner_pfi_repeats.csv', rows)
    inverse_top = np.argsort(top)
    csv_write(folder / 'inner_pfi_ranking.csv', [dict(model=name, seed=seed, group=g,
        **group_info[g], pfi_mean=float(mean_scores[g]), pfi_std=float(scores[g].std(ddof=1)),
        pfi_min=float(scores[g].min()), pfi_max=float(scores[g].max()), repeats=scores.shape[1],
        top_position=int(inverse_top[g]) + 1,
        tied_group_count=int(np.count_nonzero(mean_scores == mean_scores[g]))) for g in range(len(groups))])
    save_npz(folder / 'inner_pfi.npz', inner_ids=inner_ids, targets=y_inner,
        baseline_prediction=base, permuted_predictions=predictions, pfi=scores,
        top_order=top, low_order=low, tie_order=tie_order)
    json_write(folder / 'ranking_diagnostics.json', dict(ranking_split='inner',
        n_positive=int(np.count_nonzero(mean_scores > 0)), n_zero=int(np.count_nonzero(mean_scores == 0)),
        n_negative=int(np.count_nonzero(mean_scores < 0)), all_tied=bool(np.ptp(mean_scores) == 0),
        n_unique_scores=len(np.unique(mean_scores)),
        tie_rule='Model-independent seeded random ordering; low reverses top.',
        warning='A ranked list alone does not establish reliable importance, especially for near-zero or negative scores.'))
    mark_complete(folder, marker, ranking_stamp,
        ['inner_pfi_repeats.csv', 'inner_pfi_ranking.csv', 'inner_pfi.npz', 'ranking_diagnostics.json'])
    return top, low


def fixed_masking(predictor, raw, targets, outer_ids, fit_mean, groups, top, low,
                  random_orders, ratios, name, seed):
    baseline = np.asarray(predictor.predict(raw), dtype=np.float64)
    original = metrics(targets, baseline)
    baseline_mse = float(np.mean((baseline - targets) ** 2))
    rows, samples, selections = [], [], []
    strategies = [('top', 0, top), ('low', 0, low)]
    strategies += [('random', r, order) for r, order in enumerate(random_orders)]
    for strategy, repeat, order in strategies:
        for ratio in [0.] + list(ratios):
            k = int(math.ceil(ratio * len(groups)))
            selected_groups = order[:k]
            bands = np.sort(np.concatenate([groups[g] for g in selected_groups])) if k else np.array([], dtype=int)
            changed = raw.copy()
            changed[:, bands] = fit_mean[bands]
            prediction = np.asarray(predictor.predict(changed), dtype=np.float64) if k else baseline
            if prediction.shape != baseline.shape:
                raise ValueError('Perturbed prediction shape mismatch.')
            m = metrics(targets, prediction)
            mse = float(np.mean((prediction - targets) ** 2))
            common = dict(model=name, seed=seed, strategy=strategy, repeat=repeat, ratio=ratio,
                          n_groups=k, actual_group_fraction=k / len(groups),
                          n_bands=len(bands), actual_band_fraction=len(bands) / raw.shape[1])
            rows.append(dict(**common, **m, mse=mse, baseline_mse=baseline_mse,
                baseline_r2=original['r2'], baseline_rmse=original['rmse'], baseline_mae=original['mae'],
                delta_mse=mse - baseline_mse, delta_r2=original['r2'] - m['r2'],
                delta_rmse=m['rmse'] - original['rmse'], delta_mae=m['mae'] - original['mae'],
                prediction_change=float(np.mean(np.abs(prediction - baseline)))))
            samples.extend(dict(**common, sample_id=int(sid), target=float(targets[i]),
                original=float(baseline[i]), perturbed=float(prediction[i])) for i, sid in enumerate(outer_ids))
            selections.append(dict(**common, ranking_split='inner',
                selected_groups=selected_groups.tolist(), masked_bands=bands.tolist()))
    return rows, samples, selections, baseline


def run_one(args, name, seed, predictor, provenance, x, y, fit, inner, outer, groups,
            group_info, permutations, random_orders, tie_order, signature):
    folder = args.output / name / f'seed{seed}'
    stamp = dict(signature=signature, checkpoint_sha256=provenance['checkpoint_sha256'],
        outer_pfi_complete_sha256=provenance['outer_pfi_complete_sha256'],
        fit_ids=fit.tolist(), inner_ids=inner.tolist(), outer_ids=outer.tolist(),
        random_order_sha256=array_hash(random_orders))
    if completed(folder, 'complete.json', stamp):
        print(f'Reuse fixed perturbation: {name}/seed{seed}', flush=True)
        return
    save_split_manifest(folder, seed, fit, inner, outer, args.data,
                        selects_weights=name not in ('PLSR', 'RF', 'SVR', 'XGBoost'))
    top, low = inner_ranking(folder, predictor, x[inner], y[inner], inner, groups, group_info,
                             permutations, tie_order, stamp, name, seed)
    # Compute one shared raw-space fit mean in float64 for all nine models.
    fit_mean = x[fit].mean(axis=0)
    rows, samples, selections, baseline = fixed_masking(predictor, x[outer], y[outer], outer,
        fit_mean, groups, top, low, random_orders, args.ratios, name, seed)
    csv_write(folder / 'perturbation_runs.csv', rows)
    csv_write(folder / 'perturbation_samples.csv', samples)
    json_write(folder / 'mask_selections.json', selections)
    json_write(folder / 'provenance.json', pfi.provenance_for_folder(provenance, folder))
    csv_write(folder / 'baseline.csv', [dict(model=name, seed=seed,
        mse=float(np.mean((baseline - y[outer]) ** 2)), **metrics(y[outer], baseline))])
    save_npz(folder / 'masking_reference.npz', fit_ids=fit, fit_mean=fit_mean, outer_ids=outer,
             targets=y[outer], baseline_prediction=baseline)
    files = ['protocol_seed' + str(seed) + '.json', 'inner_complete.json', 'inner_pfi_repeats.csv',
        'inner_pfi_ranking.csv', 'inner_pfi.npz', 'ranking_diagnostics.json', 'perturbation_runs.csv',
        'perturbation_samples.csv', 'mask_selections.json', 'provenance.json', 'baseline.csv', 'masking_reference.npz']
    mark_complete(folder, 'complete.json', stamp, files)
    print(f'Completed fixed perturbation: {name}/seed{seed}', flush=True)


def aggregate(frame, keys, values):
    result = frame.groupby(keys)[values].agg(['mean', 'std', 'count'])
    result.columns = ['_'.join(column) for column in result.columns]
    return result.reset_index()


def plot_curves(frame, names, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    columns = min(3, len(names))
    nrows = int(math.ceil(len(names) / columns))
    fig, axes = plt.subplots(nrows, columns, figsize=(5 * columns, 3.8 * nrows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        for strategy in ('top', 'low', 'random'):
            part = frame[(frame.model == name) & (frame.strategy == strategy)].sort_values('actual_group_fraction')
            if part.empty:
                continue
            ax.errorbar(part.actual_group_fraction * 100, part.delta_rmse_mean,
                yerr=part.delta_rmse_std.fillna(0), marker='o', capsize=3, label=strategy)
        ax.axhline(0, color='gray', linewidth=.6)
        ax.set(title=name, xlabel='Masked wavelength groups (%)', ylabel='Delta RMSE')
        ax.legend()
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def summarize(args, signature):
    parts, baselines, rankings, missing = [], [], [], []
    for name in args.models:
        for seed in args.seeds:
            folder = args.output / name / f'seed{seed}'
            if not (folder / 'complete.json').exists():
                missing.append(dict(model=name, seed=seed))
                continue
            marker = verify_marker(folder)
            if marker['stamp']['signature'] != signature:
                raise ValueError(f'Mismatched completed run: {folder}')
            parts.append(pd.read_csv(folder / 'perturbation_runs.csv'))
            baselines.append(pd.read_csv(folder / 'baseline.csv'))
            rankings.append(pd.read_csv(folder / 'inner_pfi_ranking.csv'))
    scope = pfi.digest(dict(models=sorted(args.models), seeds=sorted(args.seeds)))[:12]
    full = set(args.models) == set(pfi.MODELS) and set(args.seeds) == set(pfi.SEEDS)
    out = args.output / ('summary' if full else 'summary_subset_' + scope)
    if parts:
        runs = pd.concat(parts, ignore_index=True)
        csv_write(out / 'perturbation_runs.csv', runs)
        values = ['mse', 'r2', 'rmse', 'mae', 'delta_mse', 'delta_r2', 'delta_rmse', 'delta_mae', 'prediction_change']
        keys = ['model', 'seed', 'strategy', 'ratio', 'n_groups', 'actual_group_fraction']
        per_seed = runs.groupby(keys)[values].mean().reset_index()
        csv_write(out / 'perturbation_per_seed.csv', per_seed)
        summary = aggregate(per_seed, ['model', 'strategy', 'ratio', 'n_groups', 'actual_group_fraction'], values)
        summary['complete_requested_seeds'] = summary.delta_rmse_count == len(args.seeds)
        csv_write(out / 'perturbation_summary.csv', summary)
        random = runs[runs.strategy == 'random']
        spread = random.groupby(['model', 'seed', 'ratio'])[values].agg(['mean', 'std', 'min', 'max', 'count'])
        spread.columns = ['_'.join(c) for c in spread.columns]
        csv_write(out / 'random_within_seed.csv', spread.reset_index())
        merge_keys = ['model', 'seed', 'ratio', 'n_groups', 'actual_group_fraction']
        top = per_seed[per_seed.strategy == 'top'].drop(columns='strategy')
        control = per_seed[per_seed.strategy == 'random'].drop(columns='strategy')
        paired = top.merge(control, on=merge_keys, suffixes=('_top', '_random'), validate='one_to_one')
        differences = []
        for metric in ('delta_mse', 'delta_r2', 'delta_rmse', 'delta_mae'):
            key = metric + '_top_minus_random'
            paired[key] = paired[metric + '_top'] - paired[metric + '_random']
            differences.append(key)
        csv_write(out / 'top_minus_random_per_seed.csv', paired)
        csv_write(out / 'top_minus_random_summary.csv', aggregate(paired,
            ['model', 'ratio', 'n_groups', 'actual_group_fraction'], differences))
        baseline = pd.concat(baselines, ignore_index=True)
        csv_write(out / 'baseline_per_seed.csv', baseline)
        csv_write(out / 'baseline_summary.csv', aggregate(baseline, ['model'], ['mse', 'r2', 'rmse', 'mae']))
        csv_write(out / 'inner_pfi_rankings.csv', pd.concat(rankings, ignore_index=True))
        plot_curves(summary, [name for name in args.models if name in set(runs.model)], out / 'fixed_masking_delta_rmse.png')
    json_write(out / 'status.json', dict(signature=signature, models=args.models, seeds=args.seeds,
        expected_runs=len(args.models) * len(args.seeds), complete_runs=len(parts), missing_runs=missing,
        all_requested_complete=not missing,
        note='Average random repeats within seed before across-seed mean/sample SD; no independence or significance claim.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data' / 'AGB.csv')
    p.add_argument('--pfi-root', type=Path, default=ROOT / 'nested_holdout_outputs' / 'unified_pfi_v1')
    p.add_argument('--output', type=Path, default=ROOT / 'nested_holdout_outputs' / 'unified_perturbation_v1')
    p.add_argument('--models', nargs='+', choices=pfi.MODELS, default=list(pfi.MODELS))
    p.add_argument('--seeds', nargs='+', type=int, default=list(pfi.SEEDS))
    p.add_argument('--group-width', type=float, default=10.)
    p.add_argument('--pfi-repeats', type=int, default=20)
    p.add_argument('--random-repeats', type=int, default=20)
    p.add_argument('--ratios', nargs='+', type=float, default=[.05, .1, .2, .3])
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=4)
    exclusive = p.add_mutually_exclusive_group()
    exclusive.add_argument('--plan-only', action='store_true', help='Read configuration and checkpoint paths; no inference, training, or output writes.')
    exclusive.add_argument('--summarize-only', action='store_true', help='Only aggregate completed perturbation outputs.')
    args = p.parse_args()
    if min(args.pfi_repeats, args.random_repeats) < 2 or args.threads < 1:
        p.error('Require >=2 repeats and positive thread count.')
    if not np.isfinite(args.group_width) or args.group_width <= 0:
        p.error('Group width must be finite and positive.')
    if any(not 0 < r < 1 for r in args.ratios) or len(set(args.ratios)) != len(args.ratios):
        p.error('Ratios must be unique, strictly between zero and one; zero is added automatically.')
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        p.error('Duplicate models or seeds.')
    if any(s < 0 or s > 2**32 - 10001 for s in args.seeds):
        p.error('Seed outside nested split range.')
    args.ratios = sorted(args.ratios)
    args.data = resolve_path(args.data)
    args.pfi_root, args.output = resolve_path(args.pfi_root), resolve_path(args.output)
    if args.output == args.pfi_root or args.output in args.pfi_root.parents or args.pfi_root in args.output.parents:
        p.error('Perturbation output must be separate from the source PFI directory.')
    args.data = require_data(args.data)
    data = pd.read_csv(args.data)
    x, y = data.iloc[:, :-1].to_numpy(float), data.iloc[:, -1].to_numpy(float)
    wl = np.asarray(data.columns[:-1], dtype=float)
    if len(x) < 15 or len(wl) < 2 or not all(np.isfinite(a).all() for a in (x, y, wl)) or np.any(np.diff(wl) <= 0):
        p.error('Require finite numeric data, >=15 samples and ordered unique wavelength columns.')
    groups = groups_for(wl, args.group_width)
    if any(math.ceil(r * len(groups)) >= len(groups) for r in args.ratios):
        p.error('Masking must leave at least one group unchanged.')
    group_info = [dict(wavelength_min=float(wl[g].min()), wavelength_max=float(wl[g].max()),
                       wavelength_center=float(wl[g].mean()), n_bands=len(g)) for g in groups]
    manifest_path = args.pfi_root / 'experiment.json'
    if not manifest_path.exists():
        raise FileNotFoundError('Run unified_pfi_runner.py first; no source PFI experiment manifest found.')
    source = pfi.read_json(manifest_path)
    if source['signature'] != pfi.digest(source['config']):
        raise ValueError('Source PFI manifest signature mismatch.')
    data_hash = hash_file(args.data)
    if (source['config']['data_sha256'] != data_hash or source['config']['group_width'] != args.group_width
            or source['config']['protocol'] != 'nested-holdout'):
        raise ValueError('Data/grouping/protocol differs from source PFI.')
    # Preserve current implementations; a changed dependency needs a new reviewed batch.
    for filename, expected in source['config']['sources'].items():
        if not pfi.compatible_source(filename, expected):
            raise ValueError(f'Source PFI implementation changed: {filename}')
    module_names = {'scikit-learn': 'sklearn', 'torch-geometric': 'torch_geometric'}
    versions = {name: package_version(name, module_names.get(name, name)) for name in source['config']['versions']}
    if versions != source['config']['versions']:
        raise ValueError('Dependency versions differ from source PFI.')
    config = dict(schema=1, source_pfi_signature=source['signature'], source_pfi_manifest_sha256=hash_file(manifest_path),
        data_sha256=data_hash, source_sha256=hash_file(Path(__file__).resolve()), versions=versions,
        device=args.device, threads=args.threads, group_width=args.group_width, pfi_repeats=args.pfi_repeats,
        random_repeats=args.random_repeats, ratios=args.ratios, ranking_split='inner', evaluation_split='outer',
        ranking_score='signed mean PFI: permuted MSE minus baseline MSE',
        replacement='per-band fit-only mean in raw space; all weights/scalers/graphs fixed',
        precision='Neural FP32, TF32 disabled; native estimator precision; float64 metrics',
        rng=dict(inner_pfi=72000, random_group_orders=73000, tie_breaking=74000))
    signature = pfi.digest(config)
    missing = []
    if not args.summarize_only:
        for name in args.models:
            for seed in args.seeds:
                run = args.pfi_root / name / f'seed{seed}'
                if not (run / 'complete.json').exists() or not (run / 'provenance.json').exists():
                    missing.append(f'{name}/seed{seed}: complete outer PFI run')
                elif not resolve_reference(pfi.read_json(run / 'provenance.json')['checkpoint'], relative_to=run).exists():
                    missing.append(f'{name}/seed{seed}: recorded checkpoint')
    print(f'{len(args.models) * len(args.seeds)} model-seed runs, {len(groups)} groups; inner PFI repeats={args.pfi_repeats}.', flush=True)
    print(f'Top/Low + {args.random_repeats} shared random orders; ratios={[0.] + args.ratios}.', flush=True)
    print(f'No training. Missing prerequisites: {missing}\nOutput: {args.output}', flush=True)
    if args.plan_only:
        return
    if missing:
        raise FileNotFoundError('Complete the listed outer PFI runs/checkpoints first; this script never trains.')
    manifest = args.output / 'experiment.json'
    if manifest.exists():
        if pfi.read_json(manifest) != dict(signature=signature, config=config):
            raise ValueError('Experiment config/data/source changed; choose a new --output.')
    else:
        if args.summarize_only:
            raise FileNotFoundError('No perturbation experiment to summarize.')
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError('Nonempty output without perturbation manifest.')
        json_write(manifest, dict(signature=signature, config=config))
    if args.summarize_only:
        summarize(args, signature)
        return
    torch.set_num_threads(args.threads)
    json_write(args.output / 'groups.json', [dict(group=i, bands=g.tolist(), **group_info[i]) for i, g in enumerate(groups)])
    for seed in args.seeds:
        fit, inner, outer = nested_holdout_split(len(x), seed)
        permutations, random_orders, tie_order = shared_design(seed, len(groups), len(inner),
                                                               args.pfi_repeats, args.random_repeats)
        shared = args.output / 'shared_design' / f'seed{seed}.npz'
        design = dict(fit_ids=fit, inner_ids=inner, outer_ids=outer, inner_permutations=permutations,
                      inner_source_sample_ids=inner[permutations], random_group_orders=random_orders, tie_order=tie_order)
        if shared.exists():
            with np.load(shared) as saved:
                for key, value in design.items():
                    np.testing.assert_array_equal(saved[key], value)
        else:
            save_npz(shared, **design)
        for name in args.models:
            fp32_mode(seed)
            predictor, provenance = load_exact_model(args, name, seed, x, y, wl, source['signature'], data_hash)
            run_one(args, name, seed, predictor, provenance, x, y, fit, inner, outer, groups,
                    group_info, permutations, random_orders, tie_order, signature)
            del predictor
    summarize(args, signature)
    print('Completed requested fixed-model perturbations; see summary/status.json or scoped subset summary.', flush=True)


if __name__ == '__main__':
    main()
