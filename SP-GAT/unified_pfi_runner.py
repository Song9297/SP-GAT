"""Nine-model grouped PFI on outer holdouts. No work occurs on import.

See UNIFIED_PFI_EXPERIMENT.md. Default is checkpoint-only; fitting missing
tabular baselines requires --fit-missing. This runner never computes IG,
selects features using outer labels, or trains band-deletion variants.
"""
import argparse
import hashlib
import json
from pathlib import Path
from release_paths import path_reference, resolve_reference, resolve_path, require_data

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import explainability_runner as deep
from evaluation_protocol import nested_holdout_split, save_split_manifest
from tabular_band_core import MODELS as TABULAR_MODELS
from tabular_band_core import FittedModel, fit_model, fp32_mode, groups_for, metrics
from tabular_band_runner import csv_write, hash_file, json_write, package_version

ROOT = Path(__file__).resolve().parent
DEEP_MODELS = ('ResNet', 'Transformer', 'GAT', 'PP-SGAT')
MODELS = tuple(TABULAR_MODELS) + DEEP_MODELS
SEEDS = (42, 123, 456, 789, 2026)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def compatible_source(filename, recorded_hash):
    """Accept only the exact source used to produce the current experiment."""
    return hash_file(ROOT / filename) == recorded_hash


def provenance_for_folder(provenance, folder):
    """Rebase in-memory project-relative references for a provenance.json file.

    Persisted paths are relative to that file's parent directory, so relocating
    the complete project and output tree preserves checkpoint discovery.
    """
    result = dict(provenance)
    for key in ('checkpoint', 'outer_pfi_directory'):
        if key in result:
            result[key] = path_reference(resolve_reference(result[key]), relative_to=folder)
    result['path_base'] = 'provenance_directory'
    return result


def plot_top10(args, signature=None):
    """Verify stored runs, recompute seed means, and export nine standalone plots.

    Plot-only mode deliberately does not load checkpoints or mutate the original
    manifest/summary. Its own receipt records the rendering code and data hashes.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    manifest = read_json(args.output / 'experiment.json')
    if manifest['signature'] != digest(manifest['config']):
        raise ValueError('Invalid source experiment manifest.')
    if signature is not None and manifest['signature'] != signature:
        raise ValueError('Plot signature mismatch.')
    tables, input_hashes = [], {}
    for name in args.models:
        for seed in args.seeds:
            folder = args.output / name / f'seed{seed}'
            marker = read_json(folder / 'complete.json')
            if marker['stamp']['signature'] != manifest['signature']:
                raise ValueError(f'Plot input signature mismatch: {folder}')
            for filename, expected in marker['files'].items():
                if hash_file(folder / filename) != expected:
                    raise ValueError(f'Changed plot source: {folder / filename}')
            path = folder / 'pfi_per_seed.csv'
            table = pd.read_csv(path)
            if not (table.model.eq(name).all() and table.seed.eq(seed).all()) or table.group.duplicated().any():
                raise ValueError(f'Unexpected plot source keys: {path}')
            input_hashes[f'{name}/seed{seed}'] = hash_file(path)
            tables.append(table)
    frame = pd.concat(tables, ignore_index=True)
    summary = frame.groupby(['model', 'group']).pfi_mean.agg(['mean', 'std', 'count']).reset_index()
    if not summary['count'].eq(len(args.seeds)).all() or len(args.seeds) < 2:
        raise ValueError('Top10 error bars require >=2 complete seeds for every group.')
    bounds = frame[['group', 'wavelength_min', 'wavelength_max']].drop_duplicates()
    summary = summary.merge(bounds, on='group', validate='many_to_one')
    full = set(args.models) == set(MODELS) and set(args.seeds) == set(SEEDS)
    scope = digest(dict(models=sorted(args.models), seeds=sorted(args.seeds)))[:12]
    output = args.output / ('summary' if full else 'summary_subset_' + scope) / 'top10_plots'
    output.mkdir(parents=True, exist_ok=True)
    selected, files = [], []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        for name in args.models:
            part = summary[summary.model == name].sort_values(['mean', 'group'], ascending=[False, True]).head(10).copy()
            part['rank'] = np.arange(1, len(part) + 1)
            selected.append(part)
            y = np.arange(len(part))
            mean, sd = part['mean'].to_numpy(), part['std'].to_numpy()
            labels = [f'{r.wavelength_min:.1f}\u2013{r.wavelength_max:.1f}' for r in part.itertuples()]
            fig, ax = plt.subplots(figsize=(9.5, 6.5))
            fig.subplots_adjust(left=.23, right=.96, top=.84, bottom=.24)
            ax.barh(y, mean, xerr=sd, height=.62, color='#347A9C',
                    error_kw={'ecolor': '#38434E', 'elinewidth': 1.1, 'capsize': 3})
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.axvline(0, color='#4C5560', lw=.8)
            ax.set_axisbelow(True)
            ax.xaxis.grid(True, color='#E1E5EA', linewidth=.7)
            ax.set_ylabel('Wavelength group (nm)')
            ax.set_xlabel('Mean PFI: increase in MSE [(g/m²)²]', labelpad=10)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, 4))
            ax.xaxis.set_major_formatter(formatter)
            ax.margins(x=.12)
            display = '1D-ResNet' if name == 'ResNet' else name
            fig.suptitle(f'{display} | Top 10 wavelength groups', x=.23, ha='left', y=.955,
                         fontsize=17, fontweight='bold')
            fig.text(.23, .898, f'Ranked by mean signed PFI across {len(args.seeds)} seeds', color='#4C5560')
            fig.text(.23, .035, f'Error bars: ±1 sample SD across seeds (ddof=1).\n'
                     f'{manifest["config"]["repeats"]} permutations/group/seed; axis scale varies by model.',
                     fontsize=9, color='#4C5560', va='bottom')
            for extension in ('png', 'pdf'):
                path = output / f'{name}_top10_pfi.{extension}'
                fig.savefig(path, dpi=300, facecolor='white')
                files.append(dict(file=path.name, sha256=hash_file(path)))
            plt.close(fig)
    csv_write(output / 'top10_plot_data.csv', pd.concat(selected, ignore_index=True).rename(
        columns={'mean': 'pfi_mean', 'std': 'pfi_std_between_seeds', 'count': 'n_seeds'}))
    json_write(output / 'plot_receipt.json', dict(source_signature=manifest['signature'],
        source_manifest_sha256=hash_file(args.output / 'experiment.json'),
        renderer_sha256=hash_file(Path(__file__).resolve()), seed_tables=input_hashes,
        models=args.models, seeds=args.seeds, files=files,
        sort='Descending signed mean PFI; tie broken by group id',
        error_bars='Sample SD of seed means, ddof=1; not a confidence interval'))
    print(f'Created {len(args.models)} Top10 plots (PNG + PDF): {output}', flush=True)


def shared_permutations(seed, n_groups, repeats, n_outer):
    """[group, repeat, destination position] -> source position, no replacement.

    Model-independent RNG streams also preserve existing repetitions when a
    future experiment increases repeats. Fixed points are allowed in standard PFI.
    """
    orders = np.empty((n_groups, repeats, n_outer), dtype=np.int64)
    for group in range(n_groups):
        for repeat in range(repeats):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 71000, group, repeat]))
            orders[group, repeat] = rng.permutation(n_outer)
    return orders


def check_permutations(orders, n_groups, repeats, n_outer):
    if orders.shape != (n_groups, repeats, n_outer):
        raise ValueError('Permutation shape mismatch.')
    if not np.all(np.sort(orders, axis=-1) == np.arange(n_outer)):
        raise ValueError('Each permutation must contain every outer position exactly once.')


def grouped_pfi(predictor, raw, targets, groups, orders):
    """Return original predictions, permuted predictions, and signed MSE increase.

    All groups share one row order across their member wavelengths. Targets stay
    attached to the destination row. Scalers/graphs/weights stay fixed.
    """
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    baseline = np.asarray(predictor.predict(raw), dtype=np.float64).reshape(-1)
    if baseline.shape != targets.shape or not np.isfinite(baseline).all():
        raise ValueError('Invalid baseline predictions.')
    baseline_mse = float(np.mean((baseline - targets) ** 2))
    predictions = np.empty((len(groups), orders.shape[1], len(raw)), dtype=np.float64)
    for group_id, bands in enumerate(groups):
        for repeat, permutation in enumerate(orders[group_id]):
            changed = raw.copy()
            changed[:, bands] = raw[permutation][:, bands]
            pred = np.asarray(predictor.predict(changed), dtype=np.float64).reshape(-1)
            if pred.shape != targets.shape or not np.isfinite(pred).all():
                raise ValueError(f'Invalid predictions for group={group_id}, repeat={repeat}.')
            predictions[group_id, repeat] = pred
    permuted_mse = np.mean((predictions - targets[None, None, :]) ** 2, axis=-1)
    # Do not take absolute values, clip negatives, or normalize by baseline MSE.
    return baseline, predictions, baseline_mse, permuted_mse, permuted_mse - baseline_mse


def deep_source_signature():
    return deep.digest(dict(core_source=deep.source_hash(), quality_source=hash_file(ROOT / 'explanation_quality.py')))


def load_deep(path, name, seed, x, y, wl, data_hash, device):
    """Validate the existing v2 checkpoint; recompute predictions in FP32."""
    cp = deep.load_checkpoint(path, device='cpu')
    config = cp['config']
    expected = dict(model=name, seed=seed, mode='all', graph_repeat=0,
        graph_seed=int(np.random.SeedSequence([seed, 1729, 0]).generate_state(1)[0]),
        epochs=300, protocol='nested-holdout', inner_ratio=.2, outer_ratio=.2,
        lr=.001, weight_decay=.0001, batch_size=32, swaps_per_edge=5,
        data_hash=data_hash, source_hash=deep_source_signature(),
        torch_version=torch.__version__, device=config.get('device'))
    if config != expected or cp['signature'] != deep.digest(config):
        raise ValueError(f'Incompatible deep checkpoint configuration/source: {path}')
    fit, inner, outer = nested_holdout_split(len(x), seed)
    for key, actual in [('fit_idx', fit), ('select_idx', inner), ('eval_idx', outer)]:
        np.testing.assert_array_equal(cp[key], actual, err_msg=f'{path}: {key} mismatch')
    np.testing.assert_array_equal(cp['wavelengths'], wl)
    # Existing deep training reads reflectance AND targets in float32.
    xf, yf = x.astype(np.float32), y.astype(np.float32)
    expected_stats = dict(x_mean=xf[fit].mean(0), x_std=xf[fit].std(0) + 1e-8,
                          y_mean=float(yf[fit].mean()), y_std=float(yf[fit].std()) + 1e-8)
    for key, value in expected_stats.items():
        np.testing.assert_allclose(cp['stats'][key], value, rtol=1e-6, atol=1e-8,
                                   err_msg=f'{path}: fit-only statistics mismatch for {key}')
    # Assert stored graph is the unmodified all-prior / chain graph.
    ei, ea, _ = deep.graph_for(name, wl, 'all', expected['graph_seed'], 5)
    for key, value in [('edge_index', ei), ('edge_attr', ea)]:
        if value is None:
            if cp[key] is not None:
                raise ValueError(f'{path}: unexpected {key}')
        elif not torch.equal(cp[key].cpu(), value.cpu()):
            raise ValueError(f'{path}: graph mismatch for {key}')
    return deep.from_checkpoint(cp, device), dict(kind='existing_deep_v2',
        checkpoint=path_reference(path), checkpoint_sha256=hash_file(path),
        checkpoint_signature=cp['signature'], training_config=config, best_epoch=cp['best_epoch'])


def validate_tabular(bundle, name, seed, x, y, fit):
    if bundle['name'] != name or bundle['seed'] != seed:
        raise ValueError('Tabular checkpoint model/seed mismatch.')
    np.testing.assert_array_equal(bundle['columns'], np.arange(x.shape[1]))
    dtype = np.float32 if name == 'MLP' else np.float64
    xf, yf = x.astype(dtype), y.astype(dtype)
    expected = StandardScaler().fit(xf[fit])
    for key in ('mean_', 'scale_', 'var_'):
        np.testing.assert_allclose(getattr(bundle['scaler'], key), getattr(expected, key), rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose([bundle['y_mean'], bundle['y_std']],
        [float(yf[fit].mean()), float(yf[fit].std()) + 1e-8], rtol=1e-7, atol=1e-10)
    if name == 'MLP' and len(bundle['history']) != 300:
        raise ValueError('MLP checkpoint must have completed 300 epochs.')


def load_tabular(args, name, seed, x, y, data_hash, signature):
    fit, inner, outer = nested_holdout_split(len(x), seed)
    local = args.output / 'baselines' / name / f'seed{seed}.joblib'
    external = args.tabular_checkpoint_root / name / f'seed{seed}' / 'checkpoints' / 'baseline.joblib'
    request = dict(signature=signature, model=name, seed=seed)
    if local.exists():
        saved = joblib.load(local)
        if saved['request'] != request:
            raise ValueError(f'Local baseline provenance mismatch: {local}')
        bundle, path, kind = saved['bundle'], local, 'pfi_baseline'
    elif external.exists():
        manifest = read_json(args.tabular_checkpoint_root / 'experiment.json')
        config = manifest['config']
        if manifest['signature'] != digest(config) or config['data_sha256'] != data_hash:
            raise ValueError('External tabular manifest/data mismatch.')
        if config['epochs'] != 300 or config['protocol'] != 'nested-holdout':
            raise ValueError('External tabular training protocol mismatch.')
        for filename in ('tabular_band_core.py', 'evaluation_protocol.py'):
            if config['sources'].get(filename) != hash_file(ROOT / filename):
                raise ValueError(f'External tabular source mismatch: {filename}')
        for package, version in config['versions'].items():
            if package_version(package, 'sklearn' if package == 'scikit-learn' else package) != version:
                raise ValueError(f'External tabular package mismatch: {package}')
        saved = joblib.load(external)
        if saved['request'] != dict(signature=manifest['signature'], model=name, seed=seed,
                                     columns=list(range(x.shape[1]))):
            raise ValueError('External tabular checkpoint request mismatch.')
        split = read_json(external.parent.parent / f'protocol_seed{seed}.json')
        for key, actual in [('fit_idx', fit), ('select_idx', inner), ('eval_idx', outer)]:
            np.testing.assert_array_equal(split[key], actual)
        if split['data_sha256'] != data_hash:
            raise ValueError('External tabular split data fingerprint mismatch.')
        bundle, path, kind = saved['bundle'], external, 'existing_tabular_baseline'
    else:
        if not args.fit_missing:
            raise FileNotFoundError(f'Missing {name}/seed{seed} baseline. Use --fit-missing when ready to fit.')
        print(f'Fit missing baseline: {name}/seed{seed}', flush=True)
        bundle = fit_model(name, seed, x, y, fit, inner, np.arange(x.shape[1]),
                           epochs=300, device=args.device, jobs=args.jobs)
        local.parent.mkdir(parents=True, exist_ok=True)
        temporary = local.with_suffix('.joblib.tmp')
        joblib.dump(dict(request=request, bundle=bundle), temporary)
        temporary.replace(local)
        path, kind = local, 'pfi_baseline'
    validate_tabular(bundle, name, seed, x, y, fit)
    provenance = dict(kind=kind, checkpoint=path_reference(path), checkpoint_sha256=hash_file(path),
        best_epoch=bundle['best_epoch'], estimator_params=bundle.get('estimator_params'),
        batch_order_sha256=bundle.get('batch_order_sha256'),
        initial_state_sha256=bundle.get('initial_state_sha256'))
    return FittedModel(bundle, args.device), provenance


def evaluate_run(args, name, seed, predictor, provenance, x, y, outer, groups, info, orders, signature):
    folder = args.output / name / f'seed{seed}'
    stamp = dict(signature=signature, checkpoint_sha256=provenance['checkpoint_sha256'],
                 permutation_sha256=hashlib.sha256(orders.tobytes()).hexdigest(), outer_ids=outer.tolist())
    marker = folder / 'complete.json'
    required = ['pfi_repeats.csv', 'pfi_per_seed.csv', 'predictions.npz', 'baseline.csv', 'provenance.json']
    if marker.exists():
        previous = read_json(marker)
        if previous['stamp'] != stamp:
            raise ValueError(f'Completed PFI run has different provenance: {folder}; use a new output.')
        if all((folder / filename).exists() and hash_file(folder / filename) == previous['files'][filename]
               for filename in required):
            print(f'Reuse completed PFI: {name}/seed{seed}', flush=True)
            return
        marker.unlink()  # incomplete/damaged output is recomputed from the same checkpoint
    baseline, prediction, mse, permuted_mse, importance = grouped_pfi(predictor, x[outer], y[outer], groups, orders)
    rows = [dict(model=name, seed=seed, group=g, repeat=r, **info[g], n_outer=len(outer),
                 baseline_mse=mse, permuted_mse=float(permuted_mse[g, r]), pfi=float(importance[g, r]))
            for g in range(len(groups)) for r in range(args.repeats)]
    frame = pd.DataFrame(rows)
    per_seed = frame.groupby(['model', 'seed', 'group']).pfi.agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    per_seed = per_seed.rename(columns={k: 'pfi_' + k for k in ('mean', 'std', 'min', 'max', 'count')})
    per_seed['baseline_mse'] = mse
    # Average rank for ties avoids interpreting tied zeros as ordered evidence.
    per_seed['rank'] = per_seed.pfi_mean.rank(ascending=False, method='average')
    per_seed = per_seed.merge(pd.DataFrame([dict(group=i, **g) for i, g in enumerate(info)]), on='group')
    csv_write(folder / 'pfi_repeats.csv', frame)
    csv_write(folder / 'pfi_per_seed.csv', per_seed)
    csv_write(folder / 'baseline.csv', [dict(model=name, seed=seed, n_outer=len(outer), mse=mse,
                                            **metrics(y[outer], baseline))])
    json_write(folder / 'provenance.json', provenance_for_folder(provenance, folder))
    temporary = folder / 'predictions.tmp.npz'
    np.savez_compressed(temporary, outer_ids=outer, targets=y[outer], baseline_prediction=baseline,
                        permuted_predictions=prediction, pfi=importance)
    temporary.replace(folder / 'predictions.npz')
    json_write(marker, dict(stamp=stamp, files={filename: hash_file(folder / filename) for filename in required}))
    print(f'Completed PFI: {name}/seed{seed}, {len(groups)} groups x {args.repeats} repeats', flush=True)


def summarize(args, signature):
    repeats, seeds, baselines, missing = [], [], [], []
    for name in args.models:
        for seed in args.seeds:
            folder = args.output / name / f'seed{seed}'
            if not (folder / 'complete.json').exists():
                missing.append(dict(model=name, seed=seed))
                continue
            marker = read_json(folder / 'complete.json')
            if marker['stamp']['signature'] != signature:
                raise ValueError(f'Mismatched completed run: {folder}')
            for filename, expected in marker['files'].items():
                if not (folder / filename).exists() or hash_file(folder / filename) != expected:
                    raise ValueError(f'Incomplete or changed result: {folder / filename}')
            repeats.append(pd.read_csv(folder / 'pfi_repeats.csv'))
            seeds.append(pd.read_csv(folder / 'pfi_per_seed.csv'))
            baselines.append(pd.read_csv(folder / 'baseline.csv'))
    # Isolate subset summaries so they cannot replace the full nine-model tables.
    scope = digest(dict(models=sorted(args.models), seeds=sorted(args.seeds)))[:12]
    out = args.output / ('summary' if set(args.models) == set(MODELS) and set(args.seeds) == set(SEEDS)
                         else 'summary_subset_' + scope)
    if seeds:
        all_seeds = pd.concat(seeds, ignore_index=True)
        csv_write(out / 'pfi_repeats.csv', pd.concat(repeats, ignore_index=True))
        csv_write(out / 'pfi_per_seed.csv', all_seeds)
        # Each seed contributes ONE mean, not 20 independent observations.
        aggregate = all_seeds.groupby(['model', 'group']).pfi_mean.agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
        aggregate = aggregate.rename(columns={'mean': 'pfi_mean', 'std': 'pfi_std_between_seeds',
                                               'min': 'pfi_min_seed', 'max': 'pfi_max_seed', 'count': 'n_seeds'})
        aggregate['complete_requested_seeds'] = aggregate.n_seeds == len(args.seeds)
        columns = ['group', 'wavelength_min', 'wavelength_max', 'wavelength_center', 'n_bands']
        aggregate = aggregate.merge(all_seeds[columns].drop_duplicates(), on='group')
        aggregate['rank'] = aggregate.groupby('model').pfi_mean.rank(ascending=False, method='average')
        csv_write(out / 'pfi_summary.csv', aggregate)
        baseline = pd.concat(baselines, ignore_index=True)
        csv_write(out / 'baseline_per_seed.csv', baseline)
        stats = baseline.groupby('model')[['mse', 'r2', 'rmse', 'mae']].agg(['mean', 'std', 'count'])
        stats.columns = ['_'.join(column) for column in stats.columns]
        csv_write(out / 'baseline_summary.csv', stats.reset_index())
    json_write(out / 'status.json', dict(models=args.models, seeds=args.seeds, signature=signature,
        expected_runs=len(args.models) * len(args.seeds), complete_runs=len(seeds), missing_runs=missing,
        all_requested_complete=not missing,
        aggregation='Mean repeats within seed, then mean and sample SD (ddof=1) across seeds.',
        note='Overlapping holdouts are not independent replications; negative PFI values are retained.'))
    if not missing and len(args.seeds) >= 2:
        plot_top10(args, signature)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data' / 'AGB.csv')
    p.add_argument('--output', type=Path, default=ROOT / 'nested_holdout_outputs' / 'unified_pfi_v1')
    p.add_argument('--deep-checkpoint-root', type=Path, default=ROOT / 'explainability_outputs_v2')
    p.add_argument('--tabular-checkpoint-root', type=Path, default=ROOT / 'nested_holdout_outputs' / 'tabular_band_v1')
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    p.add_argument('--group-width', type=float, default=10.)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--jobs', type=int, default=-1)
    p.add_argument('--fit-missing', action='store_true', help='Fit missing five-model baselines only; never overwrite existing checkpoints.')
    p.add_argument('--plan-only', action='store_true', help='Read data/config and list checkpoint paths; no fitting, inference, or writes.')
    p.add_argument('--summarize-only', action='store_true', help='Only aggregate completed PFI files.')
    p.add_argument('--plot-only', action='store_true', help='Verify completed seed tables and generate Top10 PNG/PDF plots; no training or inference.')
    args = p.parse_args()
    args.data = resolve_path(args.data)
    args.output = resolve_path(args.output)
    args.deep_checkpoint_root = resolve_path(args.deep_checkpoint_root)
    args.tabular_checkpoint_root = resolve_path(args.tabular_checkpoint_root)
    if args.plot_only:
        if args.plan_only or args.summarize_only or args.fit_missing:
            p.error('--plot-only cannot be combined with plan/summarize/fit flags.')
        if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
            p.error('Duplicate models/seeds.')
        plot_top10(args)
        return
    if args.repeats < 2 or args.threads < 1 or not np.isfinite(args.group_width) or args.group_width <= 0:
        p.error('Require repeats >=2, positive threads and finite positive group width.')
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        p.error('Duplicate models/seeds.')
    if any(seed < 0 or seed > 2**32 - 10001 for seed in args.seeds):
        p.error('Seed is outside the supported nested split range.')
    for source in (args.deep_checkpoint_root, args.tabular_checkpoint_root):
        if args.output == source or source in args.output.parents or args.output in source.parents:
            p.error('Output must be separate from source checkpoint directories.')
    args.data = require_data(args.data)
    frame = pd.read_csv(args.data)
    x, y = frame.iloc[:, :-1].to_numpy(float), frame.iloc[:, -1].to_numpy(float)
    wl = np.asarray(frame.columns[:-1], dtype=float)
    if len(x) < 15 or len(wl) < 2 or not all(np.isfinite(a).all() for a in (x, y, wl)) or np.any(np.diff(wl) <= 0):
        p.error('Expected >=15 samples, finite values, and ordered unique numeric wavelength columns.')
    groups = groups_for(wl, args.group_width)
    info = [dict(wavelength_min=float(wl[g].min()), wavelength_max=float(wl[g].max()),
                 wavelength_center=float(wl[g].mean()), n_bands=len(g)) for g in groups]
    module_names = {'scikit-learn': 'sklearn', 'torch-geometric': 'torch_geometric'}
    versions = {name: package_version(name, module_names.get(name, name))
                for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch', 'torch-geometric', 'joblib']}
    try:
        versions['xgboost'] = package_version('xgboost')
    except ImportError:
        if 'XGBoost' in args.models:
            raise
    sources = ['unified_pfi_runner.py', 'tabular_band_core.py', 'tabular_band_runner.py',
               'evaluation_protocol.py', 'explainability_core.py', 'explainability_models.py',
               'explainability_runner.py', 'explanation_quality.py']
    config = dict(schema=1, data_sha256=hash_file(args.data), sources={f: hash_file(ROOT / f) for f in sources},
        versions=versions, group_width=args.group_width, repeats=args.repeats, protocol='nested-holdout',
        score='permuted MSE minus baseline MSE; original target units squared; float64 scoring',
        permutation='outer rows without replacement; joint columns within group; fixed points allowed',
        rng='SeedSequence([seed,71000,group,repeat])', device=args.device, threads=args.threads, jobs=args.jobs,
        inference_precision='Neural FP32, TF32 disabled; native estimator precision', training_epochs=300)
    signature = digest(config)
    missing_deep, missing_tabular = [], []
    for name in args.models:
        for seed in args.seeds:
            if name in DEEP_MODELS:
                path = args.deep_checkpoint_root / 'checkpoints' / name / f'all_seed{seed}_graph0.pt'
                if not path.exists():
                    missing_deep.append(str(path))
            else:
                local = args.output / 'baselines' / name / f'seed{seed}.joblib'
                external = args.tabular_checkpoint_root / name / f'seed{seed}' / 'checkpoints' / 'baseline.joblib'
                if not local.exists() and not external.exists():
                    missing_tabular.append(f'{name}/seed{seed}')
    print(f'{len(x)} samples; {len(wl)} bands; {len(groups)} groups; {args.repeats} permutations/group.', flush=True)
    print(f'{len(args.models) * len(args.seeds)} model-seed runs. Missing deep checkpoints: {missing_deep}', flush=True)
    print(f'Missing tabular baselines: {missing_tabular}\nOutput: {args.output}', flush=True)
    if args.plan_only:
        return
    if not args.summarize_only:
        if missing_deep:
            raise FileNotFoundError('Missing deep checkpoints. Train them with python explainability_runner.py --stage train '
                                    '(use the same --data), or specify --deep-checkpoint-root for a previously trained output tree.')
        if missing_tabular and not args.fit_missing:
            raise FileNotFoundError('Missing tabular baselines; use --fit-missing when ready to fit them.')
    manifest = args.output / 'experiment.json'
    if manifest.exists():
        if read_json(manifest) != dict(signature=signature, config=config):
            raise ValueError('PFI data/config/code/runtime changed; use a new --output.')
    else:
        if args.summarize_only:
            raise FileNotFoundError('No PFI manifest to summarize.')
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError('Nonempty output directory without PFI manifest.')
        json_write(manifest, dict(signature=signature, config=config))
    if args.summarize_only:
        summarize(args, signature)
        return
    torch.set_num_threads(args.threads)
    json_write(args.output / 'groups.json', [dict(group=i, bands=g.tolist(), **info[i]) for i, g in enumerate(groups)])
    for seed in args.seeds:
        fit, inner, outer = nested_holdout_split(len(x), seed)
        orders = shared_permutations(seed, len(groups), args.repeats, len(outer))
        check_permutations(orders, len(groups), args.repeats, len(outer))
        path = args.output / 'permutations' / f'seed{seed}.npz'
        if path.exists():
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved['outer_ids'], outer)
                np.testing.assert_array_equal(saved['source_positions'], orders)
                np.testing.assert_array_equal(saved['source_sample_ids'], outer[orders])
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.stem + '.tmp.npz')
            np.savez_compressed(temporary, outer_ids=outer, source_positions=orders, source_sample_ids=outer[orders])
            temporary.replace(path)
        for name in args.models:
            fp32_mode(seed)
            if name in DEEP_MODELS:
                checkpoint = args.deep_checkpoint_root / 'checkpoints' / name / f'all_seed{seed}_graph0.pt'
                predictor, provenance = load_deep(checkpoint, name, seed, x, y, wl, config['data_sha256'], args.device)
            else:
                predictor, provenance = load_tabular(args, name, seed, x, y, config['data_sha256'], signature)
            fp32_mode(seed)  # restore inference mode after any fitting
            folder = args.output / name / f'seed{seed}'
            save_split_manifest(folder, seed, fit, inner, outer, args.data, selects_weights=name not in ('PLSR', 'RF', 'SVR', 'XGBoost'))
            evaluate_run(args, name, seed, predictor, provenance, x, y, outer, groups, info, orders, signature)
            del predictor
    summarize(args, signature)
    print('Requested PFI runs completed. See summary/status.json (or scoped subset summary).', flush=True)


if __name__ == '__main__':
    main()
