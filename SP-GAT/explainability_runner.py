"""Reproducible ablation and explanation experiments; see EXPLAINABILITY_EXPERIMENTS.md."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from release_paths import dataset_reference, resolve_path, require_data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from explanation_quality import quality_arrays, upsert_performance
from sklearn.model_selection import train_test_split

from explainability_core import (
    MODEL_NAMES, MODES, Predictor, attention_pair_scores, chain_graph, digest,
    group_scores, ig_attributions, make_model, metrics, prior_graph, rewire_graph,
    seed_everything, similarity, source_hash, wavelength_groups,
)

ROOT = Path(__file__).resolve().parent


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def load_checkpoint(path, device='cpu'):
    # Only checkpoints created by this local runner. Explicit for PyTorch >= 2.6.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def graph_for(name, wavelengths, mode, graph_seed, swaps):
    if name == 'GAT':
        return chain_graph(len(wavelengths)), None, {}
    if name != 'PP-SGAT':
        return None, None, {}
    ei, ea = prior_graph(wavelengths, 'all' if mode == 'random_vi' else mode)
    if mode == 'random_vi':
        return rewire_graph(ei, ea, len(wavelengths), graph_seed, swaps)
    return ei, ea, {}


def train_or_load(args, name, seed, mode, repeat, x, y, wl):
    graph_seed = int(np.random.SeedSequence([seed, 1729, repeat]).generate_state(1)[0])
    epochs = args.epochs if args.epochs is not None else 300
    config = dict(model=name, seed=seed, mode=mode, graph_repeat=repeat, graph_seed=graph_seed,
                  epochs=epochs, protocol=args.protocol, inner_ratio=.2, outer_ratio=.2,
                  lr=.001, weight_decay=.0001, batch_size=32, swaps_per_edge=args.swaps_per_edge,
                  data_hash=args.data_hash, source_hash=args.source_hash, torch_version=torch.__version__,
                  device=args.device)
    path = args.output / 'checkpoints' / name / f'{mode}_seed{seed}_graph{repeat}.pt'
    signature = digest(config)
    if path.exists():
        cp = load_checkpoint(path)
        if cp['signature'] != signature:
            raise ValueError(f'Checkpoint configuration mismatch: {path}. Choose a new --output directory.')
        print(f'  Reuse {name} {mode} seed={seed} graph={repeat}', flush=True)
        return cp
    train_idx, eval_idx = train_test_split(np.arange(len(x)), test_size=.2, random_state=seed)
    if args.protocol == 'nested-holdout':
        fit_idx, select_idx = train_test_split(train_idx, test_size=.2, random_state=seed + 10000)
    else:
        fit_idx, select_idx = train_idx, eval_idx
    stats = dict(x_mean=x[fit_idx].mean(0), x_std=x[fit_idx].std(0) + 1e-8,
                 y_mean=float(y[fit_idx].mean()), y_std=float(y[fit_idx].std()) + 1e-8)
    ei, ea, graph_info = graph_for(name, wl, mode, graph_seed, args.swaps_per_edge)
    # Graph RNG is separate: all variants start with the same network RNG state.
    seed_everything(seed)
    predictor = Predictor(make_model(name, len(wl)), name, wl, stats, args.device, ei, ea)
    optimizer = torch.optim.Adam(predictor.model.parameters(), lr=.001, weight_decay=.0001)
    tx = torch.as_tensor(x, dtype=torch.float32, device=args.device)
    ty = torch.as_tensor((y - stats['y_mean']) / stats['y_std'], dtype=torch.float32, device=args.device)
    order_rng = np.random.default_rng(seed + 20000)
    best_loss, best_state, best_epoch = float('inf'), None, None
    history = []
    print(f'  Train {name}/{mode} seed={seed} graph={repeat}: {epochs} epochs', flush=True)
    for epoch in range(epochs):
        predictor.model.train()
        order = order_rng.permutation(fit_idx)
        train_sum = 0.
        for start in range(0, len(order), 32):
            indices = order[start:start + 32]
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(predictor.normalized(tx[indices]), ty[indices])
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach()) * len(indices)
        predictor.model.eval()
        with torch.no_grad():
            select_sum = 0.
            for start in range(0, len(select_idx), 16):
                indices = select_idx[start:start + 16]
                select_sum += float(torch.nn.functional.mse_loss(
                    predictor.normalized(tx[indices]), ty[indices], reduction='sum'))
            select_loss = select_sum / len(select_idx)
        history.append(dict(epoch=epoch + 1, train_loss=train_sum / len(order), selection_loss=select_loss))
        if select_loss < best_loss:
            best_loss, best_epoch = select_loss, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in predictor.model.state_dict().items()}
        if (epoch + 1) % 25 == 0:
            print(f'    epoch {epoch + 1}/{epochs}, selection loss={select_loss:.5f}', flush=True)
    if best_state is None:
        raise RuntimeError('Training produced no finite selection loss.')
    predictor.model.load_state_dict(best_state)
    pred = predictor.predict(x[eval_idx])
    cp = dict(signature=signature, config=config, state_dict=best_state, stats=stats,
              fit_idx=fit_idx, select_idx=select_idx, eval_idx=eval_idx, wavelengths=wl,
              edge_index=ei, edge_attr=ea, graph_info=graph_info, history=history,
              best_epoch=best_epoch, predictions=pred, targets=y[eval_idx], metrics=metrics(y[eval_idx], pred))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(cp, temporary)
    temporary.replace(path)
    return cp


def from_checkpoint(cp, device):
    name = cp['config']['model']
    model = make_model(name, len(cp['wavelengths']))
    model.load_state_dict(cp['state_dict'])
    return Predictor(model, name, cp['wavelengths'], cp['stats'], device, cp['edge_index'], cp['edge_attr'])


def plot_curves(df, out, xlabel, ylabel='delta_rmse'):
    if df.empty:
        return
    models = list(df['model'].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    # First average repeats WITHIN a seed, then summarize seeds; repeats are not subjects.
    per_seed = df.groupby(['model', 'seed', 'strategy', 'ratio'])[ylabel].mean().reset_index()
    summary = per_seed.groupby(['model', 'strategy', 'ratio'])[ylabel].agg(['mean', 'std']).reset_index()
    csv(out.with_suffix('.csv'), summary)
    for ax, name in zip(axes[0], models):
        for strategy, part in summary[summary.model == name].groupby('strategy'):
            part = part.sort_values('ratio')
            ax.errorbar(part.ratio * 100, part['mean'], yerr=part['std'].fillna(0),
                        marker='o', capsize=3, label=strategy)
        ax.set(title=name, xlabel=xlabel, ylabel=ylabel)
        ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)


def ablation(args, x, y, wl):
    rows, diagnostics = [], []
    for mode in MODES:
        for seed in args.seeds:
            repeats = range(args.graph_repeats) if mode == 'random_vi' else range(1)
            for rep in repeats:
                cp = train_or_load(args, 'PP-SGAT', seed, mode, rep, x, y, wl)
                rows.append(dict(mode=mode, seed=seed, graph_repeat=rep,
                                 n_directed_edges=cp['edge_index'].shape[1], **cp['metrics']))
                if cp['graph_info']:
                    diagnostics.append(dict(seed=seed, graph_repeat=rep, **cp['graph_info']))
                csv(args.output / 'ablation' / 'runs.csv', rows)
    df = pd.DataFrame(rows)
    within = df.groupby(['mode', 'seed'])[['r2', 'rmse', 'mae']].mean().reset_index()
    csv(args.output / 'ablation' / 'per_seed.csv', within)
    summary = within.groupby('mode')[['r2', 'rmse', 'mae']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    csv(args.output / 'ablation' / 'summary.csv', summary.reset_index())
    random_std = df[df['mode'] == 'random_vi'].groupby('seed')[['r2', 'rmse', 'mae']].std().reset_index()
    csv(args.output / 'ablation' / 'random_graph_std_within_seed.csv', random_std)
    pivot = within.pivot(index='seed', columns='mode', values='r2')
    paired = pd.DataFrame({'seed': pivot.index, 'all_minus_random_r2': pivot['all'] - pivot['random_vi'],
                           'all_minus_no_vi_r2': pivot['all'] - pivot['no_vi']})
    csv(args.output / 'ablation' / 'paired_differences.csv', paired)
    save_json(args.output / 'ablation' / 'rewiring_diagnostics.json', diagnostics)
    fig, ax = plt.subplots(figsize=(11, 5))
    ordered = summary.reindex(list(MODES))
    ax.bar(ordered.index, ordered.r2_mean, yerr=ordered.r2_std.fillna(0), capsize=3)
    ax.set(ylabel='R2 (seed mean +/- SD)', title='Nine-mode ablation; random graphs averaged within seed')
    ax.tick_params(axis='x', rotation=25)
    fig.tight_layout(); fig.savefig(args.output / 'ablation' / 'ablation.png', dpi=300); plt.close(fig)


def band_perturbation(predictor, raw, targets, baseline, groups, scores, args, seed, name):
    original = predictor.predict(raw)
    original_metrics = metrics(targets, original)
    rows, individual = [], []
    for strategy in ['top', 'low', 'random']:
        for rep in range(args.random_repeats if strategy == 'random' else 1):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 30000, rep]))
            if strategy == 'top':
                orders = np.argsort(-scores, axis=1, kind='stable')
            elif strategy == 'low':
                orders = np.argsort(scores, axis=1, kind='stable')
            else:
                orders = np.stack([rng.permutation(len(groups)) for _ in raw])
            for ratio in [0.] + args.ratios:
                k = min(len(groups), int(math.ceil(ratio * len(groups))))
                changed = raw.copy()
                for i, order in enumerate(orders):
                    if k:
                        bands = np.concatenate([groups[g] for g in order[:k]])
                        changed[i, bands] = baseline[bands]
                pred = original if not k else predictor.predict(changed)
                m = metrics(targets, pred)
                rows.append(dict(model=name, seed=seed, strategy=strategy, repeat=rep, ratio=ratio,
                                 actual_fraction=k / len(groups), n_groups=k, **m,
                                 delta_rmse=m['rmse'] - original_metrics['rmse'],
                                 delta_mae=m['mae'] - original_metrics['mae'],
                                 prediction_change=float(np.abs(pred - original).mean())))
                individual.extend(dict(strategy=strategy, repeat=rep, ratio=ratio, eval_position=i,
                                       original=float(original[i]), perturbed=float(p), target=float(targets[i]))
                                  for i, p in enumerate(pred))
    return rows, individual


def edge_perturbation(predictor, raw, targets, args, seed):
    keys, slots, scores = attention_pair_scores(predictor, raw)
    types = np.array([k[2] for k in keys])
    original = predictor.predict(raw)
    original_metrics = metrics(targets, original)
    rows, individual = [], []
    # Rank per sample, not using labels. For each top set, controls have exact type counts.
    top_orders = np.argsort(-scores, axis=1, kind='stable')
    for strategy in ['top', 'low', 'random']:
        for rep in range(args.random_repeats if strategy == 'random' else 1):
            for ratio in [0.] + args.ratios:
                k = min(len(keys), int(math.ceil(ratio * len(keys))))
                predictions = []
                for i, raw_row in enumerate(raw):
                    top = top_orders[i, :k]
                    selected = []
                    for typ in range(3):
                        count = int(np.sum(types[top] == typ))
                        candidates = np.flatnonzero(types == typ)
                        if strategy == 'top':
                            order = candidates[np.argsort(-scores[i, candidates], kind='stable')]
                        elif strategy == 'low':
                            order = candidates[np.argsort(scores[i, candidates], kind='stable')]
                        else:
                            rng = np.random.default_rng(np.random.SeedSequence([seed, 40000, rep, i, typ]))
                            order = rng.permutation(candidates)
                        selected.extend(order[:count])
                    mask = torch.ones(predictor.ei.shape[1], dtype=torch.bool, device=predictor.device)
                    for pair in selected:
                        mask[slots[pair]] = False
                    # Preserve remaining attributes; GAT recomputes attention and automatic self-loops.
                    prediction = original[i] if not k else predictor.predict(
                        raw_row[None], predictor.ei[:, mask], predictor.ea[mask])[0]
                    predictions.append(prediction)
                    individual.append(dict(strategy=strategy, repeat=rep, ratio=ratio, eval_position=i,
                                           deleted_pairs=k, original=float(original[i]), perturbed=float(prediction),
                                           target=float(targets[i])))
                predictions = np.asarray(predictions)
                m = metrics(targets, predictions)
                rows.append(dict(model='PP-SGAT', seed=seed, strategy=strategy, repeat=rep, ratio=ratio,
                                 n_pairs=k, actual_fraction=k / len(keys), **m,
                                 delta_rmse=m['rmse'] - original_metrics['rmse'],
                                 prediction_change=float(np.abs(predictions - original).mean())))
    return rows, individual, keys, scores


def checked_ig(predictor, raw, baseline, args, quality_path):
    """Refine the whole evaluation set without changing parameters or sample selection."""
    steps = args.ig_steps
    differences = predictor.predict(raw) - predictor.predict(baseline[None])[0]
    while True:
        attrs, residual = ig_attributions(predictor, raw, baseline, steps, args.path_batch)
        quality = pd.DataFrame(quality_arrays(attrs, residual, args.ig_atol, args.ig_rtol, differences))
        quality['integration_steps'] = steps
        csv(quality_path, quality)
        if quality.passed.all():
            return attrs, residual, differences, steps
        if steps >= args.ig_max_steps:
            raise RuntimeError(f'IG completeness failed at {steps} steps: {quality_path}; no final summary written.')
        steps = min(steps * 2, args.ig_max_steps)
        print(f'    Refine IG to {steps} steps: {quality_path.name}', flush=True)


def analyze_one(args, cp, x, y, wl):
    name, seed = cp['config']['model'], cp['config']['seed']
    predictor = from_checkpoint(cp, args.device)
    indices = cp['eval_idx']
    raw, targets = x[indices], y[indices]
    baseline = x[cp['fit_idx']].mean(0) if args.baseline == 'mean' else np.median(x[cp['fit_idx']], axis=0)
    groups = wavelength_groups(wl, args.group_width)
    out = args.analysis_dir / name / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    complete_marker = out / 'run_complete.json'
    if complete_marker.exists():
        complete_marker.unlink()
    save_json(out / 'run_metadata.json', dict(checkpoint_signature=cp['signature'],
        checkpoint_config=cp['config'], fit_idx=list(map(int, cp['fit_idx'])),
        select_idx=list(map(int, cp['select_idx'])), eval_idx=list(map(int, indices))))
    print(f'  Explain {name} seed={seed}: {len(raw)} held-out samples, {len(groups)} groups', flush=True)
    attrs, residual, differences, trained_steps = checked_ig(predictor, raw, baseline, args, out/'ig_quality.csv')
    scores = group_scores(attrs, groups)
    np.savez_compressed(out / 'attributions.npz', sample_indices=indices, wavelengths=wl,
                        ig=attrs, group_scores=scores, baseline=baseline, completeness_residual=residual,
                        prediction_minus_baseline=differences, integration_steps=trained_steps)
    base_pred = predictor.predict(baseline[None])[0]
    differences = predictor.predict(raw) - base_pred
    tolerance = args.ig_atol + args.ig_rtol * np.abs(differences)
    quality = pd.DataFrame(dict(sample_index=indices, prediction_minus_baseline=differences,
                                ig_sum=attrs.sum(1), completeness_residual=residual,
                                tolerance=tolerance, passed=np.abs(residual) <= tolerance))
    quality['integration_steps'] = trained_steps
    csv(out / 'ig_quality.csv', quality)
    if not quality.passed.all():
        print(f'    WARNING: IG completeness failed for {(~quality.passed).sum()} samples; increase --ig-steps.', flush=True)
    group_rows = [dict(group=j, start_nm=float(wl[g].min()), end_nm=float(wl[g].max()),
                       n_bands=len(g), mean_abs_ig=float(scores[:, j].mean())) for j, g in enumerate(groups)]
    csv(out / 'group_importance.csv', group_rows)
    band_rows, band_individual = band_perturbation(predictor, raw, targets, baseline, groups, scores, args, seed, name)
    for row in band_individual:
        row['sample_index'] = int(indices[row['eval_position']])
    csv(out / 'band_perturbation_samples.csv', band_individual)
    csv(out / 'band_perturbation.csv', band_rows)
    edges, edge_keys, edge_scores = [], None, None
    if name == 'PP-SGAT':
        edges, edge_individual, edge_keys, edge_scores = edge_perturbation(predictor, raw, targets, args, seed)
        for row in edge_individual:
            row['sample_index'] = int(indices[row['eval_position']])
        csv(out / 'edge_perturbation.csv', edges)
        csv(out / 'edge_perturbation_samples.csv', edge_individual)
        csv(out / 'edge_attention.csv', [dict(source=a, target=b, source_nm=wl[a], target_nm=wl[b],
                                             relation_type=t, mean_attention=edge_scores[:, j].mean())
                                        for j, (a, b, t) in enumerate(edge_keys)])
        np.savez_compressed(out / 'attention.npz', keys=np.asarray(edge_keys), scores=edge_scores,
                            sample_indices=indices)
    checks = []
    for repeat in range(args.randomization_repeats):
        seed_everything(seed + 50000 + repeat)
        random_predictor = Predictor(make_model(name, len(wl)), name, wl, cp['stats'], args.device,
                                     cp['edge_index'], cp['edge_attr'])
        random_attrs, random_residual, random_diff, random_steps = checked_ig(
            random_predictor, raw, baseline, args, out/f'randomized_{repeat}_quality.csv')
        random_scores = group_scores(random_attrs, groups)
        np.savez_compressed(out / f'randomized_{repeat}.npz', ig=random_attrs, group_scores=random_scores,
                            completeness_residual=random_residual, sample_indices=indices,
                            prediction_minus_baseline=random_diff, integration_steps=random_steps)
        checks.append(dict(model=name, seed=seed, repeat=repeat, explanation='IG',
                           **similarity(scores.mean(0), random_scores.mean(0)),
                           prediction_change=float(np.abs(predictor.predict(raw) - random_predictor.predict(raw)).mean()),
                           randomized_ig_max_residual=float(np.abs(random_residual).max())))
        if name == 'PP-SGAT':
            _, _, random_attention = attention_pair_scores(random_predictor, raw)
            checks.append(dict(model=name, seed=seed, repeat=repeat, explanation='edge_attention',
                               **similarity(edge_scores.mean(0), random_attention.mean(0))))
    csv(out / 'randomization.csv', checks)
    save_json(complete_marker, dict(checkpoint_signature=cp['signature'],
                                   randomization_repeats=args.randomization_repeats,
                                   trained_steps=trained_steps, quality_passed=True))
    return dict(model=name, seed=seed, sample_indices=indices, scores=scores, edges=edge_scores,
                band_rows=band_rows, edge_rows=edges, checks=checks, quality=quality)


def summarize_analysis(args, results, wl):
    band = pd.DataFrame([r for item in results for r in item['band_rows']])
    edge = pd.DataFrame([r for item in results for r in item['edge_rows']])
    csv(args.analysis_dir / 'band_perturbation.csv', band)
    csv(args.analysis_dir / 'edge_perturbation.csv', edge)
    csv(args.analysis_dir / 'randomization.csv', [r for item in results for r in item['checks']])
    plot_curves(band, args.analysis_dir / 'band_delta_rmse.png', 'Perturbed wavelength groups (%)')
    plot_curves(band, args.analysis_dir / 'band_prediction_change.png', 'Perturbed wavelength groups (%)', 'prediction_change')
    plot_curves(edge, args.analysis_dir / 'edge_delta_rmse.png', 'Deleted reciprocal pairs (%)')
    plot_curves(edge, args.analysis_dir / 'edge_prediction_change.png', 'Deleted reciprocal pairs (%)', 'prediction_change')
    stability = []
    for name in args.models:
        runs = [r for r in results if r['model'] == name]
        for a, b in itertools.combinations(runs, 2):
            for explanation, key in [('IG', 'scores'), ('edge_attention', 'edges')]:
                if a[key] is None or b[key] is None:
                    continue
                stability.append(dict(model=name, seed_a=a['seed'], seed_b=b['seed'],
                                      explanation=explanation, scope='all_eval_samples',
                                      **similarity(a[key].mean(0), b[key].mean(0))))
                shared, ia, ib = np.intersect1d(a['sample_indices'], b['sample_indices'], return_indices=True)
                for sample, ai, bi in zip(shared, ia, ib):
                    stability.append(dict(model=name, seed_a=a['seed'], seed_b=b['seed'],
                                          explanation=explanation, scope='same_held_out_sample', sample_index=int(sample),
                                          **similarity(a[key][ai], b[key][bi])))
    csv(args.analysis_dir / 'stability.csv', stability)
    groups = wavelength_groups(wl, args.group_width)
    centers = np.array([wl[g].mean() for g in groups])
    names = list(dict.fromkeys(r['model'] for r in results))
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 3 * len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        values = np.stack([r['scores'].mean(0) for r in results if r['model'] == name])
        mean = values.mean(0)
        sd = values.std(0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
        ax.plot(centers, mean); ax.fill_between(centers, mean - sd, mean + sd, alpha=.2)
        ax.set(title=name, xlabel='Wavelength (nm)', ylabel='Sum absolute IG')
    fig.tight_layout(); fig.savefig(args.analysis_dir / 'group_importance.png', dpi=300); plt.close(fig)
    failed = sum(int((~r['quality'].passed).sum()) for r in results)
    save_json(args.analysis_dir / 'quality_summary.json', {
        'ig_failed_samples': failed, 'ig_total_samples': sum(len(r['quality']) for r in results),
        'smoke_only': args.smoke, 'protocol': args.protocol,
        'warning': 'Do not use smoke runs for scientific conclusions. Inspect IG quality before interpretation.',
        'stability_note': 'Global stability includes split/sample composition effects; shared samples are reported separately.',
        'randomization_note': 'Full parameter reinitialization, including positional embeddings and normalization buffers; fixed graph.',
    })


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['train', 'ablation', 'analyze', 'all'], default='all')
    p.add_argument('--data', type=Path, default=ROOT / 'data' / 'AGB.csv')
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=MODEL_NAMES)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456, 789, 2026])
    p.add_argument('--protocol', choices=['nested-holdout', 'legacy-holdout'], default='nested-holdout')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--graph-repeats', type=int, default=5)
    p.add_argument('--swaps-per-edge', type=int, default=5)
    p.add_argument('--group-width', type=float, default=10.)
    p.add_argument('--baseline', choices=['mean', 'median'], default='mean')
    p.add_argument('--ig-steps', type=int, default=64)
    p.add_argument('--ig-max-steps', type=int, default=4096)
    p.add_argument('--ig-atol', type=float, default=.1)
    p.add_argument('--ig-rtol', type=float, default=.05)
    p.add_argument('--path-batch', type=int, default=8)
    p.add_argument('--random-repeats', type=int, default=20)
    p.add_argument('--randomization-repeats', type=int, default=20)
    p.add_argument('--ratios', nargs='+', type=float, default=[.05, .1, .2, .3])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--smoke', action='store_true', help='Two epochs / two seeds, separate outputs, not research results')
    args = p.parse_args()
    if args.smoke:
        args.epochs, args.seeds = 2, [42, 123]
        args.graph_repeats, args.random_repeats, args.randomization_repeats = 2, 2, 1
        args.ig_steps, args.ratios = 8, [.1]
    for key in ['graph_repeats', 'swaps_per_edge', 'ig_steps', 'path_batch', 'random_repeats',
                'randomization_repeats', 'threads']:
        if getattr(args, key) < 1:
            p.error(f'--{key.replace("_", "-")} must be positive')
    if args.group_width <= 0 or (args.epochs is not None and args.epochs < 1):
        p.error('Group width and epochs must be positive')
    if any(r <= 0 or r > 1 for r in args.ratios):
        p.error('Ratios must be in (0, 1]')
    if args.ig_atol < 0 or args.ig_rtol < 0:
        p.error('IG tolerances must be nonnegative')
    if args.ig_max_steps < args.ig_steps:
        p.error('--ig-max-steps must be at least --ig-steps')
    args.ratios = sorted(set(args.ratios))
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.models)) != len(args.models):
        p.error('Duplicate seeds/models are not allowed')
    args.data = resolve_path(args.data)
    args.output = resolve_path(args.output or ROOT / ('explainability_smoke_outputs_v2' if args.smoke else 'explainability_outputs_v2'))
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    args.data = require_data(args.data)
    data = pd.read_csv(args.data)
    x, y = data.iloc[:, :-1].to_numpy(dtype=np.float32), data.iloc[:, -1].to_numpy(dtype=np.float32)
    wl = np.array(data.columns[:-1], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(wl).all():
        raise ValueError('Data must contain finite numeric reflectance, wavelength and target values.')
    if len(x) < 15 or len(wl) < 32 or not np.all(np.diff(wl) > 0):
        raise ValueError('Expected at least 15 samples, 32 ordered unique wavelengths.')
    args.data_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    args.source_hash = digest(dict(core_source=source_hash(),
        quality_source=hashlib.sha256((ROOT/'explanation_quality.py').read_bytes()).hexdigest()))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(data=dataset_reference(args.data), data_hash=args.data_hash, source_hash=args.source_hash,
                    protocol=args.protocol, epochs=args.epochs, smoke=args.smoke, device=args.device,
                    swaps_per_edge=args.swaps_per_edge)
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Output directory belongs to another experiment configuration; use a new --output.')
    save_json(manifest_path, manifest)
    print(f'Output: {args.output}\nProtocol: {args.protocol}; smoke={args.smoke}; device={args.device}', flush=True)
    if args.stage in ('ablation', 'all'):
        ablation(args, x, y, wl)
    if args.stage in ('train', 'analyze', 'all'):
        results, performance = [], []
        analysis_config = {k: getattr(args, k) for k in ['models', 'seeds', 'group_width', 'baseline', 'ig_steps',
                          'ig_atol', 'ig_rtol', 'ig_max_steps', 'path_batch', 'random_repeats', 'randomization_repeats', 'ratios']}
        args.analysis_dir = args.output / ('analysis_' + digest(analysis_config)[:12])
        if args.stage != 'train':
            # Remove stale aggregate success marker before a new/repeated analysis.
            marker = args.analysis_dir / 'analysis_complete.json'
            if marker.exists():
                marker.unlink()
            save_json(args.analysis_dir / 'config.json', analysis_config)
        for name in args.models:
            for seed in args.seeds:
                cp = train_or_load(args, name, seed, 'all', 0, x, y, wl)
                performance.append(dict(model=name, seed=seed, best_epoch=cp['best_epoch'], **cp['metrics']))
                upsert_performance(args.output / 'model_performance.csv', performance)
                if args.stage != 'train':
                    results.append(analyze_one(args, cp, x, y, wl))
        if results:
            summarize_analysis(args, results, wl)
            from summarize_randomization import summarize_directory
            summarize_directory(args.analysis_dir)
            save_json(args.analysis_dir/'analysis_complete.json', dict(quality_passed=True,
                runs=len(results), randomization_repeats=args.randomization_repeats))
    print('Completed. Review quality_summary.json and per-sample IG quality before interpreting results.', flush=True)


if __name__ == '__main__':
    main()
