"""Paired graph experiments. Importing this module never runs training.

Reuses the project's PP-SGAT, prior graph and degree-preserving null model.
All graph variants restart from the same seed, independent of graph RNG.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation_protocol import nested_holdout_split, save_split_manifest
from explainability_models import GATRegressorVI
from explainability_core import (MODES, Predictor, prior_graph, rewire_graph,
    pair_records, attention_pair_scores, metrics, seed_everything)
from release_paths import resolve_path, require_data

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 123, 456, 789, 2026]
EPOCHS = 300
SOURCE_FILES = ['controlled_graph_experiments.py', 'explainability_core.py',
                'explainability_models.py', 'evaluation_protocol.py']


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_main_architecture():
    """Reject drift in the reused class or its default hyperparameters."""
    trees = [ast.parse((ROOT / f).read_text(encoding='utf-8-sig'))
             for f in ['PP-SGAT.py', 'explainability_models.py']]
    classes = [next(n for n in t.body if isinstance(n, ast.ClassDef)
                    and n.name == 'GATRegressorVI') for t in trees]
    if ast.dump(classes[0]) != ast.dump(classes[1]):
        raise ValueError('PP-SGAT class differs from shared model; reconcile before training.')
    expected = dict(GAT_LAYERS=2, HIDDEN_DIM=64, ATTENTION_HEADS=4, DROPOUT=.2, EDGE_ATTR_DIM=4)
    for tree in trees:
        constants = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                     for t in n.targets if isinstance(t, ast.Name) and t.id in expected}
        if constants != expected:
            raise ValueError('PP-SGAT model defaults differ from the controlled experiment.')


def deletion_masks(ei, ea, scores, ratios, seed, repeats):
    """Rank on inner-set attention; controls match type counts; remove both directions."""
    keys, slots = pair_records(ei, ea)
    types = np.array([k[2] for k in keys])
    order = np.argsort(-np.asarray(scores), kind='stable')
    for ratio in ratios:
        k = min(len(keys), int(math.ceil(ratio * len(keys))))
        top = order[:k]
        counts = [int((types[top] == t).sum()) for t in range(3)]
        for strategy in ['top', 'random']:
            for rep in range(repeats if strategy == 'random' else 1):
                selected = list(top) if strategy == 'top' else []
                if strategy == 'random':
                    # Same per-type random order across ratios gives nested deletion sets.
                    for typ, count in enumerate(counts):
                        rng = np.random.default_rng(np.random.SeedSequence([seed, 70000, rep, typ]))
                        selected.extend(rng.permutation(np.flatnonzero(types == typ))[:count].tolist())
                mask = torch.ones(ei.shape[1], dtype=torch.bool)
                for pair in selected:
                    mask[slots[pair]] = False
                yield dict(strategy=strategy, ratio=float(ratio), repeat=rep, mask=mask,
                           n_pairs=k, actual_fraction=k / len(keys), type_counts=counts,
                           deleted_keys=[list(keys[i]) for i in selected])


def train_or_load(args, run_id, seed, ei, ea, x, y, wl):
    fit, inner, evaluation = nested_holdout_split(len(x), seed)
    stats = dict(x_mean=x[fit].mean(0), x_std=x[fit].std(0) + 1e-8,
                 y_mean=float(y[fit].mean()), y_std=float(y[fit].std()) + 1e-8)
    graph_hash = hashlib.sha256(ei.numpy().tobytes() + ea.numpy().tobytes()).hexdigest()
    config = dict(run_id=run_id, seed=seed, graph_hash=graph_hash, experiment=args.manifest)
    signature = digest(config)
    path = args.output / 'checkpoints' / (run_id + '.pt')
    if path.exists():
        cp = torch.load(path, map_location='cpu', weights_only=False)
        if cp['signature'] != signature:
            raise ValueError(f'Checkpoint mismatch: {path}. Use a new output directory.')
        print(f'Reuse {run_id}', flush=True)
        return cp
    seed_everything(seed)
    model = GATRegressorVI()
    initial_hash = state_digest(model.state_dict())
    predictor = Predictor(model, 'PP-SGAT', wl, stats, args.device, ei, ea)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    tx = torch.as_tensor(x, device=args.device)
    ty = torch.as_tensor((y - stats['y_mean']) / stats['y_std'], device=args.device)
    # Batch permutation RNG cannot be consumed by model dropout or graph generation.
    order_rng = np.random.default_rng(seed + 20000)
    order_hash = hashlib.sha256()
    history = []
    best_loss, best_state, best_epoch = float('inf'), None, None
    for epoch in range(EPOCHS):
        model.train()
        order = order_rng.permutation(fit)
        order_hash.update(order.astype(np.int64).tobytes())
        loss_sum = 0.
        for start in range(0, len(order), 32):
            batch = order[start:start + 32]
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(predictor.normalized(tx[batch]), ty[batch])
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss: {run_id}, epoch {epoch+1}')
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        model.eval()
        with torch.no_grad():
            inner_loss = float(torch.nn.functional.mse_loss(predictor.normalized(tx[inner]), ty[inner]))
        if not np.isfinite(inner_loss):
            raise RuntimeError(f'Non-finite inner loss: {run_id}')
        history.append(dict(epoch=epoch+1, train_loss=loss_sum/len(fit), inner_loss=inner_loss))
        if inner_loss < best_loss:
            best_loss, best_epoch = inner_loss, epoch+1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 50 == 0:
            print(f'{run_id}: {epoch+1}/{EPOCHS}, inner={inner_loss:.5f}', flush=True)
    model.load_state_dict(best_state)
    pred = predictor.predict(x[evaluation])
    cp = dict(signature=signature, config=config, state_dict=best_state, stats=stats,
              fit_idx=fit, select_idx=inner, eval_idx=evaluation, edge_index=ei, edge_attr=ea,
              initial_state_hash=initial_hash, batch_order_hash=order_hash.hexdigest(),
              best_epoch=best_epoch, best_inner_loss=best_loss, history=history,
              predictions=pred, targets=y[evaluation], metrics=metrics(y[evaluation], pred))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); torch.save(cp, temp); temp.replace(path)
    save_csv(args.output/'histories'/(run_id+'.csv'), history)
    save_csv(args.output/'predictions'/(run_id+'.csv'),
             dict(sample_index=evaluation, actual=y[evaluation], predicted=pred))
    return cp


def predictor_from(cp, wl, device):
    model = GATRegressorVI(); model.load_state_dict(cp['state_dict'])
    return Predictor(model, 'PP-SGAT', wl, cp['stats'], device, cp['edge_index'], cp['edge_attr'])


def enforce_pairing(cp, baseline):
    for key in ['initial_state_hash', 'batch_order_hash']:
        if cp[key] != baseline[key]:
            raise ValueError(f'Paired training mismatch: {key}')


def summarize(rows, output, keys):
    df = pd.DataFrame(rows)
    metrics_cols = ['r2', 'rmse', 'mae']
    metrics_cols += [c for c in ['delta_r2', 'delta_rmse', 'delta_mae'] if c in df]
    per_seed = df.groupby(keys + ['seed'], as_index=False)[metrics_cols].mean()
    per_seed.to_csv(output/'per_seed.csv', index=False)
    summary = per_seed.groupby(keys)[metrics_cols].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(output/'summary.csv', index=False)
    random_key = 'mode' if 'mode' in keys else 'strategy'
    random_name = 'random_vi' if random_key == 'mode' else 'random'
    spread = df[df[random_key] == random_name].groupby(keys+['seed'])[metrics_cols].agg(['mean','std','count'])
    spread.columns = ['_'.join(c) for c in spread.columns]
    spread.reset_index().to_csv(output/'random_within_seed.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    if keys == ['mode']:
        ordered = summary.reindex(list(MODES))
        ax.bar(ordered.index, ordered.r2_mean, yerr=ordered.r2_std.fillna(0), capsize=3)
        ax.tick_params(axis='x', rotation=35)
        ax.set(ylabel='Outer evaluation R2', title='Nine-mode ablation: seed mean and sample SD')
    else:
        for strategy in ['top','random']:
            part = summary.loc[strategy].sort_index()
            ax.errorbar(part.index*100, part.delta_rmse_mean,
                        yerr=part.delta_rmse_std.fillna(0), marker='o', capsize=3,label=strategy)
        ax.axhline(0,color='gray',linewidth=.8)
        ax.set(xlabel='Deleted reciprocal pairs (nominal %)', ylabel='RMSE change after retraining')
        ax.legend()
    fig.tight_layout(); fig.savefig(output/'summary.png',dpi=250); plt.close(fig)
    save_json(output/'summary_notes.json', dict(standard_deviation='sample SD, ddof=1',
        unit_of_summary='seed; random repeats averaged within seed',
        inference='descriptive paired comparisons, no significance tests',
        warning='Repeated holdouts overlap. Graph rewiring can retain substantial prior edges.'))
    return per_seed


def ablation(args, x, y, wl):
    rows, diagnostics = [], []
    for seed in args.seeds:
        all_ei, all_ea = prior_graph(wl, 'all')
        baseline = train_or_load(args, f'all_seed{seed}_graph0', seed, all_ei, all_ea, x, y, wl)
        for mode in MODES:
            for rep in range(args.graph_repeats if mode == 'random_vi' else 1):
                ei, ea = prior_graph(wl, 'all' if mode == 'random_vi' else mode)
                if mode == 'random_vi':
                    graph_seed = int(np.random.SeedSequence([seed,1729,rep]).generate_state(1)[0])
                    ei, ea, info = rewire_graph(ei, ea, len(wl), graph_seed, args.swaps_per_edge)
                    diagnostics.append(dict(seed=seed, repeat=rep, **info))
                run_id = f'{mode}_seed{seed}_graph{rep}'
                cp = baseline if mode == 'all' else train_or_load(args, run_id, seed, ei, ea, x, y, wl)
                enforce_pairing(cp, baseline)
                row = dict(mode=mode, seed=seed, repeat=rep, n_directed_edges=ei.shape[1], **cp['metrics'])
                row.update({'delta_'+m:cp['metrics'][m]-baseline['metrics'][m] for m in ['r2','rmse','mae']})
                rows.append(row); save_csv(args.output/'runs.csv', rows)
                save_json(args.output/'rewiring_diagnostics.json', diagnostics)
    summarize(rows, args.output, ['mode'])


def perturbation(args, x, y, wl):
    rows = []
    for seed in args.seeds:
        ei, ea = prior_graph(wl, 'all')
        baseline = train_or_load(args, f'baseline_seed{seed}', seed, ei, ea, x, y, wl)
        rows.append(dict(seed=seed, strategy='baseline', ratio=0., repeat=0, n_pairs=0,
                         actual_fraction=0., **baseline['metrics'], delta_r2=0., delta_rmse=0., delta_mae=0.))
        # External evaluation samples NEVER determine the topology used for retraining.
        predictor = predictor_from(baseline, wl, args.device)
        keys, _, scores = attention_pair_scores(predictor, x[baseline['select_idx']])
        scores = scores.mean(0)
        save_csv(args.output/'selection'/f'inner_attention_seed{seed}.csv',
                 [dict(source=a, target=b, relation_type=t, attention=float(s)) for (a,b,t),s in zip(keys,scores)])
        for case in deletion_masks(ei, ea, scores, args.ratios, seed, args.random_repeats):
            run_id = f"{case['strategy']}_ratio{case['ratio']:g}_seed{seed}_repeat{case['repeat']}"
            mask = case['mask']
            save_json(args.output/'selection'/(run_id+'.json'),
                      {k:v for k,v in case.items() if k!='mask'})
            # Keep original standardized attributes: do not renormalize after deletion.
            cp = train_or_load(args, run_id, seed, ei[:,mask], ea[mask], x, y, wl)
            enforce_pairing(cp, baseline)
            row = {k:case[k] for k in ['strategy','ratio','repeat','n_pairs','actual_fraction']}
            row.update(seed=seed, **cp['metrics'])
            row.update({'delta_'+m:cp['metrics'][m]-baseline['metrics'][m] for m in ['r2','rmse','mae']})
            rows.append(row); save_csv(args.output/'runs.csv', rows)
        del predictor
    per = summarize(rows, args.output, ['strategy','ratio'])
    top = per[per.strategy=='top'].set_index(['seed','ratio'])
    random = per[per.strategy=='random'].set_index(['seed','ratio'])
    diff = top[['r2','rmse','mae']] - random[['r2','rmse','mae']]
    diff.add_prefix('top_minus_random_').reset_index().to_csv(args.output/'paired_differences.csv',index=False)


def main(kind=None, argv=None):
    parser = argparse.ArgumentParser(description='Controlled PP-SGAT experiments; 300 fixed epochs, inner selection.')
    if kind is None:
        parser.add_argument('kind', choices=['ablation', 'perturbation'])
    parser.add_argument('--data', type=Path, default=ROOT/'data/AGB.csv')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--graph-repeats', type=int, default=5)
    parser.add_argument('--swaps-per-edge', type=int, default=5)
    parser.add_argument('--random-repeats', type=int, default=5)
    parser.add_argument('--ratios', nargs='+', type=float, default=[.05,.1,.2,.3])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--plan-only', action='store_true', help='Print configuration and run count without data, training or writes.')
    args = parser.parse_args(argv)
    kind = kind or args.kind
    args.data = resolve_path(args.data)
    args.output = resolve_path(args.output or ROOT/'nested_holdout_outputs'/(kind+'_controlled_v2'))
    if len(set(args.seeds))!=len(args.seeds) or not args.seeds or min(args.seeds)<0:
        parser.error('Seeds must be distinct non-negative integers.')
    if min(args.graph_repeats,args.swaps_per_edge,args.random_repeats)<1:
        parser.error('Repeat counts must be positive.')
    if len(set(args.ratios))!=len(args.ratios) or any(not 0<r<1 for r in args.ratios):
        parser.error('Ratios must be unique and between 0 and 1.')
    check_main_architecture()
    runs = len(args.seeds)*((8+args.graph_repeats) if kind=='ablation' else (1+len(args.ratios)*(1+args.random_repeats)))
    print(f'{kind}: {runs} training runs; {EPOCHS} epochs each; output: {args.output}', flush=True)
    if args.plan_only:
        print(f'Data required for training: {args.data}')
        print('Plan only: no data reads, training, inference or output writes.')
        return
    require_data(args.data)
    data = pd.read_csv(args.data)
    x, y = data.iloc[:,:-1].to_numpy(dtype=np.float32), data.iloc[:,-1].to_numpy(dtype=np.float32)
    wl = np.array(data.columns[:-1], dtype=float)
    if len(x)<15 or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(wl).all() or not (np.diff(wl)>0).all():
        raise ValueError('Invalid data: require finite values, ordered wavelengths and at least 15 samples.')
    args.manifest = dict(kind=kind, data_hash=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        source_hash=hashlib.sha256(b''.join((ROOT/f).read_bytes() for f in SOURCE_FILES)).hexdigest(),
        seeds=args.seeds, epochs=EPOCHS, batch_size=32, lr=.001, weight_decay=.0001,
        graph_repeats=args.graph_repeats, random_repeats=args.random_repeats, ratios=args.ratios,
        swaps_per_edge=args.swaps_per_edge, protocol='nested-holdout', device=args.device,
        torch_version=torch.__version__, batch_rng='numpy default_rng(seed+20000)',
        edge_selection='baseline inner validation samples only', no_early_stopping=True)
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=args.manifest:
        raise ValueError('Output belongs to another configuration. Choose a new --output.')
    if args.output.exists() and any(args.output.iterdir()) and not manifest_path.exists():
        raise ValueError('Nonempty output has no manifest; choose a new --output.')
    save_json(manifest_path,args.manifest)
    for seed in args.seeds:
        fit, inner, evaluation = nested_holdout_split(len(x),seed)
        save_split_manifest(args.output,seed,fit,inner,evaluation,args.data)
    (ablation if kind=='ablation' else perturbation)(args,x,y,wl)
    print('Completed. Standard deviations use ddof=1; random repeats averaged within seed.')


if __name__ == '__main__':
    main()
