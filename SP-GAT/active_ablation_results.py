"""Reproduce and summarize the active nine-mode ablation from a source-only release."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
from release_paths import resolve_path, path_reference, require_data

ROOT = Path(__file__).resolve().parent
MODES = ['ndvi', 'gndvi', 'ndre', 'ndvi_gndvi', 'ndvi_ndre',
         'ndre_gndvi', 'all', 'no_vi', 'random_mapping']


def training_commands(args):
    """Base training precedes the paired mapping control and active summary."""
    return [
        [sys.executable, str(ROOT/'controlled_graph_experiments.py'), 'ablation',
         '--data', str(args.data), '--output', str(args.base_source), '--device', args.device],
        [sys.executable, str(ROOT/'ablation_random_mapping_runner.py'),
         '--data', str(args.data), '--source', str(args.base_source), '--output', str(args.source)],
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'data/AGB.csv')
    p.add_argument('--base-source', type=Path, default=ROOT/'nested_holdout_outputs/ablation_controlled_v2')
    p.add_argument('--source', type=Path, default=ROOT/'nested_holdout_outputs/ablation_random_mapping_v1')
    p.add_argument('--output', type=Path, default=ROOT/'nested_holdout_outputs/ablation_active')
    p.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                   help='Base training device; default: CUDA when available, otherwise CPU.')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--train', action='store_true', help='Train or safely resume the complete base -> mapping -> active sequence.')
    args = p.parse_args(argv)
    for name in ['data', 'base_source', 'source', 'output']:
        setattr(args, name, resolve_path(getattr(args, name)))
    if args.device is None:
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    source, output = args.source, args.output
    directories = [args.base_source, source, output]
    for i, first in enumerate(directories):
        for second in directories[i+1:]:
            if first == second or first in second.parents or second in first.parents:
                p.error('Base, mapping and active outputs must be separate, non-nested directories.')
    if args.plan_only:
        print('Active modes:', ', '.join(MODES))
        print('Training sequence: 65 base runs (including the auxiliary random_vi control), '
              '25 additional random_mapping runs, then 65 active records.')
        print('The final nine-mode summary excludes random_vi. All runs use 300 fixed epochs.')
        for number, command in enumerate(training_commands(args), 1):
            print(f'Stage {number}:', subprocess.list2cmdline(command))
        print('Output:', output)
        print('Plan only: no data reads, training or writes.')
        return
    if args.train:
        require_data(args.data)
        for command in training_commands(args):
            subprocess.run(command, check=True, cwd=ROOT)
    if not (source/'status.json').is_file():
        raise FileNotFoundError('Missing mapping results. Use ablation_vi_runner.py --train, '
                                'or run the three stages documented in README.md.')
    status = json.loads((source/'status.json').read_text(encoding='utf-8'))
    if not status['complete']:
        raise ValueError('Source experiment is incomplete.')
    runs = pd.read_csv(source/'runs.csv')
    runs = runs[runs['mode'].isin(MODES)].copy()
    seeds = [42,123,456,789,2026]
    expected = {(m,s,r) for m in MODES for s in seeds for r in range(5 if m=='random_mapping' else 1)}
    actual = list(zip(runs['mode'],runs.seed,runs['repeat']))
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('Expected eight deterministic modes x5 plus 25 random_mapping runs.')
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output/'runs.csv',index=False)
    for filename in ['per_seed.csv','summary.csv','random_within_seed.csv']:
        frame = pd.read_csv(source/filename)
        frame = frame[frame['mode'].isin(MODES)]
        frame.to_csv(output/filename,index=False)
    paired = pd.read_csv(source/'paired_differences.csv')
    paired[paired.control=='random_mapping'].to_csv(output/'paired_differences.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = pd.read_csv(output/'summary.csv').set_index('mode').loc[MODES]
    fig, ax = plt.subplots(figsize=(11,5))
    ax.bar(summary.index,summary.r2_mean,yerr=summary.r2_std,capsize=3)
    ax.set(ylabel='Outer evaluation R2',title='Nine-mode ablation: mean and sample SD across five seeds')
    ax.tick_params(axis='x',rotation=35)
    fig.tight_layout(); fig.savefig(output/'summary.png',dpi=250); plt.close(fig)
    receipt = dict(complete=True, modes=MODES, runs=len(runs), seeds=seeds,
        retired_control='random_vi (historical files retained; excluded from active analysis)',
        source=path_reference(source, relative_to=output),
        source_runs_sha256=hashlib.sha256((source/'runs.csv').read_bytes()).hexdigest(),
        note='Random repeats averaged within seed; no significance claim.')
    (output/'status.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Active results: {len(MODES)} modes, {len(runs)} records; random_vi excluded. {output}')


if __name__ == '__main__':
    main()
