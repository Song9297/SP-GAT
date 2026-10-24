"""Run selected reproduction stages, or inspect their commands without research data."""
import argparse
import subprocess
import sys

from release_paths import ROOT, require_data


ORDER = ['main', 'deep', 'pfi', 'shap', 'masking', 'ablation', 'deletion', 'figures']
MAIN_SCRIPTS = [
    'model_PLSR.py', 'model_RF.py', 'model_SVR.py', 'model_XGBoost.py',
    'model_MLP.py', 'resnet1d_agb.py', 'transformer_agb.py', 'gat_agb.py', 'PP-SGAT.py',
]


def commands(stage, data, device):
    common = ['--data', str(data)]
    execution = ['--device', device]
    if stage == 'main':
        return [[script] + common for script in MAIN_SCRIPTS] + [
            ['summarize_performance.py'], ['collect_scatter_plots.py']]
    if stage == 'deep':
        return [['explainability_runner.py', '--stage', 'train'] + common + execution]
    if stage == 'pfi':
        return [['unified_pfi_runner.py', '--fit-missing'] + common + execution]
    if stage == 'shap':
        return [['grouped_shap_runner.py'] + common + execution]
    if stage == 'masking':
        return [['shap_perturbation_runner.py'] + common + execution]
    if stage == 'ablation':
        return [['ablation_vi_runner.py', '--train'] + common + execution]
    if stage == 'deletion':
        return [['typed_edge_deletion_runner.py'] + common + execution]
    if stage == 'figures':
        return [
            ['analyze_grouped_shap.py'],
            ['plot_grouped_shap_beeswarm.py', '--dpi', '300'] + common,
            ['plot_nine_shap_SP_GAT.py'],
            ['plot_shap_perturbation_publication.py'],
            ['plot_ablation_arcs_3x3_chain.py'] + common,
            ['plot_ablation_arcs_type_top5.py'] + common,
        ]
    raise ValueError('Unknown stage: ' + stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages', nargs='+', choices=ORDER, default=ORDER[:-1],
                        help='Run in dependency order; figures are opt-in. Prerequisites are not auto-added.')
    parser.add_argument('--data', default='data/AGB.csv')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                        help='Device for explanation and graph experiments. Main scripts retain automatic device selection.')
    parser.add_argument('--plan-only', action='store_true',
                        help='List commands using only the standard library; no data reads or training.')
    args = parser.parse_args(argv)
    if len(args.stages) != len(set(args.stages)):
        parser.error('Stages must not be repeated.')
    selected = [stage for stage in ORDER if stage in args.stages]
    if not args.plan_only:
        try:
            require_data(args.data)
        except FileNotFoundError as exc:
            parser.error(str(exc))
    for stage in selected:
        print('Stage: ' + stage, flush=True)
        for command in commands(stage, args.data, args.device):
            print(subprocess.list2cmdline(['python'] + command), flush=True)
            if not args.plan_only:
                subprocess.run([sys.executable] + command, check=True, cwd=ROOT)
    if args.plan_only:
        print('Plan only. Private data and generated weights are required for actual experiments.')


if __name__ == '__main__':
    main()
