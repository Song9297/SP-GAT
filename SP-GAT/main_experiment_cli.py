"""Dependency-free command-line setup for the published experiment scripts."""

import argparse
import json
import os

from release_paths import resolve_path


SEEDS = [42, 123, 456, 789, 2026]

MODEL_FOLDERS = {
    "PLSR": "output_PLSR",
    "RF": "output_RF",
    "SVR": "output_SVR",
    "XGBoost": "output_XGBoost",
    "MLP": "output_MLP",
    "GAT_VI": "gat_vi_outputs",
    "GAT_Baseline": "gat_baseline_outputs",
    "ResNet1D": "resnet_outputs",
    "Transformer": "transformer_outputs",
}
PERFORMANCE_INPUTS = {
    name: folder + "/results_per_seed.csv" for name, folder in MODEL_FOLDERS.items()
}
SCATTER_INPUTS = {
    name: MODEL_FOLDERS[name] + "/" + filename
    for name, filename in {
        "PLSR": "scatter_PLSR.png", "RF": "scatter_RF.png",
        "SVR": "scatter_SVR.png", "XGBoost": "scatter_XGBoost.png",
        "MLP": "scatter_MLP.png", "GAT_VI": "prediction_scatter.png",
        "GAT_Baseline": "prediction_scatter.png", "ResNet1D": "results_scatter.png",
        "Transformer": "results_scatter.png",
    }.items()
}


def _parser(description):
    parser = argparse.ArgumentParser(
        description=description,
        epilog="Relative paths are resolved from the repository root, regardless of the working directory.",
    )
    parser.add_argument(
        "--output-root", default=os.environ.get("SPGAT_OUTPUT_ROOT", "nested_holdout_outputs"),
        metavar="DIR", help="Root for experiment outputs (default: nested_holdout_outputs).",
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Print resolved paths and the experiment plan, then exit without reading data or training.",
    )
    return parser


def parse_model_cli(model_name, output_folder, neural=False):
    """Parse paths before importing scientific libraries or creating output folders."""
    parser = _parser(f"Run the published {model_name} nested-holdout experiment.")
    parser.add_argument(
        "--data", default="data/AGB.csv", metavar="CSV",
        help="Spectral CSV: wavelength columns followed by the AGB target (default: data/AGB.csv).",
    )
    args = parser.parse_args()
    args.data = resolve_path(args.data)
    args.output_root = resolve_path(args.output_root)
    if args.plan_only:
        plan = {
            "model": model_name,
            "data": str(args.data),
            "data_included": False,
            "output_directory": str(args.output_root / output_folder),
            "protocol": "nested-holdout",
            "seeds": SEEDS,
            "outer_evaluation_ratio": 0.2,
            "inner_validation_ratio_of_remaining": 0.2,
            "preprocessing_fit": "fit subset only",
            "selection": "minimum inner validation loss" if neural else "fixed model parameters; inner set reserved",
        }
        if neural:
            plan.update(epochs=300, early_stopping=False, batch_size=32)
        print(json.dumps(plan, indent=2))
        raise SystemExit(0)
    if not args.data.is_file():
        parser.error(
            f"Dataset not found: {args.data}. Research data are not included in this release. "
            "Place an authorized AGB.csv in the data/ folder or pass --data PATH. "
            "Use --plan-only to inspect the configuration without data."
        )
    os.environ["SPGAT_OUTPUT_ROOT"] = str(args.output_root)
    return args


def parse_results_cli(description, inputs, output_folder):
    """Set the shared output root for summary tools without requiring the dataset."""
    parser = _parser(description)
    args = parser.parse_args()
    args.output_root = resolve_path(args.output_root)
    if args.plan_only:
        print(json.dumps({
            "operation": description,
            "inputs": {name: str(args.output_root / path) for name, path in inputs.items()},
            "output_directory": str(args.output_root / output_folder),
            "data_required": False,
        }, indent=2))
        raise SystemExit(0)
    os.environ["SPGAT_OUTPUT_ROOT"] = str(args.output_root)
    return args
