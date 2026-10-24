# SP-GAT research code

Source release prepared on 2026-10-24 for inspection of the nine-model AGB
comparison, grouped SHAP, fixed-weight spectral masking, and graph ablation and
deletion experiments. The main model retains its research filename `PP-SGAT.py`.

**Research data, pretrained weights, saved predictions and experimental results
are not included.** Without authorized data, readers can inspect all source,
view command help/plans, and run the synthetic tests. Reproducing the paper's
numbers requires the original data and the documented training stages.

## Installation

The validation environment uses Python 3.9.24 and the direct dependency versions
in `requirements.txt`. The tested PyTorch build is 2.5.1+cu121 with PyG 2.6.1.
The NumPy runtime is 1.26.4; a stale distribution record reports 1.26.3.
The requirements and new experiment manifests use loaded runtime versions.
`environment-tested.json` retains both values where they differ.
CPU execution is supported; matching GPU and library versions matter for numeric
comparisons. A fresh dependency installation has not been tested in this release
preparation; the included tests run in the existing research environment.

Create and activate a Python 3.9 environment. For example, with conda:

```bash
conda create -n spgat python=3.9.24 pip=25.2
conda activate spgat
```

For CPU execution on Windows or Linux:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For a compatible NVIDIA system using the tested CUDA 12.1 wheel, replace the
first pip command with:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

These wheel indexes follow the [official PyTorch 2.5.1 installation
instructions](https://pytorch.org/get-started/previous-versions/#v251).
`environment.yml` is an alternative conda environment specification; run
`conda env create -f environment.yml` from the repository root. Select the desired
PyTorch build explicitly when the platform default differs from the intended one.
The requirement `torch==2.5.1` also accepts its matching local CUDA build.

## Inspect the release without private data

```bash
python reproduce.py --plan-only
python PP-SGAT.py --help
python PP-SGAT.py --plan-only
python ablation_vi_runner.py --plan-only --device cpu
python -m unittest discover -p "test_*.py" -v
```

The plan commands do not fit models or access the research dataset. The full test
suite requires the installed scientific dependencies. It uses synthetic data and
small models, not the private dataset or full research training. Tests write and
clean temporary outputs; they do not assert the paper's performance numbers.

## Paths and data

Place authorized data at `data/AGB.csv`. See [data/README.md](data/README.md) for
the schema. All relative CLI paths are anchored to the repository root, not the
shell's current directory:

```bash
python PP-SGAT.py --data data/AGB.csv --output-root nested_holdout_outputs
```

Users may explicitly pass an external absolute input path. The source contains
no author's machine-specific data or weight paths. Saved checkpoint references
are relative to their containing provenance directory and use forward slashes.
Move checkpoints and results together, preserving their relative directory tree.
See [docs/PATHS.md](docs/PATHS.md) for the precise rules and layout.

## Reproduce experiments with authorized data

The stages below generate their own weights. No external checkpoint download or
old local experiment directory is required. Run from this release's root:

```bash
# 1. Independent nine-model main performance comparison and its summaries.
python reproduce.py --stages main --data data/AGB.csv

# 2. Train four reusable neural baselines for explanations (20 checkpoints).
python explainability_runner.py --stage train --data data/AGB.csv --device cpu

# 3. Fit missing five-model baselines (25), then run nine-model grouped PFI.
python unified_pfi_runner.py --fit-missing --data data/AGB.csv --device cpu

# 4. Main grouped SHAP on the outer samples; reuse the exact PFI checkpoints.
python grouped_shap_runner.py --data data/AGB.csv --device cpu

# 5. Recompute inner-sample SHAP rankings and evaluate outer spectral masking.
python shap_perturbation_runner.py --data data/AGB.csv --device cpu

# 6. Train base graph controls, add random mapping, and summarize nine active modes.
python ablation_vi_runner.py --train --data data/AGB.csv --device cpu

# 7. Fixed-weight deletion of edges by prior type, using base all-prior checkpoints.
python typed_edge_deletion_runner.py --data data/AGB.csv --device cpu
```

Use `--device cuda` for explanation and graph stages when appropriate. The main
comparison retains the research scripts' automatic device selection. Main
comparison and explanation checkpoints come from independent training runs and
must not be treated as interchangeable records of the same run.

`python reproduce.py --data data/AGB.csv --device cpu` runs stages 1-7 in order.
The helper stops on the first failed subprocess. Selecting individual stages does
not automatically create their prerequisites. `--stages figures` is optional and
requires all the relevant experiments to have completed; see the detailed guide.
SHAP involves many synthetic-spectrum predictions and can take substantial time.

## Main model filenames

| Model | Entry script | Output subdirectory under `nested_holdout_outputs/` |
| --- | --- | --- |
| PLSR | `model_PLSR.py` | `output_PLSR` |
| RF | `model_RF.py` | `output_RF` |
| SVR | `model_SVR.py` | `output_SVR` |
| XGBoost | `model_XGBoost.py` | `output_XGBoost` |
| MLP | `model_MLP.py` | `output_MLP` |
| 1D-ResNet | `resnet1d_agb.py` | `resnet_outputs` |
| Transformer | `transformer_agb.py` | `transformer_outputs` |
| Chain GAT | `gat_agb.py` | `gat_baseline_outputs` |
| SP-GAT / PP-SGAT | `PP-SGAT.py` | `gat_vi_outputs` |

The performance summary uses the historical label `GAT_VI` for the main model and
`GAT_Baseline` for the chain baseline. Run the main scripts directly; they remain
executable experiment entries and start training when imported with valid data.

## Evaluation protocol and scope

Five repeated nested holdouts use seeds 42, 123, 456, 789 and 2026. Each run uses
51 fit, 13 inner-validation and 16 outer-evaluation samples for the original
80-sample dataset. Neural models run 300 fixed epochs and select weights using
inner-validation loss; no early stopping or systematic hyperparameter search is
performed. Traditional models use fixed hyperparameters. Metrics are reported
per seed before their means and sample SDs are computed. Repeated outer holdouts
overlap and are not independent experimental replications.

The current ablation comprises seven prior combinations, `no_vi` and
`random_mapping`. The preserved provenance workflow first trains 65 base runs
(including an auxiliary `random_vi` control), then 25 random-mapping runs. The
final active summary contains 65 records for the nine current modes and excludes
`random_vi`. Random repetitions are averaged within seed before cross-seed summaries.

The shared modules retain some auxiliary IG/PFI/control functions because formal
entries import them. Their presence does not make every historical experiment
part of the paper. Paper interpretations and runnable commands are detailed in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Reuse and publication metadata

No software license has been selected automatically, and no author names, paper
DOI, repository URL or citation have been invented. The authors should add the
chosen `LICENSE` and accurate `CITATION.cff` before a finalized public release.
See [RELEASE_NOTES.md](RELEASE_NOTES.md) for changes and validation limitations.
