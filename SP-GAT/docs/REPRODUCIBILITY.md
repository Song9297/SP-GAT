# Reproduction guide

## Main comparison and checkpoint generation

The nine main entries use `evaluation_protocol.py`, the privately supplied CSV
and their fixed research settings. Use the main entries to generate the main
comparison, and run `summarize_performance.py` for its combined table.
`collect_scatter_plots.py` collects existing figures; it is not a prediction
exporter and does not regenerate all figure styling in the submitted manuscript.

The explanation pipeline is independent. It trains neural adapters through:

```bash
python explainability_runner.py --stage train --data data/AGB.csv --device cpu
python unified_pfi_runner.py --fit-missing --data data/AGB.csv --device cpu
```

The first command produces four neural model families times five seeds, without
running IG. The second validates those checkpoints, fits missing five-model full
spectral baselines, and computes grouped PFI. Both save preprocessing/statistics
with reusable model records. Subsequent SHAP and masking stages require the
completed PFI experiment, its verified provenance and its exact checkpoints.
There is no automatic model-fitting fallback in those stages.

Main graph scripts also save a representative model and its target mean/std.
Main ResNet/Transformer save files retain their historical metadata-oriented
format; use the explanation-training command for reusable explanation weights.
Where a representative main-run seed is chosen by highest outer R2 for saving or
plotting, it is a descriptive selection and not an independent deployment-model
selection procedure. Aggregate performance still uses all five seeds.

## SHAP and fixed-weight masking

```bash
python grouped_shap_runner.py --data data/AGB.csv --device cpu
python shap_perturbation_runner.py --data data/AGB.csv --device cpu
```

The custom grouped permutation estimator treats each 10 nm wavelength group as a
Shapley participant. It uses all 51 fit samples as the empirical background and
16 forward/reverse permutation pairs by default. It is not TreeSHAP or KernelSHAP
and does not require the external `shap` package. Missing groups come from one
background spectrum at a time. The estimator is interventional, not conditional
or causal. Numerical additivity does not establish convergence or biological
plausibility; the outputs include sampling diagnostics.

The main explanation evaluates the 16 outer samples. Masking recomputes SHAP on
the 13 inner samples to determine Top/Low ordering. It replaces selected outer
reflectances with fit-set means while retaining the trained weights, preprocessing
and graph. Twenty shared random orders per seed are averaged within seed before
the five-seed summaries. This is fixed-model masking, not feature-deletion
retraining. The optional `unified_perturbation_runner.py` implements the parallel
inner-PFI-ranked masking control and is retained as a shared dependency.

Local runs can select `--models` and `--seeds`; complete-paper plots and the
numerical audit expect the full default experiment. Use a new output directory
when changing sampling budgets or source. Do not disable source or artifact-hash
checks to merge incompatible runs.

## Current graph ablation from scratch

The single command below generates its own prerequisite base results:

```bash
python ablation_vi_runner.py --train --data data/AGB.csv --device cpu
```

Its explicit equivalent is:

```bash
python controlled_graph_experiments.py ablation --data data/AGB.csv --device cpu
python ablation_random_mapping_runner.py --data data/AGB.csv
python ablation_vi_runner.py
```

The first stage trains seven prior combinations, `no_vi`, and the auxiliary
degree-preserving `random_vi` control, for 65 base runs. The second adds 25
`random_mapping` runs, each paired to the preserved base records. The third
filters to the current nine modes: seven combinations, `no_vi`, `random_mapping`.
The current table contains 40 deterministic plus 25 random-mapping records.
Omitting `--train` only regenerates that active summary from complete results.

Original model, optimization and seed settings remain unchanged. The public
workflow retains auxiliary historical base training to preserve the existing
pairing and integrity checks; it does not reinterpret `random_vi` as a current
paper control. Random mapping retains graph isomorphism while assigning graph
nodes to different wavelengths; it is not the degree-preserving rewiring null.

## Fixed-weight edge deletion

```bash
python typed_edge_deletion_runner.py --data data/AGB.csv --device cpu
```

The source is the five `all` checkpoints in the base graph experiment, not the
independently trained main performance models. Within each relation type, edge
ranking uses last-layer reciprocal attention averaged over inner-validation
samples. Top, Low and 20 Random orders are evaluated at 10%, 20%, 30% deletion on
outer samples. Both directions of a connection are removed together; other
types and model parameters remain fixed. Attention is recomputed on the modified
graph. The default has 990 modified-graph evaluations, not 990 independent fits.

## Optional analytical and publication plots

After completing all preceding stages:

```bash
python analyze_grouped_shap.py
python plot_grouped_shap_beeswarm.py --data data/AGB.csv --dpi 300
python plot_nine_shap_SP_GAT.py
python plot_shap_perturbation_publication.py
python plot_ablation_arcs_3x3_chain.py --data data/AGB.csv
python plot_ablation_arcs_type_top5.py --data data/AGB.csv
```

Equivalently use `python reproduce.py --stages figures --data data/AGB.csv`.
The attention extractor generates the saved CSV consumed by the final Top-5
per-type plot. The chain-GAT panel uses the independent explanation GAT model;
the random-mapping panel is a prespecified seed-42 graph example, not a five-seed
mean. Descriptive outer-sample attention is used for these plots; inner-sample
rankings are used for the edge-deletion evaluation. Do not conflate the two.

Portable figures use Matplotlib's bundled DejaVu fonts rather than a Windows
font directory. Font appearance may differ from the submitted figures. No
manuscript files or historical paper figure exports are included.

## What has and has not been verified

Release validation covers syntax, source-only packaging, absence of Chinese
comments/docstrings, CLI plans and missing-data behavior, synthetic algorithm
invariants, moving checkpoint/provenance trees, and rejection of modified
artifacts. See the machine-readable release validation summary supplied in this
package after preparation.

Full private-data training was not rerun for release validation. A fresh clean
environment installation and numerical identity of complete paper results have
not been established. CPU/GPU/library differences and changed source fingerprints
may require recomputing all prerequisite artifacts. Private-data unavailability
limits independent numerical reproduction; it does not prevent source inspection
or running the synthetic test suite.
