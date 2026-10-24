# Portable path conventions

The repository root is the directory containing `release_paths.py`. It is found
using `Path(__file__).resolve().parent`, so the location of the checkout is not
hardcoded. Absolute paths are used internally to execute local filesystem
operations; they are calculated at runtime, not copied from the author's machine.

## CLI inputs

Relative arguments such as `--data data/AGB.csv`, `--source results/source`,
`--output results/new` and `--deep-checkpoint-root explainability_outputs_v2`
resolve against the repository root, even when the process runs from another
working directory. An explicitly supplied absolute path is supported for local
use. No path configuration needs to be edited in Python source.

Main scripts accept `--output-root DIR`. Their per-model output subdirectories
are preserved. Summary utilities accept the same option. Explanation and graph
entries accept their documented `--output`, `--source`, `--pfi-root` and related
options; use each script's `--help` for its flags. Relative CLI defaults remain
within the repository. `SPGAT_OUTPUT_ROOT` can set the default main-result root.

## Persisted references

For PFI, SHAP and masking, `provenance.json` stores `checkpoint` as a relative
forward-slash string, with `path_base: provenance_directory`. Readers resolve
the reference against that JSON file's parent. References are rebased when copied
to a different output folder. Content hashes still bind the exact checkpoint,
data and source; a missing or modified artifact is rejected rather than replaced
by an unverified model. Other generated file references also use relative paths.

For example, a PFI record at
`nested_holdout_outputs/unified_pfi_v1/GAT/seed42/provenance.json` can refer to
the matching neural checkpoint as:

```text
../../../../explainability_outputs_v2/checkpoints/GAT/all_seed42_graph0.pt
```

Moving the whole tree preserves this relationship. Copying only the JSON file or
moving checkpoints separately does not. On Windows, portable artifact references
require the project, results and referenced checkpoints on the same drive. A clear error is
raised for cross-drive artifact layouts; keep them under a common directory.
External data can be on a different drive: metadata stores only a dataset label
and its content hash, not a private external directory.

Legacy manifests with absolute checkpoint references are intentionally rejected.
Use this release to regenerate the prerequisite artifacts; it has changed source
fingerprints and does not silently reuse the author's historical cache.

## Default generated layout

```text
SP-GAT/
  data/
    README.md
    AGB.csv                     # supplied privately, not in the release
  explainability_outputs_v2/
    manifest.json
    checkpoints/<model>/all_seed<seed>_graph0.pt
  nested_holdout_outputs/
    output_PLSR/ ... gat_vi_outputs/ ...
    unified_pfi_v1/
      experiment.json
      baselines/<tabular-model>/seed<seed>.joblib
      <model>/seed<seed>/provenance.json
    grouped_shap_v1/
    shap_perturbation_v1/
    ablation_controlled_v2/
    ablation_random_mapping_v1/
    ablation_active/
    typed_edge_deletion_v1/
  publication_figures/
```

The archive contains source and documentation only, not these generated trees.
