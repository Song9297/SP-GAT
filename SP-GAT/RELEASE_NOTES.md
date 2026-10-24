# Source release notes

Prepared on 2026-10-24. The source archive contains selected research code,
English documentation, environment specifications and synthetic tests. Original
working research scripts and private experiment outputs were not overwritten.

Changes made in the release copies:

- Added main-entry CLI paths, help and dependency-free plan commands.
- Set the private default input to `data/AGB.csv` and supplied its schema only.
- Centralized repository-root CLI resolution and relative artifact references.
- Rebased checkpoint provenance when writing PFI, SHAP or masking outputs while
  retaining source/data/checkpoint hashes and completed-file checks.
- Removed historical machine-specific path compatibility and system-font paths.
- Added a complete base -> random-mapping -> active-ablation training workflow.
- Added target scaling statistics to new main graph weight exports.
- Removed Chinese comments and Chinese docstrings from Python source. English
  comments remain; generated explanatory text and filenames use English.
- Corrected inherited display labels from kg/m2 to the paper's g/m2 unit without
  rescaling any numeric target, prediction or metric.
- Allowed a completed single-seed PFI subset to retain its tables without
  attempting a multi-seed uncertainty plot.
- Recorded loaded package versions when installed distribution metadata are
  stale; the NumPy requirement follows the tested runtime version 1.26.4.
- Preserved research model architectures, training/selection loops and numeric
  hyperparameters. Packaging and path changes do not establish identical results
  across different machines or fresh training runs.

No private CSV, raw image, pretrained weight, original prediction, historical
experiment manifest, manuscript, author contact information, credentials or
local environment prefixes are intentionally distributed. Temporary synthetic
test files and Python bytecode are excluded from the archive.

The software license and paper citation remain author decisions. Add the selected
`LICENSE` and accurate `CITATION.cff` when publication metadata are finalized.
No license or citation metadata have been assigned on the authors' behalf.
