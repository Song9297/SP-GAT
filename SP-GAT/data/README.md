# Private input data

The research dataset is intentionally not distributed. This directory contains no
measurements, sample identifiers, spectral table, or raw imagery. Model weights
and experiment results are also excluded from this source release.

After obtaining authorization to use the research data, place the CSV at
`data/AGB.csv`, or pass a different file using `--data PATH`. No public download
or data-access commitment is implied by this repository.

Expected format:

- UTF-8 CSV with one header row, no index column, and one row per sample.
- Every column except the last is a reflectance measurement; its header is a
  numeric wavelength in nm. Wavelengths must be strictly increasing and unique.
- The final column is the numeric AGB target in g/m2, normally named `AGB`.
- All input values must be finite. The published experiment uses 80 samples and
  343 spectral bands spanning approximately 450 to 900 nm.
- Preserve the authorized dataset's column and row order to reproduce the same
  splits. Preprocessing and target transformations are fitted using fit rows only.

Data identity is checked using a content hash. Do not substitute synthetic data
and interpret its results as reproductions of the paper. The unit tests create
their own synthetic inputs in temporary directories and do not need this file.
