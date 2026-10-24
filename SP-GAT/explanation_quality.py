"""Pure array quality checks; no model execution."""
import numpy as np


def quality_arrays(ig, residual, atol=.1, rtol=.05, differences=None):
    ig = np.asarray(ig)
    residual = np.asarray(residual)
    if ig.ndim != 2 or residual.shape != (len(ig),):
        raise ValueError('IG/residual shapes do not match.')
    # Older files store residual = sum(IG) - prediction_difference.
    recovered = ig.sum(axis=1) - residual
    diff = recovered if differences is None else np.asarray(differences)
    if diff.shape != residual.shape or not np.allclose(diff, recovered, atol=1e-3, rtol=1e-5):
        raise ValueError('Stored prediction differences do not match IG residuals.')
    tolerance = atol + rtol * np.abs(diff)
    finite = np.isfinite(ig).all(axis=1) & np.isfinite(residual) & np.isfinite(diff)
    return dict(prediction_minus_baseline=diff, ig_sum=ig.sum(axis=1),
                completeness_residual=residual, tolerance=tolerance,
                passed=finite & (np.abs(residual) <= tolerance))


def upsert_performance(path, rows):
    """Partial runs replace only matching model/seed records."""
    import pandas as pd
    new = pd.DataFrame(rows)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new = new.drop_duplicates(['model', 'seed'], keep='last').sort_values(['model','seed'])
    temp = path.with_suffix('.tmp')
    new.to_csv(temp, index=False)
    temp.replace(path)
