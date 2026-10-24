"""Grouped interventional Shapley estimation via antithetic permutations.

Groups are the players of the game, not sums of separately explained bands.
For a coalition S, v_x(S) = mean_b f(x_S, b_notS), with b from the fixed fit set.
This is a transparent custom permutation-SHAP estimator, not a shap-library call.
"""
import numpy as np
from scipy.stats import spearmanr


def permutation_pairs(seed, sample_id, n_groups, pairs):
    """Same orders for all models, reproducible prefixes when increasing budget."""
    return np.stack([np.random.default_rng(np.random.SeedSequence(
        [int(seed), 81000, int(sample_id), repeat])).permutation(n_groups)
        for repeat in range(pairs)])


def validate_groups(groups, n_bands):
    flat = np.concatenate(groups)
    if any(len(g) == 0 for g in groups) or not np.array_equal(np.sort(flat), np.arange(n_bands)):
        raise ValueError('Groups must partition all input columns exactly once.')


def finite_predict(predictor, raw):
    output = np.asarray(predictor.predict(raw), dtype=np.float64).reshape(-1)
    if output.shape != (len(raw),) or not np.isfinite(output).all():
        raise ValueError('Invalid/nonfinite predictions in grouped SHAP.')
    return output


def explain_sample(predictor, sample, background, groups, orders, coalition_batch=4,
                   expected_value=None, atol=1e-3, rtol=1e-5):
    """Average forward/reverse marginal contributions for each random order.

    Background integration is exact over the supplied empirical background.
    Only feature ordering is Monte Carlo approximated. Full paths telescope,
    so additivity does NOT establish sampling convergence or realistic hybrids.
    """
    sample, background = np.asarray(sample), np.asarray(background)
    if background.ndim != 2 or len(background) < 1 or sample.shape != (background.shape[1],):
        raise ValueError('Invalid sample/background shape.')
    if not np.isfinite(sample).all() or not np.isfinite(background).all():
        raise ValueError('Sample/background must be finite.')
    validate_groups(groups, len(sample))
    n = len(groups)
    if orders.ndim != 2 or orders.shape[1] != n or len(orders) < 1 or coalition_batch < 1:
        raise ValueError('Invalid permutation budget/batch.')
    if not np.all(np.sort(orders, axis=1) == np.arange(n)):
        raise ValueError('Every order must be a permutation of group ids.')
    prediction = float(finite_predict(predictor, sample[None])[0])
    if expected_value is None:
        expected_value = float(finite_predict(predictor, background).mean())
    if not np.isfinite(expected_value):
        raise ValueError('Nonfinite expected prediction.')
    full = (1 << n) - 1
    cache = {0: float(expected_value), full: prediction}
    pair_values = []
    hybrid_min, hybrid_max = float('inf'), float('-inf')
    hybrid_rows = 0
    for order in orders:
        paths = []
        for sequence in (order, order[::-1]):
            path, mask = [0], 0
            for group in sequence:
                mask |= 1 << int(group)
                path.append(mask)
            paths.append(path)
        missing = list(dict.fromkeys(mask for path in paths for mask in path if mask not in cache))
        for start in range(0, len(missing), coalition_batch):
            masks = missing[start:start + coalition_batch]
            hybrids = np.tile(background, (len(masks), 1))
            for position, mask in enumerate(masks):
                block = hybrids[position * len(background):(position + 1) * len(background)]
                for group, bands in enumerate(groups):
                    if mask & (1 << group):
                        block[:, bands] = sample[bands]
            values = finite_predict(predictor, hybrids).reshape(len(masks), len(background))
            hybrid_rows += values.size
            hybrid_min = min(hybrid_min, float(values.min()))
            hybrid_max = max(hybrid_max, float(values.max()))
            for mask, mean in zip(masks, values.mean(axis=1)):
                cache[mask] = float(mean)
        contribution = np.zeros(n, dtype=np.float64)
        for sequence, path in zip((order, order[::-1]), paths):
            changes = np.diff([cache[mask] for mask in path])
            contribution[sequence] += changes / 2.
        pair_values.append(contribution)
    pair_values = np.asarray(pair_values)
    values = pair_values.mean(axis=0)
    difference = prediction - expected_value
    residual = float(values.sum() - difference)
    tolerance = float(atol + rtol * abs(difference))
    if not np.isfinite(values).all() or abs(residual) > tolerance:
        raise ValueError(f'Grouped SHAP additivity failed: residual={residual}, tolerance={tolerance}')
    return dict(values=values, pair_values=pair_values, prediction=prediction,
                expected_value=expected_value, residual=residual, tolerance=tolerance,
                hybrid_min=hybrid_min if hybrid_rows else prediction,
                hybrid_max=hybrid_max if hybrid_rows else prediction,
                hybrid_rows=hybrid_rows, unique_coalitions=len(cache))


def ranking_similarity(first, second):
    if np.ptp(first) == 0 or np.ptp(second) == 0:
        return dict(spearman=None, top20_overlap=None, valid=False)
    k = max(1, int(np.ceil(.2 * len(first))))
    left, right = np.argsort(-first, kind='stable')[:k], np.argsort(-second, kind='stable')[:k]
    return dict(spearman=float(spearmanr(first, second).statistic),
                top20_overlap=len(set(left) & set(right)) / k, valid=True)


def sampling_diagnostics(pair_values):
    """Compare halves of paired paths, not the antithetic directions themselves."""
    # Shape: samples x random pairs x groups.
    if pair_values.ndim != 3 or pair_values.shape[1] < 4:
        raise ValueError('At least four permutation pairs required for half-budget diagnostics.')
    half = pair_values.shape[1] // 2
    first = pair_values[:, :half].mean(axis=1)
    second = pair_values[:, half:].mean(axis=1)
    denominator = float(np.mean(np.abs((first + second) / 2)))
    absolute_gap = float(np.mean(np.abs(first - second)))
    return dict(**ranking_similarity(np.abs(first).mean(0), np.abs(second).mean(0)),
                mean_absolute_half_difference=absolute_gap,
                relative_half_difference=absolute_gap / denominator if denominator > 0 else None,
                note='Diagnostic only; additivity and these summaries are not proof of convergence.')
