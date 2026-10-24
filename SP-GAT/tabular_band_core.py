"""Import-safe adapters and grouped spectral experiments for five tabular models.

No original model_*.py imports: those scripts train immediately on import.
"""
import copy
import hashlib
import math

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import nn

MODELS = ('PLSR', 'RF', 'SVR', 'XGBoost', 'MLP')


def fp32_mode(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class MLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, 64), nn.ReLU(inplace=True), nn.Dropout(.2),
                                 nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Dropout(.2),
                                 nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_estimator(name, seed, n_features, n_fit, jobs):
    if name == 'PLSR':
        return PLSRegression(n_components=min(10, n_features, n_fit - 1))
    if name == 'RF':
        return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=jobs)
    if name == 'SVR':
        return SVR(kernel='rbf', C=1., epsilon=.1, gamma='scale')
    if name == 'XGBoost':
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=100, max_depth=6, learning_rate=.3,
                            random_state=seed, n_jobs=jobs,
                            objective='reg:squarederror', verbosity=0)
    raise ValueError(name)


def metrics(y, prediction):
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite predictions; refusing to write experiment metrics.')
    return dict(r2=float(r2_score(y, prediction)),
                rmse=float(np.sqrt(mean_squared_error(y, prediction))),
                mae=float(mean_absolute_error(y, prediction)))


def groups_for(wavelengths, width=10.):
    labels = np.floor((wavelengths - np.floor(wavelengths.min() / width) * width) / width).astype(int)
    return [np.flatnonzero(labels == k) for k in np.unique(labels)]


def removed_columns(order, groups, ratio):
    k = int(math.ceil(ratio * len(groups)))
    if k >= len(groups):
        raise ValueError('Deletion must leave at least one wavelength group.')
    return np.concatenate([groups[i] for i in order[:k]]) if k else np.array([], dtype=int)


def similarity(a, b):
    # Constant vectors have no meaningful ranking, including top-k overlap.
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return dict(spearman=float('nan'), top_overlap=float('nan'), valid=False)
    k = max(1, int(math.ceil(len(a) * .2)))
    left = set(np.argsort(-a, kind='stable')[:k])
    right = set(np.argsort(-b, kind='stable')[:k])
    return dict(spearman=float(spearmanr(a, b).statistic),
                top_overlap=len(left & right) / k, valid=True)


class FittedModel:
    """Accept original-width raw spectra, apply saved column selection and scaler."""
    def __init__(self, bundle, device='cpu'):
        self.bundle = bundle
        self.device = device
        if bundle['name'] == 'MLP':
            self.model = MLP(len(bundle['columns'])).to(device)
            self.model.load_state_dict(bundle['state'])
            self.model.eval()
        else:
            self.model = bundle['estimator']

    def predict(self, raw):
        b = self.bundle
        values = np.asarray(raw, dtype=np.float32 if b['name'] == 'MLP' else np.float64)
        values = b['scaler'].transform(values[:, b['columns']])
        if b['name'] == 'MLP':
            with torch.no_grad():
                pred = np.concatenate([self.model(torch.as_tensor(values[i:i + 256],
                    dtype=torch.float32, device=self.device)).cpu().numpy()
                    for i in range(0, len(values), 256)])
        else:
            pred = self.model.predict(values).reshape(-1)
        pred = pred * b['y_std'] + b['y_mean']
        if not np.isfinite(pred).all():
            raise ValueError('Nonfinite model predictions.')
        return pred


def fit_model(name, seed, x, y, fit, inner, columns, epochs=300, device='cpu', jobs=-1):
    """Never receive outer targets for fitting/selection. All statistics use fit only.

    Reduced MLPs inherit surviving columns from the same full-width initialization;
    hidden weights, epoch sample orders and dropout RNG are paired across deletions.
    """
    fp32_mode(seed)
    columns = np.asarray(columns, dtype=int)
    dtype = np.float32 if name == 'MLP' else np.float64
    raw, target = np.asarray(x, dtype=dtype), np.asarray(y, dtype=dtype)
    scaler = StandardScaler().fit(raw[fit][:, columns])
    tx = scaler.transform(raw[fit][:, columns])
    sx = scaler.transform(raw[inner][:, columns])
    ym, ys = float(target[fit].mean()), float(target[fit].std()) + 1e-8
    ty, sy = (target[fit] - ym) / ys, (target[inner] - ym) / ys
    bundle = dict(name=name, seed=int(seed), columns=columns, scaler=scaler,
                  y_mean=ym, y_std=ys, history=[], best_epoch=None)
    if name != 'MLP':
        estimator = make_estimator(name, seed, len(columns), len(fit), jobs)
        estimator.fit(tx, ty)
        bundle.update(estimator=estimator, estimator_params=estimator.get_params())
        return bundle
    full = MLP(x.shape[1])
    initial = copy.deepcopy(full.state_dict())
    initial['net.0.weight'] = initial['net.0.weight'][:, columns].clone()
    model = MLP(len(columns)).to(device)
    model.load_state_dict(initial)
    bundle['initial_state_sha256'] = hashlib.sha256(b''.join(
        v.numpy().tobytes() for v in initial.values())).hexdigest()
    fp32_mode(seed + 100000)  # independent of input width / model construction RNG
    train_x, train_y = torch.tensor(tx, device=device), torch.tensor(ty, device=device)
    select_x, select_y = torch.tensor(sx, device=device), torch.tensor(sy, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    rng = np.random.default_rng(seed + 20000)
    best, best_state = float('inf'), None
    order_hash = hashlib.sha256()
    for epoch in range(epochs):
        order = rng.permutation(len(fit))
        order_hash.update(np.asarray(fit[order], dtype=np.int64).tobytes())
        model.train()
        total = 0.
        for start in range(0, len(order), 32):
            ix = order[start:start + 32]
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(train_x[ix]), train_y[ix])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite MLP training loss.')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ix)
        model.eval()
        with torch.no_grad():
            val = nn.functional.mse_loss(model(select_x), select_y).item()
        bundle['history'].append(dict(epoch=epoch + 1, fit_loss=total / len(fit), inner_loss=val))
        if np.isfinite(val) and val < best:
            best, bundle['best_epoch'] = val, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise ValueError('MLP produced no finite validation checkpoint.')
    bundle.update(state=best_state, batch_order_sha256=order_hash.hexdigest(), best_inner_loss=best)
    return bundle


def donor_importance(predictor, raw, donors, groups, seed, repeats=20):
    """Mean |prediction change| per sample/group using joint fit-row donors.

    This is model reliance, not additive attribution, conditional importance, or
    loss-based permutation importance. No evaluation labels are used.
    """
    original = predictor.predict(raw)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 61000]))
    # Same donor identities across groups and models; replacement within group joint.
    donor_rows = rng.integers(0, len(donors), size=(repeats, len(raw)))
    scores = np.empty((repeats, len(raw), len(groups)))
    for j, group in enumerate(groups):
        changed = np.tile(raw, (repeats, 1))
        changed[:, group] = donors[donor_rows.reshape(-1)][:, group]
        pred = predictor.predict(changed).reshape(repeats, len(raw))
        scores[:, :, j] = np.abs(pred - original)
    return scores.mean(0), scores, donor_rows


def fixed_perturbation(predictor, raw, targets, sample_ids, baseline, groups, scores,
                       seed, ratios, random_repeats):
    original = predictor.predict(raw)
    reference = metrics(targets, original)
    rows, individuals = [], []
    for strategy in ('top', 'low', 'random'):
        for repeat in range(random_repeats if strategy == 'random' else 1):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 30000, repeat]))
            if strategy == 'random':
                orders = np.stack([rng.permutation(len(groups)) for _ in raw])
            else:
                orders = np.argsort(-scores if strategy == 'top' else scores, axis=1, kind='stable')
            for ratio in [0.] + list(ratios):
                changed = raw.copy()
                selections = []
                for i, order in enumerate(orders):
                    bands = removed_columns(order, groups, ratio)
                    changed[i, bands] = baseline[bands]
                    selections.append(bands)
                prediction = predictor.predict(changed) if ratio else original
                m = metrics(targets, prediction)
                common = dict(strategy=strategy, repeat=repeat, ratio=ratio,
                              n_groups=int(math.ceil(ratio * len(groups))),
                              actual_fraction=math.ceil(ratio * len(groups)) / len(groups))
                rows.append(dict(**common, **m, delta_r2=reference['r2'] - m['r2'],
                    delta_rmse=m['rmse'] - reference['rmse'], delta_mae=m['mae'] - reference['mae'],
                    prediction_change=float(np.abs(prediction - original).mean())))
                for i, sid in enumerate(sample_ids):
                    individuals.append(dict(**common, sample_id=int(sid), original=float(original[i]),
                        perturbed=float(prediction[i]), target=float(targets[i]),
                        n_bands=len(selections[i]), removed_bands=';'.join(map(str, selections[i]))))
    return rows, individuals
