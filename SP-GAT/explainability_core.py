"""Shared graph construction, model adapters and explanation measurements."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Data

from explainability_models import ResNet1D, SpectralTransformer, GATBaseline, GATRegressorVI

MODES = {
    'ndvi': ['ndvi'], 'ndre': ['ndre'], 'gndvi': ['gndvi'],
    'ndvi_ndre': ['ndvi', 'ndre'], 'ndvi_gndvi': ['ndvi', 'gndvi'],
    'ndre_gndvi': ['ndre', 'gndvi'], 'all': ['ndvi', 'ndre', 'gndvi'],
    'random_vi': [], 'no_vi': [],
}
MODEL_NAMES = ['ResNet', 'Transformer', 'GAT', 'PP-SGAT']
REGIONS = {'Green': (500, 570, 22), 'Red': (650, 700, 18),
           'RedEdge': (700, 750, 18), 'NIR': (750, 900, 55)}
RELATIONS = {'ndvi': ('Red', 'NIR', 0), 'gndvi': ('Green', 'NIR', 1),
             'ndre': ('RedEdge', 'NIR', 2)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_hash():
    root = Path(__file__).resolve().parent
    return hashlib.sha256(b''.join((root / n).read_bytes() for n in
        ['explainability_core.py', 'explainability_models.py', 'explainability_runner.py'])).hexdigest()


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prior_graph(wavelengths, mode='all', topk=2):
    """Same hard-bounded Gaussian membership / Top-K rule as PP-SGAT.py."""
    members = {}
    n = len(wavelengths)
    for key, (lo, hi, sigma) in REGIONS.items():
        m = np.exp(-0.5 * ((wavelengths - (lo + hi) / 2) / sigma) ** 2)
        m[(wavelengths < lo) | (wavelengths > hi)] = 0
        members[key] = m
    edges, attrs = [], []
    for relation in MODES[mode]:
        a, b, typ = RELATIONS[relation]
        prior = np.outer(members[a], members[b]) + np.outer(members[b], members[a])
        pairs = {}
        for i in range(n):
            row = prior[i].copy()
            row[max(0, i - 1):min(n, i + 2)] = 0
            valid = np.flatnonzero(row > 0)
            order = valid[np.argsort(row[valid])[::-1]]
            for j in order[:topk]:
                pairs[(i, int(j))] = float(row[j])
        for (i, j), s in list(pairs.items()):
            pairs[(j, i)] = max(pairs.get((j, i), 0), float(prior[j, i]))
        for (i, j), s in pairs.items():
            edges.append((i, j))
            attr = [0., 0., 0., s]
            attr[typ] = 1.
            attrs.append(attr)
    ei = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous()
    ea = torch.tensor(attrs, dtype=torch.float32).reshape(-1, 4)
    if len(ea):
        ea[:, 3] = (ea[:, 3] - ea[:, 3].mean()) / (ea[:, 3].std() + 1e-8)
    return ei, ea


def pair_records(ei, ea):
    """Return undirected pairs and their two directed edge indices."""
    records = {}
    for k, (i, j) in enumerate(ei.t().cpu().tolist()):
        if i == j:
            continue
        typ = int(ea[k, :3].argmax())
        key = (min(i, j), max(i, j), typ)
        records.setdefault(key, []).append(k)
    if any(len(v) != 2 for v in records.values()):
        raise ValueError('Expected exactly two reciprocal edges per relation pair.')
    return list(records), list(records.values())


def rewire_graph(ei, ea, n_bands, seed, swaps_per_edge=5):
    """Undirected double-edge swaps preserve each node's degree per relation.

    Preserve reciprocal edges, edge count and attribute multiset, without forcing
    original region pairs. Attribute-to-endpoint assignment is intentionally changed.
    Reject loops, adjacent edges and duplicate pairs (also across relation types).
    """
    rng = np.random.default_rng(seed)
    keys, slots = pair_records(ei, ea)
    pairs = [(a, b) for a, b, _ in keys]
    occupied = set(pairs)
    if len(occupied) != len(pairs):
        raise ValueError('Parallel relation pairs are not supported by this null model.')
    initial = set(pairs)
    successes = 0
    diagnostics = []
    for typ in range(3):
        indices = np.array([k for k, key in enumerate(keys) if key[2] == typ])
        target = swaps_per_edge * len(indices)
        done, attempts = 0, 0
        while done < target and attempts < max(1000, target * 100):
            attempts += 1
            if len(indices) < 2:
                break
            p, q = rng.choice(indices, 2, replace=False)
            a, b = pairs[p]
            c, d = pairs[q]
            if rng.random() < 0.5:
                a, b = b, a
            if rng.random() < 0.5:
                c, d = d, c
            if len({a, b, c, d}) < 4:
                continue
            u, v = tuple(sorted((a, d))), tuple(sorted((c, b)))
            if abs(u[0] - u[1]) <= 1 or abs(v[0] - v[1]) <= 1:
                continue
            if u in occupied or v in occupied:
                continue
            occupied.remove(pairs[p]); occupied.remove(pairs[q])
            occupied.update([u, v])
            pairs[p], pairs[q] = u, v
            done += 1
        if done < target:
            raise RuntimeError(f'Rewiring did not mix sufficiently: type={typ}, {done}/{target} swaps.')
        successes += done
        diagnostics.append({'type': typ, 'swaps': done, 'attempts': attempts})
    out = ei.clone()
    for (a, b), directed in zip(pairs, slots):
        out[:, directed[0]] = torch.tensor([a, b])
        out[:, directed[1]] = torch.tensor([b, a])
    # These invariants must hold for every generated random graph, not just tests.
    for typ in range(3):
        mask = ea[:, typ] == 1
        for axis in (0, 1):
            assert torch.equal(torch.bincount(ei[axis, mask], minlength=n_bands),
                               torch.bincount(out[axis, mask], minlength=n_bands))
    return out, ea.clone(), {
        'successful_swaps': successes, 'pair_overlap_fraction': len(initial & occupied) / max(1, len(initial)),
        'details': diagnostics, 'graph_seed': int(seed),
    }


def chain_graph(n):
    pairs = [(i, i + 1) for i in range(n - 1)] + [(i + 1, i) for i in range(n - 1)]
    pairs += [(i, i) for i in range(n)]
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def make_model(name, n):
    if name == 'ResNet':
        return ResNet1D(in_ch=1, block_chs=[32, 32, 64, 64], kernel_size=5, dropout=.2, use_bn=True)
    if name == 'Transformer':
        return SpectralTransformer(seq_len=n, d_model=64, nheads=4, nlayers=2, ff_dim=128, dropout=.2)
    if name == 'GAT':
        return GATBaseline()
    if name == 'PP-SGAT':
        return GATRegressorVI()
    raise ValueError(name)


class Predictor:
    """Differentiable raw-reflectance -> original-unit AGB adapter."""
    def __init__(self, model, name, wavelengths, stats, device='cpu', ei=None, ea=None):
        self.model = model.to(device)
        self.name, self.device = name, device
        self.stats = stats
        self.wl = torch.tensor((wavelengths - wavelengths.min()) / np.ptp(wavelengths),
                               dtype=torch.float32, device=device)
        self.bm = torch.as_tensor(stats['x_mean'], dtype=torch.float32, device=device)
        self.bs = torch.as_tensor(stats['x_std'], dtype=torch.float32, device=device)
        self.ei = ei.to(device) if ei is not None else None
        self.ea = ea.to(device) if ea is not None else None

    def graph_data(self, raw, ei=None, ea=None):
        b, n = raw.shape
        edge = self.ei if ei is None else ei
        attr = self.ea if ea is None else ea
        offsets = torch.arange(b, device=raw.device) * n
        batched_ei = (edge[None, :, :] + offsets[:, None, None]).permute(1, 0, 2).reshape(2, -1)
        return Data(x=torch.stack([self.wl.expand(b, -1), raw], -1).reshape(b * n, 2),
                    edge_index=batched_ei, edge_attr=attr.repeat(b, 1) if attr is not None else None,
                    batch=torch.arange(b, device=raw.device).repeat_interleave(n))

    def normalized(self, raw, ei=None, ea=None):
        if self.name in ('ResNet', 'Transformer'):
            x = (raw - self.bm) / self.bs
            return self.model(x[:, None, :] if self.name == 'ResNet' else x)
        return self.model(self.graph_data(raw, ei, ea))

    def __call__(self, raw, ei=None, ea=None):
        return self.normalized(raw, ei, ea) * self.stats['y_std'] + self.stats['y_mean']

    def predict(self, raw, ei=None, ea=None, batch_size=16):
        self.model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(raw), batch_size):
                x = torch.as_tensor(raw[start:start + batch_size], dtype=torch.float32, device=self.device)
                out.append(self(x, ei, ea).cpu().numpy())
        return np.concatenate(out)


def metrics(y, pred):
    return {'r2': float(r2_score(y, pred)), 'rmse': float(np.sqrt(mean_squared_error(y, pred))),
            'mae': float(mean_absolute_error(y, pred))}


def ig_attributions(predictor, raw, baseline, steps=64, path_batch=8):
    """Trapezoidal IG in raw reflectance; wavelength coordinates remain fixed."""
    predictor.model.eval()
    attrs, residuals = [], []
    base = torch.as_tensor(baseline, dtype=torch.float32, device=predictor.device)
    for row in raw:
        x = torch.as_tensor(row, dtype=torch.float32, device=predictor.device)
        total = torch.zeros_like(x)
        for start in range(0, steps + 1, path_batch):
            indices = torch.arange(start, min(start + path_batch, steps + 1), device=x.device)
            alpha = indices.to(x.dtype) / steps
            path = (base + alpha[:, None] * (x - base)).detach().requires_grad_(True)
            grad = torch.autograd.grad(predictor(path).sum(), path)[0]
            weights = torch.ones_like(alpha)
            weights[(indices == 0) | (indices == steps)] = .5
            total += (grad * weights[:, None]).sum(0)
        attribution = (x - base) * total / steps
        with torch.no_grad():
            difference = (predictor(x[None]) - predictor(base[None])).item()
        attrs.append(attribution.detach().cpu().numpy())
        residuals.append(float(attribution.sum().item() - difference))
    return np.stack(attrs), np.array(residuals)


def wavelength_groups(wavelengths, width):
    labels = np.floor((wavelengths - np.floor(wavelengths.min() / width) * width) / width).astype(int)
    return [np.flatnonzero(labels == k) for k in np.unique(labels)]


def group_scores(attrs, groups):
    # Sum absolute band attributions (not abs(sum), which would cancel directions).
    return np.stack([np.abs(attrs[:, g]).sum(1) for g in groups], axis=1)


def similarity(a, b, fraction=.2):
    k = max(1, int(math.ceil(len(a) * fraction)))
    rho = float(spearmanr(a, b).statistic) if np.std(a) > 0 and np.std(b) > 0 else float('nan')
    ta = set(np.argsort(-a, kind='stable')[:k])
    tb = set(np.argsort(-b, kind='stable')[:k])
    return {'spearman': rho, 'top_overlap': len(ta & tb) / k, 'top_k': k}


def attention_pair_scores(predictor, raw):
    keys, slots = pair_records(predictor.ei, predictor.ea)
    scores = []
    predictor.model.eval()
    with torch.no_grad():
        for row in raw:
            x = torch.as_tensor(row[None], dtype=torch.float32, device=predictor.device)
            _, layers = predictor.model(predictor.graph_data(x), return_attention=True)
            ei, alpha = layers[-1]
            values = alpha.mean(-1).cpu().numpy()
            lookup = {tuple(p): v for p, v in zip(ei.t().cpu().tolist(), values)}
            scores.append([(lookup[(a, b)] + lookup[(b, a)]) / 2 for a, b, _ in keys])
    return keys, slots, np.asarray(scores)
