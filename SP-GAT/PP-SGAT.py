# Protocol: nested-holdout. Fit-only preprocessing; inner validation selects weights;
# reported metrics/attention visualizations use outer evaluation (except retraining edge selection).
"""Vegetation-index-prior GAT regression with NDVI, GNDVI and NDRE edges.

Two GAT layers, four attention heads and hidden dimension 64.
Five seeded nested-holdout runs; 300 fixed epochs with checkpoint selection by inner validation loss.
"""

from main_experiment_cli import parse_model_cli

_CLI = parse_model_cli('PP-SGAT', 'gat_vi_outputs', neural=True)

import os
import sys
import copy
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from evaluation_protocol import (nested_holdout_split, save_split_manifest,
                                 protocol_output, OUTER_RATIO, INNER_RATIO)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

# ======================================================================

# ======================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = str(_CLI.data)
OUTPUT_FOLDER = protocol_output(_SCRIPT_DIR, 'gat_vi_outputs')
MODEL_SAVE_PATH = os.path.join(OUTPUT_FOLDER, 'gat_vi_model.pth')


GAT_LAYERS       = 2
HIDDEN_DIM       = 64
ATTENTION_HEADS  = 4
DROPOUT          = 0.2


LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 1e-4
NUM_EPOCHS       = 300
BATCH_SIZE       = 32
PATIENCE         = 50
# Shared split ratios: outer 20%, inner 20% of the remaining samples.
REPEAT_SEEDS     = [42, 123, 456, 789, 2026]


PER_RELATION_TOPK = 2
EDGE_ATTR_DIM     = 4    # [one-hot(3): NDVI/GNDVI/NDRE, prior_strength(1)]
STRENGTH_COL      = 3
MEMBERSHIP_MODE   = 'gaussian'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ======================================================================

# ======================================================================
print("=" * 60)
print("  Step 1: Defining spectral region membership functions")
print("=" * 60)

REGIONS = {
    'Blue':     (450, 500),
    'Green':    (500, 570),
    'Red':      (650, 700),
    'RedEdge':  (700, 750),
    'NIR':      (750, 900),
}

REGION_SIGMAS = {
    'Blue': 18.0, 'Green': 22.0, 'Red': 18.0,
    'RedEdge': 18.0, 'NIR': 55.0,
}

def gaussian_membership(wavelengths, region_range, sigma):
    wl_min, wl_max = region_range
    center = (wl_min + wl_max) / 2.0
    membership = np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)
    membership[(wavelengths < wl_min) | (wavelengths > wl_max)] = 0.0
    return membership

# ======================================================================

# ======================================================================
print("\n" + "=" * 60)
print("  Step 2: Loading spectral data")
print("=" * 60)

data = pd.read_csv(DATA_PATH)
wavelength_cols = data.columns[:-1].tolist()
agb_col = data.columns[-1]

wavelengths = data[wavelength_cols].values
agb_values  = data[agb_col].values
n_samples, n_bands = wavelengths.shape

wavelength_values = np.array([float(col) for col in wavelength_cols])
wl_min, wl_max = wavelength_values.min(), wavelength_values.max()
normalized_wavelengths = (wavelength_values - wl_min) / (wl_max - wl_min)

print(f"  Samples: {n_samples}, Bands: {n_bands}")
print(f"  Wavelength range: {wavelength_values[0]:.2f} ~ {wavelength_values[-1]:.2f} nm")

# ======================================================================

# ======================================================================
print("\n" + "=" * 60)
print("  Step 3: Building region membership matrices")
print("=" * 60)

region_members = {}
for name, rng in REGIONS.items():
    m = gaussian_membership(wavelength_values, rng, REGION_SIGMAS[name])
    region_members[name] = m
    n_active = int((m > 0).sum())
    avg_strength = m[m > 0].mean() if n_active > 0 else 0
    print(f"  {name:>8s} ({rng[0]:.0f}-{rng[1]:.0f} nm): {n_active} bands, avg={avg_strength:.3f}")

fig, ax = plt.subplots(figsize=(10, 4))
colors = ['#2196F3', '#4CAF50', '#F44336', '#FF9800', '#9C27B0']
for idx, (name, m) in enumerate(region_members.items()):
    ax.plot(wavelength_values, m, color=colors[idx], lw=2, label=name)
    ax.fill_between(wavelength_values, 0, m, color=colors[idx], alpha=0.15)
ax.set_xlabel('Wavelength (nm)', fontsize=12); ax.set_ylabel('Membership', fontsize=12)
ax.set_title('Spectral Region Membership Functions', fontsize=13)
ax.legend(fontsize=9, ncol=5); ax.set_ylim(-0.05, 1.1); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'region_membership.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print("  [OK] region_membership.png")

# ======================================================================

# ======================================================================
print("\n" + "=" * 60)
print("  Step 4: Building VI prior matrices")
print("=" * 60)

def build_prior_matrix(m_a, m_b):
    return np.outer(m_a, m_b) + np.outer(m_b, m_a)

prior_ndvi  = build_prior_matrix(region_members['Red'],      region_members['NIR'])
prior_gndvi = build_prior_matrix(region_members['Green'],    region_members['NIR'])
prior_ndre  = build_prior_matrix(region_members['RedEdge'],  region_members['NIR'])

for name, mat in [('NDVI', prior_ndvi), ('GNDVI', prior_gndvi), ('NDRE', prior_ndre)]:
    print(f"  {name:>5s}: non-zero={int((mat > 0).sum()):>5d}, max={mat.max():.3f}, "
          f"mean={mat[mat > 0].mean():.3f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (title, mat) in zip(axes, [('NDVI', prior_ndvi), ('GNDVI', prior_gndvi),
                                    ('NDRE', prior_ndre)]):
    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd')
    ax.set_title(title, fontsize=11); ax.set_xlabel('Band j', fontsize=9)
plt.colorbar(im, ax=axes.tolist(), label='Prior Strength')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'prior_matrices.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print("  [OK] prior_matrices.png")

# ======================================================================

# ======================================================================
print("\n" + "=" * 60)
print("  Step 5: Building augmented graph topology")
print("=" * 60)


chain_edge_index = torch.zeros(2, 0, dtype=torch.long)
chain_edge_attr  = torch.zeros(0, EDGE_ATTR_DIM)
n_chain = 0

print(f"  Chain edges: {n_chain} (cancelled, VI-only topology)")

def build_region_pair_mask(m_a, m_b):
    return np.outer(m_a > 0, m_b > 0) | np.outer(m_b > 0, m_a > 0)

def sparsify_per_node(prior_mat, region_mask, topk, edge_type_id):
    candidates = {}
    for i in range(n_bands):
        valid = region_mask[i].copy()
        valid[i] = False
        if n_bands > 1:
            valid[max(0, i - 1):min(n_bands, i + 2)] = False
        row_vals = prior_mat[i].copy()
        row_vals[~valid] = 0.0
        valid_idx = np.flatnonzero(row_vals > 0)
        if valid_idx.size == 0: continue
        order = valid_idx[np.argsort(row_vals[valid_idx])[::-1]]
        for j in order[:topk]:
            candidates[(i, int(j))] = float(row_vals[j])
    for (i, j), strength in list(candidates.items()):
        rs = float(prior_mat[j, i])
        if rs > 0:
            candidates[(j, i)] = max(candidates.get((j, i), 0.0), rs)
    if not candidates:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, EDGE_ATTR_DIM)
    edges = list(candidates.keys())
    strengths = [candidates[e] for e in edges]
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    ea = torch.zeros(len(edges), EDGE_ATTR_DIM)
    ea[:, edge_type_id] = 1.0
    ea[:, STRENGTH_COL] = torch.tensor(strengths, dtype=torch.float32)
    return ei, ea

ndvi_ei, ndvi_ea   = sparsify_per_node(prior_ndvi,  build_region_pair_mask(region_members['Red'], region_members['NIR']),     PER_RELATION_TOPK, 0)
gndvi_ei, gndvi_ea = sparsify_per_node(prior_gndvi, build_region_pair_mask(region_members['Green'], region_members['NIR']),   PER_RELATION_TOPK, 1)
ndre_ei, ndre_ea   = sparsify_per_node(prior_ndre,  build_region_pair_mask(region_members['RedEdge'], region_members['NIR']), PER_RELATION_TOPK, 2)

all_edge_index = torch.cat([chain_edge_index, ndvi_ei, gndvi_ei, ndre_ei], dim=1)
all_edge_attr  = torch.cat([chain_edge_attr,  ndvi_ea,  gndvi_ea,  ndre_ea], dim=0)
assert all_edge_index.shape[1] == all_edge_attr.shape[0]

ea_mean = torch.zeros(EDGE_ATTR_DIM)
ea_std  = torch.ones(EDGE_ATTR_DIM)
ea_mean[STRENGTH_COL] = all_edge_attr[:, STRENGTH_COL].mean()
ea_std[STRENGTH_COL]  = all_edge_attr[:, STRENGTH_COL].std() + 1e-8
all_edge_attr[:, STRENGTH_COL] = (all_edge_attr[:, STRENGTH_COL] - ea_mean[STRENGTH_COL]) / ea_std[STRENGTH_COL]

n_total_edges = all_edge_index.shape[1]
print(f"  Total edges: {n_total_edges}  "
      f"(Chain={n_chain}, NDVI={ndvi_ei.shape[1]}, GNDVI={gndvi_ei.shape[1]}, "
      f"NDRE={ndre_ei.shape[1]})")

# ======================================================================

# ======================================================================
print("\n" + "=" * 60)
print("  Step 6: Defining GAT model + run_once function")
print("=" * 60)

class GATRegressorVI(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=HIDDEN_DIM,
                 num_heads=ATTENTION_HEADS, num_layers=GAT_LAYERS,
                 dropout=DROPOUT, edge_attr_dim=EDGE_ATTR_DIM):
        super().__init__()
        self.dropout = dropout
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(in_channels, hidden_dim // num_heads, num_heads,
                                       concat=True, dropout=dropout, edge_dim=edge_attr_dim,
                                       add_self_loops=True))
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim // num_heads, num_heads,
                                           concat=True, dropout=dropout, edge_dim=edge_attr_dim,
                                           add_self_loops=True))
        self.gat_layers.append(GATConv(hidden_dim, hidden_dim, 1, concat=False, dropout=dropout,
                                       edge_dim=edge_attr_dim, add_self_loops=True))
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data, return_attention=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        attn_w_list = []
        for gat_layer in self.gat_layers:
            if return_attention:
                x, (attn_ei, attn_w) = gat_layer(x, edge_index, edge_attr=edge_attr,
                                                  return_attention_weights=True)
                attn_w_list.append((attn_ei, attn_w))
            else:
                x = gat_layer(x, edge_index, edge_attr=edge_attr)
            x = nn.functional.elu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.regressor(x)
        if return_attention:
            return x.squeeze(-1), attn_w_list
        return x.squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_params = sum(p.numel() for p in GATRegressorVI().parameters())

def run_once(seed):

    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    fit_idx, inner_idx, eval_idx = nested_holdout_split(n_samples, seed)
    save_split_manifest(OUTPUT_FOLDER, seed, fit_idx, inner_idx, eval_idx, DATA_PATH)

    agb_mean = float(agb_values[fit_idx].mean())
    agb_std  = float(agb_values[fit_idx].std()) + 1e-8
    norm_y = (agb_values - agb_mean) / agb_std

    graph_list = []
    for idx in range(n_samples):
        nf = np.column_stack([normalized_wavelengths, wavelengths[idx]])
        graph_list.append(Data(
            x=torch.tensor(nf, dtype=torch.float32),
            edge_index=all_edge_index, edge_attr=all_edge_attr,
            y=torch.tensor([norm_y[idx]], dtype=torch.float32),
            sample_id=torch.tensor([idx], dtype=torch.long)
        ))

    train_graphs = [graph_list[i] for i in fit_idx]
    eval_graphs   = [graph_list[i] for i in eval_idx]
    inner_graphs = [graph_list[i] for i in inner_idx]
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader   = DataLoader(eval_graphs,   batch_size=BATCH_SIZE, shuffle=False)
    inner_loader = DataLoader(inner_graphs, batch_size=BATCH_SIZE, shuffle=False)

    model = GATRegressorVI().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    tl_list, vl_list = [], []
    best_vl, pat, best_w = float('inf'), 0, None
    for epoch in range(NUM_EPOCHS):
        model.train(); rl = 0.0; nb = 0
        for bd in train_loader:
            bd = bd.to(device); optimizer.zero_grad()
            loss = criterion(model(bd), bd.y)
            loss.backward(); optimizer.step()
            rl += loss.item() * bd.num_graphs; nb += bd.num_graphs
        tl = rl / max(len(train_loader.dataset), 1)
        tl_list.append(tl)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for bd in inner_loader:
                bd = bd.to(device)
                vl += criterion(model(bd), bd.y).item() * bd.num_graphs
        vl /= max(len(inner_loader.dataset), 1)
        vl_list.append(vl)

        if vl < best_vl:
            best_vl, pat = vl, 0
            best_w = copy.deepcopy(model.state_dict())
        else:
            pat += 1

    if best_w is not None:
        model.load_state_dict(best_w)

    model.eval(); pn, tn = [], []
    with torch.no_grad():
        for bd in eval_loader:
            bd = bd.to(device)
            pn.extend(model(bd).cpu().numpy())
            tn.extend(bd.y.cpu().numpy())
    preds   = np.array(pn) * agb_std + agb_mean
    targets = np.array(tn) * agb_std + agb_mean

    return {
        'seed': seed,
        'r2': r2_score(targets, preds),
        'rmse': np.sqrt(mean_squared_error(targets, preds)),
        'mae': mean_absolute_error(targets, preds),
        'best_inner_loss': best_vl,
        'agb_mean': agb_mean, 'agb_std': agb_std,
        'model': model, 'eval_graphs': eval_graphs,
        'history': {'train_loss': tl_list, 'val_loss': vl_list},
        'preds': preds, 'targets': targets,
    }

# ======================================================================

# ======================================================================
print(f"\n  Params: {total_params:,}")
print(f"  Seeds:  {REPEAT_SEEDS}\n")

all_results = []
for seed in REPEAT_SEEDS:
    print(f"[seed={seed:>3d}] Training...", end=" ", flush=True)
    r = run_once(seed)
    all_results.append(r)
    print(f"R²={r['r2']:.4f}  RMSE={r['rmse']:.2f}  (selected by inner loss={r['best_inner_loss']:.6f})")

# ======================================================================

# ======================================================================
r2s   = [r['r2']   for r in all_results]
rmses = [r['rmse'] for r in all_results]
maes  = [r['mae']  for r in all_results]
r2_m, r2_s     = np.mean(r2s),   np.std(r2s)
rmse_m, rmse_s = np.mean(rmses), np.std(rmses)
mae_m, mae_s   = np.mean(maes),  np.std(maes)

print("\n" + "=" * 60)
print(f"  GAT+VI Final Results ({len(REPEAT_SEEDS)} seeds, NDVI+GNDVI+NDRE only)")
print("=" * 60)
print(f"  R²:   {r2_m:.4f} ± {r2_s:.4f}")
print(f"  RMSE: {rmse_m:.4f} ± {rmse_s:.4f}  g/m²")
print(f"  MAE:  {mae_m:.4f} ± {mae_s:.4f}  g/m²")

pd.DataFrame([{
    'seed': r['seed'], 'r2': r['r2'], 'rmse': r['rmse'], 'mae': r['mae'],
    'best_inner_loss': r['best_inner_loss'], 'param_count': total_params,
} for r in all_results]).to_csv(
    os.path.join(OUTPUT_FOLDER, 'results_per_seed.csv'),
    index=False, encoding='utf-8-sig'
)

# ======================================================================

# ======================================================================
best_idx = int(np.argmax(r2s))
br = all_results[best_idx]
print(f"\n  Best seed: {br['seed']} (R²={br['r2']:.4f}) — visualizations & model save")

model = br['model']
train_losses, val_losses = br['history']['train_loss'], br['history']['val_loss']

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(train_losses) + 1), train_losses, 'b-', lw=1.5, label='Train')
ax.plot(range(1, len(val_losses) + 1),   val_losses,   'r-', lw=1.5, label='Inner validation')
ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Loss (MSE)', fontsize=12)
ax.set_title(f'GAT+VI Training (seed={br["seed"]})', fontsize=14)
ax.legend(); ax.tick_params(direction='in'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'training_curves.png'), dpi=300, bbox_inches='tight')
plt.close(fig)


fig, ax = plt.subplots(figsize=(5.5, 5))
seed_colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
for r, c in zip(all_results, seed_colors):
    ax.scatter(r['targets'], r['preds'], alpha=0.55, s=40,
               color=c, edgecolors='k', lw=0.3, label=f"seed={r['seed']}")
all_t = np.concatenate([r['targets'] for r in all_results])
all_p = np.concatenate([r['preds'] for r in all_results])
mi, ma = min(all_t.min(), all_p.min()), max(all_t.max(), all_p.max())
ax.plot([mi, ma], [mi, ma], 'r--', lw=2, label='1:1 line')
coef = np.polyfit(all_t, all_p, 1)
xx = np.linspace(mi, ma, 100)
ax.plot(xx, coef[0] * xx + coef[1], 'k-', lw=2, label='Fit line')
textstr = f'R² = {r2_m:.3f}±{r2_s:.3f}\nRMSE = {rmse_m:.2f}±{rmse_s:.2f}\nMAE = {mae_m:.2f}±{mae_s:.2f}\nN = {len(REPEAT_SEEDS)} seeds'
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.8))
ax.set_xlabel('Measured AGB (g/m²)', fontsize=12); ax.set_ylabel('Predicted AGB (g/m²)', fontsize=12)
ax.set_title(f'GAT+VI', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=7); ax.tick_params(direction='out', labelsize=12); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'prediction_scatter.png'), dpi=600, bbox_inches='tight')
plt.close(fig)


attn_sum = np.zeros((n_bands, n_bands), dtype=np.float64)
attn_count = np.zeros((n_bands, n_bands), dtype=np.int32)
for vg in br['eval_graphs']:
    vg_d = vg.to(device)
    with torch.no_grad():
        _, attn_list = model(vg_d, return_attention=True)
    attn_ei, attn_w = attn_list[-1]
    attn_ei = attn_ei.detach().cpu().numpy()
    attn_w = attn_w.detach().cpu().numpy()
    edge_attn = attn_w.mean(axis=1) if attn_w.ndim == 2 else attn_w.reshape(-1)
    for e in range(attn_ei.shape[1]):
        attn_sum[int(attn_ei[0, e]), int(attn_ei[1, e])] += float(edge_attn[e])
        attn_count[int(attn_ei[0, e]), int(attn_ei[1, e])] += 1

mean_attn = np.divide(attn_sum, np.maximum(attn_count, 1),
                      where=attn_count > 0, out=np.zeros_like(attn_sum))
base_ei = all_edge_index.cpu().numpy()
presence = np.zeros((n_bands, n_bands), dtype=bool)
for e in range(base_ei.shape[1]):
    presence[base_ei[0, e], base_ei[1, e]] = True
mean_attn[~presence] = 0.0
np.save(os.path.join(OUTPUT_FOLDER, 'mean_attention_matrix.npy'), mean_attn)


fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(mean_attn, aspect='auto', cmap='viridis')
ax.set_xlabel('Target index'); ax.set_ylabel('Source index')
ax.set_title('GAT Attention Matrix (best seed outer evaluation set)', fontsize=12)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'attention_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close(fig)


edge_records = []
for e in range(all_edge_index.shape[1]):
    src, dst = int(all_edge_index[0, e]), int(all_edge_index[1, e])
    if src == dst: continue
    etype = int(torch.argmax(all_edge_attr[e, :STRENGTH_COL]).item())
    score = float(mean_attn[src, dst])
    if score <= 0: continue
    edge_records.append({
        'source_idx': src, 'target_idx': dst,
        'source_nm': float(wavelength_values[src]),
        'target_nm': float(wavelength_values[dst]),
        'edge_type': etype, 'prior_strength': float(all_edge_attr[e, STRENGTH_COL].item()),
        'attention': score
    })

pair_dict = {}
for rec in edge_records:
    key = tuple(sorted([rec['source_idx'], rec['target_idx']])) + (rec['edge_type'],)
    pair_dict.setdefault(key, []).append(rec)

all_pair_edges, type_names = [], {0: 'NDVI', 1: 'GNDVI', 2: 'NDRE'}
for (a, b, etype), records in pair_dict.items():
    all_pair_edges.append({
        'source_idx': a, 'target_idx': b,
        'source_nm': float(wavelength_values[a]),
        'target_nm': float(wavelength_values[b]),
        'edge_type': etype, 'edge_type_name': type_names.get(etype, f'Type{etype}'),
        'prior_strength': float(np.mean([r['prior_strength'] for r in records])),
        'attention': float(np.mean([r['attention'] for r in records])),
    })


PER_TYPE_TOPK = 5
top_edges = []
for t in sorted(type_names.keys()):
    type_edges = [e for e in all_pair_edges if e['edge_type'] == t]
    type_edges = sorted(type_edges, key=lambda x: x['attention'], reverse=True)[:PER_TYPE_TOPK]
    top_edges.extend(type_edges)
top_edges = sorted(top_edges, key=lambda x: x['attention'], reverse=True)
pd.DataFrame(top_edges).to_csv(
    os.path.join(OUTPUT_FOLDER, 'top_k_spectral_relations.csv'),
    index=False, encoding='utf-8-sig'
)


TYPE_COLORS = {0: 'tab:red', 1: 'tab:green', 2: 'tab:blue'}

fig, ax = plt.subplots(figsize=(12, 6))
ax.hlines(0, wavelength_values.min(), wavelength_values.max(), lw=1.2, color='gray', zorder=1)
used = sorted(set([r['source_nm'] for r in top_edges] + [r['target_nm'] for r in top_edges]))
if used: ax.scatter(used, np.zeros(len(used)), s=35, zorder=3, color='k')
if top_edges:
    scores = np.array([r['attention'] for r in top_edges])
    sm, sM = scores.min(), scores.max()
    def w_from(v):
        return 2.0 if sM - sm < 1e-12 else 1.0 + 4.0 * (v - sm) / (sM - sm)
    wl_span = wavelength_values.max() - wavelength_values.min()
    for rank, rec in enumerate(top_edges):
        color = TYPE_COLORS.get(rec['edge_type'], 'k')
        x1, x2 = rec['source_nm'], rec['target_nm']
        lf, rt = min(x1, x2), max(x1, x2); md = (lf + rt) / 2.0
        span = max(rt - lf, 1.0); x = np.linspace(lf, rt, 120)
        h = (0.08 + 0.20 * (span / wl_span)) * (1.0 + 0.12 * (rank % 4))
        y = h * (1.0 - ((x - md) / (span / 2.0)) ** 2)
        ax.plot(x, y, lw=w_from(rec['attention']), alpha=0.65, color=color)
        if rank < 10:
            ax.text(md, y.max() + 0.012, f"{int(round(x1))}-{int(round(x2))}",
                    ha='center', va='bottom', fontsize=7, rotation=45)
ax.set_xlim(wavelength_values.min() - 5, wavelength_values.max() + 5)
ax.set_xlabel('Wavelength (nm)'); ax.grid(axis='x', alpha=0.2)

handles = [plt.Line2D([0], [0], color=c, lw=3, alpha=0.65, label=type_names[t])
           for t, c in TYPE_COLORS.items() if t in type_names]
ax.legend(handles=handles, fontsize=9, loc='upper left', framealpha=0.8)
ax.set_title(f'Top Spectral Interactions per VI type (top-{PER_TYPE_TOPK} each, seed={br["seed"]})',
             fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'top_k_spectral_arc.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# ======================================================================

# ======================================================================
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'in_channels': 2, 'hidden_dim': HIDDEN_DIM,
        'num_heads': ATTENTION_HEADS, 'num_layers': GAT_LAYERS,
        'dropout': DROPOUT, 'edge_attr_dim': EDGE_ATTR_DIM,
        'agb_mean': br['agb_mean'], 'agb_std': br['agb_std'],
        'n_bands': n_bands, 'wavelength_values': wavelength_values,
        'normalized_wavelengths': normalized_wavelengths,
        'edge_index': all_edge_index, 'edge_attr': all_edge_attr,
        'edge_attr_mean': ea_mean, 'edge_attr_std': ea_std,
        'region_members': region_members,
        'prior_matrices': {'NDVI': prior_ndvi, 'GNDVI': prior_gndvi, 'NDRE': prior_ndre},
        'per_relation_topk': PER_RELATION_TOPK, 'membership_mode': MEMBERSHIP_MODE,
        'region_sigmas': REGION_SIGMAS, 'regions': REGIONS,
    },
    'best_seed': br['seed'],
    'agb_mean': br['agb_mean'], 'agb_std': br['agb_std'],
}, MODEL_SAVE_PATH)

pd.DataFrame({
    'Actual_AGB': br['targets'], 'Predicted_AGB': br['preds'],
    'Absolute_Error': np.abs(br['targets'] - br['preds'])
}).to_csv(os.path.join(OUTPUT_FOLDER, 'prediction_results.csv'), index=False, encoding='utf-8')

with open(os.path.join(OUTPUT_FOLDER, 'metrics.txt'), 'w', encoding='utf-8') as f:
    f.write("GAT+VI Multi-Seed Results (NDVI+GNDVI+NDRE)\n" + "=" * 50 + "\n\n")
    f.write(f"  R²:   {r2_m:.4f} ± {r2_s:.4f}\n")
    f.write(f"  RMSE: {rmse_m:.4f} ± {rmse_s:.4f} g/m²\n")
    f.write(f"  MAE:  {mae_m:.4f} ± {mae_s:.4f} g/m²\n\n")
    f.write(f"  Per-seed:\n")
    for r in all_results:
        f.write(f"    seed={r['seed']:>3d}: R²={r['r2']:.4f}  RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}\n")
    f.write(f"\n  Best seed: {br['seed']}\n")
    f.write(f"  Params: {total_params:,}\n")
    f.write(f"\n  Total Edges: {n_total_edges}\n")
    f.write(f"    Chain={n_chain}, NDVI={ndvi_ei.shape[1]}, GNDVI={gndvi_ei.shape[1]}, "
            f"NDRE={ndre_ei.shape[1]}\n")
    f.write(f"\n  Edge Attr Mean: {ea_mean.tolist()}\n")
    f.write(f"  Edge Attr Std:  {ea_std.tolist()}\n")

print(f"\n  [OK] All outputs -> {OUTPUT_FOLDER}")
print("\n" + "=" * 60)
print("  GAT+VI Multi-Seed Training Complete!")
print("=" * 60)
print(f"  Final: R²={r2_m:.4f} ± {r2_s:.4f}, RMSE={rmse_m:.2f}")
print("=" * 60)
