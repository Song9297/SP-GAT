# Protocol: nested-holdout. Fit-only preprocessing; inner validation selects weights;
# reported metrics/attention visualizations use outer evaluation (except retraining edge selection).
"""Chain-topology GAT baseline for AGB regression.

Adjacent bidirectional edges and self-loops; no vegetation-index edge attributes.
Five seeded nested-holdout runs; 300 fixed epochs with checkpoint selection by inner validation loss.
"""

from main_experiment_cli import parse_model_cli

_CLI = parse_model_cli('GAT baseline', 'gat_baseline_outputs', neural=True)

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
OUTPUT_FOLDER = protocol_output(_SCRIPT_DIR, 'gat_baseline_outputs')
MODEL_SAVE_PATH = os.path.join(OUTPUT_FOLDER, 'gat_baseline_model.pth')


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

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ======================================================================

# ======================================================================
print("=" * 60)
print("  Step 1: Loading spectral data")
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
print("  Step 2: Building chain topology (adjacent bidirectional + self-loop)")
print("=" * 60)

edges = []
for i in range(n_bands - 1):
    edges.append([i, i + 1]); edges.append([i + 1, i])
for i in range(n_bands):
    edges.append([i, i])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print(f"  Edges: {edge_index.shape[1]}  (chain adjacent bidirectional + self-loop)")

# ======================================================================

# ======================================================================
print("\n" + "=" * 60)
print("  Step 3: Defining GAT model + run_once function")
print("=" * 60)

class GATBaseline(nn.Module):

    def __init__(self, in_channels=2, hidden_dim=HIDDEN_DIM,
                 num_heads=ATTENTION_HEADS, num_layers=GAT_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.dropout = dropout
        self.gat_layers = nn.ModuleList()

        self.gat_layers.append(GATConv(in_channels, hidden_dim // num_heads, num_heads,
                                       concat=True, dropout=dropout,
                                       add_self_loops=False))

        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim // num_heads, num_heads,
                                           concat=True, dropout=dropout,
                                           add_self_loops=False))

        self.gat_layers.append(GATConv(hidden_dim, hidden_dim, 1, concat=False, dropout=dropout,
                                       add_self_loops=False))
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data, return_attention=False):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        attn_w_list = []
        for gat_layer in self.gat_layers:
            if return_attention:
                x, (attn_ei, attn_w) = gat_layer(x, edge_index,
                                                  return_attention_weights=True)
                attn_w_list.append((attn_ei, attn_w))
            else:
                x = gat_layer(x, edge_index)
            x = nn.functional.elu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.regressor(x)
        if return_attention:
            return x.squeeze(-1), attn_w_list
        return x.squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_params = sum(p.numel() for p in GATBaseline().parameters())

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
            edge_index=edge_index,
            y=torch.tensor([norm_y[idx]], dtype=torch.float32),
            sample_id=torch.tensor([idx], dtype=torch.long)
        ))

    train_graphs = [graph_list[i] for i in fit_idx]
    eval_graphs   = [graph_list[i] for i in eval_idx]
    inner_graphs = [graph_list[i] for i in inner_idx]
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader   = DataLoader(eval_graphs,   batch_size=BATCH_SIZE, shuffle=False)
    inner_loader = DataLoader(inner_graphs, batch_size=BATCH_SIZE, shuffle=False)

    model = GATBaseline().to(device)
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
print(f"\n  Model: GAT Baseline (chain topology, no edge_attr)")
print(f"  Params: {total_params:,}")
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
print(f"  GAT Baseline Final Results ({len(REPEAT_SEEDS)} seeds)")
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
ax.set_title(f'GAT Baseline Training (seed={br["seed"]})', fontsize=14)
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
ax.set_title(f'GAT Baseline', fontsize=12, fontweight='bold')
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
presence = np.zeros((n_bands, n_bands), dtype=bool)
base_ei = edge_index.cpu().numpy()
for e in range(base_ei.shape[1]):
    presence[base_ei[0, e], base_ei[1, e]] = True
mean_attn[~presence] = 0.0
np.save(os.path.join(OUTPUT_FOLDER, 'mean_attention_matrix.npy'), mean_attn)


fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(mean_attn, aspect='auto', cmap='viridis')
ax.set_xlabel('Target index'); ax.set_ylabel('Source index')
ax.set_title('GAT Baseline Attention Matrix (best seed outer evaluation set)', fontsize=12)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'attention_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close(fig)


edge_records = []
for e in range(base_ei.shape[1]):
    src, dst = int(base_ei[0, e]), int(base_ei[1, e])
    if src == dst: continue
    score = float(mean_attn[src, dst])
    if score <= 0: continue
    edge_records.append((src, dst, score))

pair_dict = {}
for src, dst, score in edge_records:
    key = tuple(sorted([src, dst]))
    pair_dict.setdefault(key, []).append(score)

top_edges = [{
    'source_idx': a, 'target_idx': b,
    'source_nm': float(wavelength_values[a]),
    'target_nm': float(wavelength_values[b]),
    'edge_type': 'Chain', 'attention': float(np.mean(scores)),
} for (a, b), scores in pair_dict.items()]
top_edges = sorted(top_edges, key=lambda x: x['attention'], reverse=True)[:30]
pd.DataFrame(top_edges).to_csv(
    os.path.join(OUTPUT_FOLDER, 'top_k_spectral_relations.csv'),
    index=False, encoding='utf-8-sig'
)


fig, ax = plt.subplots(figsize=(12, 6))
ax.hlines(0, wavelength_values.min(), wavelength_values.max(), lw=1.2)
used = sorted(set([r['source_nm'] for r in top_edges] + [r['target_nm'] for r in top_edges]))
if used: ax.scatter(used, np.zeros(len(used)), s=35, zorder=3)
if top_edges:
    scores = np.array([r['attention'] for r in top_edges])
    sm, sM = scores.min(), scores.max()
    def w_from(v):
        return 2.0 if sM - sm < 1e-12 else 1.0 + 4.0 * (v - sm) / (sM - sm)
    wl_span = wavelength_values.max() - wavelength_values.min()
    for rank, rec in enumerate(top_edges):
        x1, x2 = rec['source_nm'], rec['target_nm']
        lf, rt = min(x1, x2), max(x1, x2); md = (lf + rt) / 2.0
        span = max(rt - lf, 1.0); x = np.linspace(lf, rt, 120)
        h = (0.08 + 0.20 * (span / wl_span)) * (1.0 + 0.12 * (rank % 4))
        y = h * (1.0 - ((x - md) / (span / 2.0)) ** 2)
        ax.plot(x, y, lw=w_from(rec['attention']), alpha=0.65)
        if rank < 10:
            ax.text(md, y.max() + 0.012, f"{int(round(x1))}-{int(round(x2))}",
                    ha='center', va='bottom', fontsize=7, rotation=45)
ax.set_xlim(wavelength_values.min() - 5, wavelength_values.max() + 5)
ax.set_xlabel('Wavelength (nm)'); ax.grid(axis='x', alpha=0.2)
ax.set_title(f'Top Spectral Interactions — Chain GAT (seed={br["seed"]})', fontsize=12)
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
        'dropout': DROPOUT, 'n_bands': n_bands,
        'agb_mean': br['agb_mean'], 'agb_std': br['agb_std'],
        'wavelength_values': wavelength_values,
        'normalized_wavelengths': normalized_wavelengths,
        'edge_index': edge_index,
    },
    'best_seed': br['seed'],
    'agb_mean': br['agb_mean'], 'agb_std': br['agb_std'],
}, MODEL_SAVE_PATH)

pd.DataFrame({
    'Actual_AGB': br['targets'], 'Predicted_AGB': br['preds'],
    'Absolute_Error': np.abs(br['targets'] - br['preds'])
}).to_csv(os.path.join(OUTPUT_FOLDER, 'prediction_results.csv'), index=False, encoding='utf-8')

with open(os.path.join(OUTPUT_FOLDER, 'metrics.txt'), 'w', encoding='utf-8') as f:
    f.write("GAT Baseline Multi-Seed Results (Chain Topology Only)\n" + "=" * 50 + "\n\n")
    f.write(f"  R²:   {r2_m:.4f} ± {r2_s:.4f}\n")
    f.write(f"  RMSE: {rmse_m:.4f} ± {rmse_s:.4f} g/m²\n")
    f.write(f"  MAE:  {mae_m:.4f} ± {mae_s:.4f} g/m²\n\n")
    f.write(f"  Per-seed:\n")
    for r in all_results:
        f.write(f"    seed={r['seed']:>3d}: R²={r['r2']:.4f}  RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}\n")
    f.write(f"\n  Best seed: {br['seed']}\n")
    f.write(f"  Params: {total_params:,}\n")
    f.write(f"  Topology: chain adjacent bidirectional + self-loop ({edge_index.shape[1]} edges)\n")
    f.write(f"  Edge attr: None (plain GAT, no edge_dim)\n")
    f.write(f"  Readout: global_mean_pool\n")

print(f"\n  [OK] All outputs -> {OUTPUT_FOLDER}")
print("\n" + "=" * 60)
print("  GAT Baseline Multi-Seed Training Complete!")
print("=" * 60)
print(f"  Final: R²={r2_m:.4f} ± {r2_s:.4f}, RMSE={rmse_m:.2f}")
print("=" * 60)
