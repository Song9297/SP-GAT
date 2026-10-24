# Protocol: nested-holdout. Fit-only preprocessing; inner validation selects weights;
# reported metrics/attention visualizations use outer evaluation (except retraining edge selection).
"""MLP baseline for spectral AGB regression.

Input projection to 64 features followed by a 64-to-32-to-1 regression head.
Five seeded nested-holdout runs; 300 fixed epochs with checkpoint selection by inner validation loss.
"""
from main_experiment_cli import parse_model_cli

_CLI = parse_model_cli('MLP', 'output_MLP', neural=True)

import os, sys, io, copy, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from evaluation_protocol import (nested_holdout_split, save_split_manifest,
                                 protocol_output, OUTER_RATIO, INNER_RATIO)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = str(_CLI.data)
OUTPUT_FOLDER = protocol_output(_SCRIPT_DIR, 'output_MLP')
# Shared split ratios: outer 20%, inner 20% of the remaining samples.
REPEAT_SEEDS  = [42, 123, 456, 789, 2026]


HIDDEN_DIM = 64
DROPOUT    = 0.2


LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS    = 300
BATCH_SIZE    = 32
PATIENCE      = 50

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print("Loading data...")
df = pd.read_csv(DATA_PATH)
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.float32)
n_samples, n_bands = X.shape
print(f"  {n_samples} samples × {n_bands} bands")


class MLPRegressor(nn.Module):


    def __init__(self, in_dim=n_bands, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        half = hidden_dim // 2
        self.net = nn.Sequential(

            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, half),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(half, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

total_params = sum(p.numel() for p in MLPRegressor().parameters())
print(f"  Params: {total_params:,}")

# ===== 3. run_once =====
def run_once(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    fit_idx, inner_idx, eval_idx = nested_holdout_split(n_samples, seed)
    save_split_manifest(OUTPUT_FOLDER, seed, fit_idx, inner_idx, eval_idx, DATA_PATH)
    X_tr, X_eval = X[fit_idx], X[eval_idx]
    y_tr, y_eval = y[fit_idx], y[eval_idx]


    y_mean, y_std = float(y_tr.mean()), float(y_tr.std()) + 1e-8
    y_tr_s  = (y_tr  - y_mean) / y_std
    y_eval_s = (y_eval - y_mean) / y_std

    x_sc = StandardScaler()
    X_tr_s  = x_sc.fit_transform(X_tr)
    X_eval_s = x_sc.transform(X_eval)

    # DataLoader
    ds_tr = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr_s, dtype=torch.float32))
    ds_eval = TensorDataset(
        torch.tensor(X_eval_s, dtype=torch.float32),
        torch.tensor(y_eval_s, dtype=torch.float32))
    loader_tr  = DataLoader(ds_tr,  batch_size=BATCH_SIZE, shuffle=True)
    loader_eval = DataLoader(ds_eval, batch_size=BATCH_SIZE, shuffle=False)

    X_inner_s = x_sc.transform(X[inner_idx])
    y_inner_s = (y[inner_idx] - y_mean) / y_std
    loader_inner = DataLoader(TensorDataset(
        torch.tensor(X_inner_s, dtype=torch.float32),
        torch.tensor(y_inner_s, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False)


    model = MLPRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


    best_vl, pat, best_w = float('inf'), 0, None
    for epoch in range(NUM_EPOCHS):
        model.train()
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()

        model.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in loader_inner:
                xb, yb = xb.to(device), yb.to(device)
                vl += criterion(model(xb), yb).item() * xb.size(0)
        vl /= len(loader_inner.dataset)

        if vl < best_vl:
            best_vl, pat = vl, 0
            best_w = copy.deepcopy(model.state_dict())
        else:
            pat += 1

    if best_w is not None:
        model.load_state_dict(best_w)


    model.eval()
    preds_norm, targets_norm = [], []
    with torch.no_grad():
        for xb, yb in loader_eval:
            xb = xb.to(device)
            preds_norm.extend(model(xb).cpu().numpy())
            targets_norm.extend(yb.numpy())

    preds   = np.array(preds_norm)   * y_std + y_mean
    targets = np.array(targets_norm) * y_std + y_mean

    return {
        'seed': seed,
        'r2':   r2_score(targets, preds),
        'rmse': np.sqrt(mean_squared_error(targets, preds)),
        'mae':  mean_absolute_error(targets, preds),
        'best_inner_loss': best_vl,
        'param_count': total_params,
        'eval_true': targets, 'eval_pred': preds,
    }


print(f"\nTraining MLP (GAT regressor head arch) over {len(REPEAT_SEEDS)} seeds...\n")
all_results = []
for seed in REPEAT_SEEDS:
    print(f"  [seed={seed:>3d}]", end=" ", flush=True)
    r = run_once(seed)
    all_results.append(r)
    print(f"R²={r['r2']:.4f}  RMSE={r['rmse']:.2f}  (selected by inner loss={r['best_inner_loss']:.6f})")


r2s   = [r['r2']   for r in all_results]
rmses = [r['rmse'] for r in all_results]
maes  = [r['mae']  for r in all_results]
r2_m, r2_s     = np.mean(r2s),   np.std(r2s)
rmse_m, rmse_s = np.mean(rmses), np.std(rmses)
mae_m, mae_s   = np.mean(maes),  np.std(maes)

print(f"\n  MLP Summary (arch = GAT regressor head):")
print(f"    R²   = {r2_m:.4f} ± {r2_s:.4f}")
print(f"    RMSE = {rmse_m:.2f} ± {rmse_s:.2f}")
print(f"    MAE  = {mae_m:.2f} ± {mae_s:.2f}")

pd.DataFrame(all_results).drop(columns=['eval_true', 'eval_pred']).to_csv(
    os.path.join(OUTPUT_FOLDER, 'results_per_seed.csv'), index=False, encoding='utf-8-sig')


plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(5.5, 5))
seed_colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
for r, c in zip(all_results, seed_colors):
    ax.scatter(r['eval_true'], r['eval_pred'], alpha=0.55, s=40,
               color=c, edgecolors='k', lw=0.3, label=f"seed={r['seed']}")
all_t = np.concatenate([r['eval_true'] for r in all_results])
all_p = np.concatenate([r['eval_pred'] for r in all_results])
mi, ma = min(all_t.min(), all_p.min()), max(all_t.max(), all_p.max())
ax.plot([mi, ma], [mi, ma], 'r--', lw=2, label='1:1 line')
coef = np.polyfit(all_t, all_p, 1)
xx = np.linspace(mi, ma, 100)
ax.plot(xx, coef[0] * xx + coef[1], 'k-', lw=2, label='Fit line')
textstr = f'R² = {r2_m:.3f}±{r2_s:.3f}\nRMSE = {rmse_m:.2f}±{rmse_s:.2f}\nMAE = {mae_m:.2f}±{mae_s:.2f}\nN = {len(REPEAT_SEEDS)} seeds'
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.8))
ax.set_xlabel('Measured AGB (g/m²)', fontsize=12)
ax.set_ylabel('Predicted AGB (g/m²)', fontsize=12)
ax.set_title(f'MLP (GAT regressor head)', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=7); ax.tick_params(direction='out', labelsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'scatter_MLP.png'), dpi=600, bbox_inches='tight')
plt.close(fig)


with open(os.path.join(OUTPUT_FOLDER, 'report.txt'), 'w', encoding='utf-8') as f:
    f.write(f"MLP (GAT regressor head arch) — Multi-Seed Results\n{'='*40}\n")
    f.write(f"Seeds: {REPEAT_SEEDS}\nOuter evaluation ratio: {OUTER_RATIO}\n")
    f.write(f"\nArchitecture (matches PP-SGAT.py regressor head):\n")
    f.write(f"  Linear(343→64) + ReLU + Dropout   # input projection\n")
    f.write(f"  Linear(64→32)  + ReLU + Dropout   # GAT regressor layer 1\n")
    f.write(f"  Linear(32→1)                      # GAT regressor layer 2\n")
    f.write(f"\nGAT regressor original:\n")
    f.write(f"  Linear(hidden_dim, hidden_dim//2) + ReLU + Dropout + Linear(hidden_dim//2, 1)\n")
    f.write(f"  = Linear(64→32) + ReLU + Dropout + Linear(32→1)\n\n")
    f.write(f"Params: {total_params:,}\n")
    f.write(f"Optimizer: Adam(lr={LEARNING_RATE}, wd={WEIGHT_DECAY})\n")
    f.write(f"Epochs: {NUM_EPOCHS}, Patience: {PATIENCE}, Batch: {BATCH_SIZE}\n")
    f.write(f"X scaler: StandardScaler (train-only)\ny scaler: train-only mean/std\n\n")
    f.write("Protocol: nested-holdout; fit-only preprocessing; metrics: outer evaluation\n")
    f.write(f"Mean ± Std:\n")
    f.write(f"  R²   = {r2_m:.4f} ± {r2_s:.4f}\n")
    f.write(f"  RMSE = {rmse_m:.2f} ± {rmse_s:.2f}\n")
    f.write(f"  MAE  = {mae_m:.2f} ± {mae_s:.2f}\n\n")
    f.write(f"Per-seed:\n")
    for r in all_results:
        f.write(f"  seed={r['seed']:>3d}: R²={r['r2']:.4f} RMSE={r['rmse']:.2f} MAE={r['mae']:.2f}\n")

print(f"\n  [OK] Output -> {OUTPUT_FOLDER}")
print(f"  Final: R²={r2_m:.4f}±{r2_s:.4f}, RMSE={rmse_m:.2f}±{rmse_s:.2f}")
