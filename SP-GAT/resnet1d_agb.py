# Protocol: nested-holdout. Fit-only preprocessing; inner validation selects weights;
# reported metrics/attention visualizations use outer evaluation (except retraining edge selection).
"""One-dimensional residual CNN baseline for spectral AGB regression.

Five seeded nested-holdout runs; 300 fixed epochs with checkpoint selection by inner validation loss.
"""

from main_experiment_cli import parse_model_cli

_CLI = parse_model_cli('ResNet1D', 'resnet_outputs', neural=True)

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
from torch.utils.data import DataLoader, TensorDataset
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
OUTPUT_FOLDER = protocol_output(_SCRIPT_DIR, 'resnet_outputs')


DROPOUT       = 0.2
RESNET_BLOCKS = [32, 32, 64, 64]
KERNEL_SIZE   = 5
USE_BN        = True


LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS    = 300
BATCH_SIZE    = 32
PATIENCE      = 50
# Shared split ratios: outer 20%, inner 20% of the remaining samples.
REPEAT_SEEDS  = [42, 123, 456, 789, 2026]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ======================================================================

# ======================================================================
print("=" * 65)
print("  1D-ResNet AGB Prediction  (vs GAT+VI Ablation)")
print("=" * 65)

data = pd.read_csv(DATA_PATH)
wavelength_cols = data.columns[:-1].tolist()
agb_col = data.columns[-1]

wavelengths = data[wavelength_cols].values   # (n_samples, n_bands)
agb_values  = data[agb_col].values            # (n_samples,)
n_samples, n_bands = wavelengths.shape

print(f"  Samples: {n_samples}, Bands: {n_bands}")
print(f"  AGB range: {agb_values.min():.2f} ~ {agb_values.max():.2f} g/m²")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======================================================================

# ======================================================================
class ResBlock1D(nn.Module):
    """Basic 1D residual block with optional downsample shortcut."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1,
                 dropout=0.0, use_bn=True):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad)
        self.bn1   = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad)
        self.bn2   = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
        self.relu  = nn.ReLU(inplace=True)
        self.dp    = nn.Dropout(dropout)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm1d(out_ch) if use_bn else nn.Identity(),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dp(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet1D(nn.Module):


    def __init__(self, in_ch=1, block_chs=[32, 32, 64, 64],
                 kernel_size=7, dropout=0.2, use_bn=True):
        super().__init__()
        self.dropout = dropout

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, block_chs[0], kernel_size, stride=2, padding=kernel_size//2),
            nn.BatchNorm1d(block_chs[0]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
        )

        blocks = []
        for i in range(len(block_chs) - 1):
            ic, oc = block_chs[i], block_chs[i + 1]
            st = 1 if ic == oc else 2
            blocks.append(ResBlock1D(ic, oc, kernel_size=3, stride=st,
                                     dropout=dropout, use_bn=use_bn))
        self.res_blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        fc_in = block_chs[-1]
        self.regressor = nn.Sequential(
            nn.Linear(fc_in, fc_in // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_in // 2, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.regressor(x).squeeze(-1)


_tmp = ResNet1D(in_ch=1, block_chs=RESNET_BLOCKS,
                kernel_size=KERNEL_SIZE, dropout=DROPOUT, use_bn=USE_BN)
TOTAL_PARAMS = sum(p.numel() for p in _tmp.parameters())
del _tmp

# ======================================================================

# ======================================================================
def run_once(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    fit_idx, inner_idx, eval_idx = nested_holdout_split(n_samples, seed)
    save_split_manifest(OUTPUT_FOLDER, seed, fit_idx, inner_idx, eval_idx, DATA_PATH)


    agb_mean = float(agb_values[fit_idx].mean())
    agb_std  = float(agb_values[fit_idx].std()) + 1e-8
    norm_y = (agb_values - agb_mean) / agb_std


    tr = wavelengths[fit_idx]
    bm, bs = tr.mean(0), tr.std(0) + 1e-8
    norm_x = (wavelengths - bm) / bs

    X_train = torch.tensor(norm_x[fit_idx, np.newaxis, :].astype(np.float32))
    y_train = torch.tensor(norm_y[fit_idx].astype(np.float32))
    X_eval   = torch.tensor(norm_x[eval_idx,   np.newaxis, :].astype(np.float32))
    y_eval   = torch.tensor(norm_y[eval_idx].astype(np.float32))

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    eval_loader   = DataLoader(TensorDataset(X_eval,   y_eval),
                              batch_size=BATCH_SIZE, shuffle=False)


    X_inner = torch.tensor(norm_x[inner_idx, np.newaxis, :].astype(np.float32))
    y_inner = torch.tensor(norm_y[inner_idx].astype(np.float32))
    inner_loader = DataLoader(TensorDataset(X_inner, y_inner),
                              batch_size=BATCH_SIZE, shuffle=False)

    model = ResNet1D(in_ch=1, block_chs=RESNET_BLOCKS,
                     kernel_size=KERNEL_SIZE, dropout=DROPOUT, use_bn=USE_BN).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    hist = {'train_loss': [], 'val_loss': []}
    best_vl, pat, best_w = float('inf'), 0, None

    for epoch in range(NUM_EPOCHS):
        model.train()
        tl, nb = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()
            tl += loss.item(); nb += 1
        tl /= max(nb, 1)

        model.eval()
        vl, nb = 0.0, 0
        with torch.no_grad():
            for xb, yb in inner_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl += criterion(model(xb), yb).item(); nb += 1
        vl /= max(nb, 1)

        hist['train_loss'].append(tl)
        hist['val_loss'].append(vl)

        if vl < best_vl:
            best_vl, pat = vl, 0
            best_w = copy.deepcopy(model.state_dict())
        else:
            pat += 1

    if best_w is not None:
        model.load_state_dict(best_w)


    model.eval()
    pn, tn = [], []
    with torch.no_grad():
        for xb, yb in eval_loader:
            pn.extend(model(xb.to(device)).cpu().numpy())
            tn.extend(yb.numpy())

    preds   = np.array(pn) * agb_std + agb_mean
    targets = np.array(tn) * agb_std + agb_mean

    return {
        'seed': seed,
        'r2': r2_score(targets, preds),
        'rmse': np.sqrt(mean_squared_error(targets, preds)),
        'mae': mean_absolute_error(targets, preds),
        'best_inner_loss': best_vl,
        'agb_mean': agb_mean, 'agb_std': agb_std,
        'history': hist, 'preds': preds, 'targets': targets,
    }

# ======================================================================

# ======================================================================
print(f"\n  Model: 1D-ResNet  (4-stage BasicBlock)")
print(f"  Params: {TOTAL_PARAMS:,}")
print(f"  Seeds:  {REPEAT_SEEDS}  →  {len(REPEAT_SEEDS)} runs\n")

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

r2_m, r2_s   = np.mean(r2s),   np.std(r2s)
rmse_m, rmse_s = np.mean(rmses), np.std(rmses)
mae_m, mae_s = np.mean(maes), np.std(maes)

print("\n" + "=" * 65)
print("  1D-ResNet Final Results  ({} seeds)".format(len(REPEAT_SEEDS)))
print("=" * 65)
print(f"  R²:   {r2_m:.4f} ± {r2_s:.4f}")
print(f"  RMSE: {rmse_m:.4f} ± {rmse_s:.4f}  g/m²")
print(f"  MAE:  {mae_m:.4f} ± {mae_s:.4f}  g/m²")

# ======================================================================

# ======================================================================
pd.DataFrame([{
    'seed': r['seed'], 'r2': r['r2'], 'rmse': r['rmse'], 'mae': r['mae'],
    'best_inner_loss': r['best_inner_loss'], 'param_count': TOTAL_PARAMS,
} for r in all_results]).to_csv(
    os.path.join(OUTPUT_FOLDER, 'results_per_seed.csv'),
    index=False, encoding='utf-8-sig'
)

metrics_path = os.path.join(OUTPUT_FOLDER, 'metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write("1D-ResNet AGB Prediction — Multi-Seed Results\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"  R²:   {r2_m:.4f} ± {r2_s:.4f}\n")
    f.write(f"  RMSE: {rmse_m:.4f} ± {rmse_s:.4f} g/m²\n")
    f.write(f"  MAE:  {mae_m:.4f} ± {mae_s:.4f} g/m²\n\n")
    f.write(f"  Per-seed:\n")
    for r in all_results:
        f.write(f"    seed={r['seed']:>3d}: R²={r['r2']:.4f}  "
                f"RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}\n")
    f.write(f"\n  Seeds: {REPEAT_SEEDS}\n")
    f.write(f"\nModel:\n")
    f.write(f"  Type: 1D-ResNet (BasicBlock, 4 stages)\n")
    f.write(f"  Channels: {RESNET_BLOCKS}, Kernel={KERNEL_SIZE}, BN={USE_BN}\n")
    f.write(f"  Params: {TOTAL_PARAMS:,}\n")
    f.write(f"\nTraining (matched to GAT+VI):\n")
    f.write(f"  Adam(lr={LEARNING_RATE}, wd={WEIGHT_DECAY}), "
            f"{NUM_EPOCHS}/{PATIENCE}, batch={BATCH_SIZE}\n")
    f.write(f"  Input:  per-band StandardScaler (train-only)\n")
    f.write(f"  Target: StandardScaler (train-only)\n")

print(f"\n  [OK] Metrics -> {metrics_path}")

# ======================================================================

# ======================================================================
best_idx = int(np.argmax(r2s))
br = all_results[best_idx]

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(br['history']['train_loss'], label='Train', color='#2196F3')
ax1.plot(br['history']['val_loss'],   label='Inner validation',   color='#F44336')
ax1.set_xlabel('Epoch', fontsize=12); ax1.set_ylabel('MSE Loss', fontsize=12)
ax1.set_title(f'1D-ResNet Training (seed={br["seed"]})', fontsize=12)
ax1.legend(fontsize=12); ax1.grid(True, alpha=0.3); ax1.tick_params(direction='out', labelsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'training_curves.png'), dpi=600, bbox_inches='tight')
plt.close(fig)


fig, ax2 = plt.subplots(figsize=(5.5, 5))
seed_colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
for r, c in zip(all_results, seed_colors):
    ax2.scatter(r['targets'], r['preds'], alpha=0.55, s=40,
                color=c, edgecolors='white', lw=0.3, label=f"seed={r['seed']}")
all_t = np.concatenate([r['targets'] for r in all_results])
all_p = np.concatenate([r['preds'] for r in all_results])
lm, lM = min(all_t.min(), all_p.min()) - 5, max(all_t.max(), all_p.max()) + 5
ax2.plot([lm, lM], [lm, lM], 'r--', lw=1.5, label='1:1 line')
coef = np.polyfit(all_t, all_p, 1)
xx = np.linspace(lm, lM, 100)
ax2.plot(xx, coef[0] * xx + coef[1], 'k-', lw=1.5, label='Fit line')
textstr = f'R² = {r2_m:.3f}±{r2_s:.3f}\nRMSE = {rmse_m:.2f}±{rmse_s:.2f}\nMAE = {mae_m:.2f}±{mae_s:.2f}\nN = {len(REPEAT_SEEDS)} seeds'
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.8))
ax2.set_xlabel('Measured AGB (g/m²)', fontsize=12)
ax2.set_ylabel('Predicted AGB (g/m²)', fontsize=12)
ax2.set_title(f'1D-ResNet', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=7); ax2.grid(True, alpha=0.3); ax2.tick_params(direction='out', labelsize=12)
ax2.set_xlim(lm, lM); ax2.set_ylim(lm, lM)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'results_scatter.png'), dpi=600, bbox_inches='tight')
plt.close(fig)

# ======================================================================

# ======================================================================
torch.save({
    'model_config': {
        'in_ch': 1, 'block_chs': RESNET_BLOCKS,
        'kernel_size': KERNEL_SIZE, 'dropout': DROPOUT, 'use_bn': USE_BN,
    },
    'best_seed': br['seed'], 'best_r2': br['r2'],
    'agb_mean': br['agb_mean'], 'agb_std': br['agb_std'],
    'total_params': TOTAL_PARAMS,
}, os.path.join(OUTPUT_FOLDER, 'resnet1d_agb.pt'))

print(f"  [OK] All outputs -> {OUTPUT_FOLDER}")
print("\n" + "=" * 65)
print("  1D-ResNet Complete!")
print("=" * 65)
