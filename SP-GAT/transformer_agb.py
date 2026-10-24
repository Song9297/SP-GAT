# Protocol: nested-holdout. Fit-only preprocessing; inner validation selects weights;
# reported metrics/attention visualizations use outer evaluation (except retraining edge selection).
"""Transformer encoder baseline for spectral AGB regression.

Five seeded nested-holdout runs; 300 fixed epochs with checkpoint selection by inner validation loss.
"""

from main_experiment_cli import parse_model_cli

_CLI = parse_model_cli('Transformer', 'transformer_outputs', neural=True)

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
OUTPUT_FOLDER = protocol_output(_SCRIPT_DIR, 'transformer_outputs')


D_MODEL       = 64
NHEADS        = 4
N_LAYERS      = 2
FF_DIM        = 128
DROPOUT       = 0.2
USE_LN        = True     # LayerNorm


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
print("  Pure Transformer AGB Prediction  (vs GAT+VI / ResNet)")
print("=" * 65)

data = pd.read_csv(DATA_PATH)
wavelength_cols = data.columns[:-1].tolist()
agb_col = data.columns[-1]

wavelengths = data[wavelength_cols].values   # (n_samples, n_bands)
agb_values  = data[agb_col].values
n_samples, n_bands = wavelengths.shape

print(f"  Samples: {n_samples}, Bands: {n_bands}")
print(f"  AGB range: {agb_values.min():.2f} ~ {agb_values.max():.2f} g/m²")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======================================================================

# ======================================================================
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderBlock(nn.Module):
    """Pre-LayerNorm Transformer Encoder block."""
    def __init__(self, d_model, nheads, ff_dim, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nheads,
            dropout=dropout, batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dp = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):

        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed)
        x = x + self.dp(attn_out)

        x = x + self.dp(self.ff(self.norm2(x)))
        if return_attention:
            return x, attn_weights
        return x


class SpectralTransformer(nn.Module):


    def __init__(self, seq_len=n_bands, d_model=64, nheads=4,
                 nlayers=2, ff_dim=128, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model


        self.embed = nn.Linear(1, d_model)
        self.pe = PositionalEncoding(d_model, max_len=seq_len + 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(d_model, nheads, ff_dim, dropout)
            for _ in range(nlayers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.regressor = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, return_attention=False):

        B = x.size(0)
        tokens = self.embed(x.unsqueeze(-1))      # (B, 343, d_model)

        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 344, d_model)
        tokens = self.pe(tokens)

        attn_list = []
        if return_attention:
            for block in self.encoder:
                tokens, aw = block(tokens, return_attention=True)
                attn_list.append(aw)
        else:
            tokens = self.encoder(tokens)

        tokens = self.final_norm(tokens)

        cls_out  = tokens[:, 0, :]                # (B, d_model)
        mean_out = tokens[:, 1:, :].mean(dim=1)   # (B, d_model)
        pooled = torch.cat([cls_out, mean_out], dim=-1)  # (B, 2*d_model)

        out = self.regressor(pooled).squeeze(-1)
        if return_attention:
            return out, attn_list
        return out


_tmp = SpectralTransformer(seq_len=n_bands, d_model=D_MODEL, nheads=NHEADS,
                           nlayers=N_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT)
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

    X_train = torch.tensor(norm_x[fit_idx].astype(np.float32))
    y_train = torch.tensor(norm_y[fit_idx].astype(np.float32))
    X_eval   = torch.tensor(norm_x[eval_idx].astype(np.float32))
    y_eval   = torch.tensor(norm_y[eval_idx].astype(np.float32))

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    eval_loader   = DataLoader(TensorDataset(X_eval,   y_eval),
                              batch_size=BATCH_SIZE, shuffle=False)

    X_inner = torch.tensor(norm_x[inner_idx].astype(np.float32))
    y_inner = torch.tensor(norm_y[inner_idx].astype(np.float32))
    inner_loader = DataLoader(TensorDataset(X_inner, y_inner),
                              batch_size=BATCH_SIZE, shuffle=False)

    model = SpectralTransformer(seq_len=n_bands, d_model=D_MODEL, nheads=NHEADS,
                                nlayers=N_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT).to(device)
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
        'model': model,
        'eval_loader': eval_loader,
        'bm': bm, 'bs': bs,  # per-band mean/std for inverse norm (if needed)
    }

# ======================================================================

# ======================================================================
print(f"\n  Model: Pure Transformer Encoder")
print(f"  d_model={D_MODEL}, heads={NHEADS}, layers={N_LAYERS}, "
      f"ff={FF_DIM}, dropout={DROPOUT}")
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
print("  Pure Transformer Final Results  ({} seeds)".format(len(REPEAT_SEEDS)))
print("=" * 65)
print(f"  R²:   {r2_m:.4f} ± {r2_s:.4f}")
print(f"  RMSE: {rmse_m:.4f} ± {rmse_s:.4f}  g/m²")
print(f"  MAE:  {mae_m:.4f} ± {mae_s:.4f}  g/m²")


pd.DataFrame([{
    'seed': r['seed'], 'r2': r['r2'], 'rmse': r['rmse'], 'mae': r['mae'],
    'best_inner_loss': r['best_inner_loss'], 'param_count': TOTAL_PARAMS,
} for r in all_results]).to_csv(
    os.path.join(OUTPUT_FOLDER, 'results_per_seed.csv'),
    index=False, encoding='utf-8-sig'
)


metrics_path = os.path.join(OUTPUT_FOLDER, 'metrics.txt')
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write("Pure Transformer AGB Prediction — Multi-Seed Results\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"  R²:   {r2_m:.4f} ± {r2_s:.4f}\n")
    f.write(f"  RMSE: {rmse_m:.4f} ± {rmse_s:.4f} g/m²\n")
    f.write(f"  MAE:  {mae_m:.4f} ± {mae_s:.4f} g/m²\n\n")
    f.write(f"  Per-seed:\n")
    for r in all_results:
        f.write(f"    seed={r['seed']:>3d}: R²={r['r2']:.4f}  "
                f"RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}\n")
    f.write(f"\nModel:\n")
    f.write(f"  Type: Transformer Encoder (Pre-LN)\n")
    f.write(f"  d_model={D_MODEL}, heads={NHEADS}, layers={N_LAYERS}, "
            f"ff={FF_DIM}, dropout={DROPOUT}\n")
    f.write(f"  Token: per-band scalar -> Linear embed + learnable PE\n")
    f.write(f"  Pool: <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> mean → FC regressor\n")
    f.write(f"  Params: {TOTAL_PARAMS:,}\n")
    f.write(f"\nTraining (matched to GAT+VI / ResNet):\n")
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
ax1.set_title(f'Transformer Training (seed={br["seed"]})', fontsize=12)
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
ax2.set_title(f'Transformer', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=7); ax2.grid(True, alpha=0.3); ax2.tick_params(direction='out', labelsize=12)
ax2.set_xlim(lm, lM); ax2.set_ylim(lm, lM)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'results_scatter.png'), dpi=600, bbox_inches='tight')
plt.close(fig)

# ======================================================================

# ======================================================================
print("\n  Generating band attention heatmaps...")
best_model = br['model']
best_model.eval()


# L = 344 (CLS token + 343 bands)
all_layer_attn = [[] for _ in range(N_LAYERS)]
with torch.no_grad():
    for xb, _ in br['eval_loader']:
        _, attn_list = best_model(xb.to(device), return_attention=True)
        for li, aw in enumerate(attn_list):
            all_layer_attn[li].append(aw.cpu())


avg_attn_per_layer = []
for li in range(N_LAYERS):
    avg_attn = torch.cat(all_layer_attn[li], dim=0).mean(dim=0).numpy()  # (344, 344)
    avg_attn_per_layer.append(avg_attn)


wavelength_values = np.array([float(c) for c in wavelength_cols])


fig, axes = plt.subplots(N_LAYERS, 1, figsize=(14, 4 * N_LAYERS), sharex=True)
if N_LAYERS == 1:
    axes = [axes]

for li, avg_attn in enumerate(avg_attn_per_layer):
    cls_to_bands = avg_attn[0, 1:]
    axes[li].plot(wavelength_values, cls_to_bands, color='#E91E63', lw=1.0)
    axes[li].fill_between(wavelength_values, 0, cls_to_bands, alpha=0.25, color='#E91E63')
    axes[li].set_ylabel('Attention Weight', fontsize=11)
    axes[li].set_title(f'Layer {li}: <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> Token → Band Attention', fontsize=12)
    axes[li].grid(True, alpha=0.3)

axes[-1].set_xlabel('Wavelength (nm)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'attention_cls_to_bands.png'),
            dpi=300, bbox_inches='tight')
plt.close(fig)
print("    [OK] attention_cls_to_bands.png")


fig, axes = plt.subplots(1, N_LAYERS, figsize=(6 * N_LAYERS, 5))
if N_LAYERS == 1:
    axes = [axes]

for li, avg_attn in enumerate(avg_attn_per_layer):

    band_attn = avg_attn[1:, 1:]   # (343, 343)
    im = axes[li].imshow(band_attn, cmap='YlOrRd', aspect='auto',
                         extent=[wavelength_values[0], wavelength_values[-1],
                                 wavelength_values[-1], wavelength_values[0]])
    axes[li].set_title(f'Layer {li}: Band-Band Self-Attention', fontsize=12)
    axes[li].set_xlabel('Key Band (nm)', fontsize=11)
    axes[li].set_ylabel('Query Band (nm)', fontsize=11)

plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label='Attention Weight')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'attention_band_band_heatmap.png'),
            dpi=300, bbox_inches='tight')
plt.close(fig)
print("    [OK] attention_band_band_heatmap.png")


avg_all = np.mean(avg_attn_per_layer, axis=0)   # (344, 344)
band_avg = avg_all[1:, 1:]                       # (343, 343)
cls_band = avg_all[0, 1:]                        # (343,)

fig = plt.figure(figsize=(14, 9))

ax1 = fig.add_axes([0.08, 0.55, 0.85, 0.38])
ax1.plot(wavelength_values, cls_band, color='#E91E63', lw=1.0)
ax1.fill_between(wavelength_values, 0, cls_band, alpha=0.25, color='#E91E63')
ax1.set_ylabel('Attention from CLS', fontsize=12)
ax1.set_title(f'CLS Token → Band Attention  (avg over {N_LAYERS} layers, best seed={br["seed"]})', fontsize=13)
ax1.grid(True, alpha=0.3)


ax2 = fig.add_axes([0.08, 0.08, 0.85, 0.38])
im2 = ax2.imshow(band_avg, cmap='YlOrRd', aspect='auto',
                 extent=[wavelength_values[0], wavelength_values[-1],
                         wavelength_values[-1], wavelength_values[0]])
ax2.set_xlabel('Key Band (nm)', fontsize=12)
ax2.set_ylabel('Query Band (nm)', fontsize=12)
ax2.set_title('Band × Band Self-Attention  (avg over all layers)', fontsize=13)
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Attention Weight')

plt.savefig(os.path.join(OUTPUT_FOLDER, 'attention_summary_heatmap.png'),
            dpi=300, bbox_inches='tight')
plt.close(fig)
print("    [OK] attention_summary_heatmap.png")


np.savez(os.path.join(OUTPUT_FOLDER, 'attention_maps.npz'),
         **{f'layer_{li}': avg_attn_per_layer[li] for li in range(N_LAYERS)},
         wavelengths=wavelength_values)
print("    [OK] attention_maps.npz")
torch.save({
    'model_config': {
        'seq_len': n_bands, 'd_model': D_MODEL, 'nheads': NHEADS,
        'nlayers': N_LAYERS, 'ff_dim': FF_DIM, 'dropout': DROPOUT,
    },
    'best_seed': br['seed'], 'best_r2': br['r2'],
    'agb_mean': br['agb_mean'], 'agb_std': br['agb_std'],
    'total_params': TOTAL_PARAMS,
}, os.path.join(OUTPUT_FOLDER, 'transformer_agb.pt'))

print(f"  [OK] All outputs -> {OUTPUT_FOLDER}")
print("\n" + "=" * 65)
print("  Pure Transformer Complete!")
print("=" * 65)
