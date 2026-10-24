# Protocol: nested-holdout. Fit-only preprocessing; inner validation selects weights;
# reported metrics/attention visualizations use outer evaluation (except retraining edge selection).
"""Partial least-squares regression baseline.

Uses min(10, number_of_bands) components and five seeded nested-holdout runs.
"""
from main_experiment_cli import parse_model_cli

_CLI = parse_model_cli('PLSR', 'output_PLSR', neural=False)

import os, sys, io, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluation_protocol import (nested_holdout_split, save_split_manifest,
                                 protocol_output, OUTER_RATIO, INNER_RATIO)
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = str(_CLI.data)
OUTPUT_FOLDER = protocol_output(_SCRIPT_DIR, 'output_PLSR')
# Shared split ratios: outer 20%, inner 20% of the remaining samples.
REPEAT_SEEDS  = [42, 123, 456, 789, 2026]
MODEL_NAME    = 'PLSR'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


print("Loading data...")
df = pd.read_csv(DATA_PATH)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
n_samples, n_bands = X.shape
N_COMPONENTS = min(10, n_bands)
print(f"  {n_samples} samples × {n_bands} bands")

# ===== 2. run_once =====
def run_once(seed):
    np.random.seed(seed)
    fit_idx, inner_idx, eval_idx = nested_holdout_split(n_samples, seed)
    save_split_manifest(OUTPUT_FOLDER, seed, fit_idx, inner_idx, eval_idx, DATA_PATH, selects_weights=False)
    X_tr, X_eval = X[fit_idx], X[eval_idx]
    y_tr, y_eval = y[fit_idx], y[eval_idx]


    y_mean, y_std = float(y_tr.mean()), float(y_tr.std()) + 1e-8
    y_tr_s  = (y_tr  - y_mean) / y_std
    x_sc = StandardScaler()
    X_tr_s  = x_sc.fit_transform(X_tr)
    X_eval_s = x_sc.transform(X_eval)

    model = PLSRegression(n_components=N_COMPONENTS)
    model.fit(X_tr_s, y_tr_s)
    preds = model.predict(X_eval_s).flatten() * y_std + y_mean


    param_count = int(model.coef_.size)

    return {
        'seed': seed,
        'r2':   r2_score(y_eval, preds),
        'rmse': np.sqrt(mean_squared_error(y_eval, preds)),
        'mae':  mean_absolute_error(y_eval, preds),
        'param_count': param_count,
        'eval_true': y_eval, 'eval_pred': preds,
    }


print(f"\nTraining {MODEL_NAME} over {len(REPEAT_SEEDS)} seeds...\n")
all_results = []
for seed in REPEAT_SEEDS:
    print(f"  [seed={seed:>3d}]", end=" ", flush=True)
    r = run_once(seed)
    all_results.append(r)
    print(f"R²={r['r2']:.4f}  RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}")


r2s   = [r['r2']   for r in all_results]
rmses = [r['rmse'] for r in all_results]
maes  = [r['mae']  for r in all_results]
r2_m, r2_s     = np.mean(r2s),   np.std(r2s)
rmse_m, rmse_s = np.mean(rmses), np.std(rmses)
mae_m, mae_s   = np.mean(maes),  np.std(maes)

print(f"\n  {MODEL_NAME} Summary:")
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
ax.set_title(f'PLSR', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=7); ax.tick_params(direction='out', labelsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'scatter_PLSR.png'), dpi=600, bbox_inches='tight')
plt.close(fig)


with open(os.path.join(OUTPUT_FOLDER, 'report.txt'), 'w', encoding='utf-8') as f:
    f.write(f"{MODEL_NAME} — Multi-Seed Results\n{'='*40}\n")
    f.write(f"Seeds: {REPEAT_SEEDS}\nOuter evaluation ratio: {OUTER_RATIO}\n")
    f.write(f"X scaler: StandardScaler (train-only)\ny scaler: train-only mean/std\n")
    f.write(f"n_components = {N_COMPONENTS}\n\n")
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
