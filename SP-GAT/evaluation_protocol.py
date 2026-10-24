"""Shared nested-holdout protocol; importing this module does not train or write files."""
import hashlib
import json
import os
from pathlib import Path

from release_paths import dataset_reference, resolve_path

import numpy as np
from sklearn.model_selection import train_test_split

OUTER_RATIO = 0.2
INNER_RATIO = 0.2
INNER_SEED_OFFSET = 10000
PROTOCOL_NAME = 'nested-holdout'


def nested_holdout_split(n_samples, seed, outer_ratio=OUTER_RATIO, inner_ratio=INNER_RATIO):
    """Return fit, inner selection and outer evaluation row indices, in original split order."""
    candidate_idx, eval_idx = train_test_split(
        np.arange(n_samples), test_size=outer_ratio, random_state=seed)
    fit_idx, inner_idx = train_test_split(
        candidate_idx, test_size=inner_ratio, random_state=seed + INNER_SEED_OFFSET)
    return fit_idx, inner_idx, eval_idx


def protocol_output(script_dir, folder):
    """Separate future nested-holdout outputs from historical 64/16 outputs."""
    output_root = os.environ.get('SPGAT_OUTPUT_ROOT')
    root = resolve_path(output_root) if output_root else Path(script_dir) / 'nested_holdout_outputs'
    return str(root / folder)


def save_split_manifest(output_dir, seed, fit_idx, inner_idx, eval_idx, data_path,
                        selects_weights=True):
    """Save the actual split and dataset fingerprint when the user runs training."""
    fit, inner, evaluation = map(lambda a: list(map(int, a)), (fit_idx, inner_idx, eval_idx))
    if set(fit) & set(inner) or set(fit) & set(evaluation) or set(inner) & set(evaluation):
        raise ValueError('Fit, inner validation and outer evaluation must be disjoint.')
    manifest = dict(protocol=PROTOCOL_NAME, seed=int(seed), inner_seed=int(seed) + INNER_SEED_OFFSET,
                    outer_ratio=OUTER_RATIO, inner_ratio=INNER_RATIO,
                    fit_idx=fit, select_idx=inner, eval_idx=evaluation,
                    n_fit=len(fit), n_inner=len(inner), n_eval=len(evaluation),
                    preprocessing_fit='fit_idx only', metrics_split='eval_idx',
                    weight_selection='inner validation loss' if selects_weights else 'fixed parameters; inner set reserved',
                    data_path=dataset_reference(data_path),
                    data_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest())
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'protocol_seed{seed}.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'  [nested-holdout seed={seed}] fit={len(fit)}, inner={len(inner)}, outer_eval={len(evaluation)}')

