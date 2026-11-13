#!/usr/bin/env python
"""Round 7: re-score all 3 WavLM seeds on VAL using batch_size=1 (no
padding at all) -- the exact code path evaluate.py uses for the real
test-set run. This supersedes the 0.7027 +/- 0.0076 figure reported in
round 6, which turned out to depend on an arbitrary historical batch
grouping from the Kaggle run (see RESULTS.md round 7 for the full
account): the frozen WavLM CNN feature encoder has no attention masking
internally, so padded zeros in a batch shift GroupNorm statistics and can
move a clip's prediction substantially depending on who else shares its
batch. batch_size=1 removes that dependency entirely -- a clip's score
depends only on its own audio and the checkpoint weights.

Determinism check: each seed's inference is run TWICE and compared
bit-for-bit before being trusted.

Usage:
    python scripts/25_wavlm_rescore_batch1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate as ev
from fer.uncertainty import bootstrap_actor_level_macro_f1

REPO_ROOT = ev.REPO_ROOT
OUT_DIR = REPO_ROOT / "outputs"
SEEDS = [0, 1, 2]
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0


def main() -> None:
    sf = ev.load_split(REPO_ROOT / ev.DATA_CFG["paths"]["splits_dir"] / "split_v1.json")
    val_actors = set(sf.assignment["val"])
    test_actors = set(sf.assignment["test"])
    metadata_all = pd.read_csv(REPO_ROOT / ev.DATA_CFG["paths"]["outputs_dir"] / "metadata.csv", dtype={"actor_id": str})
    eval_metadata = metadata_all[metadata_all["actor_id"].isin(val_actors)].copy()
    assert not (set(eval_metadata["actor_id"]) & test_actors)
    print(f"val metadata: {len(eval_metadata)} clips, {eval_metadata['actor_id'].nunique()} actors")

    device = ev.device_for_torch()
    print(f"device: {device}")

    all_rows = []
    summary_rows = []
    for seed in SEEDS:
        print(f"\nseed={seed}: running batch_size=1 inference twice to check determinism...")
        df1 = ev.fetch_wavlm_probs(eval_metadata, seed, device)
        df2 = ev.fetch_wavlm_probs(eval_metadata, seed, device)
        prob_cols = [f"prob_{c}" for c in ev.CLASS_ORDER]
        d1 = df1.set_index("clip_id")[prob_cols].to_numpy()
        d2 = df2.set_index("clip_id")[prob_cols].reindex(df1.set_index("clip_id").index).to_numpy()
        identical = np.array_equal(d1, d2)
        max_diff = float(np.max(np.abs(d1 - d2)))
        print(f"  two runs identical: {identical} (max diff: {max_diff})")
        if not identical:
            raise RuntimeError(f"seed={seed}: two batch_size=1 runs on {device} were not bit-identical (max diff {max_diff}) -- re-run on CPU instead.")

        df1["seed"] = seed
        y_pred = [ev.CLASS_ORDER[i] for i in df1[prob_cols].to_numpy().argmax(axis=1)]
        df1["y_pred"] = y_pred
        all_rows.append(df1)

        macro_f1 = f1_score(df1["y_true"], y_pred, labels=ev.CLASS_ORDER, average="macro")
        accuracy = accuracy_score(df1["y_true"], y_pred)
        per_class_f1 = dict(zip(ev.CLASS_ORDER, f1_score(df1["y_true"], y_pred, labels=ev.CLASS_ORDER, average=None)))
        cm = confusion_matrix(df1["y_true"], y_pred, labels=ev.CLASS_ORDER)

        actor_ids = df1["actor_id"].to_numpy()
        boot_scores = bootstrap_actor_level_macro_f1(actor_ids, df1["y_true"].to_numpy(), np.array(y_pred), ev.CLASS_ORDER, N_BOOTSTRAP, BOOTSTRAP_SEED)
        ci_lo, ci_hi = np.percentile(boot_scores, [2.5, 97.5])

        print(f"  macro_f1={macro_f1:.4f}  accuracy={accuracy:.4f}  bootstrap 95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  per-class F1: {{{', '.join(f'{c}={v:.4f}' for c, v in per_class_f1.items())}}}")
        pd.DataFrame(cm, index=ev.CLASS_ORDER, columns=ev.CLASS_ORDER).to_csv(OUT_DIR / f"confusion_matrix_wavlm_batch1_seed{seed}.csv")

        summary_rows.append({"seed": seed, "macro_f1": macro_f1, "accuracy": accuracy, "ci95_lo": ci_lo, "ci95_hi": ci_hi, **{f"f1_{c}": v for c, v in per_class_f1.items()}})

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df.to_csv(OUT_DIR / "wavlm_val_probs_batch1.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "wavlm_batch1_seed_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("summary over 3 seeds (batch_size=1, supersedes round 6's batched numbers):")
    print(summary_df.to_string(index=False))
    print(f"\nmacro_f1: {summary_df['macro_f1'].mean():.4f} +/- {summary_df['macro_f1'].std():.4f}")
    print(f"accuracy: {summary_df['accuracy'].mean():.4f} +/- {summary_df['accuracy'].std():.4f}")


if __name__ == "__main__":
    main()
