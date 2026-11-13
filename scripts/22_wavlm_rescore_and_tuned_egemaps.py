#!/usr/bin/env python
"""Phase 5, round 6/7: re-score WavLM locally using the exact metric code
eGeMAPS was scored with (sklearn f1_score/accuracy_score/confusion_matrix
directly, no abstention since WavLM never abstains), confirm the val clip
set matches eGeMAPS exactly, and fix the eGeMAPS/WavLM tuning asymmetry by
tuning the eGeMAPS logistic regression's C on val actors (WavLM's head
hyperparameters were already chosen with early stopping on val macro-F1
inside the Kaggle notebook; eGeMAPS had never had anything tuned on val
before this).

Round 7 update: WavLM's input is now outputs/wavlm_val_probs_batch1.csv
(scripts/25_wavlm_rescore_batch1.py), NOT the original Kaggle-produced
outputs/wavlm_val_probs.csv -- the latter's numbers turned out to depend
on an arbitrary historical batch grouping (see RESULTS.md round 7), so
the "Kaggle-reported" comparison below is kept only as a labeled,
explained discrepancy, not a correctness check.

Inputs:
    outputs/wavlm_val_probs_batch1.csv -- per-clip, per-seed val probabilities, batch_size=1
    outputs/wavlm_seed_summary.csv     -- Kaggle's own self-reported val macro-F1 (batch_size=16, superseded)

Usage:
    python scripts/22_wavlm_rescore_and_tuned_egemaps.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.audio_features import feature_columns, load_features
from fer.splits import load_split
from fer.uncertainty import bootstrap_actor_level_macro_f1

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"
SEEDS = [0, 1, 2]
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def evaluate(y_val, y_pred, class_order) -> dict:
    macro_f1 = f1_score(y_val, y_pred, labels=class_order, average="macro")
    accuracy = accuracy_score(y_val, y_pred)
    per_class_f1 = f1_score(y_val, y_pred, labels=class_order, average=None)
    cm = confusion_matrix(y_val, y_pred, labels=class_order)
    return {"macro_f1": macro_f1, "accuracy": accuracy, "per_class_f1": dict(zip(class_order, per_class_f1)), "confusion_matrix": cm}


def main() -> None:
    audio_cfg = yaml.safe_load((REPO_ROOT / "configs" / "audio_baseline.yaml").read_text())
    data_cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    class_order = audio_cfg["class_order"]

    sf = load_split(REPO_ROOT / data_cfg["paths"]["splits_dir"] / "split_v1.json")
    if sf.status != "frozen":
        raise RuntimeError(f"split is not frozen (status={sf.status}) -- refusing to proceed")
    train_actors, val_actors, test_actors = set(sf.assignment["train"]), set(sf.assignment["val"]), set(sf.assignment["test"])

    # ---------------------------------------------------------------
    # Item 1: re-score WavLM
    # ---------------------------------------------------------------
    wavlm_df = pd.read_csv(OUT_DIR / "wavlm_val_probs_batch1.csv")
    wavlm_actors = {c.split("_")[0] for c in wavlm_df["clip_id"]}
    assert not wavlm_actors & test_actors, f"WavLM val probs touch test actors: {wavlm_actors & test_actors}"
    assert wavlm_actors == val_actors, f"WavLM val actors {wavlm_actors} != official val actors {val_actors}"

    features_df = load_features(REPO_ROOT / audio_cfg["features"]["cache_path"])
    assert not set(features_df["actor_id"]) & test_actors
    egemaps_val_clips = set(features_df[features_df["actor_id"].isin(val_actors)]["clip_id"])
    for seed in SEEDS:
        seed_clips = set(wavlm_df[wavlm_df["seed"] == seed]["clip_id"])
        assert seed_clips == egemaps_val_clips, (
            f"WavLM seed={seed} val clip set differs from eGeMAPS val clip set: "
            f"wavlm-only={len(seed_clips - egemaps_val_clips)} egemaps-only={len(egemaps_val_clips - seed_clips)}"
        )
    print(f"CONFIRMED: WavLM val clip set is identical to eGeMAPS val clip set "
          f"({len(egemaps_val_clips)} clips, {len(val_actors)} val actors) for all 3 seeds.")

    kaggle_summary = pd.read_csv(OUT_DIR / "wavlm_seed_summary.csv").set_index("seed")

    wavlm_results = []
    print("\n" + "=" * 70)
    print("WavLM re-scored locally (sklearn f1_score/accuracy_score/confusion_matrix,")
    print("identical metric code to the eGeMAPS script -- no abstention, WavLM never abstains)")
    print("=" * 70)
    for seed in SEEDS:
        sub = wavlm_df[wavlm_df["seed"] == seed]
        metrics = evaluate(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(), class_order)
        wavlm_results.append(metrics)

        actor_ids = sub["clip_id"].map(lambda c: c.split("_")[0]).to_numpy()
        boot_scores = bootstrap_actor_level_macro_f1(actor_ids, sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(), class_order, N_BOOTSTRAP, BOOTSTRAP_SEED)
        ci_lo, ci_hi = np.percentile(boot_scores, [2.5, 97.5])

        kaggle_mf1 = kaggle_summary.loc[seed, "val_macro_f1"]
        diff = metrics["macro_f1"] - kaggle_mf1
        print(f"\nseed={seed}: local macro_f1={metrics['macro_f1']:.4f}  accuracy={metrics['accuracy']:.4f}  (batch_size=1, reproducible)")
        print(f"  Kaggle-reported val_macro_f1={kaggle_mf1:.4f}  (diff={diff:+.4f}) -- batch_size=16, batch-order-dependent, SUPERSEDED (see RESULTS.md round 7)")
        print(f"  actor-level bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]  (bootstrap mean={boot_scores.mean():.4f}, std={boot_scores.std():.4f})")
        print("  per-class F1:")
        for cls in class_order:
            print(f"    {cls:10s}: {metrics['per_class_f1'][cls]:.4f}")
        cm_path = OUT_DIR / f"confusion_matrix_wavlm_seed{seed}.csv"
        pd.DataFrame(metrics["confusion_matrix"], index=class_order, columns=class_order).to_csv(cm_path)
        print(f"  confusion matrix written to {cm_path}")

    macro_f1s = [r["macro_f1"] for r in wavlm_results]
    accs = [r["accuracy"] for r in wavlm_results]
    print(f"\nWavLM summary over {len(SEEDS)} seeds:")
    print(f"  macro_f1: {np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s):.4f}  (per-seed: {[f'{v:.4f}' for v in macro_f1s]})")
    print(f"  accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}  (per-seed: {[f'{v:.4f}' for v in accs]})")
    print("  per-class F1 (mean +/- std across seeds):")
    for cls in class_order:
        vals = [r["per_class_f1"][cls] for r in wavlm_results]
        print(f"    {cls:10s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    wavlm_summary_df = pd.DataFrame({
        "seed": SEEDS,
        "macro_f1": macro_f1s,
        "accuracy": accs,
        "kaggle_reported_macro_f1": [kaggle_summary.loc[s, "val_macro_f1"] for s in SEEDS],
    })
    wavlm_summary_df.to_csv(OUT_DIR / "wavlm_rescored_summary.csv", index=False)

    # ---------------------------------------------------------------
    # Item 2: tune eGeMAPS LR's C on val actors
    # ---------------------------------------------------------------
    train_df = features_df[features_df["actor_id"].isin(train_actors)]
    val_df = features_df[features_df["actor_id"].isin(val_actors)]
    cols = feature_columns(features_df)
    X_train_raw = train_df[cols].to_numpy(dtype=np.float64)
    y_train = train_df["emotion"].to_numpy()
    X_val_raw = val_df[cols].to_numpy(dtype=np.float64)
    y_val = val_df["emotion"].to_numpy()

    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    untuned_C = audio_cfg["model"]["C"]
    lr_untuned = LogisticRegression(class_weight=audio_cfg["model"]["class_weight"], solver=audio_cfg["model"]["solver"], max_iter=audio_cfg["model"]["max_iter"], C=untuned_C, random_state=0).fit(X_train, y_train)
    untuned_metrics = evaluate(y_val, lr_untuned.predict(X_val), class_order)

    print("\n" + "=" * 70)
    print(f"eGeMAPS LR: tuning C on val macro-F1, grid={C_GRID}")
    print("=" * 70)
    grid_results = []
    for C in C_GRID:
        lr = LogisticRegression(class_weight=audio_cfg["model"]["class_weight"], solver=audio_cfg["model"]["solver"], max_iter=audio_cfg["model"]["max_iter"], C=C, random_state=0).fit(X_train, y_train)
        mf1 = f1_score(y_val, lr.predict(X_val), labels=class_order, average="macro")
        grid_results.append({"C": C, "val_macro_f1": mf1})
        print(f"  C={C:<8g} val_macro_f1={mf1:.4f}")

    grid_df = pd.DataFrame(grid_results)
    best_row = grid_df.loc[grid_df["val_macro_f1"].idxmax()]
    best_C = float(best_row["C"])
    lr_tuned = LogisticRegression(class_weight=audio_cfg["model"]["class_weight"], solver=audio_cfg["model"]["solver"], max_iter=audio_cfg["model"]["max_iter"], C=best_C, random_state=0).fit(X_train, y_train)
    tuned_metrics = evaluate(y_val, lr_tuned.predict(X_val), class_order)

    print(f"\nchosen C = {best_C}")
    print(f"untuned (C={untuned_C}): val macro_f1={untuned_metrics['macro_f1']:.4f}  accuracy={untuned_metrics['accuracy']:.4f}")
    print(f"tuned   (C={best_C}):  val macro_f1={tuned_metrics['macro_f1']:.4f}  accuracy={tuned_metrics['accuracy']:.4f}")
    print(f"improvement: {tuned_metrics['macro_f1'] - untuned_metrics['macro_f1']:+.4f}")
    print("(deterministic lbfgs solver on a fixed design matrix -- no seed variance, same as the original eGeMAPS script.)")

    grid_df.to_csv(OUT_DIR / "egemaps_C_tuning_grid.csv", index=False)
    tuning_summary = pd.DataFrame([
        {"variant": "untuned", "C": untuned_C, "val_macro_f1": untuned_metrics["macro_f1"], "val_accuracy": untuned_metrics["accuracy"]},
        {"variant": "tuned", "C": best_C, "val_macro_f1": tuned_metrics["macro_f1"], "val_accuracy": tuned_metrics["accuracy"]},
    ])
    tuning_summary.to_csv(OUT_DIR / "egemaps_tuned_vs_untuned.csv", index=False)
    for name, metrics in [("untuned", untuned_metrics), ("tuned", tuned_metrics)]:
        cm_path = OUT_DIR / f"confusion_matrix_egemaps_lr_{name}.csv"
        pd.DataFrame(metrics["confusion_matrix"], index=class_order, columns=class_order).to_csv(cm_path)


if __name__ == "__main__":
    main()
