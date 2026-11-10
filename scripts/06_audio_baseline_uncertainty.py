#!/usr/bin/env python
"""Speaker-level uncertainty for the Phase 3 audio baseline.

The earlier 3-seed run showed std=0 because logistic regression with the
lbfgs solver is deterministic — that only proves the model is stable, not
that its performance would generalize to a different sample of speakers.
This script estimates genuine speaker-level uncertainty two ways, neither
of which touches the test set:

  (a) A cluster bootstrap over the 15 VAL actors (resampling actors, not
      clips, 1000 times) around the exact model from script 05 — a 95% CI
      for "how would macro-F1 on val look if we'd drawn a different sample
      of val speakers from the same population".
  (b) Actor-level 5-fold GroupKFold CV over all 79 train+val actors — a
      genuine refit-and-reevaluate estimate of generalization variability
      across different speaker splits, which the single train/val split
      cannot provide on its own.

Usage:
    python scripts/06_audio_baseline_uncertainty.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.audio_features import feature_columns, load_features
from fer.envinfo import get_env_report
from fer.splits import load_split
from fer.uncertainty import actor_level_kfold_cv, bootstrap_actor_level_macro_f1

REPO_ROOT = Path(__file__).resolve().parents[1]
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0
N_CV_FOLDS = 5


def main() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "audio_baseline.yaml").read_text())
    data_cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    class_order = cfg["class_order"]

    split_path = REPO_ROOT / data_cfg["paths"]["splits_dir"] / "split_v1.json"
    sf = load_split(split_path)
    if sf.status != "frozen":
        raise RuntimeError(f"{split_path} is not frozen — refusing to proceed")
    train_actors, val_actors, test_actors = set(sf.assignment["train"]), set(sf.assignment["val"]), set(sf.assignment["test"])

    features_df = load_features(REPO_ROOT / cfg["features"]["cache_path"])
    loaded_actors = set(features_df["actor_id"])
    assert not loaded_actors & test_actors, "test actors present in feature cache — refusing to proceed"
    assert loaded_actors == (train_actors | val_actors), "feature cache does not exactly cover train+val actors"

    cols = feature_columns(features_df)
    env = get_env_report(repo_root=str(REPO_ROOT))

    tracking_uri = cfg["mlflow"]["tracking_uri"]
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        tracking_uri = "sqlite:///" + str(REPO_ROOT / tracking_uri.removeprefix("sqlite:///"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # ---------- (a) cluster bootstrap over VAL actors ----------
    train_df = features_df[features_df["actor_id"].isin(train_actors)]
    val_df = features_df[features_df["actor_id"].isin(val_actors)]

    X_train_raw = train_df[cols].to_numpy(dtype=np.float64)
    y_train = train_df["emotion"].to_numpy()
    X_val_raw = val_df[cols].to_numpy(dtype=np.float64)
    y_val = val_df["emotion"].to_numpy()
    val_actor_ids = val_df["actor_id"].to_numpy()

    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    lr = LogisticRegression(
        class_weight=cfg["model"]["class_weight"],
        solver=cfg["model"]["solver"],
        max_iter=cfg["model"]["max_iter"],
        C=cfg["model"]["C"],
        random_state=0,
    ).fit(X_train, y_train)
    y_val_pred = lr.predict(X_val)
    point_macro_f1 = f1_score(y_val, y_val_pred, labels=class_order, average="macro")

    boot_scores = bootstrap_actor_level_macro_f1(
        val_actor_ids, y_val, y_val_pred, class_order, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
    )
    ci_lo, ci_hi = np.percentile(boot_scores, [2.5, 97.5])

    print("=" * 70)
    print(f"(a) CLUSTER BOOTSTRAP over {len(val_actors)} VAL actors, n_bootstrap={N_BOOTSTRAP}")
    print("=" * 70)
    print(f"point estimate (all 15 val actors once each): macro_f1 = {point_macro_f1:.4f}")
    print(f"bootstrap mean: {boot_scores.mean():.4f}, std: {boot_scores.std():.4f}")
    print(f"95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    val_cm = confusion_matrix(y_val, y_val_pred, labels=class_order)
    print(f"\nconfusion matrix (point estimate, true rows x pred cols, order={class_order}):")
    print(pd.DataFrame(val_cm, index=class_order, columns=class_order).to_string())

    with mlflow.start_run(run_name="val_actor_bootstrap"):
        mlflow.set_tag("model_name", "egemaps_lr")
        mlflow.set_tag("analysis", "bootstrap_val_actors")
        mlflow.log_param("n_bootstrap", N_BOOTSTRAP)
        mlflow.log_param("bootstrap_seed", BOOTSTRAP_SEED)
        mlflow.log_param("split_hash", sf.content_hash)
        mlflow.log_param("git_commit", env["git_commit"])
        mlflow.log_metric("point_macro_f1", point_macro_f1)
        mlflow.log_metric("bootstrap_mean_macro_f1", boot_scores.mean())
        mlflow.log_metric("bootstrap_std_macro_f1", boot_scores.std())
        mlflow.log_metric("ci95_lo", ci_lo)
        mlflow.log_metric("ci95_hi", ci_hi)
        cm_path = REPO_ROOT / "outputs" / "confusion_matrix_val_bootstrap_point_estimate.csv"
        pd.DataFrame(val_cm, index=class_order, columns=class_order).to_csv(cm_path)
        mlflow.log_artifact(str(cm_path))

    # ---------- (b) actor-level 5-fold GroupKFold CV over 79 train+val actors ----------
    X_all = features_df[cols].to_numpy(dtype=np.float64)
    y_all = features_df["emotion"].to_numpy()
    all_actor_ids = features_df["actor_id"].to_numpy()

    def model_factory():
        return LogisticRegression(
            class_weight=cfg["model"]["class_weight"],
            solver=cfg["model"]["solver"],
            max_iter=cfg["model"]["max_iter"],
            C=cfg["model"]["C"],
            random_state=0,
        )

    per_fold, oof_true, oof_pred, oof_actors = actor_level_kfold_cv(
        X_all, y_all, all_actor_ids, class_order, n_splits=N_CV_FOLDS, model_factory=model_factory
    )

    fold_df = pd.DataFrame(per_fold)
    print()
    print("=" * 70)
    print(f"(b) ACTOR-LEVEL {N_CV_FOLDS}-FOLD CV over {len(set(all_actor_ids))} train+val actors")
    print("=" * 70)
    print(fold_df.to_string(index=False))
    print(f"\nmacro_f1: {fold_df['macro_f1'].mean():.4f} +/- {fold_df['macro_f1'].std():.4f}")
    print(f"accuracy: {fold_df['accuracy'].mean():.4f} +/- {fold_df['accuracy'].std():.4f}")

    assert set(oof_actors) == set(all_actor_ids), "out-of-fold predictions do not cover every actor exactly once"
    oof_cm = confusion_matrix(oof_true, oof_pred, labels=class_order)
    oof_macro_f1 = f1_score(oof_true, oof_pred, labels=class_order, average="macro")
    print(f"\npooled out-of-fold macro_f1 (all 79 actors, each predicted exactly once by its held-out fold): {oof_macro_f1:.4f}")
    print(f"pooled out-of-fold confusion matrix (true rows x pred cols, order={class_order}):")
    print(pd.DataFrame(oof_cm, index=class_order, columns=class_order).to_string())

    with mlflow.start_run(run_name="actor_level_cv_summary"):
        mlflow.set_tag("model_name", "egemaps_lr")
        mlflow.set_tag("analysis", "actor_level_cv")
        mlflow.log_param("n_splits", N_CV_FOLDS)
        mlflow.log_param("n_actors", len(set(all_actor_ids)))
        mlflow.log_param("split_hash", sf.content_hash)
        mlflow.log_param("git_commit", env["git_commit"])
        mlflow.log_metric("cv_mean_macro_f1", fold_df["macro_f1"].mean())
        mlflow.log_metric("cv_std_macro_f1", fold_df["macro_f1"].std())
        mlflow.log_metric("cv_mean_accuracy", fold_df["accuracy"].mean())
        mlflow.log_metric("cv_std_accuracy", fold_df["accuracy"].std())
        mlflow.log_metric("oof_macro_f1", oof_macro_f1)
        for _, row in fold_df.iterrows():
            mlflow.log_metric(f"fold{int(row['fold'])}_macro_f1", row["macro_f1"])
        fold_csv = REPO_ROOT / "outputs" / "cv_per_fold_results.csv"
        fold_df.to_csv(fold_csv, index=False)
        mlflow.log_artifact(str(fold_csv))
        oof_cm_path = REPO_ROOT / "outputs" / "confusion_matrix_cv_out_of_fold.csv"
        pd.DataFrame(oof_cm, index=class_order, columns=class_order).to_csv(oof_cm_path)
        mlflow.log_artifact(str(oof_cm_path))


if __name__ == "__main__":
    main()
