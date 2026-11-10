#!/usr/bin/env python
"""Phase 3 classical audio baseline.

StandardScaler + multinomial logistic regression on openSMILE eGeMAPSv02
functionals, trained on TRAIN actors and evaluated on VAL actors only,
compared against a stratified chance-level DummyClassifier under the
identical setup. 3 seeds; every run logged to MLflow.

The test set is never loaded here — see the assertion right after the
split is loaded.

Usage:
    python scripts/05_audio_baseline_egemaps_lr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.audio_features import feature_columns, load_features
from fer.envinfo import get_env_report
from fer.splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]


def evaluate_model(model, X_val_scaled, y_val, class_order) -> dict:
    y_pred = model.predict(X_val_scaled)
    macro_f1 = f1_score(y_val, y_pred, labels=class_order, average="macro")
    accuracy = accuracy_score(y_val, y_pred)
    per_class_f1 = f1_score(y_val, y_pred, labels=class_order, average=None)
    cm = confusion_matrix(y_val, y_pred, labels=class_order)
    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "per_class_f1": dict(zip(class_order, per_class_f1)),
        "confusion_matrix": cm,
    }


def log_run(run_name: str, model_name: str, seed: int, params: dict, metrics: dict, cm: np.ndarray, class_order: list[str], env: dict):
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_name", model_name)
        mlflow.log_param("seed", seed)
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("git_commit", env["git_commit"])
        for pkg in ("opensmile", "scikit-learn", "pandas", "numpy"):
            mlflow.log_param(f"version_{pkg}", env["package_versions"].get(pkg, "n/a"))

        mlflow.log_metric("macro_f1", metrics["macro_f1"])
        mlflow.log_metric("accuracy", metrics["accuracy"])
        for cls, f1 in metrics["per_class_f1"].items():
            mlflow.log_metric(f"f1_{cls}", f1)

        cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in class_order], columns=[f"pred_{c}" for c in class_order])
        cm_path = REPO_ROOT / "outputs" / f"confusion_matrix_{model_name}_seed{seed}.csv"
        cm_df.to_csv(cm_path)
        mlflow.log_artifact(str(cm_path))


def main() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "audio_baseline.yaml").read_text())
    data_cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())

    split_path = REPO_ROOT / data_cfg["paths"]["splits_dir"] / "split_v1.json"
    sf = load_split(split_path)
    if sf.status != "frozen":
        raise RuntimeError(f"{split_path} is not frozen (status={sf.status}) — refusing to proceed")

    train_actors = set(sf.assignment["train"])
    val_actors = set(sf.assignment["val"])
    test_actors = set(sf.assignment["test"])

    features_path = REPO_ROOT / cfg["features"]["cache_path"]
    features_df = load_features(features_path)

    loaded_actors = set(features_df["actor_id"])
    assert not loaded_actors & test_actors, (
        f"test actors found in loaded feature cache: {loaded_actors & test_actors} — refusing to proceed"
    )
    assert loaded_actors <= (train_actors | val_actors), "unexpected actors in feature cache"

    train_df = features_df[features_df["actor_id"].isin(train_actors)]
    val_df = features_df[features_df["actor_id"].isin(val_actors)]
    print(f"train: {len(train_df)} clips, {train_df['actor_id'].nunique()} actors")
    print(f"val:   {len(val_df)} clips, {val_df['actor_id'].nunique()} actors")

    cols = feature_columns(features_df)
    X_train_raw = train_df[cols].to_numpy(dtype=np.float64)
    y_train = train_df["emotion"].to_numpy()
    X_val_raw = val_df[cols].to_numpy(dtype=np.float64)
    y_val = val_df["emotion"].to_numpy()

    class_order = cfg["class_order"]
    assert set(class_order) == set(np.unique(y_train)) == set(np.unique(y_val)), (
        f"class_order {class_order} does not match observed labels "
        f"train={sorted(set(y_train))} val={sorted(set(y_val))}"
    )

    env = get_env_report(repo_root=str(REPO_ROOT))

    tracking_uri = cfg["mlflow"]["tracking_uri"]
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        tracking_uri = "sqlite:///" + str(REPO_ROOT / tracking_uri.removeprefix("sqlite:///"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    lr_results, dummy_results = [], []

    for seed in cfg["seeds"]:
        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)

        lr = LogisticRegression(
            class_weight=cfg["model"]["class_weight"],
            solver=cfg["model"]["solver"],
            max_iter=cfg["model"]["max_iter"],
            C=cfg["model"]["C"],
            random_state=seed,
        ).fit(X_train, y_train)
        lr_metrics = evaluate_model(lr, X_val, y_val, class_order)
        lr_results.append(lr_metrics)
        log_run(
            f"egemaps_lr_seed{seed}", "egemaps_lr", seed,
            {"split_hash": sf.content_hash, "n_features": len(cols), "n_train": len(train_df), "n_val": len(val_df), **cfg["model"]},
            lr_metrics, lr_metrics["confusion_matrix"], class_order, env,
        )

        dummy = DummyClassifier(strategy="stratified", random_state=seed).fit(X_train, y_train)
        dummy_metrics = evaluate_model(dummy, X_val, y_val, class_order)
        dummy_results.append(dummy_metrics)
        log_run(
            f"dummy_stratified_seed{seed}", "dummy_stratified", seed,
            {"split_hash": sf.content_hash, "n_train": len(train_df), "n_val": len(val_df), "strategy": "stratified"},
            dummy_metrics, dummy_metrics["confusion_matrix"], class_order, env,
        )

    def summarize(results: list[dict], name: str) -> None:
        macro_f1s = [r["macro_f1"] for r in results]
        accs = [r["accuracy"] for r in results]
        print(f"\n=== {name} (n_seeds={len(results)}) ===")
        print(f"macro_f1: {np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s):.4f}  (per-seed: {[f'{v:.4f}' for v in macro_f1s]})")
        print(f"accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}  (per-seed: {[f'{v:.4f}' for v in accs]})")
        print("per-class F1 (mean +/- std across seeds):")
        for cls in class_order:
            vals = [r["per_class_f1"][cls] for r in results]
            print(f"  {cls:10s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
        print(f"confusion matrix (seed {cfg['seeds'][0]}, true rows x pred cols, order={class_order}):")
        print(pd.DataFrame(results[0]["confusion_matrix"], index=class_order, columns=class_order).to_string())

    summarize(lr_results, "eGeMAPS + StandardScaler + LogisticRegression (balanced)")
    summarize(dummy_results, "Chance baseline: DummyClassifier(strategy='stratified')")

    print("\nNote: logistic regression uses the deterministic 'lbfgs' solver on a fixed")
    print("design matrix, so its 3-seed std is expected to be ~0 (see explanation before")
    print("this script was written). The dummy baseline's std reflects genuine stochasticity.")


if __name__ == "__main__":
    main()
