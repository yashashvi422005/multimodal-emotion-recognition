#!/usr/bin/env python
"""Phase 4, first version: fine-tune ResNet-18 (timm, ImageNet weights) on
clean face crops. Frame-level training (TRAIN actors, 4 subsampled frames
per clip per epoch — see fer.face_model for the throughput rationale),
clip-level prediction by averaging frame softmax probabilities. Evaluated
on VAL actors only; test is never loaded. 3 seeds, early stopping on a
monitoring val macro-F1, final numbers from the best checkpoint's
full-frame val evaluation. MPS device.

Usage:
    python scripts/10_face_model_train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.envinfo import get_env_report
from fer.face_model import train_one_seed
from fer.splits import load_split
from fer.uncertainty import bootstrap_actor_level_macro_f1

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_ORDER = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
SEEDS = [0, 1, 2]
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0
CHECKPOINT_DIR = REPO_ROOT / "models" / "face_resnet18"


def main() -> None:
    data_cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    sf = load_split(REPO_ROOT / data_cfg["paths"]["splits_dir"] / "split_v1.json")
    if sf.status != "frozen":
        raise RuntimeError("split is not frozen — refusing to proceed")
    train_actors, val_actors, test_actors = set(sf.assignment["train"]), set(sf.assignment["val"]), set(sf.assignment["test"])

    metadata_df = pd.read_csv(REPO_ROOT / data_cfg["paths"]["outputs_dir"] / "metadata.csv", dtype={"actor_id": str})
    train_metadata = metadata_df[metadata_df["actor_id"].isin(train_actors)]
    val_metadata = metadata_df[metadata_df["actor_id"].isin(val_actors)]
    assert not set(train_metadata["actor_id"]) & test_actors
    assert not set(val_metadata["actor_id"]) & test_actors
    print(f"train: {len(train_metadata)} clips, {train_metadata['actor_id'].nunique()} actors")
    print(f"val:   {len(val_metadata)} clips, {val_metadata['actor_id'].nunique()} actors")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    crops_dir = REPO_ROOT / "cache" / "face_crops"
    env = get_env_report(repo_root=str(REPO_ROOT))

    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT}/mlflow.db")
    mlflow.set_experiment("phase4_face_resnet18_clean")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for seed in SEEDS:
        print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")
        result = train_one_seed(
            train_metadata, val_metadata, crops_dir, CLASS_ORDER, seed=seed, device=device,
        )
        ckpt_path = CHECKPOINT_DIR / f"resnet18_clean_seed{seed}.pt"
        torch.save(result["model"].state_dict(), ckpt_path)

        with mlflow.start_run(run_name=f"resnet18_clean_seed{seed}"):
            mlflow.set_tag("model_name", "face_resnet18_clean")
            mlflow.log_param("seed", seed)
            mlflow.log_param("split_hash", sf.content_hash)
            mlflow.log_param("git_commit", env["git_commit"])
            mlflow.log_param("best_epoch", result["best_epoch"])
            mlflow.log_param("n_train_clips", len(train_metadata))
            mlflow.log_param("n_val_clips", len(val_metadata))
            for pkg in ("torch", "timm", "scikit-learn"):
                mlflow.log_param(f"version_{pkg}", env["package_versions"].get(pkg, "n/a"))
            mlflow.log_metric("val_macro_f1", result["val_macro_f1"])
            mlflow.log_metric("val_accuracy", result["val_accuracy"])
            for cls, f1 in result["per_class_f1"].items():
                mlflow.log_metric(f"f1_{cls}", f1)
            for h in result["history"]:
                mlflow.log_metrics({"train_loss": h["train_loss"], "monitor_val_macro_f1": h["monitor_val_macro_f1"]}, step=h["epoch"])
            cm_path = REPO_ROOT / "outputs" / f"confusion_matrix_face_resnet18_seed{seed}.csv"
            pd.DataFrame(result["confusion_matrix"], index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(cm_path)
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(ckpt_path))

        results.append(result)
        print(f"seed={seed} FINAL val_macro_f1={result['val_macro_f1']:.4f} val_accuracy={result['val_accuracy']:.4f}")

    macro_f1s = [r["val_macro_f1"] for r in results]
    accs = [r["val_accuracy"] for r in results]
    print(f"\n{'='*70}\nSUMMARY (clean val, 3 seeds)\n{'='*70}")
    print(f"macro_f1: {np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s):.4f}  (per-seed: {[f'{v:.4f}' for v in macro_f1s]})")
    print(f"accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print("per-class F1 (mean +/- std across seeds):")
    for cls in CLASS_ORDER:
        vals = [r["per_class_f1"][cls] for r in results]
        print(f"  {cls:10s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    print(f"\nconfusion matrix (seed {SEEDS[0]}):")
    print(pd.DataFrame(results[0]["confusion_matrix"], index=CLASS_ORDER, columns=CLASS_ORDER).to_string())

    rep = results[0]
    boot_scores = bootstrap_actor_level_macro_f1(
        rep["actor_ids"], rep["y_true"], rep["y_pred"], CLASS_ORDER, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
    )
    ci_lo, ci_hi = np.percentile(boot_scores, [2.5, 97.5])
    print(f"\nBootstrap CI (seed {SEEDS[0]}'s model, {N_BOOTSTRAP} resamples over {len(val_actors)} val actors):")
    print(f"point estimate: {rep['val_macro_f1']:.4f}, bootstrap mean: {boot_scores.mean():.4f}, 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    with mlflow.start_run(run_name="resnet18_clean_summary"):
        mlflow.set_tag("model_name", "face_resnet18_clean")
        mlflow.set_tag("analysis", "summary")
        mlflow.log_param("split_hash", sf.content_hash)
        mlflow.log_param("git_commit", env["git_commit"])
        mlflow.log_metric("mean_macro_f1", np.mean(macro_f1s))
        mlflow.log_metric("std_macro_f1", np.std(macro_f1s))
        mlflow.log_metric("bootstrap_ci_lo", ci_lo)
        mlflow.log_metric("bootstrap_ci_hi", ci_hi)

    summary_path = REPO_ROOT / "outputs" / "phase4_face_resnet18_clean_summary.csv"
    pd.DataFrame(
        [{"seed": s, "val_macro_f1": r["val_macro_f1"], "val_accuracy": r["val_accuracy"], "best_epoch": r["best_epoch"]} for s, r in zip(SEEDS, results)]
    ).to_csv(summary_path, index=False)
    print(f"\nSummary written to: {summary_path}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
