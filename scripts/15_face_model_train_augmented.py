#!/usr/bin/env python
"""Phase 4 ablation: occlusion-augmented face model training. Two
variants, 3 seeds each:
  (a) aug_all        -- all 4 occlusion types available during training
  (b) aug_no_mask    -- mask held out of training; unseen at evaluation

Each selected training frame is independently clean (50%) or occluded
with a random allowed type + random severity (50%) — see
fer.face_model_augmented. Monitoring/early-stopping and the final
reported metric stay on CLEAN val, matching the clean-trained model's
methodology, so the comparison isolates the effect of training-time
occlusion exposure. Time estimated beforehand (~2.88 min/epoch, ~5
epochs/seed average, ~100-115 min projected for all 6 seed-runs) before
launching this script.

Usage:
    python scripts/15_face_model_train_augmented.py
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
from fer.face_model_augmented import train_one_seed_augmented
from fer.splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_ORDER = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
SEEDS = [0, 1, 2]
VARIANTS = {
    "aug_all": ["mask", "sunglasses", "patch", "hand_over_mouth"],
    "aug_no_mask": ["sunglasses", "patch", "hand_over_mouth"],
}


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
    landmarks_dir = REPO_ROOT / "cache" / "landmarks"
    env = get_env_report(repo_root=str(REPO_ROOT))

    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT}/mlflow.db")
    mlflow.set_experiment("phase4_face_resnet18_augmented")

    for variant_name, allowed_types in VARIANTS.items():
        ckpt_dir = REPO_ROOT / "models" / f"face_resnet18_{variant_name}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for seed in SEEDS:
            print(f"\n{'=' * 70}\nVARIANT {variant_name} (allowed_types={allowed_types}) SEED {seed}\n{'=' * 70}")
            result = train_one_seed_augmented(
                train_metadata, val_metadata, crops_dir, landmarks_dir, CLASS_ORDER, seed=seed, device=device, allowed_types=allowed_types,
            )
            ckpt_path = ckpt_dir / f"resnet18_{variant_name}_seed{seed}.pt"
            torch.save(result["model"].state_dict(), ckpt_path)

            with mlflow.start_run(run_name=f"{variant_name}_seed{seed}"):
                mlflow.set_tag("model_name", f"face_resnet18_{variant_name}")
                mlflow.log_param("seed", seed)
                mlflow.log_param("allowed_types", ",".join(allowed_types))
                mlflow.log_param("split_hash", sf.content_hash)
                mlflow.log_param("git_commit", env["git_commit"])
                mlflow.log_param("best_epoch", result["best_epoch"])
                mlflow.log_metric("val_macro_f1_clean", result["val_macro_f1"])
                mlflow.log_metric("val_accuracy_clean", result["val_accuracy"])
                for cls, f1 in result["per_class_f1"].items():
                    mlflow.log_metric(f"f1_{cls}", f1)
                for h in result["history"]:
                    mlflow.log_metrics({"train_loss": h["train_loss"], "monitor_val_macro_f1": h["monitor_val_macro_f1"]}, step=h["epoch"])
                mlflow.log_artifact(str(ckpt_path))

            results.append(result)
            print(f"{variant_name} seed={seed} FINAL clean val_macro_f1={result['val_macro_f1']:.4f} val_accuracy={result['val_accuracy']:.4f}")

        macro_f1s = [r["val_macro_f1"] for r in results]
        print(f"\n{variant_name}: clean val macro_f1 = {np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s):.4f} (per-seed: {[f'{v:.4f}' for v in macro_f1s]})")

        summary_path = REPO_ROOT / "outputs" / f"phase4_face_resnet18_{variant_name}_summary.csv"
        pd.DataFrame(
            [{"seed": s, "val_macro_f1_clean": r["val_macro_f1"], "val_accuracy_clean": r["val_accuracy"], "best_epoch": r["best_epoch"]} for s, r in zip(SEEDS, results)]
        ).to_csv(summary_path, index=False)
        print(f"Summary written to: {summary_path}")

    print(f"\nAll checkpoints saved under: {REPO_ROOT / 'models'}/face_resnet18_<variant>/")


if __name__ == "__main__":
    main()
