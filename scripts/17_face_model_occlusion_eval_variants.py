#!/usr/bin/env python
"""Evaluate the occlusion-augmented face model checkpoints (aug_all,
aug_no_mask; 3 seeds each) on VAL, clean + every occlusion condition,
reusing the EXISTING detection cache (detection depends only on the
occluded image, not model weights — already rebuilt with the fixed,
monotonic mask-severity geometry). Appends their per-clip probability
vectors, tagged with a `model_variant` column, to
outputs/phase4_face_probs.parquet alongside the clean-trained model's
rows (re-tagged "clean" if not already present).

Usage:
    python scripts/17_face_model_occlusion_eval_variants.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.metrics import score_with_abstention
from fer.occlusion_eval import evaluate_clean_probs, evaluate_occlusion_probs, load_resnet18_checkpoint
from fer.splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]
FACE_CFG = yaml.safe_load((REPO_ROOT / "configs" / "face_model.yaml").read_text())
OCC_CFG = FACE_CFG["occlusion_eval"]

CLASS_ORDER = FACE_CFG["class_order"]
OCCLUSION_TYPES = OCC_CFG["occlusion_types"]
SEVERITIES = OCC_CFG["severities"]
CONDITIONS = [(t, s) for t in OCCLUSION_TYPES for s in SEVERITIES]
FRAMES_PER_CLIP_EVAL = OCC_CFG["frames_per_clip_eval"]
SEEDS = [0, 1, 2]
OCCLUSION_SEED = 0
VARIANTS = ["aug_all", "aug_no_mask"]

CROPS_DIR = REPO_ROOT / "cache" / "face_crops"
LANDMARKS_DIR = REPO_ROOT / "cache" / "landmarks"
DETECTION_CACHE_PATH = REPO_ROOT / OCC_CFG["detection_cache_path"]
OUT_DIR = REPO_ROOT / "outputs"
FACE_PROBS_PATH = OUT_DIR / "phase4_face_probs.parquet"


def load_val_tasks():
    data_cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    sf = load_split(REPO_ROOT / data_cfg["paths"]["splits_dir"] / "split_v1.json")
    if sf.status != "frozen":
        raise RuntimeError("split is not frozen — refusing to proceed")
    val_actors, test_actors = set(sf.assignment["val"]), set(sf.assignment["test"])

    metadata_df = pd.read_csv(REPO_ROOT / data_cfg["paths"]["outputs_dir"] / "metadata.csv", dtype={"actor_id": str})
    val_metadata = metadata_df[metadata_df["actor_id"].isin(val_actors)]
    assert not set(val_metadata["actor_id"]) & test_actors

    tasks, clip_true = [], {}
    for _, r in val_metadata.iterrows():
        clip_id, true_label = r["clip_id"], r["emotion"]
        npz_path = LANDMARKS_DIR / f"{clip_id}.npz"
        if not npz_path.exists():
            continue
        lm_data = np.load(npz_path)
        tasks.append((clip_id, lm_data["frame_indices"], lm_data["landmarks"]))
        clip_true[clip_id] = true_label
    return tasks, clip_true, val_metadata


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    tasks, clip_true, val_metadata = load_val_tasks()
    detection_df = pd.read_parquet(DETECTION_CACHE_PATH)
    det_lookup = detection_df.set_index(["clip_id", "frame_idx", "occlusion_type", "severity"])["detected"]

    existing = pd.read_parquet(FACE_PROBS_PATH)
    if "model_variant" not in existing.columns:
        existing.insert(0, "model_variant", "clean")  # script13's output predates this column; all its rows are the clean-trained model

    all_new = []
    for variant in VARIANTS:
        ckpt_dir = REPO_ROOT / "models" / f"face_resnet18_{variant}"
        for seed in SEEDS:
            ckpt = ckpt_dir / f"resnet18_{variant}_seed{seed}.pt"
            print(f"\n=== variant={variant} seed={seed} ===")
            model = load_resnet18_checkpoint(ckpt, len(CLASS_ORDER), device)

            t0 = time.perf_counter()
            occ_df = evaluate_occlusion_probs(tasks, clip_true, det_lookup, model, CLASS_ORDER, CONDITIONS, device, CROPS_DIR, FRAMES_PER_CLIP_EVAL, OCCLUSION_SEED)
            print(f"  occlusion eval: {(time.perf_counter() - t0) / 60:.1f} min")

            t0 = time.perf_counter()
            clean_df = evaluate_clean_probs(val_metadata, model, CLASS_ORDER, device, CROPS_DIR)
            print(f"  clean eval: {(time.perf_counter() - t0) / 60:.1f} min")

            occ_df.insert(0, "seed", seed)
            occ_df.insert(0, "model_variant", variant)
            clean_df.insert(0, "seed", seed)
            clean_df.insert(0, "model_variant", variant)
            clean_df.insert(2, "occlusion_type", "clean")
            clean_df.insert(3, "severity", np.nan)
            all_new.append(occ_df)
            all_new.append(clean_df)

    combined = pd.concat([existing] + all_new, ignore_index=True)
    combined.to_parquet(FACE_PROBS_PATH, index=False)
    print(f"\nSaved combined per-clip face probabilities ({len(combined)} rows, variants={sorted(combined['model_variant'].unique())}) to {FACE_PROBS_PATH}")

    # Quick occlusion-table printout, all variants, official scoring
    rows = []
    for variant in ["clean"] + VARIANTS:
        for seed in SEEDS:
            for occ_type, severity in [("clean", np.nan)] + CONDITIONS:
                if occ_type == "clean":
                    sub = combined[(combined["model_variant"] == variant) & (combined["seed"] == seed) & (combined["occlusion_type"] == "clean")]
                else:
                    sub = combined[(combined["model_variant"] == variant) & (combined["seed"] == seed) & (combined["occlusion_type"] == occ_type) & (combined["severity"] == severity)]
                if len(sub) == 0:
                    continue
                macro_f1, acc, _ = score_with_abstention(sub["y_true"].tolist(), sub["y_pred"].tolist(), sub["abstained"].tolist(), CLASS_ORDER)
                rows.append({"model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "macro_f1": macro_f1, "accuracy": acc, "n_abstained": int(sub["abstained"].sum()), "n_clips": len(sub)})

    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(OUT_DIR / "phase4_occlusion_by_variant_by_seed.csv", index=False)
    agg = scores_df.groupby(["model_variant", "occlusion_type", "severity"], dropna=False).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        n_abstained=("n_abstained", "first"), n_clips=("n_clips", "first"),
    ).reset_index()
    agg.to_csv(OUT_DIR / "phase4_occlusion_table_by_variant.csv", index=False)
    print("\nocclusion table, clean-trained vs augmented variants, mean +/- std over 3 seeds:")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
