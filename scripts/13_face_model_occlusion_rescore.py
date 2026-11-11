#!/usr/bin/env python
"""Three fixes to the Phase 4 occlusion results, applied together since
they share the same re-evaluation pass:

1. Official abstention scoring (fer.metrics.score_with_abstention):
   abstained clips are a false negative for their true class, never a
   false positive, and stay in the accuracy denominator. This replaces
   BOTH prior variants (counting the uniform-distribution argmax
   tie-break as a real prediction, and dropping abstained clips
   entirely) as the one scoring method used from here on.
2. Reproducibility: see tests/test_occlusion_reproducibility.py and
   fer.occlusion_eval's docstring — fixed by averaging frame
   probabilities on CPU rather than on the MPS device.
3. Mask severity redefined by vertical extent from real landmarks (see
   fer.face_pipeline.apply_mask): 0.3 = mouth+chin, 0.6 = up to the nose
   bridge midpoint, 0.9 = up to just below the eyes. Only mask's
   detection-cache rows are rebuilt; sunglasses/patch/hand_over_mouth are
   untouched and reused as-is.

Also writes outputs/phase4_face_probs.parquet — full per-clip softmax
probability vectors (not just the argmax), for clean + every occlusion
condition, all 3 seeds. This is the master table Phase 5's fusion
development reads from (fix 4), so face-branch inference isn't run a
third time.

Usage:
    python scripts/13_face_model_occlusion_rescore.py
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.face_pipeline import FaceDetector, apply_occlusion
from fer.metrics import score_with_abstention
from fer.occlusion_eval import evaluate_clean_probs, evaluate_occlusion_probs, load_resnet18_checkpoint, select_eval_frame_positions
from fer.splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]
FACE_CFG = yaml.safe_load((REPO_ROOT / "configs" / "face_model.yaml").read_text())
OCC_CFG = FACE_CFG["occlusion_eval"]

CLASS_ORDER = FACE_CFG["class_order"]
OCCLUSION_TYPES = OCC_CFG["occlusion_types"]
SEVERITIES = OCC_CFG["severities"]
CONDITIONS = [(t, s) for t in OCCLUSION_TYPES for s in SEVERITIES]
FRAMES_PER_CLIP_EVAL = OCC_CFG["frames_per_clip_eval"]
N_WORKERS = max(1, (os.cpu_count() or 4) - OCC_CFG["cpu_cores_reserved"])
SEEDS = [0, 1, 2]
OCCLUSION_SEED = 0

CROPS_DIR = REPO_ROOT / "cache" / "face_crops"
LANDMARKS_DIR = REPO_ROOT / "cache" / "landmarks"
MEDIAPIPE_MODEL_PATH = REPO_ROOT / "models" / "mediapipe" / "face_landmarker.task"
DETECTION_CACHE_PATH = REPO_ROOT / OCC_CFG["detection_cache_path"]
CHECKPOINT_DIR = REPO_ROOT / "models" / "face_resnet18"
OUT_DIR = REPO_ROOT / "outputs"


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


# --- mask-only detection cache rebuild (CPU, multiprocess) ---

_detector: FaceDetector | None = None


def _init_detector_worker() -> None:
    global _detector
    _detector = FaceDetector(MEDIAPIPE_MODEL_PATH)


def _detect_mask_for_clip(args: tuple) -> list[tuple]:
    clip_id, frame_indices, landmarks = args
    positions = select_eval_frame_positions(len(frame_indices), FRAMES_PER_CLIP_EVAL)
    rows = []
    for pos in positions:
        fi, lm = frame_indices[pos], landmarks[pos]
        crop_path = CROPS_DIR / clip_id / f"frame_{int(fi):03d}.jpg"
        if not crop_path.exists():
            continue
        crop = np.array(Image.open(crop_path))
        lm_valid = None if np.isnan(lm).any() else lm
        for severity in SEVERITIES:
            occluded = apply_occlusion("mask", crop, lm_valid, severity, clip_id, OCCLUSION_SEED)
            det = _detector.detect_mediapipe_only(occluded)
            rows.append((clip_id, int(fi), "mask", severity, bool(det.success)))
    return rows


def rebuild_mask_detection_cache(tasks: list[tuple]) -> pd.DataFrame:
    existing = pd.read_parquet(DETECTION_CACHE_PATH)
    kept = existing[existing["occlusion_type"] != "mask"]
    print(f"detection cache: dropping {len(existing) - len(kept)} stale mask rows, keeping {len(kept)} non-mask rows")

    t_start = time.perf_counter()
    new_rows = []
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=N_WORKERS, initializer=_init_detector_worker) as pool:
        for i, rows in enumerate(pool.imap_unordered(_detect_mask_for_clip, tasks, chunksize=2)):
            new_rows.extend(rows)
            if (i + 1) % 200 == 0 or (i + 1) == len(tasks):
                print(f"  mask-detect [{i + 1}/{len(tasks)}] elapsed={time.perf_counter() - t_start:.1f}s")

    new_df = pd.DataFrame(new_rows, columns=["clip_id", "frame_idx", "occlusion_type", "severity", "detected"])
    full_df = pd.concat([kept, new_df], ignore_index=True)
    full_df.to_parquet(DETECTION_CACHE_PATH, index=False)
    print(f"mask detection rebuilt: {len(new_df)} rows in {(time.perf_counter() - t_start) / 60:.2f} min")
    return full_df


def compute_mask_coverage(tasks: list[tuple], n_samples: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(tasks), size=min(n_samples, len(tasks)), replace=False)
    rows = []
    for severity in SEVERITIES:
        fracs = []
        for i in sample_idx:
            clip_id, frame_indices, landmarks = tasks[i]
            pos = int(rng.integers(0, len(frame_indices)))
            lm = landmarks[pos]
            if np.isnan(lm).any():
                continue
            blank = np.full((224, 224, 3), (255, 0, 255), dtype=np.uint8)
            occluded = apply_occlusion("mask", blank, lm, severity, clip_id, OCCLUSION_SEED)
            changed = np.any(occluded != blank, axis=-1)
            fracs.append(changed.mean())
        rows.append({"occlusion_type": "mask", "severity": severity, "mean_coverage_frac": float(np.mean(fracs)), "n_samples": len(fracs)})
    return pd.DataFrame(rows)


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    tasks, clip_true, val_metadata = load_val_tasks()
    print(f"val: {len(tasks)} clips")

    # --- Fix 3: rebuild mask cache with new severity geometry, report coverage ---
    rebuild_mask_detection_cache(tasks)
    coverage_df = compute_mask_coverage(tasks)
    coverage_df.to_csv(OUT_DIR / "phase4_mask_coverage_v2.csv", index=False)
    print("\n[Fix 3] new mask coverage:")
    print(coverage_df.to_string(index=False))

    detection_df = pd.read_parquet(DETECTION_CACHE_PATH)
    det_lookup = detection_df.set_index(["clip_id", "frame_idx", "occlusion_type", "severity"])["detected"]

    # --- Re-evaluate all conditions, all 3 seeds, saving full probability vectors ---
    all_probs = []
    for seed in SEEDS:
        ckpt = CHECKPOINT_DIR / f"resnet18_clean_seed{seed}.pt"
        print(f"\n=== seed {seed} ===")
        model = load_resnet18_checkpoint(ckpt, len(CLASS_ORDER), device)

        t0 = time.perf_counter()
        occ_df = evaluate_occlusion_probs(tasks, clip_true, det_lookup, model, CLASS_ORDER, CONDITIONS, device, CROPS_DIR, FRAMES_PER_CLIP_EVAL, OCCLUSION_SEED)
        print(f"  occlusion eval: {(time.perf_counter() - t0) / 60:.1f} min")

        t0 = time.perf_counter()
        clean_df = evaluate_clean_probs(val_metadata, model, CLASS_ORDER, device, CROPS_DIR)
        print(f"  clean eval: {(time.perf_counter() - t0) / 60:.1f} min")

        occ_df.insert(0, "seed", seed)
        clean_df.insert(0, "seed", seed)
        clean_df.insert(1, "occlusion_type", "clean")
        clean_df.insert(2, "severity", np.nan)
        all_probs.append(occ_df)
        all_probs.append(clean_df)

    probs_df = pd.concat(all_probs, ignore_index=True)
    probs_df.to_parquet(OUT_DIR / "phase4_face_probs.parquet", index=False)
    print(f"\nSaved per-clip face probabilities to {OUT_DIR / 'phase4_face_probs.parquet'} ({len(probs_df)} rows)")

    # --- Fix 1: official scoring, per seed, then mean +/- std over seeds ---
    score_rows = []
    for seed in SEEDS:
        for occ_type, severity in [("clean", np.nan)] + CONDITIONS:
            if occ_type == "clean":
                sub = probs_df[(probs_df["seed"] == seed) & (probs_df["occlusion_type"] == "clean")]
            else:
                sub = probs_df[(probs_df["seed"] == seed) & (probs_df["occlusion_type"] == occ_type) & (probs_df["severity"] == severity)]
            macro_f1, acc, _ = score_with_abstention(sub["y_true"].tolist(), sub["y_pred"].tolist(), sub["abstained"].tolist(), CLASS_ORDER)
            score_rows.append({"seed": seed, "occlusion_type": occ_type, "severity": severity, "macro_f1": macro_f1, "accuracy": acc, "n_abstained": int(sub["abstained"].sum()), "n_clips": len(sub)})

    scores_df = pd.DataFrame(score_rows)
    scores_df.to_csv(OUT_DIR / "phase4_occlusion_eval_by_seed_official.csv", index=False)

    agg = scores_df.groupby(["occlusion_type", "severity"], dropna=False).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        n_abstained=("n_abstained", "first"), n_clips=("n_clips", "first"),
    ).reset_index()
    agg.to_csv(OUT_DIR / "phase4_occlusion_eval_table_official.csv", index=False)
    print("\n[Fix 1] official occlusion table (abstention = FN only, no FP, stays in denominator), mean +/- std over 3 seeds:")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
