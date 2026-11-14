#!/usr/bin/env python
"""Build the occlusion-detection cache for the 12 TEST actors -- the one
prerequisite FINAL_EVAL_PLAN.md and evaluate.py both flag as not yet
built (the existing cache/phase4_occlusion_detection_cache.parquet
covers VAL clips only). Same methodology as scripts/13's mask-only
rebuild, generalized to all 4 occlusion types x 3 severities (detection
only, no model inference -- CPU, multiprocess).

Does NOT read emotion labels or run any model; this step only answers
"did MediaPipe still find a face on this occluded crop", which is a
precondition for scoring, not a result in itself. Test labels/clips are
touched here only in the sense that occlusion images get *generated* for
them -- the frozen test set's labels are never read by this script.

Workers fixed at 2 (not os.cpu_count()-based), per explicit instruction.

Usage:
    python scripts/26_build_test_detection_cache.py
"""

from __future__ import annotations

import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.face_pipeline import FaceDetector, apply_occlusion
from fer.occlusion_eval import select_eval_frame_positions
from fer.splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]
FACE_CFG = yaml.safe_load((REPO_ROOT / "configs" / "face_model.yaml").read_text())
OCC_CFG = FACE_CFG["occlusion_eval"]

OCCLUSION_TYPES = OCC_CFG["occlusion_types"]
SEVERITIES = OCC_CFG["severities"]
CONDITIONS = [(t, s) for t in OCCLUSION_TYPES for s in SEVERITIES]
FRAMES_PER_CLIP_EVAL = OCC_CFG["frames_per_clip_eval"]
N_WORKERS = 2  # fixed, per explicit instruction -- not os.cpu_count()-based
OCCLUSION_SEED = 0

CROPS_DIR = REPO_ROOT / "cache" / "face_crops"
LANDMARKS_DIR = REPO_ROOT / "cache" / "landmarks"
MEDIAPIPE_MODEL_PATH = REPO_ROOT / "models" / "mediapipe" / "face_landmarker.task"
OUT_CACHE_PATH = REPO_ROOT / "cache" / "phase4_test_occlusion_detection_cache.parquet"

_detector: FaceDetector | None = None


def _init_detector_worker() -> None:
    global _detector
    _detector = FaceDetector(MEDIAPIPE_MODEL_PATH)


def _detect_all_conditions_for_clip(args: tuple) -> list[tuple]:
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
        for occ_type, severity in CONDITIONS:
            occluded = apply_occlusion(occ_type, crop, lm_valid, severity, clip_id, OCCLUSION_SEED)
            det = _detector.detect_mediapipe_only(occluded)
            rows.append((clip_id, int(fi), occ_type, severity, bool(det.success)))
    return rows


def main() -> None:
    data_cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    sf = load_split(REPO_ROOT / data_cfg["paths"]["splits_dir"] / "split_v1.json")
    if sf.status != "frozen":
        raise RuntimeError("split is not frozen -- refusing to proceed")
    test_actors = set(sf.assignment["test"])
    val_actors, train_actors = set(sf.assignment["val"]), set(sf.assignment["train"])

    metadata_df = pd.read_csv(REPO_ROOT / data_cfg["paths"]["outputs_dir"] / "metadata.csv", dtype={"actor_id": str})
    test_metadata = metadata_df[metadata_df["actor_id"].isin(test_actors)]
    assert set(test_metadata["actor_id"]) == test_actors
    assert not (set(test_metadata["actor_id"]) & (val_actors | train_actors))
    print(f"test: {len(test_metadata)} clips, {test_metadata['actor_id'].nunique()} actors")

    tasks = []
    for _, r in test_metadata.iterrows():
        npz_path = LANDMARKS_DIR / f"{r['clip_id']}.npz"
        if not npz_path.exists():
            continue
        lm_data = np.load(npz_path)
        tasks.append((r["clip_id"], lm_data["frame_indices"], lm_data["landmarks"]))
    print(f"{len(tasks)} clips have cached landmarks; building detection cache for {len(CONDITIONS)} conditions x up to {FRAMES_PER_CLIP_EVAL} frames each, workers={N_WORKERS}")

    t_start = time.perf_counter()
    all_rows = []
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=N_WORKERS, initializer=_init_detector_worker) as pool:
        for i, rows in enumerate(pool.imap_unordered(_detect_all_conditions_for_clip, tasks, chunksize=2)):
            all_rows.extend(rows)
            if (i + 1) % 100 == 0 or (i + 1) == len(tasks):
                print(f"  [{i + 1}/{len(tasks)}] elapsed={time.perf_counter() - t_start:.1f}s")

    detection_df = pd.DataFrame(all_rows, columns=["clip_id", "frame_idx", "occlusion_type", "severity", "detected"])
    detection_df.to_parquet(OUT_CACHE_PATH, index=False)
    print(f"\nbuilt {len(detection_df)} rows in {(time.perf_counter() - t_start) / 60:.2f} min -> {OUT_CACHE_PATH}")
    print(f"detection rate: {detection_df['detected'].mean():.4f}")


if __name__ == "__main__":
    main()
