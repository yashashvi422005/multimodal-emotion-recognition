#!/usr/bin/env python
"""Evaluate the clean-trained face model (seed 0's checkpoint — singular,
not re-run per seed) on VAL under every occlusion type x severity.

Two phases, run separately, per a redesign after a first attempt
projected ~12 hours (re-running MediaPipe detection inside the model-eval
loop, on every one of ~20-30 cached frames/clip x 12 conditions, single
process):

  Phase 1 (build_detection_cache): CPU-bound, multiprocess, run ONCE.
  For each val clip, 8 EVENLY SPACED frames (same 8 frames for every
  occlusion type, severity, and model — see configs/face_model.yaml) x 12
  conditions: apply the occlusion, run MediaPipe detection, record
  True/False. Cached to disk (cache/phase4_occlusion_detection_cache.parquet)
  so this expensive step is never repeated, including across future face
  models (detection depends only on the occluded image, not on model
  weights). Parallelized across CPU cores, reserving 2 cores so the Mac
  stays usable.

  Phase 2 (evaluate_model): single process, MPS, batched, torch.no_grad(),
  model in eval() mode. Reads the cached True/False table — no detection
  re-run here. For each clip x condition, regenerates (cheap, deterministic)
  the occluded crops for frames the cache marked as detected, batches them
  through the model, and averages softmax probabilities.

Face-branch missing-detection rule (two sentences, as requested): a frame
where the face isn't detected casts no vote, so a clip's prediction is the
average of only its surviving frames. If a clip has zero surviving frames
under some condition, the face branch doesn't skip it — it abstains by
predicting a uniform distribution over classes, and that clip still counts
in macro-F1 (as whichever class the uniform distribution's argmax
tie-break resolves to, which is deterministic and documented, not a
special out-of-label-set sentinel).

Usage:
    python scripts/11_face_model_occlusion_eval.py --limit 20   # dry run, time it
    python scripts/11_face_model_occlusion_eval.py              # full run
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import yaml
from PIL import Image
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.face_model import IMAGENET_MEAN, IMAGENET_STD
from fer.face_pipeline import FaceDetector, apply_occlusion
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
SEED = 0  # only used for the within-frame occlusion placement RNG (patch/hand), not frame selection

CHECKPOINT_PATH = REPO_ROOT / "models" / "face_resnet18" / "resnet18_clean_seed0.pt"
CROPS_DIR = REPO_ROOT / "cache" / "face_crops"
LANDMARKS_DIR = REPO_ROOT / "cache" / "landmarks"
MEDIAPIPE_MODEL_PATH = REPO_ROOT / "models" / "mediapipe" / "face_landmarker.task"
DETECTION_CACHE_PATH = REPO_ROOT / OCC_CFG["detection_cache_path"]


def image_to_tensor(arr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)
    return (t - IMAGENET_MEAN) / IMAGENET_STD


def select_eval_frame_positions(n_frames: int, k: int = FRAMES_PER_CLIP_EVAL) -> np.ndarray:
    """Evenly spaced positions into a clip's frame_indices/landmarks arrays.
    Depends only on n_frames — same positions for every occlusion
    type/severity/model, per configs/face_model.yaml's frame_selection."""
    if n_frames <= k:
        return np.arange(n_frames)
    return np.unique(np.round(np.linspace(0, n_frames - 1, k)).astype(int))


def load_val_tasks() -> tuple[list[tuple], dict]:
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
    return tasks, clip_true


# ---------------------------------------------------------------------------
# Phase 1: detection cache (CPU, multiprocess)
# ---------------------------------------------------------------------------

_detector: FaceDetector | None = None


def _init_detector_worker() -> None:
    global _detector
    _detector = FaceDetector(MEDIAPIPE_MODEL_PATH)


def _detect_clip(args: tuple) -> list[tuple]:
    clip_id, frame_indices, landmarks = args
    positions = select_eval_frame_positions(len(frame_indices))
    rows = []
    for pos in positions:
        fi, lm = frame_indices[pos], landmarks[pos]
        crop_path = CROPS_DIR / clip_id / f"frame_{int(fi):03d}.jpg"
        if not crop_path.exists():
            continue
        crop = np.array(Image.open(crop_path))
        lm_valid = None if np.isnan(lm).any() else lm
        for occ_type, severity in CONDITIONS:
            occluded = apply_occlusion(occ_type, crop, lm_valid, severity, clip_id, SEED)
            det = _detector.detect_mediapipe_only(occluded)
            rows.append((clip_id, int(fi), occ_type, severity, bool(det.success)))
    return rows


def build_detection_cache(tasks: list[tuple], log_fn=print) -> pd.DataFrame:
    """Incremental: clips already present in the on-disk cache are skipped,
    so a prior dry run's work is reused by the full run rather than redone."""
    existing = pd.DataFrame(columns=["clip_id", "frame_idx", "occlusion_type", "severity", "detected"])
    if DETECTION_CACHE_PATH.exists():
        existing = pd.read_parquet(DETECTION_CACHE_PATH)
        done_ids = set(existing["clip_id"].unique())
        tasks = [t for t in tasks if t[0] not in done_ids]
        log_fn(f"detection cache: {len(done_ids)} clips already cached, {len(tasks)} remaining")

    if not tasks:
        return existing

    t_start = time.perf_counter()
    new_rows = []
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=N_WORKERS, initializer=_init_detector_worker) as pool:
        for i, rows in enumerate(pool.imap_unordered(_detect_clip, tasks, chunksize=1)):
            new_rows.extend(rows)
            if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
                elapsed = time.perf_counter() - t_start
                rate = (i + 1) / elapsed
                eta = (len(tasks) - (i + 1)) / rate
                log_fn(f"  detect [{i + 1}/{len(tasks)}] elapsed={elapsed:.1f}s rate={rate:.3f} clips/s ETA={eta / 60:.1f}min")

    new_df = pd.DataFrame(new_rows, columns=["clip_id", "frame_idx", "occlusion_type", "severity", "detected"])
    full_df = pd.concat([existing, new_df], ignore_index=True)
    DETECTION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(DETECTION_CACHE_PATH, index=False)
    return full_df


# ---------------------------------------------------------------------------
# Phase 2: model evaluation (MPS, single process, batched)
# ---------------------------------------------------------------------------

def evaluate_model(tasks: list[tuple], clip_true: dict, detection_df: pd.DataFrame, log_fn=print) -> pd.DataFrame:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = timm.create_model("resnet18", pretrained=False, num_classes=len(CLASS_ORDER)).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    landmarks_by_clip = {clip_id: (fi, lm) for clip_id, fi, lm in tasks}
    det_lookup = detection_df.set_index(["clip_id", "frame_idx", "occlusion_type", "severity"])["detected"]

    uniform_pred = CLASS_ORDER[int(np.argmax(np.ones(len(CLASS_ORDER))))]  # deterministic tie-break, documented

    t_start = time.perf_counter()
    rows = []
    for occ_type, severity in CONDITIONS:
        y_true, y_pred, n_abstain = [], [], 0
        for clip_id, (frame_indices, landmarks) in landmarks_by_clip.items():
            positions = select_eval_frame_positions(len(frame_indices))
            batch_imgs = []
            for pos in positions:
                fi, lm = frame_indices[pos], landmarks[pos]
                key = (clip_id, int(fi), occ_type, severity)
                if key not in det_lookup.index or not det_lookup.loc[key]:
                    continue
                crop_path = CROPS_DIR / clip_id / f"frame_{int(fi):03d}.jpg"
                crop = np.array(Image.open(crop_path))
                lm_valid = None if np.isnan(lm).any() else lm
                occluded = apply_occlusion(occ_type, crop, lm_valid, severity, clip_id, SEED)
                batch_imgs.append(image_to_tensor(occluded))

            y_true.append(clip_true[clip_id])
            if not batch_imgs:
                y_pred.append(uniform_pred)
                n_abstain += 1
                continue
            with torch.no_grad():
                x = torch.stack(batch_imgs).to(device)
                probs = torch.softmax(model(x), dim=1).mean(dim=0).cpu().numpy()
            y_pred.append(CLASS_ORDER[int(np.argmax(probs))])

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        macro_f1 = f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro")
        acc = (y_true == y_pred).mean()
        rows.append({"occlusion_type": occ_type, "severity": severity, "macro_f1": macro_f1, "accuracy": acc, "n_abstained": n_abstain, "n_clips": len(y_true)})
        log_fn(f"{occ_type:16s} severity={severity}: macro_f1={macro_f1:.4f} accuracy={acc:.4f} abstained={n_abstain}/{len(y_true)}")

    log_fn(f"eval phase elapsed: {(time.perf_counter() - t_start) / 60:.1f} min")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="limit to N val clips, for timing a dry run")
    args = parser.parse_args()

    tasks, clip_true = load_val_tasks()
    print(f"val: {len(tasks)} clips with cached landmarks")
    print(f"frames_per_clip_eval={FRAMES_PER_CLIP_EVAL} (evenly spaced), n_workers={N_WORKERS}, conditions={len(CONDITIONS)}")

    if args.limit:
        tasks = tasks[: args.limit]
        print(f"--limit {args.limit}: dry run on {len(tasks)} clips")

    t0 = time.perf_counter()
    detection_df = build_detection_cache(tasks)
    t1 = time.perf_counter()
    print(f"\nPhase 1 (detection cache) wall time for this invocation: {(t1 - t0) / 60:.2f} min")

    relevant_ids = {t[0] for t in tasks}
    detection_df = detection_df[detection_df["clip_id"].isin(relevant_ids)]

    results_df = evaluate_model(tasks, clip_true, detection_df)
    t2 = time.perf_counter()
    print(f"Phase 2 (model eval) wall time: {(t2 - t1) / 60:.2f} min")
    print(f"Total wall time this invocation: {(t2 - t0) / 60:.2f} min")

    if args.limit:
        print(f"\n--limit {args.limit} dry run complete — not saving the results table (results are partial).")
    else:
        out_path = REPO_ROOT / "outputs" / "phase4_occlusion_eval_table.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
