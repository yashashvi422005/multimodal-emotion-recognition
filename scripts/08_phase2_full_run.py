#!/usr/bin/env python
"""Phase 2 Part B: full run over all 7,439 canonical clips (including
test-actor clips — this is deterministic preprocessing with no fitted
parameters, explicitly requested to cover "all 7,439 clips", unlike the
Phase 3 audio-feature extraction which deliberately stayed train+val-only).

Per clip, caches:
  - cache/face_crops/{clip_id}/frame_{i:03d}.jpg  — clean 224x224 crops for
    every ~10fps-sampled frame
  - cache/landmarks/{clip_id}.npz                  — 478-pt landmarks for
    every successfully-detected sampled frame (frame_indices, landmarks, methods)
  - cache/audio/{clip_id}.npy                       — mono 16kHz waveform
  - cache/occluded_crops/{clip_id}/{type}_{severity}.jpg — all 4 occlusion
    types x 3 severities, applied to one representative frame per clip

And accumulates, flushed incrementally every CHECKPOINT_EVERY clips (so a
crash partway through doesn't lose completed work):
  - outputs/phase2_full_run_summary.csv            — per-clip stats
  - outputs/phase2_occlusion_detection_table.csv   — per clip x type x
    severity, whether MediaPipe (no fallback) still detects a face
  - outputs/phase2_full_run_failures.csv           — clips that errored,
    with the error, so the run continues rather than aborting entirely

Usage:
    python scripts/08_phase2_full_run.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import av
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.face_pipeline import SEVERITIES, FaceDetector, apply_occlusion, crop_face
from fer.splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_FPS = 10
MAX_FRAMES_PER_CLIP = 60
OCCLUSION_TYPES = ["mask", "sunglasses", "patch", "hand_over_mouth"]
SEED = 42
CHECKPOINT_EVERY = 200

CACHE_CROPS = REPO_ROOT / "cache" / "face_crops"
CACHE_LANDMARKS = REPO_ROOT / "cache" / "landmarks"
CACHE_AUDIO = REPO_ROOT / "cache" / "audio"
CACHE_OCCLUDED = REPO_ROOT / "cache" / "occluded_crops"
OUT_DIR = REPO_ROOT / "outputs"

SUMMARY_PATH = OUT_DIR / "phase2_full_run_summary.csv"
DETECTION_TABLE_PATH = OUT_DIR / "phase2_occlusion_detection_table.csv"
FAILURES_PATH = OUT_DIR / "phase2_full_run_failures.csv"


def sample_frame_indices(n_total_frames: int, container_fps: float, target_fps: float, max_frames: int) -> list[int]:
    step = max(1, round(container_fps / target_fps))
    idxs = list(range(0, n_total_frames, step))
    return idxs[:max_frames]


def process_clip(detector: FaceDetector, clip_id: str, video_path: str, audio_path: str):
    for d in (CACHE_CROPS / clip_id, CACHE_OCCLUDED / clip_id):
        d.mkdir(parents=True, exist_ok=True)
    CACHE_LANDMARKS.mkdir(parents=True, exist_ok=True)
    CACHE_AUDIO.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    container = av.open(video_path)
    vstream = container.streams.video[0]
    container_fps = float(vstream.average_rate) if vstream.average_rate else TARGET_FPS
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()

    idxs = sample_frame_indices(len(frames), container_fps, TARGET_FPS, MAX_FRAMES_PER_CLIP)

    frame_indices, landmarks_list, methods = [], [], []
    detected_crops = {}
    for i in idxs:
        det = detector.detect(frames[i])
        if not det.success:
            continue
        crop, transform = crop_face(frames[i], det.bbox, out_size=224)
        Image.fromarray(crop).save(CACHE_CROPS / clip_id / f"frame_{i:03d}.jpg", quality=90)
        lm = transform(det.landmarks_478) if det.landmarks_478 is not None else np.full((478, 2), np.nan)
        frame_indices.append(i)
        landmarks_list.append(lm)
        methods.append(det.method)
        detected_crops[i] = {"crop": crop, "landmarks": det.landmarks_478 is not None, "landmarks_crop": lm}

    n_sampled, n_detected = len(idxs), len(frame_indices)

    if frame_indices:
        np.savez_compressed(
            CACHE_LANDMARKS / f"{clip_id}.npz",
            frame_indices=np.array(frame_indices),
            landmarks=np.stack(landmarks_list),
            methods=np.array(methods),
        )

    wav, samplerate = sf.read(audio_path, dtype="float32")
    audio_ok = True
    if samplerate != 16000 or (wav.ndim == 2 and wav.shape[1] != 1):
        audio_ok = False
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    np.save(CACHE_AUDIO / f"{clip_id}.npy", wav.astype(np.float32))

    detection_rows = []
    if detected_crops:
        mid_idx = idxs[len(idxs) // 2]
        rep_idx = min(detected_crops.keys(), key=lambda i: abs(i - mid_idx))
        rep = detected_crops[rep_idx]
        rep_landmarks = rep["landmarks_crop"] if rep["landmarks"] else None
        for occ_type in OCCLUSION_TYPES:
            for severity in SEVERITIES:
                occluded = apply_occlusion(occ_type, rep["crop"], rep_landmarks, severity, clip_id, SEED)
                Image.fromarray(occluded).save(CACHE_OCCLUDED / clip_id / f"{occ_type}_{severity}.jpg", quality=90)
                det_occ = detector.detect_mediapipe_only(occluded)
                detection_rows.append(
                    {"clip_id": clip_id, "occlusion_type": occ_type, "severity": severity, "mediapipe_detected": det_occ.success}
                )

    elapsed = time.perf_counter() - t0
    summary_row = {
        "clip_id": clip_id,
        "n_frames_sampled": n_sampled,
        "n_frames_detected": n_detected,
        "detection_failure_rate": 1 - (n_detected / n_sampled) if n_sampled else float("nan"),
        "audio_samplerate": samplerate,
        "audio_ok": audio_ok,
        "elapsed_s": elapsed,
    }
    return summary_row, detection_rows


def flush(buffer: list[dict], path: Path, first_write: bool) -> None:
    if not buffer:
        return
    df = pd.DataFrame(buffer)
    df.to_csv(path, mode="w" if first_write else "a", header=first_write, index=False)


def main() -> None:
    data_cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    metadata_df = pd.read_csv(REPO_ROOT / data_cfg["paths"]["outputs_dir"] / "metadata.csv", dtype={"actor_id": str})

    sf_split = load_split(REPO_ROOT / data_cfg["paths"]["splits_dir"] / "split_v1.json")
    if sf_split.status != "frozen":
        raise RuntimeError("split is not frozen — refusing to proceed")

    print(f"Processing all {len(metadata_df)} canonical clips (train+val+test — deterministic preprocessing, no fitting).")

    model_path = REPO_ROOT / "models" / "mediapipe" / "face_landmarker.task"
    detector = FaceDetector(model_path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in (SUMMARY_PATH, DETECTION_TABLE_PATH, FAILURES_PATH):
        if p.exists():
            p.unlink()

    summary_buf, detection_buf, failures_buf = [], [], []
    summary_first, detection_first, failures_first = True, True, True
    t_start = time.perf_counter()

    for n, (_, r) in enumerate(metadata_df.iterrows(), start=1):
        try:
            summary_row, detection_rows = process_clip(detector, r["clip_id"], r["video_path"], r["audio_path"])
            summary_buf.append(summary_row)
            detection_buf.extend(detection_rows)
        except Exception as e:  # noqa: BLE001 — one bad clip must not kill a multi-hour run
            failures_buf.append({"clip_id": r["clip_id"], "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})

        if n % CHECKPOINT_EVERY == 0 or n == len(metadata_df):
            flush(summary_buf, SUMMARY_PATH, summary_first)
            flush(detection_buf, DETECTION_TABLE_PATH, detection_first)
            flush(failures_buf, FAILURES_PATH, failures_first)
            summary_first = summary_first and not summary_buf
            detection_first = detection_first and not detection_buf
            failures_first = failures_first and not failures_buf
            summary_buf, detection_buf, failures_buf = [], [], []
            elapsed = time.perf_counter() - t_start
            rate = n / elapsed
            eta_s = (len(metadata_df) - n) / rate if rate > 0 else float("nan")
            print(f"[{n}/{len(metadata_df)}] elapsed={elapsed/60:.1f}min rate={rate:.2f} clips/s ETA={eta_s/60:.1f}min", flush=True)

    total_elapsed = time.perf_counter() - t_start
    print(f"\nDONE. Total elapsed: {total_elapsed/60:.1f} min for {len(metadata_df)} clips.")


if __name__ == "__main__":
    main()
