#!/usr/bin/env python
"""Phase 2 pilot (Part A only): 20 clips, to validate the face
preprocessing pipeline before committing to it at scale.

Steps:
  1. Select 20 clips from TRAIN+VAL actors only (test untouched), stratified
     across the 6 emotions for variety.
  2. Inspect the raw .flv files with the `ffprobe` CLI directly (independent
     of the PyAV-based probing already done in Phase 1).
  3. Face detection: MediaPipe FaceLandmarker primary, RetinaFace fallback,
     no Haar cascades. Crop to 224x224, remap landmarks into crop space.
  4. Apply all 4 synthetic occlusion types (mask, sunglasses, patch,
     hand_over_mouth) at a fixed severity, deterministic per clip+seed.
  5. Save a contact sheet (original vs each occlusion) for a few clips.
  6. Report: per-frame detection failure rate, wall time per clip, and an
     extrapolated estimate for all 7,439 clips.

Usage:
    python scripts/07_phase2_pilot.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.face_pipeline import FaceDetector, apply_occlusion, crop_face
from fer.splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]
N_PILOT_CLIPS = 20
TARGET_FPS = 10
OCCLUSION_SEED = 42
OCCLUSION_SEVERITY = 0.6
OCCLUSION_TYPES = ["mask", "sunglasses", "patch", "hand_over_mouth"]
N_CONTACT_SHEET_CLIPS = 4


def select_pilot_clips(metadata_df: pd.DataFrame, train_val_actors: set[str], n: int, seed: int) -> pd.DataFrame:
    pool = metadata_df[metadata_df["actor_id"].isin(train_val_actors)]
    rng = np.random.default_rng(seed)
    per_class = n // 6
    rows = []
    for emotion in sorted(pool["emotion"].unique()):
        sub = pool[pool["emotion"] == emotion]
        k = min(per_class, len(sub))
        rows.append(sub.sample(n=k, random_state=seed))
    chosen = pd.concat(rows)
    remaining = n - len(chosen)
    if remaining > 0:
        leftover_pool = pool[~pool["clip_id"].isin(chosen["clip_id"])]
        chosen = pd.concat([chosen, leftover_pool.sample(n=remaining, random_state=seed)])
    return chosen.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def ffprobe_inspect(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def sample_frame_indices(n_total_frames: int, container_fps: float, target_fps: float) -> list[int]:
    step = max(1, round(container_fps / target_fps))
    return list(range(0, n_total_frames, step))


def build_contact_sheet(rows: list[dict], out_path: Path) -> None:
    cell = 160
    labels = ["original"] + OCCLUSION_TYPES
    sheet = Image.new("RGB", (cell * len(labels), cell * len(rows) + 24), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for col, label in enumerate(labels):
        draw.text((col * cell + 8, 4), label, fill=(0, 0, 0), font=font)
    for r, row in enumerate(rows):
        y0 = 24 + r * cell
        imgs = [row["original"]] + [row[occ] for occ in OCCLUSION_TYPES]
        for col, arr in enumerate(imgs):
            thumb = Image.fromarray(arr).resize((cell - 8, cell - 8))
            sheet.paste(thumb, (col * cell + 4, y0 + 4))
        draw.text((4, y0 + 4), row["clip_id"], fill=(255, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    data_cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    metadata_df = pd.read_csv(REPO_ROOT / data_cfg["paths"]["outputs_dir"] / "metadata.csv", dtype={"actor_id": str})

    sf = load_split(REPO_ROOT / data_cfg["paths"]["splits_dir"] / "split_v1.json")
    if sf.status != "frozen":
        raise RuntimeError("split is not frozen — refusing to proceed")
    train_val_actors = set(sf.assignment["train"]) | set(sf.assignment["val"])
    test_actors = set(sf.assignment["test"])

    pilot = select_pilot_clips(metadata_df, train_val_actors, N_PILOT_CLIPS, seed=OCCLUSION_SEED)
    assert not set(pilot["actor_id"]) & test_actors, "test actor leaked into pilot selection"
    print(f"Selected {len(pilot)} pilot clips from train+val actors.")
    print(pilot[["clip_id", "actor_id", "emotion"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("ffprobe inspection (independent of Phase 1's PyAV-based probing)")
    print("=" * 70)
    ffprobe_summaries = []
    for _, r in pilot.iterrows():
        info = ffprobe_inspect(r["video_path"])
        vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
        ffprobe_summaries.append(
            {
                "clip_id": r["clip_id"],
                "codec": vstream.get("codec_name"),
                "width": vstream.get("width"),
                "height": vstream.get("height"),
                "r_frame_rate": vstream.get("r_frame_rate"),
                "duration": info["format"].get("duration"),
            }
        )
    ffprobe_df = pd.DataFrame(ffprobe_summaries)
    print(ffprobe_df.to_string(index=False))

    model_path = REPO_ROOT / "models" / "mediapipe" / "face_landmarker.task"
    detector = FaceDetector(model_path)

    per_clip_times = []
    total_frames, failed_frames = 0, 0
    method_counts = {"mediapipe": 0, "retinaface": 0, "none": 0}
    contact_sheet_rows = []

    print("\n" + "=" * 70)
    print(f"FACE DETECTION + CROP + OCCLUSION, target ~{TARGET_FPS}fps per clip")
    print("=" * 70)

    for _, r in pilot.iterrows():
        t0 = time.perf_counter()
        container = av.open(r["video_path"])
        vstream = container.streams.video[0]
        container_fps = float(vstream.average_rate) if vstream.average_rate else TARGET_FPS
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()

        idxs = sample_frame_indices(len(frames), container_fps, TARGET_FPS)
        # Detect on EVERY sampled frame (not just until one succeeds) — this
        # loop's timing is what extrapolates to the full 7,439-clip estimate,
        # and it must reflect the real per-frame workload, not a shortcut.
        frame_results = {}
        for i in idxs:
            total_frames += 1
            det = detector.detect(frames[i])
            if not det.success:
                failed_frames += 1
                method_counts["none"] += 1
                continue
            method_counts[det.method] += 1
            crop, transform = crop_face(frames[i], det.bbox, out_size=224)
            lm_crop = transform(det.landmarks_478) if det.landmarks_478 is not None else None
            frame_results[i] = {"crop": crop, "landmarks": lm_crop}

        best_frame = None
        if frame_results:
            mid_idx = idxs[len(idxs) // 2]
            nearest = min(frame_results.keys(), key=lambda i: abs(i - mid_idx))
            best_frame = frame_results[nearest]

        elapsed = time.perf_counter() - t0
        per_clip_times.append({"clip_id": r["clip_id"], "n_frames_sampled": len(idxs), "elapsed_s": elapsed})
        print(f"  {r['clip_id']}: {len(idxs)} frames sampled, {elapsed:.2f}s, best_frame={'ok' if best_frame else 'FAILED'}")

        if best_frame is not None and len(contact_sheet_rows) < N_CONTACT_SHEET_CLIPS:
            row = {"clip_id": r["clip_id"], "original": best_frame["crop"]}
            for occ in OCCLUSION_TYPES:
                row[occ] = apply_occlusion(occ, best_frame["crop"], best_frame["landmarks"], OCCLUSION_SEVERITY, r["clip_id"], OCCLUSION_SEED)
            contact_sheet_rows.append(row)

    times_df = pd.DataFrame(per_clip_times)
    failure_rate = failed_frames / total_frames if total_frames else float("nan")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total frames sampled across {len(pilot)} clips: {total_frames}")
    print(f"Detection method breakdown: {method_counts}")
    print(f"Per-frame detection failure rate: {failure_rate:.4f} ({failed_frames}/{total_frames})")
    print(f"\nPer-clip wall time (this Mac): mean={times_df['elapsed_s'].mean():.2f}s, "
          f"median={times_df['elapsed_s'].median():.2f}s, max={times_df['elapsed_s'].max():.2f}s")

    n_total_clips = 7439
    est_total_s = times_df["elapsed_s"].mean() * n_total_clips
    print(f"\nExtrapolated estimate for all {n_total_clips} clips: "
          f"{est_total_s / 60:.1f} min ({est_total_s / 3600:.2f} hours), "
          f"based on mean per-clip time x {n_total_clips}")

    contact_sheet_path = REPO_ROOT / "outputs" / "phase2_pilot" / "contact_sheet.png"
    build_contact_sheet(contact_sheet_rows, contact_sheet_path)
    print(f"\nContact sheet saved to: {contact_sheet_path}")

    ffprobe_df.to_csv(REPO_ROOT / "outputs" / "phase2_pilot" / "ffprobe_summary.csv", index=False)
    times_df.to_csv(REPO_ROOT / "outputs" / "phase2_pilot" / "per_clip_times.csv", index=False)


if __name__ == "__main__":
    main()
