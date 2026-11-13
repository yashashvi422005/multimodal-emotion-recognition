#!/usr/bin/env python
"""Quick personal webcam check -- NOT part of the evaluation, NOT a new
result. No training, tuning, or test-set access happens here. Uses the
already-frozen occlusion-augmented face model (aug_all, seed 0) and the
already-frozen WavLM audio model (seed 0), with exactly the same
preprocessing as the evaluation pipeline:

  - face: fer.face_pipeline.FaceDetector.detect() (MediaPipe, RetinaFace
    fallback) -> fer.face_pipeline.crop_face(bbox, out_size=224, margin=0.3)
    -> ImageNet-normalized tensor (fer.face_model.load_image_tensor's
    normalization, applied in-memory instead of from a cached file).
  - audio: mono, resampled to 16 kHz, fed through the same
    Wav2Vec2FeatureExtractor + WavLMModel (frozen CNN encoder) + small
    classification head used in notebooks/wavlm_finetune_kaggle.ipynb.

Fusion here is a plain fixed-50/50 average of the two branches' raw
softmax outputs -- there is no held-out data to cross-fit a temperature
or a gate on a single live clip, so (unlike the pre-registered evaluation)
no calibration is applied. This is a live demo, not a scored result.

Nothing is written to disk: video frames and the audio buffer are kept in
memory only, for the duration of one 4-second recording, then discarded.

Usage:
    python scripts/webcam_try.py
Controls:
    q  quit
    r  record 4 seconds of video+audio, then show face/voice/fused results
    (any key, once results are shown)  back to live view
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import timm
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.face_pipeline import FaceDetector, crop_face
from fer.wavlm_model import TARGET_SR, WavLMClassifier, get_feature_extractor

REPO_ROOT = Path(__file__).resolve().parents[1]
FACE_CFG = yaml.safe_load((REPO_ROOT / "configs" / "face_model.yaml").read_text())
CLASS_ORDER = FACE_CFG["class_order"]
MEDIAPIPE_MODEL_PATH = REPO_ROOT / "models" / "mediapipe" / "face_landmarker.task"
FACE_CKPT_PATH = REPO_ROOT / "models" / "face_resnet18_aug_all" / "resnet18_aug_all_seed0.pt"
WAVLM_CKPT_PATH = REPO_ROOT / "models" / "wavlm" / "wavlm_base_plus_seed0.pt"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
RECORD_SECONDS = 4.0
LIVE_INFER_INTERVAL = 0.3  # seconds between face-model calls in live mode ("a few times per second")
N_RECORD_FRAMES_FOR_FACE = 8  # matches frames_per_clip_eval in configs/face_model.yaml
DISCLAIMER = "Research demo, trained on acted speech (CREMA-D), not validated on live input."
DEVICE = torch.device("cpu")  # CPU is the reliable default for this one-off check


def load_models():
    print("loading face model (aug_all, seed 0)...")
    face_model = timm.create_model("resnet18", pretrained=False, num_classes=len(CLASS_ORDER))
    face_model.load_state_dict(torch.load(FACE_CKPT_PATH, map_location="cpu", weights_only=False))
    face_model.eval().to(DEVICE)

    print("loading WavLM model (seed 0) -- only a small config is fetched from the HF Hub, not pretrained weights...")
    feature_extractor = get_feature_extractor()
    wavlm_model = WavLMClassifier(len(CLASS_ORDER))
    wavlm_model.load_state_dict(torch.load(WAVLM_CKPT_PATH, map_location="cpu", weights_only=False))
    wavlm_model.eval().to(DEVICE)

    print("loading face detector (MediaPipe)...")
    detector = FaceDetector(MEDIAPIPE_MODEL_PATH)
    print("all models loaded.\n")
    return face_model, wavlm_model, feature_extractor, detector


def crop_to_tensor(crop_rgb: np.ndarray) -> torch.Tensor:
    img = crop_rgb.astype(np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1)
    return (t - IMAGENET_MEAN) / IMAGENET_STD


@torch.no_grad()
def face_probs_for_frame(frame_rgb: np.ndarray, face_model, detector: FaceDetector):
    """Returns (probs or None, bbox or None) -- None if no face detected,
    same no-detection-abstains convention documented in configs/face_model.yaml."""
    det = detector.detect(frame_rgb)
    if not det.success:
        return None, None
    crop, _ = crop_face(frame_rgb, det.bbox, out_size=224, margin=0.3)
    t = crop_to_tensor(crop).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(face_model(t), dim=1).cpu().numpy()[0]
    return probs, det.bbox


@torch.no_grad()
def wavlm_probs_for_audio(wav: np.ndarray, wavlm_model, feature_extractor) -> np.ndarray:
    processed = feature_extractor([wav], sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = wavlm_model(processed["input_values"].to(DEVICE), processed["attention_mask"].to(DEVICE))
    return torch.softmax(logits, dim=1).cpu().numpy()[0]


def draw_disclaimer(frame_bgr: np.ndarray) -> None:
    h, w = frame_bgr.shape[:2]
    cv2.rectangle(frame_bgr, (0, h - 24), (w, h), (0, 0, 0), -1)
    cv2.putText(frame_bgr, DISCLAIMER, (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def draw_probs_overlay(frame_bgr: np.ndarray, probs: np.ndarray | None, bbox: tuple | None, label: str) -> None:
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 220, 0), 2)
    y = 20
    cv2.putText(frame_bgr, label, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    y += 20
    if probs is None:
        cv2.putText(frame_bgr, "no face detected", (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return
    for cls, p in zip(CLASS_ORDER, probs):
        cv2.putText(frame_bgr, f"{cls:8s} {p:.2f}", (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18


def record_and_fuse(cap, face_model, wavlm_model, feature_extractor, detector) -> None:
    print("recording 4 seconds of video + audio (kept in memory only)...")
    audio_buf = sd.rec(int(RECORD_SECONDS * TARGET_SR), samplerate=TARGET_SR, channels=1, dtype="float32")

    frames_rgb = []
    t0 = time.time()
    while time.time() - t0 < RECORD_SECONDS:
        ret, frame_bgr = cap.read()
        if not ret:
            continue
        frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        preview = frame_bgr.copy()
        cv2.putText(preview, f"RECORDING {time.time() - t0:.1f}s / {RECORD_SECONDS:.0f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        draw_disclaimer(preview)
        cv2.imshow("FER webcam demo", preview)
        cv2.waitKey(1)
    sd.wait()
    print(f"captured {len(frames_rgb)} video frames and {audio_buf.shape[0]} audio samples.")

    if frames_rgb:
        idx = np.linspace(0, len(frames_rgb) - 1, min(N_RECORD_FRAMES_FOR_FACE, len(frames_rgb))).astype(int)
        chosen_frames = [frames_rgb[i] for i in idx]
    else:
        chosen_frames = []

    prob_sum = np.zeros(len(CLASS_ORDER))
    n_detected = 0
    last_bbox = None
    for fr in chosen_frames:
        p, bbox = face_probs_for_frame(fr, face_model, detector)
        if p is not None:
            prob_sum += p
            n_detected += 1
            last_bbox = bbox
    face_probs = (prob_sum / n_detected) if n_detected > 0 else np.full(len(CLASS_ORDER), 1.0 / len(CLASS_ORDER))
    print(f"face branch: detected a face in {n_detected}/{len(chosen_frames)} sampled frames.")

    audio_mono = audio_buf[:, 0]
    audio_probs = wavlm_probs_for_audio(audio_mono, wavlm_model, feature_extractor)

    fused_probs = 0.5 * face_probs + 0.5 * audio_probs

    canvas = np.zeros((420, 760, 3), dtype=np.uint8)
    cv2.putText(canvas, "RESULT (fixed 50/50 fusion, raw softmax, no calibration)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    columns = [
        ("FACE ONLY (aug_all, seed 0)", face_probs, 10, (0, 255, 0) if n_detected > 0 else (0, 0, 255)),
        ("VOICE ONLY (WavLM, seed 0)", audio_probs, 270, (0, 255, 255)),
        ("FUSED (50/50)", fused_probs, 530, (255, 255, 0)),
    ]
    for title, probs, x, color in columns:
        cv2.putText(canvas, title, (x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        top_idx = int(np.argmax(probs))
        cv2.putText(canvas, f"-> {CLASS_ORDER[top_idx]}", (x, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        y = 110
        for cls, p in zip(CLASS_ORDER, probs):
            bar_len = int(p * 150)
            cv2.rectangle(canvas, (x, y - 10), (x + bar_len, y), color, -1)
            cv2.putText(canvas, f"{cls:8s} {p:.2f}", (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            y += 32
    if n_detected == 0:
        cv2.putText(canvas, "(no face detected in any sampled frame -- face branch abstained to uniform)", (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    draw_disclaimer(canvas)
    cv2.putText(canvas, "press any key to go back to live view", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imshow("FER webcam demo", canvas)
    cv2.waitKey(0)


def main() -> None:
    face_model, wavlm_model, feature_extractor, detector = load_models()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("could not open the webcam (index 0) -- check macOS Camera permission for this app")

    print("live view running. Press 'r' to record 4s, 'q' to quit.")
    last_infer = 0.0
    cached_probs, cached_bbox = None, None
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                continue
            now = time.time()
            if now - last_infer > LIVE_INFER_INTERVAL:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                cached_probs, cached_bbox = face_probs_for_frame(frame_rgb, face_model, detector)
                last_infer = now

            draw_probs_overlay(frame_bgr, cached_probs, cached_bbox, "face-only (live)")
            draw_disclaimer(frame_bgr)
            cv2.putText(frame_bgr, "r: record 4s   q: quit", (6, frame_bgr.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow("FER webcam demo", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                record_and_fuse(cap, face_model, wavlm_model, feature_extractor, detector)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
