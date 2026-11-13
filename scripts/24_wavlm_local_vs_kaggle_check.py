#!/usr/bin/env python
"""Dress-rehearsal item 1: run local inference with the 3 downloaded WavLM
checkpoints on the VAL clips (from this project's local audio files, not
Kaggle's) and compare against the per-clip probabilities Kaggle itself
produced (outputs/wavlm_val_probs.csv). Uses the exact same preprocessing
as notebooks/wavlm_finetune_kaggle.ipynb: mono, resampled to 16 kHz (local
files are already 16 kHz, so this is a no-op here), truncated to 6.0s,
Wav2Vec2FeatureExtractor + WavLMModel (frozen CNN encoder) + the same
small classification head, and the SAME per-clip order and batch_size=16
(no shuffle) the notebook's val_loader used -- reconstructed from
wavlm_val_probs.csv's own row order (identical across all 3 seeds, since
the notebook built val_df/val_loader once and reused it for every seed),
so any batch-boundary padding effects are reproduced identically rather
than introduced by a different grouping.

Does not touch test actors.

Usage:
    python scripts/24_wavlm_local_vs_kaggle_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.splits import load_split
from fer.wavlm_model import TARGET_SR, WavLMClassifier, get_feature_extractor, truncate_to_max_seconds

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"
AUDIO_DIR = REPO_ROOT / "data" / "CREMA-D" / "AudioWAV"
CLASS_ORDER = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
SEEDS = [0, 1, 2]
BATCH_SIZE = 16  # matches the notebook's val_loader: batch_size=8 (train) * 2


def load_waveform(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, TARGET_SR).numpy()
    return truncate_to_max_seconds(wav, TARGET_SR)


class AudioClipDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.rows = list(zip(df["wav_path"], df["emotion"], df["clip_id"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, emotion, clip_id = self.rows[i]
        return load_waveform(path), LABEL_TO_IDX[emotion], clip_id


def main() -> None:
    data_cfg = __import__("yaml").safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    sf_split = load_split(REPO_ROOT / data_cfg["paths"]["splits_dir"] / "split_v1.json")
    if sf_split.status != "frozen":
        raise RuntimeError("split not frozen")
    val_actors = set(sf_split.assignment["val"])
    test_actors = set(sf_split.assignment["test"])

    kaggle_df = pd.read_csv(OUT_DIR / "wavlm_val_probs.csv")
    kaggle_actors = {c.split("_")[0] for c in kaggle_df["clip_id"]}
    assert not kaggle_actors & test_actors, "Kaggle val probs touch test actors"
    assert kaggle_actors == val_actors, "Kaggle val actor set != official val actor set"

    # Reconstruct the exact clip order the notebook's val_loader used (identical for all 3 seeds).
    clip_order = kaggle_df[kaggle_df["seed"] == 0]["clip_id"].tolist()
    for s in [1, 2]:
        assert kaggle_df[kaggle_df["seed"] == s]["clip_id"].tolist() == clip_order, f"seed {s} row order differs from seed 0 -- cannot assume a fixed val_loader order"
    print(f"reconstructed val clip order from wavlm_val_probs.csv: {len(clip_order)} clips, identical across all 3 seeds.")

    metadata_df = pd.read_csv(OUT_DIR / "metadata.csv", dtype={"actor_id": str})
    clip_to_emotion = dict(zip(metadata_df["clip_id"], metadata_df["emotion"]))
    rows = []
    for clip_id in clip_order:
        wav_path = AUDIO_DIR / f"{clip_id}.wav"
        assert wav_path.exists(), f"local audio file missing: {wav_path}"
        rows.append({"clip_id": clip_id, "emotion": clip_to_emotion[clip_id], "wav_path": str(wav_path)})
    val_df = pd.DataFrame(rows)
    print(f"local val_df built: {len(val_df)} clips, all with local audio present.")

    feature_extractor = get_feature_extractor()

    def collate_batch(batch):
        wavs, labels, clip_ids = zip(*batch)
        processed = feature_extractor(list(wavs), sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        return processed["input_values"], processed["attention_mask"], torch.tensor(labels, dtype=torch.long), clip_ids

    loader = DataLoader(AudioClipDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch, num_workers=0)

    summary_rows = []
    all_diffs = []
    for seed in SEEDS:
        ckpt_path = REPO_ROOT / "models" / "wavlm" / f"wavlm_base_plus_seed{seed}.pt"
        model = WavLMClassifier(len(CLASS_ORDER))
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
        model.eval()

        local_probs: dict[str, np.ndarray] = {}
        local_pred: dict[str, str] = {}
        with torch.no_grad():
            for input_values, attention_mask, labels, clip_ids in loader:
                probs = torch.softmax(model(input_values, attention_mask), dim=1).numpy()
                for cid, p in zip(clip_ids, probs):
                    local_probs[cid] = p
                    local_pred[cid] = CLASS_ORDER[int(np.argmax(p))]

        kaggle_sub = kaggle_df[kaggle_df["seed"] == seed].set_index("clip_id")
        prob_cols = [f"prob_{c}" for c in CLASS_ORDER]
        max_abs_diff = 0.0
        n_pred_mismatch = 0
        for cid in clip_order:
            kaggle_p = kaggle_sub.loc[cid, prob_cols].to_numpy(dtype=np.float64)
            diff = np.max(np.abs(local_probs[cid] - kaggle_p))
            max_abs_diff = max(max_abs_diff, diff)
            if local_pred[cid] != kaggle_sub.loc[cid, "y_pred"]:
                n_pred_mismatch += 1
        all_diffs.append(max_abs_diff)

        y_true = val_df["emotion"].tolist()
        y_pred_local = [local_pred[cid] for cid in clip_order]
        local_macro_f1 = f1_score(y_true, y_pred_local, labels=CLASS_ORDER, average="macro")
        kaggle_macro_f1 = f1_score(y_true, kaggle_sub.loc[clip_order, "y_pred"].tolist(), labels=CLASS_ORDER, average="macro")

        print(f"\nseed={seed}:")
        print(f"  max abs probability difference (local vs Kaggle, over all {len(clip_order)} clips x 6 classes): {max_abs_diff:.6f}")
        print(f"  argmax prediction mismatches: {n_pred_mismatch}/{len(clip_order)}")
        print(f"  local macro_f1={local_macro_f1:.4f}  kaggle macro_f1={kaggle_macro_f1:.4f}  exact match: {local_macro_f1 == kaggle_macro_f1}")
        summary_rows.append({"seed": seed, "max_abs_prob_diff": max_abs_diff, "n_pred_mismatch": n_pred_mismatch, "local_macro_f1": local_macro_f1, "kaggle_macro_f1": kaggle_macro_f1})

    print("\n" + "=" * 70)
    print(f"overall max abs probability difference across all 3 seeds: {max(all_diffs):.6f}")
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "wavlm_local_vs_kaggle_check.csv", index=False)


if __name__ == "__main__":
    main()
