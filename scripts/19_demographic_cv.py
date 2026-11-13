#!/usr/bin/env python
"""Demographic fairness check: actor-level 5-fold cross-validation over
the 79 non-test actors (train+val combined), for audio-only, face-only
(clean-trained and aug_all), and fixed 50/50 fusion (both face variants).
Reports macro-F1 by sex, age band, and race on CLEAN clips only (no
occlusion conditions here -- this checks base-task fairness, not
occlusion robustness).

Scope reduction (agreed before running, given a 10-17 hour full-spec
estimate): 1 seed per fold instead of 3, for the face models only (audio
has no meaningful seed variance -- lbfgs is deterministic). This trades
away this analysis's mean+/-std across seeds; that limitation is stated
in RESULTS.md. 10 face-model training runs total (2 variants x 5 folds),
not 30.

Within each outer fold, the ~63-64 training actors are further split
80/20 (deterministic, by sorted actor id) into an inner-train/inner-monitor
set, so early stopping never sees the outer held-out fold's actors --
the only actors a fold's final metric is computed on are actors that
fold's model never saw at all, for any purpose.

Usage:
    python scripts/19_demographic_cv.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.audio_features import feature_columns, load_features
from fer.face_model import build_eval_rows, predict_clip_level, train_one_seed
from fer.face_model_augmented import train_one_seed_augmented
from fer.splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_ORDER = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
N_FOLDS = 5
SEED = 0
AUG_ALL_TYPES = ["mask", "sunglasses", "patch", "hand_over_mouth"]


def make_folds(actors: list[str], n_folds: int) -> list[list[str]]:
    ordered = sorted(actors)
    return [list(a) for a in np.array_split(ordered, n_folds)]


def inner_split(actors: list[str], monitor_frac: float = 0.2) -> tuple[list[str], list[str]]:
    ordered = sorted(actors)
    k = max(1, round(1 / monitor_frac))
    monitor = ordered[::k]
    train = [a for a in ordered if a not in set(monitor)]
    return train, monitor


def main() -> None:
    data_cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    audio_cfg = yaml.safe_load((REPO_ROOT / "configs" / "audio_baseline.yaml").read_text())
    sf = load_split(REPO_ROOT / data_cfg["paths"]["splits_dir"] / "split_v1.json")
    if sf.status != "frozen":
        raise RuntimeError("split is not frozen -- refusing to proceed")
    test_actors = set(sf.assignment["test"])
    non_test_actors = sorted(set(sf.assignment["train"]) | set(sf.assignment["val"]))
    print(f"non-test actors: {len(non_test_actors)}")

    metadata_df_all = pd.read_csv(REPO_ROOT / data_cfg["paths"]["outputs_dir"] / "metadata.csv", dtype={"actor_id": str})
    metadata_df = metadata_df_all[metadata_df_all["actor_id"].isin(non_test_actors)].copy()
    assert not set(metadata_df["actor_id"]) & test_actors, "test actors leaked into metadata after filtering"

    features_df = load_features(REPO_ROOT / audio_cfg["features"]["cache_path"])
    feat_cols = feature_columns(features_df)
    assert not set(features_df["actor_id"]) & test_actors, "test actors leaked into audio features"

    crops_dir = REPO_ROOT / "cache" / "face_crops"
    landmarks_dir = REPO_ROOT / "cache" / "landmarks"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    folds = make_folds(non_test_actors, N_FOLDS)
    for i, f in enumerate(folds):
        print(f"fold {i}: {len(f)} held-out actors")

    all_rows = []  # clip_id, actor_id, y_true, method, y_pred
    fold_timings = []

    for fold_idx, held_out in enumerate(folds):
        held_out_set = set(held_out)
        train_pool = [a for a in non_test_actors if a not in held_out_set]
        inner_train_actors, inner_monitor_actors = inner_split(train_pool)
        print(f"\n{'='*70}\nFOLD {fold_idx}: held_out={len(held_out)} train_pool={len(train_pool)} "
              f"inner_train={len(inner_train_actors)} inner_monitor={len(inner_monitor_actors)}\n{'='*70}")

        held_out_metadata = metadata_df[metadata_df["actor_id"].isin(held_out_set)]
        inner_train_metadata = metadata_df[metadata_df["actor_id"].isin(inner_train_actors)]
        inner_monitor_metadata = metadata_df[metadata_df["actor_id"].isin(inner_monitor_actors)]
        held_out_true = dict(zip(held_out_metadata["clip_id"], held_out_metadata["emotion"]))
        held_out_actor_of_clip = dict(zip(held_out_metadata["clip_id"], held_out_metadata["actor_id"]))

        # --- audio-only ---
        t0 = time.perf_counter()
        train_pool_set = set(train_pool)
        audio_train_df = features_df[features_df["actor_id"].isin(train_pool_set)]
        audio_held_df = features_df[features_df["actor_id"].isin(held_out_set)]
        scaler = StandardScaler().fit(audio_train_df[feat_cols].to_numpy(dtype=np.float64))
        X_train = scaler.transform(audio_train_df[feat_cols].to_numpy(dtype=np.float64))
        X_held = scaler.transform(audio_held_df[feat_cols].to_numpy(dtype=np.float64))
        lr = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, C=1.0, random_state=SEED).fit(
            X_train, audio_train_df["emotion"].to_numpy()
        )
        audio_pred = lr.predict(X_held)
        audio_clip_ids = audio_held_df["clip_id"].tolist()
        audio_pred_by_clip = dict(zip(audio_clip_ids, audio_pred))
        audio_proba = lr.predict_proba(X_held)
        audio_classes = list(lr.classes_)
        audio_proba_by_clip = {cid: dict(zip(audio_classes, p)) for cid, p in zip(audio_clip_ids, audio_proba)}
        print(f"  audio-only: {time.perf_counter()-t0:.1f}s, {len(audio_clip_ids)} held-out clips")

        for cid in audio_clip_ids:
            all_rows.append({"clip_id": cid, "actor_id": held_out_actor_of_clip[cid],
                              "fold": fold_idx, "y_true": held_out_true[cid], "method": "audio_only", "y_pred": audio_pred_by_clip[cid]})

        # --- face clean-trained ---
        t0 = time.perf_counter()
        result_clean = train_one_seed(inner_train_metadata, inner_monitor_metadata, crops_dir, CLASS_ORDER, seed=SEED, device=device)
        model_clean = result_clean["model"]
        held_eval_rows = build_eval_rows(held_out_metadata, crops_dir, CLASS_ORDER)
        clip_ids_c, y_true_c, y_pred_c = predict_clip_level(model_clean, held_eval_rows, CLASS_ORDER, device)
        face_clean_dur = time.perf_counter() - t0
        print(f"  face clean-trained: {face_clean_dur/60:.1f} min, best_epoch={result_clean['best_epoch']}")

        # per-clip probabilities for fusion (re-run predict with probs)
        face_clean_probs = _clip_probs(model_clean, held_eval_rows, CLASS_ORDER, device)

        for cid, yt, yp in zip(clip_ids_c, y_true_c, y_pred_c):
            all_rows.append({"clip_id": cid, "actor_id": held_out_actor_of_clip[cid],
                              "fold": fold_idx, "y_true": yt, "method": "face_only_clean", "y_pred": yp})

        # --- face aug_all ---
        t0 = time.perf_counter()
        result_aug = train_one_seed_augmented(inner_train_metadata, inner_monitor_metadata, crops_dir, landmarks_dir, CLASS_ORDER, seed=SEED, device=device, allowed_types=AUG_ALL_TYPES)
        model_aug = result_aug["model"]
        clip_ids_a, y_true_a, y_pred_a = predict_clip_level(model_aug, held_eval_rows, CLASS_ORDER, device)
        face_aug_dur = time.perf_counter() - t0
        print(f"  face aug_all: {face_aug_dur/60:.1f} min, best_epoch={result_aug['best_epoch']}")

        face_aug_probs = _clip_probs(model_aug, held_eval_rows, CLASS_ORDER, device)

        for cid, yt, yp in zip(clip_ids_a, y_true_a, y_pred_a):
            all_rows.append({"clip_id": cid, "actor_id": held_out_actor_of_clip[cid],
                              "fold": fold_idx, "y_true": yt, "method": "face_only_aug_all", "y_pred": yp})

        # --- fixed 50/50 fusion, raw (uncalibrated) probabilities, both face variants ---
        for face_probs, tag in [(face_clean_probs, "fusion_clean"), (face_aug_probs, "fusion_aug_all")]:
            for cid in face_probs:
                if cid not in audio_proba_by_clip:
                    continue
                fused = {c: 0.5 * face_probs[cid].get(c, 0.0) + 0.5 * audio_proba_by_clip[cid].get(c, 0.0) for c in CLASS_ORDER}
                pred = max(fused, key=fused.get)
                all_rows.append({"clip_id": cid, "actor_id": held_out_actor_of_clip[cid],
                                  "fold": fold_idx, "y_true": held_out_true[cid], "method": tag, "y_pred": pred})

        fold_timings.append({"fold": fold_idx, "face_clean_min": face_clean_dur / 60, "face_aug_all_min": face_aug_dur / 60})

    results_df = pd.DataFrame(all_rows)
    out_path = REPO_ROOT / "outputs" / "phase_demographic_cv_predictions.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(results_df)} predictions to {out_path}")
    pd.DataFrame(fold_timings).to_csv(REPO_ROOT / "outputs" / "phase_demographic_cv_timings.csv", index=False)

    # overall macro-F1 per method (sanity check, pooled OOF over all 79 actors)
    print("\noverall OOF macro-F1 per method:")
    for method in results_df["method"].unique():
        sub = results_df[results_df["method"] == method]
        mf1 = f1_score(sub["y_true"], sub["y_pred"], labels=CLASS_ORDER, average="macro")
        print(f"  {method:20s} macro_f1={mf1:.4f}  n={len(sub)}")


def _clip_probs(model, rows, class_order, device, batch_size=128):
    """Per-clip averaged softmax probabilities (same aggregation as
    predict_clip_level, but returns the full probability dict instead of
    just the argmax -- needed for fusion)."""
    import torch
    from torch.utils.data import DataLoader

    from fer.face_model import FrameDataset

    model.eval()
    ds = FrameDataset(rows, class_order)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    clip_prob_sums: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for images, _labels, clip_ids in loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1).cpu().numpy()
            for p, cid in zip(probs, clip_ids):
                clip_prob_sums.setdefault(cid, np.zeros(len(class_order)))
                clip_prob_sums[cid] += p
    return {cid: {c: v[i] / v.sum() for i, c in enumerate(class_order)} for cid, v in clip_prob_sums.items()}


if __name__ == "__main__":
    main()
