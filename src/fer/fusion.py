"""Shared Phase 5 fusion utilities: actor-level 2-fold splitting,
clean-data-only calibration cross-fitting, and the four fusion-weighting
methods used in the comparison (fixed, confidence-weighted by
max-probability, confidence-weighted by negative entropy, and a learned
logistic-regression gate). No other fusion variants are added here —
these four are fixed in advance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .calibration import apply_temperature, fit_temperature


def two_fold_actor_split(actors: list[str]) -> tuple[set[str], set[str]]:
    ordered = sorted(actors)
    return set(ordered[0::2]), set(ordered[1::2])


def probs_matrix(df: pd.DataFrame, class_order: list[str]) -> np.ndarray:
    return df[[f"prob_{c}" for c in class_order]].to_numpy(dtype=np.float64)


def y_true_idx(df: pd.DataFrame, class_order: list[str]) -> np.ndarray:
    label_to_idx = {c: i for i, c in enumerate(class_order)}
    return df["y_true"].map(label_to_idx).to_numpy()


def cross_fit_calibrate_clean_only(
    df: pd.DataFrame, clean_mask: np.ndarray, clip_actor: dict, fold_a: set, fold_b: set, class_order: list[str],
) -> tuple[np.ndarray, dict]:
    """Temperature fit on CLEAN rows only, 2-fold cross-fit over actors,
    then applied to EVERY row (clean or occluded) for a clip in that
    fold. Returns (calibrated_probs, {'T_for_fold_a': .., 'T_for_fold_b': ..})."""
    actor_of_row = df["clip_id"].map(clip_actor)
    in_a = actor_of_row.isin(fold_a).to_numpy()
    in_b = actor_of_row.isin(fold_b).to_numpy()
    P, yt = probs_matrix(df, class_order), y_true_idx(df, class_order)

    T_for_a = fit_temperature(P[clean_mask & in_b], yt[clean_mask & in_b])  # fit on B, applied to A
    T_for_b = fit_temperature(P[clean_mask & in_a], yt[clean_mask & in_a])  # fit on A, applied to B

    calibrated = P.copy()
    calibrated[in_a] = apply_temperature(P[in_a], T_for_a)
    calibrated[in_b] = apply_temperature(P[in_b], T_for_b)
    return calibrated, {"T_for_fold_a": T_for_a, "T_for_fold_b": T_for_b}


def maxprob_confidence(probs: np.ndarray) -> np.ndarray:
    return probs.max(axis=-1)


def negentropy_confidence(probs: np.ndarray, n_classes: int) -> np.ndarray:
    """0 at a uniform distribution, log(n_classes) at a one-hot
    distribution -- the negative entropy, shifted non-negative so it reads
    like a confidence score the same direction as max-probability."""
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    h = -(p * np.log(p)).sum(axis=-1)
    return np.log(n_classes) - h


def fit_gate_cross_fit(face_conf: np.ndarray, audio_conf: np.ndarray, face_correct: np.ndarray, actor_of_row: pd.Series, fold_a: set, fold_b: set) -> np.ndarray:
    """Pooled (clean + every occlusion condition) logistic-regression gate:
    P(face branch is correct | [face_conf, audio_conf]), used as the face
    fusion weight. 2-fold cross-fit over actors: the gate fit on fold B is
    applied to fold A and vice versa, so no clip is scored by a gate fit
    on itself or its own fold."""
    in_a = actor_of_row.isin(fold_a).to_numpy()
    in_b = actor_of_row.isin(fold_b).to_numpy()
    X = np.stack([face_conf, audio_conf], axis=1)
    y = face_correct.astype(int)

    w_face = np.zeros(len(X))
    for train_mask, apply_mask in [(in_b, in_a), (in_a, in_b)]:
        if len(np.unique(y[train_mask])) < 2:
            w_face[apply_mask] = float(y[train_mask].mean())
            continue
        lr = LogisticRegression().fit(X[train_mask], y[train_mask])
        pos_col = list(lr.classes_).index(1)
        w_face[apply_mask] = lr.predict_proba(X[apply_mask])[:, pos_col]
    return w_face
