#!/usr/bin/env python
"""Phase 5, development version: first confidence-weighted late-fusion
result, VAL only (test untouched).

1. Audio branch: refit the exact Phase 3 eGeMAPS + StandardScaler +
   LogisticRegression pipeline (deterministic — std~=0 across seeds, so
   seed 0 is refit once) and take predict_proba on val. Audio has no
   occlusion variants (occluding the face doesn't touch the waveform),
   so its per-clip probabilities are reused unchanged across every
   "condition" in the comparison below.
2. Face branch: read outputs/phase4_face_probs.parquet (written by
   scripts/13), which already has clean + every occlusion condition,
   for all 3 seeds.
3. Calibration: temperature-scale each branch (fer.calibration), fit on
   CLEAN data only via 2-fold cross-fitting over val actors (fold A's
   temperature is fit on fold B and applied to fold A, and vice versa —
   so no clip is ever scored with a temperature fit on itself or its own
   fold). The same fold-assigned temperature (fit on clean) is then
   applied to that branch's occluded probabilities too, to see whether a
   clean-fit calibration holds up under occlusion.
4. Fusion: per clip, per condition, per seed — audio-only, face-only
   (both scored with fer.metrics.score_with_abstention — the sole
   official scoring rule, so face-only here reproduces scripts/13's
   table exactly, a built-in consistency check), fixed 50/50 late
   fusion, and confidence-weighted late fusion (per-clip weight from
   each branch's own calibrated max-probability, normalized to sum to
   1). Mean +/- std over 3 seeds.

Usage:
    python scripts/14_phase5_fusion_dev.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.audio_features import feature_columns, load_features
from fer.calibration import apply_temperature, fit_temperature
from fer.metrics import expected_calibration_error, score_with_abstention
from fer.splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]
FACE_CFG = yaml.safe_load((REPO_ROOT / "configs" / "face_model.yaml").read_text())
AUDIO_CFG = yaml.safe_load((REPO_ROOT / "configs" / "audio_baseline.yaml").read_text())
CLASS_ORDER = FACE_CFG["class_order"]
SEEDS = [0, 1, 2]
OUT_DIR = REPO_ROOT / "outputs"
FACE_PROBS_PATH = OUT_DIR / "phase4_face_probs.parquet"


def prob_cols() -> list[str]:
    return [f"prob_{c}" for c in CLASS_ORDER]


def probs_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[prob_cols()].to_numpy(dtype=np.float64)


def y_true_idx(df: pd.DataFrame) -> np.ndarray:
    label_to_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    return df["y_true"].map(label_to_idx).to_numpy()


def two_fold_actor_split(val_actors: list[str]) -> tuple[set[str], set[str]]:
    ordered = sorted(val_actors)
    return set(ordered[0::2]), set(ordered[1::2])


def cross_fit_calibrate(df: pd.DataFrame, clean_mask: np.ndarray, clip_actor: dict, fold_a: set, fold_b: set) -> tuple[np.ndarray, dict]:
    """Fits temperature on CLEAN rows only (2-fold cross-fit over val
    actors), then applies the clip's fold-assigned temperature to EVERY
    row for that clip (clean or occluded). Returns (calibrated_probs for
    all rows in df, {'T_fit_on_b_applied_to_a': .., 'T_fit_on_a_applied_to_b': ..})."""
    actor_of_row = df["clip_id"].map(clip_actor)
    in_fold_a = actor_of_row.isin(fold_a).to_numpy()
    in_fold_b = actor_of_row.isin(fold_b).to_numpy()

    clean_a = clean_mask & in_fold_a
    clean_b = clean_mask & in_fold_b
    P, yt = probs_matrix(df), y_true_idx(df)

    T_fit_on_b = fit_temperature(P[clean_b], yt[clean_b])   # applied to fold A
    T_fit_on_a = fit_temperature(P[clean_a], yt[clean_a])   # applied to fold B

    calibrated = P.copy()
    calibrated[in_fold_a] = apply_temperature(P[in_fold_a], T_fit_on_b)
    calibrated[in_fold_b] = apply_temperature(P[in_fold_b], T_fit_on_a)
    return calibrated, {"T_fit_on_b_applied_to_fold_a": T_fit_on_b, "T_fit_on_a_applied_to_fold_b": T_fit_on_a}


def ece_for_mask(df: pd.DataFrame, calibrated: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Returns (ece_before, ece_after) for the rows selected by mask."""
    P = probs_matrix(df)[mask]
    cal = calibrated[mask]
    y_pred_idx = np.argmax(P, axis=1)  # argmax unaffected by temperature scaling
    label_to_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    yt = df["y_true"].to_numpy()[mask]
    yt_idx = np.array([label_to_idx[c] for c in yt])
    abstained = df["abstained"].to_numpy()[mask] if "abstained" in df.columns else np.zeros(mask.sum(), dtype=bool)

    correct = (y_pred_idx == yt_idx) & (~abstained)
    conf_before = P.max(axis=1)
    conf_after = cal.max(axis=1)
    return expected_calibration_error(conf_before, correct), expected_calibration_error(conf_after, correct)


def fetch_audio_probs() -> pd.DataFrame:
    data_cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    sf = load_split(REPO_ROOT / data_cfg["paths"]["splits_dir"] / "split_v1.json")
    train_actors, val_actors, test_actors = set(sf.assignment["train"]), set(sf.assignment["val"]), set(sf.assignment["test"])

    features_df = load_features(REPO_ROOT / AUDIO_CFG["features"]["cache_path"])
    assert not set(features_df["actor_id"]) & test_actors

    train_df = features_df[features_df["actor_id"].isin(train_actors)]
    val_df = features_df[features_df["actor_id"].isin(val_actors)]
    cols = feature_columns(features_df)

    scaler = StandardScaler().fit(train_df[cols].to_numpy(dtype=np.float64))
    X_train = scaler.transform(train_df[cols].to_numpy(dtype=np.float64))
    X_val = scaler.transform(val_df[cols].to_numpy(dtype=np.float64))

    lr = LogisticRegression(
        class_weight=AUDIO_CFG["model"]["class_weight"], solver=AUDIO_CFG["model"]["solver"],
        max_iter=AUDIO_CFG["model"]["max_iter"], C=AUDIO_CFG["model"]["C"], random_state=0,
    ).fit(X_train, train_df["emotion"].to_numpy())

    proba = lr.predict_proba(X_val)
    class_to_col = {c: i for i, c in enumerate(lr.classes_)}
    reordered = proba[:, [class_to_col[c] for c in CLASS_ORDER]]

    rows = []
    for i, (_, r) in enumerate(val_df.iterrows()):
        row = {"clip_id": r["clip_id"], "actor_id": r["actor_id"], "y_true": r["emotion"]}
        for c, p in zip(CLASS_ORDER, reordered[i]):
            row[f"prob_{c}"] = float(p)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    data_cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    sf = load_split(REPO_ROOT / data_cfg["paths"]["splits_dir"] / "split_v1.json")
    val_actors = sf.assignment["val"]
    fold_a, fold_b = two_fold_actor_split(val_actors)
    print(f"val actors: {len(val_actors)}; fold A: {len(fold_a)}, fold B: {len(fold_b)}")

    print("fitting audio branch (eGeMAPS + LR, deterministic)...")
    audio_df = fetch_audio_probs()
    clip_actor_audio = dict(zip(audio_df["clip_id"], audio_df["actor_id"]))
    audio_df["abstained"] = False
    audio_calibrated, audio_T = cross_fit_calibrate(audio_df, np.ones(len(audio_df), dtype=bool), clip_actor_audio, fold_a, fold_b)
    audio_ece_before, audio_ece_after = ece_for_mask(audio_df, audio_calibrated, np.ones(len(audio_df), dtype=bool))
    print(f"audio temperatures: {audio_T}")
    print(f"audio ECE before/after: {audio_ece_before:.4f} / {audio_ece_after:.4f}")

    audio_P_raw = probs_matrix(audio_df)
    audio_lookup_raw = {cid: p for cid, p in zip(audio_df["clip_id"], audio_P_raw)}
    audio_lookup_cal = {cid: p for cid, p in zip(audio_df["clip_id"], audio_calibrated)}

    print("\nloading face probabilities...")
    face_df = pd.read_parquet(FACE_PROBS_PATH)
    conditions = sorted(
        {(row.occlusion_type, row.severity) for row in face_df[["occlusion_type", "severity"]].drop_duplicates().itertuples()},
        key=lambda x: (x[0] != "clean", x[0], (x[1] if pd.notna(x[1]) else -1)),
    )

    ece_rows, fusion_rows = [], []

    for seed in SEEDS:
        sub_all = face_df[face_df["seed"] == seed].copy()
        clip_actor_face = clip_actor_audio  # same val clips/actors as the audio branch
        clean_mask_all = (sub_all["occlusion_type"] == "clean").to_numpy()
        face_calibrated_all, face_T = cross_fit_calibrate(sub_all, clean_mask_all, clip_actor_face, fold_a, fold_b)
        sub_all = sub_all.reset_index(drop=True)
        print(f"seed {seed} face temperatures: {face_T}")

        for occ_type, severity in conditions:
            if occ_type == "clean":
                cond_mask = (sub_all["occlusion_type"] == "clean").to_numpy()
            else:
                cond_mask = ((sub_all["occlusion_type"] == occ_type) & (sub_all["severity"] == severity)).to_numpy()

            cond_df = sub_all[cond_mask].reset_index(drop=True)
            cond_calibrated = face_calibrated_all[cond_mask]

            ece_before, ece_after = ece_for_mask(sub_all, face_calibrated_all, cond_mask)
            ece_rows.append({"seed": seed, "occlusion_type": occ_type, "severity": severity, "branch": "face", "ece_before": ece_before, "ece_after": ece_after})

            # align audio rows to this condition's clip order
            audio_P_cal = np.stack([audio_lookup_cal[c] for c in cond_df["clip_id"]])
            audio_P_raw_aligned = np.stack([audio_lookup_raw[c] for c in cond_df["clip_id"]])
            face_P_raw = probs_matrix(cond_df)
            face_P_cal = cond_calibrated
            abstained = cond_df["abstained"].to_numpy()
            y_true = cond_df["y_true"].tolist()

            audio_pred = [CLASS_ORDER[i] for i in np.argmax(audio_P_raw_aligned, axis=1)]
            face_pred = [CLASS_ORDER[i] for i in np.argmax(face_P_raw, axis=1)]
            audio_mf1, audio_acc, _ = score_with_abstention(y_true, audio_pred, [False] * len(y_true), CLASS_ORDER)
            face_mf1, face_acc, _ = score_with_abstention(y_true, face_pred, abstained.tolist(), CLASS_ORDER)

            fused_fixed_P = 0.5 * audio_P_cal + 0.5 * face_P_cal
            fused_fixed_pred = [CLASS_ORDER[i] for i in np.argmax(fused_fixed_P, axis=1)]
            fixed_mf1, fixed_acc, _ = score_with_abstention(y_true, fused_fixed_pred, [False] * len(y_true), CLASS_ORDER)

            audio_conf = audio_P_cal.max(axis=1)
            face_conf = face_P_cal.max(axis=1)
            w_face = face_conf / (face_conf + audio_conf)
            w_audio = 1.0 - w_face
            fused_conf_P = w_audio[:, None] * audio_P_cal + w_face[:, None] * face_P_cal
            fused_conf_pred = [CLASS_ORDER[i] for i in np.argmax(fused_conf_P, axis=1)]
            conf_mf1, conf_acc, _ = score_with_abstention(y_true, fused_conf_pred, [False] * len(y_true), CLASS_ORDER)

            fusion_rows.append({
                "seed": seed, "occlusion_type": occ_type, "severity": severity,
                "audio_macro_f1": audio_mf1, "audio_accuracy": audio_acc,
                "face_macro_f1": face_mf1, "face_accuracy": face_acc,
                "fixed_fusion_macro_f1": fixed_mf1, "fixed_fusion_accuracy": fixed_acc,
                "conf_fusion_macro_f1": conf_mf1, "conf_fusion_accuracy": conf_acc,
                "mean_face_weight": float(w_face.mean()),
            })

    ece_df = pd.DataFrame(ece_rows)
    ece_df.to_csv(OUT_DIR / "phase5_ece_by_seed.csv", index=False)
    ece_agg = ece_df.groupby(["occlusion_type", "severity"], dropna=False).agg(
        ece_before_mean=("ece_before", "mean"), ece_before_std=("ece_before", "std"),
        ece_after_mean=("ece_after", "mean"), ece_after_std=("ece_after", "std"),
    ).reset_index()
    ece_agg.to_csv(OUT_DIR / "phase5_ece_table.csv", index=False)
    print("\n[Fix 5] face-branch ECE before/after calibration, mean +/- std over 3 seeds:")
    print(ece_agg.to_string(index=False))
    print(f"\naudio ECE before/after (condition-invariant): {audio_ece_before:.4f} / {audio_ece_after:.4f}")

    fusion_df = pd.DataFrame(fusion_rows)
    fusion_df.to_csv(OUT_DIR / "phase5_fusion_by_seed.csv", index=False)
    fusion_agg = fusion_df.groupby(["occlusion_type", "severity"], dropna=False).agg(
        audio_macro_f1_mean=("audio_macro_f1", "mean"), audio_macro_f1_std=("audio_macro_f1", "std"),
        face_macro_f1_mean=("face_macro_f1", "mean"), face_macro_f1_std=("face_macro_f1", "std"),
        fixed_fusion_macro_f1_mean=("fixed_fusion_macro_f1", "mean"), fixed_fusion_macro_f1_std=("fixed_fusion_macro_f1", "std"),
        conf_fusion_macro_f1_mean=("conf_fusion_macro_f1", "mean"), conf_fusion_macro_f1_std=("conf_fusion_macro_f1", "std"),
        mean_face_weight_mean=("mean_face_weight", "mean"), mean_face_weight_std=("mean_face_weight", "std"),
    ).reset_index()
    fusion_agg.to_csv(OUT_DIR / "phase5_fusion_table.csv", index=False)
    print("\n[Fix 6] fusion comparison, mean +/- std over 3 seeds:")
    print(fusion_agg.to_string(index=False))


if __name__ == "__main__":
    main()
