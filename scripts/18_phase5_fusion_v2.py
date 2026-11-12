#!/usr/bin/env python
"""Phase 5, round 2: fusion comparison across all three face-model
variants (clean-trained, aug_all, aug_no_mask), with the four fusion
methods fixed in advance — fixed 50/50, confidence-weighted (max-prob),
confidence-weighted (negative entropy), and a learned logistic-regression
gate. No other fusion variants are added.

Audio branch: same refit eGeMAPS+LR as round 1, calibrated once
(condition-invariant). Face branch: each (variant, seed)'s probabilities
(outputs/phase4_face_probs.parquet) are temperature-calibrated on CLEAN
data only via 2-fold actor cross-fitting (fer.fusion), same as round 1.

The learned gate uses max-probability confidence from both branches as
its two input features (the same "confidence" used by the max-prob
fusion method), predicting P(face branch correct | [face_conf,
audio_conf]) via logistic regression, fit POOLED across clean + every
occlusion condition (not clean-only — a gate fit only on clean would
never see occluded confidence patterns) with the same 2-fold actor
cross-fit as calibration.

Note: patch severity 0.9 has 493/1224 (40.3%) zero-detection clips, each
scored as an exactly-uniform face distribution — ANY confidence-weighted
method will push face's weight to ~0 there and fusion will converge to
~audio-only BY CONSTRUCTION, not because the weighting scheme "solved"
that condition.

Usage:
    python scripts/18_phase5_fusion_v2.py
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
from fer.fusion import cross_fit_calibrate_clean_only, fit_gate_cross_fit, maxprob_confidence, negentropy_confidence, probs_matrix, two_fold_actor_split, y_true_idx
from fer.metrics import score_with_abstention
from fer.splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]
FACE_CFG = yaml.safe_load((REPO_ROOT / "configs" / "face_model.yaml").read_text())
AUDIO_CFG = yaml.safe_load((REPO_ROOT / "configs" / "audio_baseline.yaml").read_text())
CLASS_ORDER = FACE_CFG["class_order"]
N_CLASSES = len(CLASS_ORDER)
SEEDS = [0, 1, 2]
VARIANTS = ["clean", "aug_all", "aug_no_mask"]
OUT_DIR = REPO_ROOT / "outputs"
FACE_PROBS_PATH = OUT_DIR / "phase4_face_probs.parquet"
EPS = 1e-9


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
        row = {"clip_id": r["clip_id"], "actor_id": r["actor_id"], "y_true": r["emotion"], "abstained": False}
        for c, p in zip(CLASS_ORDER, reordered[i]):
            row[f"prob_{c}"] = float(p)
        rows.append(row)
    return pd.DataFrame(rows)


def fuse_and_score(audio_cal: np.ndarray, face_cal: np.ndarray, w_face: np.ndarray, y_true: list, class_order: list[str]) -> tuple[float, float]:
    w_audio = 1.0 - w_face
    fused = w_audio[:, None] * audio_cal + w_face[:, None] * face_cal
    pred = [class_order[i] for i in np.argmax(fused, axis=1)]
    macro_f1, acc, _ = score_with_abstention(y_true, pred, [False] * len(y_true), class_order)
    return macro_f1, acc


def main() -> None:
    data_cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    sf = load_split(REPO_ROOT / data_cfg["paths"]["splits_dir"] / "split_v1.json")
    val_actors = sf.assignment["val"]
    fold_a, fold_b = two_fold_actor_split(val_actors)
    print(f"val actors: {len(val_actors)}; fold A: {len(fold_a)}, fold B: {len(fold_b)}")

    print("fitting audio branch...")
    audio_df = fetch_audio_probs()
    clip_actor = dict(zip(audio_df["clip_id"], audio_df["actor_id"]))
    audio_calibrated, audio_T = cross_fit_calibrate_clean_only(audio_df, np.ones(len(audio_df), dtype=bool), clip_actor, fold_a, fold_b, CLASS_ORDER)
    print(f"audio temperatures: {audio_T}")

    audio_raw = probs_matrix(audio_df, CLASS_ORDER)
    audio_lookup_raw = dict(zip(audio_df["clip_id"], audio_raw))
    audio_lookup_cal = dict(zip(audio_df["clip_id"], audio_calibrated))
    audio_lookup_conf_max = {cid: maxprob_confidence(v[None, :])[0] for cid, v in audio_lookup_cal.items()}
    audio_lookup_conf_ent = {cid: negentropy_confidence(v[None, :], N_CLASSES)[0] for cid, v in audio_lookup_cal.items()}

    print("loading face probabilities...")
    face_all = pd.read_parquet(FACE_PROBS_PATH)
    conditions = sorted(
        {(row.occlusion_type, row.severity) for row in face_all[["occlusion_type", "severity"]].drop_duplicates().itertuples()},
        key=lambda x: (x[0] != "clean", x[0], (x[1] if pd.notna(x[1]) else -1)),
    )

    all_rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            sub_all = face_all[(face_all["model_variant"] == variant) & (face_all["seed"] == seed)].reset_index(drop=True)
            if len(sub_all) == 0:
                print(f"  (skip: no rows for variant={variant} seed={seed})")
                continue
            clean_mask_all = (sub_all["occlusion_type"] == "clean").to_numpy()
            face_calibrated_all, face_T = cross_fit_calibrate_clean_only(sub_all, clean_mask_all, clip_actor, fold_a, fold_b, CLASS_ORDER)
            print(f"variant={variant} seed={seed} face temperatures: {face_T}")

            actor_of_row = sub_all["clip_id"].map(clip_actor)
            face_raw_all = probs_matrix(sub_all, CLASS_ORDER)
            face_pred_idx_all = np.argmax(face_raw_all, axis=1)
            y_true_all = sub_all["y_true"].to_numpy()
            abstained_all = sub_all["abstained"].to_numpy()
            label_to_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
            face_correct_all = np.array([CLASS_ORDER[p] == yt for p, yt in zip(face_pred_idx_all, y_true_all)]) & (~abstained_all)

            face_conf_max_all = maxprob_confidence(face_calibrated_all)
            face_conf_ent_all = negentropy_confidence(face_calibrated_all, N_CLASSES)
            audio_conf_max_all = np.array([audio_lookup_conf_max[c] for c in sub_all["clip_id"]])
            audio_conf_ent_all = np.array([audio_lookup_conf_ent[c] for c in sub_all["clip_id"]])

            # learned gate: fit POOLED across clean + every occlusion condition
            w_gate_all = fit_gate_cross_fit(face_conf_max_all, audio_conf_max_all, face_correct_all, actor_of_row, fold_a, fold_b)

            for occ_type, severity in conditions:
                if occ_type == "clean":
                    cond_mask = clean_mask_all
                else:
                    cond_mask = ((sub_all["occlusion_type"] == occ_type) & (sub_all["severity"] == severity)).to_numpy()
                if cond_mask.sum() == 0:
                    continue

                cond_df = sub_all[cond_mask]
                y_true = cond_df["y_true"].tolist()
                abstained = cond_df["abstained"].tolist()
                face_pred = [CLASS_ORDER[i] for i in face_pred_idx_all[cond_mask]]

                audio_cal_cond = np.stack([audio_lookup_cal[c] for c in cond_df["clip_id"]])
                audio_raw_cond = np.stack([audio_lookup_raw[c] for c in cond_df["clip_id"]])
                audio_pred = [CLASS_ORDER[i] for i in np.argmax(audio_raw_cond, axis=1)]

                audio_mf1, audio_acc, _ = score_with_abstention(y_true, audio_pred, [False] * len(y_true), CLASS_ORDER)
                face_mf1, face_acc, _ = score_with_abstention(y_true, face_pred, abstained, CLASS_ORDER)
                all_rows.append({"model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "audio_only", "macro_f1": audio_mf1, "accuracy": audio_acc, "mean_face_weight": np.nan})
                all_rows.append({"model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "face_only", "macro_f1": face_mf1, "accuracy": face_acc, "mean_face_weight": np.nan})

                face_cal_cond = face_calibrated_all[cond_mask]

                w_fixed = np.full(cond_mask.sum(), 0.5)
                mf1, acc = fuse_and_score(audio_cal_cond, face_cal_cond, w_fixed, y_true, CLASS_ORDER)
                all_rows.append({"model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "fixed_5050", "macro_f1": mf1, "accuracy": acc, "mean_face_weight": 0.5})

                fc_max, ac_max = face_conf_max_all[cond_mask], audio_conf_max_all[cond_mask]
                w_max = fc_max / (fc_max + ac_max + EPS)
                mf1, acc = fuse_and_score(audio_cal_cond, face_cal_cond, w_max, y_true, CLASS_ORDER)
                all_rows.append({"model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "conf_maxprob", "macro_f1": mf1, "accuracy": acc, "mean_face_weight": float(w_max.mean())})

                fc_ent, ac_ent = face_conf_ent_all[cond_mask], audio_conf_ent_all[cond_mask]
                w_ent = fc_ent / (fc_ent + ac_ent + EPS)
                mf1, acc = fuse_and_score(audio_cal_cond, face_cal_cond, w_ent, y_true, CLASS_ORDER)
                all_rows.append({"model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "conf_negentropy", "macro_f1": mf1, "accuracy": acc, "mean_face_weight": float(w_ent.mean())})

                w_gate = w_gate_all[cond_mask]
                mf1, acc = fuse_and_score(audio_cal_cond, face_cal_cond, w_gate, y_true, CLASS_ORDER)
                all_rows.append({"model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "learned_gate", "macro_f1": mf1, "accuracy": acc, "mean_face_weight": float(w_gate.mean())})

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUT_DIR / "phase5_fusion_v2_by_seed.csv", index=False)

    agg = results_df.groupby(["model_variant", "occlusion_type", "severity", "method"], dropna=False).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        mean_face_weight_mean=("mean_face_weight", "mean"), mean_face_weight_std=("mean_face_weight", "std"),
    ).reset_index()
    agg.to_csv(OUT_DIR / "phase5_fusion_v2_table.csv", index=False)
    print("\nfull fusion table (mean +/- std over 3 seeds):")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
