#!/usr/bin/env python
"""Phase 5, round 6: fusion comparison on VAL with WavLM as the audio
branch, for all three face-model variants (clean-trained, aug_all,
aug_no_mask), using the exact same four pre-registered fusion methods as
scripts/18 (fixed 50/50, confidence-weighted max-prob, confidence-weighted
negative entropy, learned gate) and the same temperature-calibration /
gate cross-fitting methodology (fer.fusion). This is a straightforward
substitution of the audio branch, not a new fusion design.

Unlike the classical eGeMAPS audio branch (deterministic, one fit reused
across all 3 "seeds"), WavLM has genuine per-seed variation, so each face
seed i is paired with WavLM seed i (both already use seeds [0, 1, 2], same
convention as the face/audio seed pairing implied throughout this
project). Audio-side calibration is refit per WavLM seed, independently
of the classical eGeMAPS branch's calibration.

Usage:
    python scripts/23_phase5_fusion_wavlm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.fusion import cross_fit_calibrate_clean_only, fit_gate_cross_fit, maxprob_confidence, negentropy_confidence, probs_matrix, two_fold_actor_split
from fer.metrics import score_with_abstention
from fer.splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]
FACE_CFG = yaml.safe_load((REPO_ROOT / "configs" / "face_model.yaml").read_text())
CLASS_ORDER = FACE_CFG["class_order"]
N_CLASSES = len(CLASS_ORDER)
SEEDS = [0, 1, 2]
VARIANTS = ["clean", "aug_all", "aug_no_mask"]
OUT_DIR = REPO_ROOT / "outputs"
FACE_PROBS_PATH = OUT_DIR / "phase4_face_probs.parquet"
WAVLM_PROBS_PATH = OUT_DIR / "wavlm_val_probs_batch1.csv"  # round 7: batch_size=1, reproducible -- supersedes the original Kaggle batch_size=16 CSV
EPS = 1e-9


def load_wavlm_audio_df(seed: int) -> pd.DataFrame:
    wavlm_df = pd.read_csv(WAVLM_PROBS_PATH)
    sub = wavlm_df[wavlm_df["seed"] == seed].copy()
    sub["actor_id"] = sub["clip_id"].map(lambda c: c.split("_")[0])
    sub["abstained"] = False
    return sub[["clip_id", "actor_id", "y_true", "abstained"] + [f"prob_{c}" for c in CLASS_ORDER]]


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

    face_all = pd.read_parquet(FACE_PROBS_PATH)
    conditions = sorted(
        {(row.occlusion_type, row.severity) for row in face_all[["occlusion_type", "severity"]].drop_duplicates().itertuples()},
        key=lambda x: (x[0] != "clean", x[0], (x[1] if pd.notna(x[1]) else -1)),
    )

    all_rows = []
    for seed in SEEDS:
        print(f"\nfitting WavLM audio branch, seed={seed}...")
        audio_df = load_wavlm_audio_df(seed)
        clip_actor = dict(zip(audio_df["clip_id"], audio_df["actor_id"]))
        audio_calibrated, audio_T = cross_fit_calibrate_clean_only(audio_df, np.ones(len(audio_df), dtype=bool), clip_actor, fold_a, fold_b, CLASS_ORDER)
        print(f"  WavLM seed={seed} temperatures: {audio_T}")

        audio_raw = probs_matrix(audio_df, CLASS_ORDER)
        audio_lookup_raw = dict(zip(audio_df["clip_id"], audio_raw))
        audio_lookup_cal = dict(zip(audio_df["clip_id"], audio_calibrated))
        audio_lookup_conf_max = {cid: maxprob_confidence(v[None, :])[0] for cid, v in audio_lookup_cal.items()}
        audio_lookup_conf_ent = {cid: negentropy_confidence(v[None, :], N_CLASSES)[0] for cid, v in audio_lookup_cal.items()}

        for variant in VARIANTS:
            sub_all = face_all[(face_all["model_variant"] == variant) & (face_all["seed"] == seed)].reset_index(drop=True)
            if len(sub_all) == 0:
                print(f"  (skip: no face rows for variant={variant} seed={seed})")
                continue
            # WavLM's val set and the face branch's val set must be the exact same clips.
            assert set(sub_all["clip_id"]) == set(audio_df["clip_id"]), "face and WavLM clip sets differ"

            clean_mask_all = (sub_all["occlusion_type"] == "clean").to_numpy()
            face_calibrated_all, face_T = cross_fit_calibrate_clean_only(sub_all, clean_mask_all, clip_actor, fold_a, fold_b, CLASS_ORDER)
            print(f"  variant={variant} seed={seed} face temperatures: {face_T}")

            actor_of_row = sub_all["clip_id"].map(clip_actor)
            face_raw_all = probs_matrix(sub_all, CLASS_ORDER)
            face_pred_idx_all = np.argmax(face_raw_all, axis=1)
            y_true_all = sub_all["y_true"].to_numpy()
            abstained_all = sub_all["abstained"].to_numpy()
            face_correct_all = np.array([CLASS_ORDER[p] == yt for p, yt in zip(face_pred_idx_all, y_true_all)]) & (~abstained_all)

            face_conf_max_all = maxprob_confidence(face_calibrated_all)
            face_conf_ent_all = negentropy_confidence(face_calibrated_all, N_CLASSES)
            audio_conf_max_all = np.array([audio_lookup_conf_max[c] for c in sub_all["clip_id"]])
            audio_conf_ent_all = np.array([audio_lookup_conf_ent[c] for c in sub_all["clip_id"]])

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
                all_rows.append({"audio_branch": "wavlm", "model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "audio_only", "macro_f1": audio_mf1, "accuracy": audio_acc, "mean_face_weight": np.nan})
                all_rows.append({"audio_branch": "wavlm", "model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "face_only", "macro_f1": face_mf1, "accuracy": face_acc, "mean_face_weight": np.nan})

                face_cal_cond = face_calibrated_all[cond_mask]

                w_fixed = np.full(cond_mask.sum(), 0.5)
                mf1, acc = fuse_and_score(audio_cal_cond, face_cal_cond, w_fixed, y_true, CLASS_ORDER)
                all_rows.append({"audio_branch": "wavlm", "model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "fixed_5050", "macro_f1": mf1, "accuracy": acc, "mean_face_weight": 0.5})

                fc_max, ac_max = face_conf_max_all[cond_mask], audio_conf_max_all[cond_mask]
                w_max = fc_max / (fc_max + ac_max + EPS)
                mf1, acc = fuse_and_score(audio_cal_cond, face_cal_cond, w_max, y_true, CLASS_ORDER)
                all_rows.append({"audio_branch": "wavlm", "model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "conf_maxprob", "macro_f1": mf1, "accuracy": acc, "mean_face_weight": float(w_max.mean())})

                fc_ent, ac_ent = face_conf_ent_all[cond_mask], audio_conf_ent_all[cond_mask]
                w_ent = fc_ent / (fc_ent + ac_ent + EPS)
                mf1, acc = fuse_and_score(audio_cal_cond, face_cal_cond, w_ent, y_true, CLASS_ORDER)
                all_rows.append({"audio_branch": "wavlm", "model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "conf_negentropy", "macro_f1": mf1, "accuracy": acc, "mean_face_weight": float(w_ent.mean())})

                w_gate = w_gate_all[cond_mask]
                mf1, acc = fuse_and_score(audio_cal_cond, face_cal_cond, w_gate, y_true, CLASS_ORDER)
                all_rows.append({"audio_branch": "wavlm", "model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "learned_gate", "macro_f1": mf1, "accuracy": acc, "mean_face_weight": float(w_gate.mean())})

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUT_DIR / "phase5_fusion_wavlm_by_seed.csv", index=False)

    agg = results_df.groupby(["model_variant", "occlusion_type", "severity", "method"], dropna=False).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        mean_face_weight_mean=("mean_face_weight", "mean"), mean_face_weight_std=("mean_face_weight", "std"),
    ).reset_index()
    agg.to_csv(OUT_DIR / "phase5_fusion_wavlm_table.csv", index=False)
    print("\nfull WavLM-audio fusion table (mean +/- std over 3 seeds):")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
