#!/usr/bin/env python
"""The final evaluation script specified by FINAL_EVAL_PLAN.md (as
amended 2026-11-13): tuned eGeMAPS (C=0.3) and WavLM audio branches, all
3 face-model variants (clean-trained, aug_all, aug_no_mask), all 6
fusion methods (audio_only, face_only, fixed_5050, conf_maxprob,
conf_negentropy, learned_gate), all 13 conditions, macro-F1 as primary
metric, and the pre-registered paired actor-level bootstrap comparisons
P1/P2/P3 (plus the original eGeMAPS-based comparisons, now secondary).

Calibration (temperature) and the learned gate are fit ONCE on ALL VAL
actors -- exactly the methodology used throughout development, just
without the 2-fold cross-fit split -- and FROZEN. The frozen parameters
are applied directly to whichever split is being scored; test labels are
only ever used for scoring, never for fitting anything. When --split val
(the dry run), the fitting set and the scoring set are the same actors,
so this script fits and scores using the same frozen parameters as a
code-path check before trusting the --split test run.

--split test is refused unless --final is also passed (defense in
depth), and additionally requires a pre-built test occlusion-detection
cache (this script does not build one itself -- see FINAL_EVAL_PLAN.md's
note that the cache "must be rebuilt for test clips" as a separate,
deliberate step).

Usage:
    python evaluate.py                         # dry run on VAL, seeds 0,1,2
    python evaluate.py --seeds 0                # dry run on VAL, seed 0 only (fast)
    python evaluate.py --split test --final     # the real, frozen-test-set run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fer.audio_features import extract_egemaps_for_clips, feature_columns, load_features, save_features
from fer.calibration import apply_temperature, fit_temperature
from fer.fusion import apply_gate, fit_gate_on_all, maxprob_confidence, negentropy_confidence, probs_matrix, y_true_idx
from fer.metrics import score_with_abstention
from fer.occlusion_eval import evaluate_clean_probs, evaluate_occlusion_probs, load_resnet18_checkpoint
from fer.splits import load_split
from fer.wavlm_model import TARGET_SR, WavLMClassifier, get_feature_extractor, truncate_to_max_seconds

REPO_ROOT = Path(__file__).resolve().parent
FACE_CFG = yaml.safe_load((REPO_ROOT / "configs" / "face_model.yaml").read_text())
AUDIO_CFG = yaml.safe_load((REPO_ROOT / "configs" / "audio_baseline.yaml").read_text())
DATA_CFG = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
CLASS_ORDER = FACE_CFG["class_order"]
N_CLASSES = len(CLASS_ORDER)
OCC_CFG = FACE_CFG["occlusion_eval"]
OCC_CONDITIONS = [(t, s) for t in OCC_CFG["occlusion_types"] for s in OCC_CFG["severities"]]
ALL_CONDITIONS = [("clean", None)] + OCC_CONDITIONS
FRAMES_PER_CLIP_EVAL = OCC_CFG["frames_per_clip_eval"]
OCCLUSION_SEED = 0
EGEMAPS_TUNED_C = 0.3  # chosen in RESULTS.md round 6 / outputs/egemaps_C_tuning_grid.csv
EPS = 1e-9

FACE_VARIANT_CKPT = {
    "clean": (REPO_ROOT / "models" / "face_resnet18", "resnet18_clean_seed{seed}.pt"),
    "aug_all": (REPO_ROOT / "models" / "face_resnet18_aug_all", "resnet18_aug_all_seed{seed}.pt"),
    "aug_no_mask": (REPO_ROOT / "models" / "face_resnet18_aug_no_mask", "resnet18_aug_no_mask_seed{seed}.pt"),
}
CROPS_DIR = REPO_ROOT / "cache" / "face_crops"
LANDMARKS_DIR = REPO_ROOT / "cache" / "landmarks"
WAVLM_CKPT_DIR = REPO_ROOT / "models" / "wavlm"
# NOT cache/audio/*.npy -- those were extracted from the VIDEO track via PyAV
# (scripts/08_phase2_full_run.py), a different decode path than the canonical
# AudioWAV files WavLM was actually trained/validated on (both the Kaggle run
# and scripts/24's local check read AudioWAV directly). Using cache/audio here
# silently shifted WavLM's audio_only macro-F1 by ~0.007 in an earlier version
# of this script -- see RESULTS.md round 7 for the full account.
AUDIO_WAV_DIR = REPO_ROOT / DATA_CFG["paths"]["audio_dir"]
OUT_DIR = REPO_ROOT / "outputs"
VAL_DETECTION_CACHE = REPO_ROOT / OCC_CFG["detection_cache_path"]
TEST_DETECTION_CACHE = REPO_ROOT / "cache" / "phase4_test_occlusion_detection_cache.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--final", action="store_true", help="required to run --split test")
    p.add_argument("--seeds", type=str, default="0,1,2", help="comma-separated seed list, e.g. '0' for a fast dry run")
    return p.parse_args()


def device_for_torch() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------
# eGeMAPS (tuned) audio branch -- deterministic, no seed variation
# ---------------------------------------------------------------

def fetch_egemaps_probs(eval_metadata: pd.DataFrame, train_actors: set, test_actors: set, split: str) -> pd.DataFrame:
    cached = load_features(REPO_ROOT / AUDIO_CFG["features"]["cache_path"])
    assert not (set(cached["actor_id"]) & test_actors), "cached eGeMAPS training features touch test actors"
    train_rows = cached[cached["actor_id"].isin(train_actors)]
    cols = feature_columns(cached)

    if split == "val":
        eval_rows = cached[cached["actor_id"].isin(set(eval_metadata["actor_id"]))]
        assert set(eval_rows["clip_id"]) == set(eval_metadata["clip_id"]), "cached val eGeMAPS features don't match eval metadata"
    else:
        cache_path = REPO_ROOT / "cache" / "features" / "egemaps_functionals_test.parquet"
        if cache_path.exists():
            eval_rows = load_features(cache_path)
            assert set(eval_rows["clip_id"]) == set(eval_metadata["clip_id"]), "cached test eGeMAPS features don't match eval metadata"
        else:
            print(f"extracting eGeMAPS functionals for {len(eval_metadata)} test clips (not cached yet)...")
            eval_rows, failures = extract_egemaps_for_clips(eval_metadata)
            if failures:
                raise RuntimeError(f"eGeMAPS extraction failed for {len(failures)} test clips: {failures[:5]}")
            save_features(eval_rows, cache_path)

    scaler = StandardScaler().fit(train_rows[cols].to_numpy(dtype=np.float64))
    X_train = scaler.transform(train_rows[cols].to_numpy(dtype=np.float64))
    X_eval = scaler.transform(eval_rows[cols].to_numpy(dtype=np.float64))
    lr = LogisticRegression(
        class_weight=AUDIO_CFG["model"]["class_weight"], solver=AUDIO_CFG["model"]["solver"],
        max_iter=AUDIO_CFG["model"]["max_iter"], C=EGEMAPS_TUNED_C, random_state=0,
    ).fit(X_train, train_rows["emotion"].to_numpy())

    proba = lr.predict_proba(X_eval)
    class_to_col = {c: i for i, c in enumerate(lr.classes_)}
    reordered = proba[:, [class_to_col[c] for c in CLASS_ORDER]]

    rows = []
    for i, (_, r) in enumerate(eval_rows.iterrows()):
        row = {"clip_id": r["clip_id"], "actor_id": r["actor_id"], "y_true": r["emotion"], "abstained": False}
        row.update({f"prob_{c}": float(p) for c, p in zip(CLASS_ORDER, reordered[i])})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# WavLM audio branch -- one fine-tuned checkpoint per seed
# ---------------------------------------------------------------

def load_wavlm_waveform(clip_id: str) -> np.ndarray:
    import soundfile as sf
    import torchaudio
    wav, sr = sf.read(str(AUDIO_WAV_DIR / f"{clip_id}.wav"), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, TARGET_SR).numpy()
    return truncate_to_max_seconds(wav, TARGET_SR)


def fetch_wavlm_probs(eval_metadata: pd.DataFrame, seed: int, device: torch.device) -> pd.DataFrame:
    feature_extractor = get_feature_extractor()
    model = WavLMClassifier(len(CLASS_ORDER))
    ckpt_path = WAVLM_CKPT_DIR / f"wavlm_base_plus_seed{seed}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    model.eval().to(device)

    clip_ids = sorted(eval_metadata["clip_id"])
    clip_to_emotion = dict(zip(eval_metadata["clip_id"], eval_metadata["emotion"]))
    clip_to_actor = dict(zip(eval_metadata["clip_id"], eval_metadata["actor_id"]))

    # batch_size=1 (no padding at all) is DELIBERATE, not a missed optimization: the
    # frozen CNN feature encoder has no attention mask internally, so a zero-padded
    # batch's GroupNorm statistics shift with how much padding a clip happens to share
    # a batch with -- found empirically to move some clips' softmax by up to ~0.6 in
    # probability depending on arbitrary batch composition (see RESULTS.md round 7).
    # batch_size=1 is the only way to get a per-clip result that depends on nothing
    # but that clip's own audio and the checkpoint weights.
    rows = []
    with torch.no_grad():
        for cid in clip_ids:
            wav = load_wavlm_waveform(cid)
            processed = feature_extractor([wav], sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
            logits = model(processed["input_values"].to(device), processed["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            row = {"clip_id": cid, "actor_id": clip_to_actor[cid], "y_true": clip_to_emotion[cid], "abstained": False}
            row.update({f"prob_{c}": float(pp) for c, pp in zip(CLASS_ORDER, probs)})
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Face branch -- 3 variants x seeds, clean + 12 occlusion conditions
# ---------------------------------------------------------------

def build_face_tasks(eval_metadata: pd.DataFrame):
    tasks, clip_true = [], {}
    for _, r in eval_metadata.iterrows():
        npz_path = LANDMARKS_DIR / f"{r['clip_id']}.npz"
        if not npz_path.exists():
            continue
        lm_data = np.load(npz_path)
        tasks.append((r["clip_id"], lm_data["frame_indices"], lm_data["landmarks"]))
        clip_true[r["clip_id"]] = r["emotion"]
    return tasks, clip_true


def fetch_face_probs(variant: str, seed: int, eval_metadata: pd.DataFrame, det_lookup: pd.Series, device: torch.device) -> pd.DataFrame:
    ckpt_dir, name_tmpl = FACE_VARIANT_CKPT[variant]
    ckpt_path = ckpt_dir / name_tmpl.format(seed=seed)
    model = load_resnet18_checkpoint(ckpt_path, N_CLASSES, device)

    clean_df = evaluate_clean_probs(eval_metadata, model, CLASS_ORDER, device, CROPS_DIR)
    clean_df["occlusion_type"] = "clean"
    clean_df["severity"] = np.nan

    tasks, clip_true = build_face_tasks(eval_metadata)
    occ_df = evaluate_occlusion_probs(tasks, clip_true, det_lookup, model, CLASS_ORDER, OCC_CONDITIONS, device, CROPS_DIR, FRAMES_PER_CLIP_EVAL, OCCLUSION_SEED)

    combined = pd.concat([clean_df, occ_df], ignore_index=True)
    combined["model_variant"] = variant
    combined["seed"] = seed
    return combined


# ---------------------------------------------------------------
# Fusion (shared across both audio branches)
# ---------------------------------------------------------------

def fuse_and_score(audio_cal: np.ndarray, face_cal: np.ndarray, w_face: np.ndarray, y_true: list, class_order: list[str]) -> tuple[float, float, list[str]]:
    w_audio = 1.0 - w_face
    fused = w_audio[:, None] * audio_cal + w_face[:, None] * face_cal
    pred = [class_order[i] for i in np.argmax(fused, axis=1)]
    macro_f1, acc, _ = score_with_abstention(y_true, pred, [False] * len(y_true), class_order)
    return macro_f1, acc, pred


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.split == "test" and not args.final:
        raise SystemExit("--split test requires --final (defense in depth -- the test set stays frozen otherwise).")

    sf = load_split(REPO_ROOT / DATA_CFG["paths"]["splits_dir"] / "split_v1.json")
    if sf.status != "frozen":
        raise RuntimeError(f"split is not frozen (status={sf.status}) -- refusing to proceed")
    train_actors = set(sf.assignment["train"])
    val_actors = set(sf.assignment["val"])
    test_actors = set(sf.assignment["test"])
    eval_actors = val_actors if args.split == "val" else test_actors

    print(f"evaluate.py: split={args.split} final={args.final} seeds={seeds} eval_actors={len(eval_actors)}")
    if args.split == "test":
        assert not (eval_actors & val_actors) and not (eval_actors & train_actors)
        if not TEST_DETECTION_CACHE.exists():
            raise RuntimeError(
                f"{TEST_DETECTION_CACHE} does not exist. This script does not build the test occlusion-"
                f"detection cache itself (a separate, deliberate step -- see FINAL_EVAL_PLAN.md). Build it "
                f"first, then re-run."
            )
        eval_det_cache_path = TEST_DETECTION_CACHE
    else:
        eval_det_cache_path = VAL_DETECTION_CACHE

    metadata_all = pd.read_csv(REPO_ROOT / DATA_CFG["paths"]["outputs_dir"] / "metadata.csv", dtype={"actor_id": str})
    val_metadata = metadata_all[metadata_all["actor_id"].isin(val_actors)].copy()
    eval_metadata = val_metadata if args.split == "val" else metadata_all[metadata_all["actor_id"].isin(eval_actors)].copy()
    assert set(eval_metadata["actor_id"]) == eval_actors, "eval metadata does not cover every eval actor"
    assert not (set(eval_metadata["actor_id"]) & (train_actors | (val_actors if args.split == "test" else test_actors)))
    print(f"calibration/gate fitting set (frozen, per FINAL_EVAL_PLAN.md amendment): VAL, {len(val_metadata)} clips, {val_metadata['actor_id'].nunique()} actors")
    print(f"scoring set: {args.split.upper()}, {len(eval_metadata)} clips, {eval_metadata['actor_id'].nunique()} actors")

    device = device_for_torch()
    print(f"device: {device}")

    val_detection_df = pd.read_parquet(VAL_DETECTION_CACHE)
    val_det_lookup = val_detection_df.set_index(["clip_id", "frame_idx", "occlusion_type", "severity"])["detected"]
    if args.split == "val":
        eval_det_lookup = val_det_lookup
    else:
        eval_detection_df = pd.read_parquet(eval_det_cache_path)
        eval_det_lookup = eval_detection_df.set_index(["clip_id", "frame_idx", "occlusion_type", "severity"])["detected"]

    print("\nfetching eGeMAPS (tuned, C={}) audio probs for VAL (calibration fitting)...".format(EGEMAPS_TUNED_C))
    egemaps_df_val = fetch_egemaps_probs(val_metadata, train_actors, test_actors, "val")
    if args.split == "val":
        egemaps_df_eval = egemaps_df_val
    else:
        print("fetching eGeMAPS (tuned) audio probs for TEST scoring...")
        egemaps_df_eval = fetch_egemaps_probs(eval_metadata, train_actors, test_actors, "test")

    all_rows = []
    per_clip_rows = []

    for seed in seeds:
        print(f"\nfetching WavLM (seed={seed}) audio probs for VAL (calibration fitting)...")
        wavlm_df_val = fetch_wavlm_probs(val_metadata, seed, device)
        if args.split == "val":
            wavlm_df_eval = wavlm_df_val
        else:
            print(f"fetching WavLM (seed={seed}) audio probs for TEST scoring...")
            wavlm_df_eval = fetch_wavlm_probs(eval_metadata, seed, device)
        assert set(wavlm_df_eval["clip_id"]) == set(eval_metadata["clip_id"])

        for audio_name, audio_df_val, audio_df_eval in [("egemaps_tuned", egemaps_df_val, egemaps_df_eval), ("wavlm", wavlm_df_val, wavlm_df_eval)]:
            # Calibration: fit ONCE on ALL val rows (audio has no occlusion condition, every
            # row is "clean"), freeze, apply to the scoring set. Test labels are never used here.
            P_val, yt_val = probs_matrix(audio_df_val, CLASS_ORDER), y_true_idx(audio_df_val, CLASS_ORDER)
            audio_T = fit_temperature(P_val, yt_val)
            print(f"  audio={audio_name} seed={seed}: temperature (fit on ALL {len(val_metadata['actor_id'].unique())} val actors, frozen) = {audio_T:.4f}")

            audio_calibrated_eval = apply_temperature(probs_matrix(audio_df_eval, CLASS_ORDER), audio_T)
            audio_raw_eval = probs_matrix(audio_df_eval, CLASS_ORDER)
            audio_lookup_raw = dict(zip(audio_df_eval["clip_id"], audio_raw_eval))
            audio_lookup_cal = dict(zip(audio_df_eval["clip_id"], audio_calibrated_eval))
            audio_lookup_conf_max = {cid: maxprob_confidence(v[None, :])[0] for cid, v in audio_lookup_cal.items()}
            audio_lookup_conf_ent = {cid: negentropy_confidence(v[None, :], N_CLASSES)[0] for cid, v in audio_lookup_cal.items()}

            audio_calibrated_val = apply_temperature(P_val, audio_T)
            audio_conf_max_val_lookup = {cid: maxprob_confidence(v[None, :])[0] for cid, v in zip(audio_df_val["clip_id"], audio_calibrated_val)}

            for variant in FACE_VARIANT_CKPT:
                print(f"  seed={seed} audio={audio_name} face={variant}: fetching face probs for VAL (calibration/gate fitting)...")
                face_all_val = fetch_face_probs(variant, seed, val_metadata, val_det_lookup, device)
                if args.split == "val":
                    face_all_eval = face_all_val
                else:
                    print(f"  seed={seed} audio={audio_name} face={variant}: fetching face probs for TEST scoring...")
                    face_all_eval = fetch_face_probs(variant, seed, eval_metadata, eval_det_lookup, device)

                clean_mask_val = (face_all_val["occlusion_type"] == "clean").to_numpy()
                P_face_val_clean = probs_matrix(face_all_val, CLASS_ORDER)[clean_mask_val]
                yt_face_val_clean = y_true_idx(face_all_val, CLASS_ORDER)[clean_mask_val]
                face_T = fit_temperature(P_face_val_clean, yt_face_val_clean)
                print(f"    face temperature (fit on ALL val clean rows, frozen) = {face_T:.4f}")

                face_calibrated_eval = apply_temperature(probs_matrix(face_all_eval, CLASS_ORDER), face_T)
                face_calibrated_val_all = apply_temperature(probs_matrix(face_all_val, CLASS_ORDER), face_T)

                # Gate: fit ONCE on ALL val rows (pooled clean + every occlusion condition), freeze.
                face_raw_val_all = probs_matrix(face_all_val, CLASS_ORDER)
                face_pred_idx_val_all = np.argmax(face_raw_val_all, axis=1)
                y_true_val_all = face_all_val["y_true"].to_numpy()
                abstained_val_all = face_all_val["abstained"].to_numpy()
                face_correct_val_all = np.array([CLASS_ORDER[p] == yt for p, yt in zip(face_pred_idx_val_all, y_true_val_all)]) & (~abstained_val_all)
                face_conf_max_val_all = maxprob_confidence(face_calibrated_val_all)
                audio_conf_max_val_all = np.array([audio_conf_max_val_lookup[c] for c in face_all_val["clip_id"]])
                gate_model, gate_fallback = fit_gate_on_all(face_conf_max_val_all, audio_conf_max_val_all, face_correct_val_all)

                face_raw_eval = probs_matrix(face_all_eval, CLASS_ORDER)
                face_pred_idx_eval = np.argmax(face_raw_eval, axis=1)
                clean_mask_eval = (face_all_eval["occlusion_type"] == "clean").to_numpy()
                abstained_eval = face_all_eval["abstained"].to_numpy()

                face_conf_max_eval = maxprob_confidence(face_calibrated_eval)
                face_conf_ent_eval = negentropy_confidence(face_calibrated_eval, N_CLASSES)
                audio_conf_max_eval = np.array([audio_lookup_conf_max[c] for c in face_all_eval["clip_id"]])
                audio_conf_ent_eval = np.array([audio_lookup_conf_ent[c] for c in face_all_eval["clip_id"]])

                w_gate_eval = apply_gate(gate_model, gate_fallback, face_conf_max_eval, audio_conf_max_eval)

                for occ_type, severity in ALL_CONDITIONS:
                    if occ_type == "clean":
                        cond_mask = clean_mask_eval
                    else:
                        cond_mask = ((face_all_eval["occlusion_type"] == occ_type) & (face_all_eval["severity"] == severity)).to_numpy()
                    if cond_mask.sum() == 0:
                        continue

                    cond_df = face_all_eval[cond_mask]
                    y_true = cond_df["y_true"].tolist()
                    abstained = abstained_eval[cond_mask].tolist()
                    face_pred = [CLASS_ORDER[i] for i in face_pred_idx_eval[cond_mask]]
                    cond_clip_ids = cond_df["clip_id"].tolist()

                    audio_cal_cond = np.stack([audio_lookup_cal[c] for c in cond_clip_ids])
                    audio_raw_cond = np.stack([audio_lookup_raw[c] for c in cond_clip_ids])
                    audio_pred = [CLASS_ORDER[i] for i in np.argmax(audio_raw_cond, axis=1)]

                    audio_mf1, audio_acc, _ = score_with_abstention(y_true, audio_pred, [False] * len(y_true), CLASS_ORDER)
                    face_mf1, face_acc, _ = score_with_abstention(y_true, face_pred, abstained, CLASS_ORDER)
                    all_rows.append({"audio_branch": audio_name, "model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "audio_only", "macro_f1": audio_mf1, "accuracy": audio_acc, "mean_face_weight": np.nan})
                    all_rows.append({"audio_branch": audio_name, "model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "face_only", "macro_f1": face_mf1, "accuracy": face_acc, "mean_face_weight": np.nan})
                    for cid, yt, yp in zip(cond_clip_ids, y_true, audio_pred):
                        per_clip_rows.append({"audio_branch": audio_name, "model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "audio_only", "clip_id": cid, "y_true": yt, "y_pred": yp})
                    for cid, yt, yp in zip(cond_clip_ids, y_true, face_pred):
                        per_clip_rows.append({"audio_branch": audio_name, "model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "face_only", "clip_id": cid, "y_true": yt, "y_pred": yp})

                    face_cal_cond = face_calibrated_eval[cond_mask]

                    w_fixed = np.full(cond_mask.sum(), 0.5)
                    mf1, acc, pred = fuse_and_score(audio_cal_cond, face_cal_cond, w_fixed, y_true, CLASS_ORDER)
                    all_rows.append({"audio_branch": audio_name, "model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "fixed_5050", "macro_f1": mf1, "accuracy": acc, "mean_face_weight": 0.5})
                    for cid, yt, yp in zip(cond_clip_ids, y_true, pred):
                        per_clip_rows.append({"audio_branch": audio_name, "model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "fixed_5050", "clip_id": cid, "y_true": yt, "y_pred": yp})

                    fc_max, ac_max = face_conf_max_eval[cond_mask], audio_conf_max_eval[cond_mask]
                    w_max = fc_max / (fc_max + ac_max + EPS)
                    mf1, acc, pred = fuse_and_score(audio_cal_cond, face_cal_cond, w_max, y_true, CLASS_ORDER)
                    all_rows.append({"audio_branch": audio_name, "model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "conf_maxprob", "macro_f1": mf1, "accuracy": acc, "mean_face_weight": float(w_max.mean())})
                    for cid, yt, yp in zip(cond_clip_ids, y_true, pred):
                        per_clip_rows.append({"audio_branch": audio_name, "model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "conf_maxprob", "clip_id": cid, "y_true": yt, "y_pred": yp})

                    fc_ent, ac_ent = face_conf_ent_eval[cond_mask], audio_conf_ent_eval[cond_mask]
                    w_ent = fc_ent / (fc_ent + ac_ent + EPS)
                    mf1, acc, pred = fuse_and_score(audio_cal_cond, face_cal_cond, w_ent, y_true, CLASS_ORDER)
                    all_rows.append({"audio_branch": audio_name, "model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "conf_negentropy", "macro_f1": mf1, "accuracy": acc, "mean_face_weight": float(w_ent.mean())})

                    w_gate = w_gate_eval[cond_mask]
                    mf1, acc, pred = fuse_and_score(audio_cal_cond, face_cal_cond, w_gate, y_true, CLASS_ORDER)
                    all_rows.append({"audio_branch": audio_name, "model_variant": variant, "seed": seed, "occlusion_type": occ_type, "severity": severity, "method": "learned_gate", "macro_f1": mf1, "accuracy": acc, "mean_face_weight": float(w_gate.mean())})

    results_df = pd.DataFrame(all_rows)
    per_clip_df = pd.DataFrame(per_clip_rows)
    results_df.to_csv(OUT_DIR / f"evaluate_{args.split}_by_seed.csv", index=False)
    per_clip_df.to_csv(OUT_DIR / f"evaluate_{args.split}_per_clip.csv", index=False)

    agg = results_df.groupby(["audio_branch", "model_variant", "occlusion_type", "severity", "method"], dropna=False).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        mean_face_weight_mean=("mean_face_weight", "mean"), mean_face_weight_std=("mean_face_weight", "std"),
    ).reset_index()
    agg.to_csv(OUT_DIR / f"evaluate_{args.split}_table.csv", index=False)
    print(f"\n{'DRY RUN on VAL -- not a test-set result' if args.split == 'val' else 'FINAL TEST-SET EVALUATION'}")
    print(f"full table (mean +/- std over {len(seeds)} seed(s) {seeds}):")
    print(agg.to_string(index=False))

    print("\n--- pre-registered paired bootstrap comparisons ---")
    if args.split == "val":
        print("(DRY RUN on VAL -- these numbers are a code-path check, NOT the pre-registered test-set result.)")
    run_bootstrap_comparisons(per_clip_df, eval_actors, eval_metadata)


def run_bootstrap_comparisons(per_clip_df: pd.DataFrame, eval_actors: set, eval_metadata: pd.DataFrame) -> None:
    from sklearn.metrics import f1_score
    clip_to_actor = dict(zip(eval_metadata["clip_id"], eval_metadata["actor_id"]))
    actor_list = sorted(eval_actors)

    seeds_present = sorted(per_clip_df["seed"].unique())
    if len(seeds_present) < 3:
        print(f"NOTE: only seed(s) {seeds_present} present this run -- the real test-set run uses all 3 seeds per FINAL_EVAL_PLAN.md.")

    def select(method: str, variant: str, audio_name: str, occ_type: str | None = None, severity: float | None = None):
        def _sel(s: int) -> pd.DataFrame:
            m = (per_clip_df["seed"] == s) & (per_clip_df["method"] == method) & (per_clip_df["model_variant"] == variant) & (per_clip_df["audio_branch"] == audio_name)
            if occ_type is not None:
                m = m & (per_clip_df["occlusion_type"] == occ_type)
            if severity is not None:
                m = m & (per_clip_df["severity"] == severity)
            return per_clip_df[m]
        return _sel

    def _actor_index(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
        y_true, y_pred = sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()
        actor = sub["clip_id"].map(clip_to_actor).to_numpy()
        idx = {a: np.where(actor == a)[0] for a in actor_list}
        return y_true, y_pred, idx

    def paired_bootstrap(sel_a, sel_b, label: str, n_bootstrap: int = 2000, boot_seed: int = 0) -> None:
        """Amendment (b): the resampling unit is the ACTOR. Each of the
        n_bootstrap draws picks actor_list-many actors with replacement, and
        for a drawn actor every one of its clips (across whatever condition
        scope sel_a/sel_b already filtered to) is pooled together -- no
        separate clip-level or condition-level resampling.
        Amendment (c): when more than one seed is present, EVERY draw reuses
        the SAME resampled actor set across all seeds, computes that draw's
        paired diff per seed, and averages those diffs -- the primary result
        is this seed-averaged diff and its 95% CI, not a per-seed CI average."""
        per_seed_data = {}
        for s in seeds_present:
            sub_a, sub_b = sel_a(s), sel_b(s)
            if sub_a.empty or sub_b.empty:
                continue
            per_seed_data[s] = (_actor_index(sub_a), _actor_index(sub_b))
        if not per_seed_data:
            print(f"{label}: no data")
            return

        seeds_here = sorted(per_seed_data.keys())
        per_seed_point = {}
        for s in seeds_here:
            (yt_a, yp_a, _), (yt_b, yp_b, _) = per_seed_data[s]
            per_seed_point[s] = f1_score(yt_a, yp_a, labels=CLASS_ORDER, average="macro") - f1_score(yt_b, yp_b, labels=CLASS_ORDER, average="macro")
        point_avg = float(np.mean(list(per_seed_point.values())))

        rng = np.random.default_rng(boot_seed)
        n_seeds_here = len(seeds_here)
        diffs = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            resampled = rng.choice(actor_list, size=len(actor_list), replace=True)  # same resampled actor set reused for every seed in this draw
            seed_diffs = np.empty(n_seeds_here)
            for i, s in enumerate(seeds_here):
                (yt_a, yp_a, idx_a), (yt_b, yp_b, idx_b) = per_seed_data[s]
                ia = np.concatenate([idx_a[a] for a in resampled])
                ib = np.concatenate([idx_b[a] for a in resampled])
                fa = f1_score(yt_a[ia], yp_a[ia], labels=CLASS_ORDER, average="macro")
                fb = f1_score(yt_b[ib], yp_b[ib], labels=CLASS_ORDER, average="macro")
                seed_diffs[i] = fa - fb
            diffs[b] = seed_diffs.mean()
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
        p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())

        per_seed_str = ", ".join(f"seed{s}={d:+.4f}" for s, d in per_seed_point.items())
        if n_seeds_here == 1:
            print(f"{label}: diff={point_avg:+.4f}  95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  p={p_value:.4f}  (seed {seeds_here[0]} only)")
        else:
            print(f"{label}: diff={point_avg:+.4f} (seed-averaged, {n_seeds_here} seeds)  95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  p={p_value:.4f}  [per-seed: {per_seed_str}]")

    print("\n### PRIMARY comparisons (amendment, 2026-11-13) ###")

    print("\nP1: fixed 50/50 fusion (aug_all + WavLM) vs WavLM alone -- does the face add to a strong voice model?")
    paired_bootstrap(select("fixed_5050", "aug_all", "wavlm"), select("audio_only", "aug_all", "wavlm"), "  P1 pooled (all 13 conditions)")
    for occ_type, severity in ALL_CONDITIONS:
        paired_bootstrap(select("fixed_5050", "aug_all", "wavlm", occ_type, severity), select("audio_only", "aug_all", "wavlm", occ_type, severity), f"  P1 [{occ_type} {severity}]")

    print("\nP2: aug_no_mask vs clean-trained face-only, under mask only (unchanged from the original plan)")
    for severity in OCC_CFG["severities"]:
        paired_bootstrap(select("face_only", "aug_no_mask", "wavlm", "mask", severity), select("face_only", "clean", "wavlm", "mask", severity), f"  P2 [mask {severity}]")

    print("\nP3: confidence-weighted fusion (max-prob, calibrated) vs fixed 50/50 (aug_all + WavLM) -- the original hypothesis test")
    paired_bootstrap(select("conf_maxprob", "aug_all", "wavlm"), select("fixed_5050", "aug_all", "wavlm"), "  P3 pooled (all 13 conditions)")
    for occ_type, severity in ALL_CONDITIONS:
        paired_bootstrap(select("conf_maxprob", "aug_all", "wavlm", occ_type, severity), select("fixed_5050", "aug_all", "wavlm", occ_type, severity), f"  P3 [{occ_type} {severity}]")

    print("\n### SECONDARY comparisons (original eGeMAPS-based tests, still reported) ###")
    for variant in FACE_VARIANT_CKPT:
        for audio_name in ["egemaps_tuned", "wavlm"]:
            paired_bootstrap(select("fixed_5050", variant, audio_name), select("audio_only", variant, audio_name), f"  fusion(fixed_5050) vs audio_only [{audio_name}, face={variant}, pooled]")


if __name__ == "__main__":
    main()
