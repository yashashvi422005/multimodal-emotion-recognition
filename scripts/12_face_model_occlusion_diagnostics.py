#!/usr/bin/env python
"""Post-hoc diagnostics on the Phase 4 occlusion evaluation, requested
before building fusion (Phase 5). Five checks, all read-only w.r.t. the
already-trained checkpoints (no retraining):

1. Abstention scoring: scripts/11 counts a zero-detection clip as a
   prediction of CLASS_ORDER[0] ("anger") — argmax of a uniform
   distribution, which is a deterministic tie-break, NOT "no prediction"
   (dropping the clip from the metric). Both variants are recomputed and
   reported side by side.
2. Mean fraction of the 224x224 crop covered by `mask`/`sunglasses` at
   each severity, measured geometrically (draw the occlusion onto a
   blank canvas of a color distinct from any fill color, measure the
   changed-pixel fraction) — independent of any model or detection run.
3. Predicted-class distribution under each `mask` severity vs. clean
   (seed 0), to check whether the model is collapsing onto one or two
   classes rather than failing broadly.
4. Mean max-softmax confidence vs. accuracy, for clean and every
   occlusion condition — distinguishes "appropriately uncertain" from
   "confidently wrong," which matters directly for confidence-weighted
   fusion.
5. The full occlusion table (scripts/11) re-run for seeds 1 and 2,
   reusing the EXISTING detection cache (cache/phase4_occlusion_detection_cache.parquet)
   since MediaPipe detection depends only on the occluded image, not on
   model weights — so only the cheap model-eval phase needs to re-run.
   Reported as mean +/- std across all 3 seeds.

Also saves a small contact sheet of mask-occluded crops at each severity
(4 clips, 4 actors, 4 emotions) to help interpret check 3's finding.

Usage:
    python scripts/12_face_model_occlusion_diagnostics.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import yaml
from PIL import Image, ImageDraw
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.face_model import IMAGENET_MEAN, IMAGENET_STD, FrameDataset, build_eval_rows
from fer.face_pipeline import apply_occlusion
from fer.splits import load_split
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
FACE_CFG = yaml.safe_load((REPO_ROOT / "configs" / "face_model.yaml").read_text())
OCC_CFG = FACE_CFG["occlusion_eval"]

CLASS_ORDER = FACE_CFG["class_order"]
OCCLUSION_TYPES = OCC_CFG["occlusion_types"]
SEVERITIES = OCC_CFG["severities"]
CONDITIONS = [(t, s) for t in OCCLUSION_TYPES for s in SEVERITIES]
FRAMES_PER_CLIP_EVAL = OCC_CFG["frames_per_clip_eval"]
SEEDS = [0, 1, 2]
SEED = 0  # occlusion placement RNG seed (patch/hand_over_mouth), as in scripts/10 & /11

CROPS_DIR = REPO_ROOT / "cache" / "face_crops"
LANDMARKS_DIR = REPO_ROOT / "cache" / "landmarks"
DETECTION_CACHE_PATH = REPO_ROOT / OCC_CFG["detection_cache_path"]
CHECKPOINT_DIR = REPO_ROOT / "models" / "face_resnet18"
OUT_DIR = REPO_ROOT / "outputs"


def image_to_tensor(arr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)
    return (t - IMAGENET_MEAN) / IMAGENET_STD


def select_eval_frame_positions(n_frames: int, k: int = FRAMES_PER_CLIP_EVAL) -> np.ndarray:
    if n_frames <= k:
        return np.arange(n_frames)
    return np.unique(np.round(np.linspace(0, n_frames - 1, k)).astype(int))


def load_val_tasks():
    data_cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    sf = load_split(REPO_ROOT / data_cfg["paths"]["splits_dir"] / "split_v1.json")
    if sf.status != "frozen":
        raise RuntimeError("split is not frozen — refusing to proceed")
    val_actors, test_actors = set(sf.assignment["val"]), set(sf.assignment["test"])

    metadata_df = pd.read_csv(REPO_ROOT / data_cfg["paths"]["outputs_dir"] / "metadata.csv", dtype={"actor_id": str})
    val_metadata = metadata_df[metadata_df["actor_id"].isin(val_actors)]
    assert not set(val_metadata["actor_id"]) & test_actors

    tasks, clip_true = [], {}
    for _, r in val_metadata.iterrows():
        clip_id, true_label = r["clip_id"], r["emotion"]
        npz_path = LANDMARKS_DIR / f"{clip_id}.npz"
        if not npz_path.exists():
            continue
        lm_data = np.load(npz_path)
        tasks.append((clip_id, lm_data["frame_indices"], lm_data["landmarks"]))
        clip_true[clip_id] = true_label
    return tasks, clip_true, val_metadata


# ---------------------------------------------------------------------------
# Check 2: geometric coverage (no model, no detection)
# ---------------------------------------------------------------------------

def compute_coverage(tasks: list[tuple], n_samples: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(tasks), size=min(n_samples, len(tasks)), replace=False)
    rows = []
    for occ_type in ("mask", "sunglasses"):
        for severity in SEVERITIES:
            fracs = []
            for i in sample_idx:
                clip_id, frame_indices, landmarks = tasks[i]
                pos = int(rng.integers(0, len(frame_indices)))
                lm = landmarks[pos]
                if np.isnan(lm).any():
                    continue
                blank = np.full((224, 224, 3), (255, 0, 255), dtype=np.uint8)
                occluded = apply_occlusion(occ_type, blank, lm, severity, clip_id, SEED)
                changed = np.any(occluded != blank, axis=-1)
                fracs.append(changed.mean())
            rows.append({"occlusion_type": occ_type, "severity": severity, "mean_coverage_frac": float(np.mean(fracs)), "n_samples": len(fracs)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Checks 1, 3, 4, 5: per-seed detailed occlusion eval + clean confidence
# ---------------------------------------------------------------------------

def evaluate_occlusion_detailed(tasks, clip_true, det_lookup, checkpoint_path, device) -> pd.DataFrame:
    model = timm.create_model("resnet18", pretrained=False, num_classes=len(CLASS_ORDER)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    records = []
    for occ_type, severity in CONDITIONS:
        for clip_id, frame_indices, landmarks in tasks:
            positions = select_eval_frame_positions(len(frame_indices))
            batch_imgs = []
            for pos in positions:
                fi, lm = frame_indices[pos], landmarks[pos]
                key = (clip_id, int(fi), occ_type, severity)
                if key not in det_lookup.index or not det_lookup.loc[key]:
                    continue
                crop_path = CROPS_DIR / clip_id / f"frame_{int(fi):03d}.jpg"
                crop = np.array(Image.open(crop_path))
                lm_valid = None if np.isnan(lm).any() else lm
                occluded = apply_occlusion(occ_type, crop, lm_valid, severity, clip_id, SEED)
                batch_imgs.append(image_to_tensor(occluded))

            if batch_imgs:
                with torch.no_grad():
                    x = torch.stack(batch_imgs).to(device)
                    probs = torch.softmax(model(x), dim=1).mean(dim=0).cpu().numpy()
                abstained, n_surv = False, len(batch_imgs)
            else:
                probs = np.ones(len(CLASS_ORDER)) / len(CLASS_ORDER)
                abstained, n_surv = True, 0

            pred_idx = int(np.argmax(probs))
            records.append({
                "clip_id": clip_id, "occlusion_type": occ_type, "severity": severity,
                "y_true": clip_true[clip_id], "y_pred": CLASS_ORDER[pred_idx],
                "confidence": float(probs[pred_idx]), "abstained": abstained, "n_surviving": n_surv,
            })
    return pd.DataFrame(records)


def evaluate_clean_detailed(val_metadata, checkpoint_path, device) -> pd.DataFrame:
    model = timm.create_model("resnet18", pretrained=False, num_classes=len(CLASS_ORDER)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    rows = build_eval_rows(val_metadata, CROPS_DIR, CLASS_ORDER)
    ds = FrameDataset(rows, CLASS_ORDER)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    clip_prob_sums, clip_counts, clip_true = {}, {}, {}
    for images, labels, clip_ids in loader:
        images = images.to(device)
        with torch.no_grad():
            probs = torch.softmax(model(images), dim=1).cpu().numpy()
        for p, lbl, cid in zip(probs, labels.numpy(), clip_ids):
            clip_prob_sums.setdefault(cid, np.zeros(len(CLASS_ORDER)))
            clip_prob_sums[cid] += p
            clip_counts[cid] = clip_counts.get(cid, 0) + 1
            clip_true[cid] = int(lbl)

    records = []
    for cid in sorted(clip_prob_sums):
        mean_probs = clip_prob_sums[cid] / clip_counts[cid]
        pred_idx = int(np.argmax(mean_probs))
        records.append({"clip_id": cid, "y_true": CLASS_ORDER[clip_true[cid]], "y_pred": CLASS_ORDER[pred_idx], "confidence": float(mean_probs[pred_idx])})
    return pd.DataFrame(records)


def score_variant(df: pd.DataFrame, exclude_abstained: bool) -> tuple[float, float]:
    sub = df[~df["abstained"]] if exclude_abstained else df
    if len(sub) == 0:
        return float("nan"), float("nan")
    macro_f1 = f1_score(sub["y_true"], sub["y_pred"], labels=CLASS_ORDER, average="macro", zero_division=0)
    acc = (sub["y_true"] == sub["y_pred"]).mean()
    return macro_f1, acc


# ---------------------------------------------------------------------------
# Contact sheet: mask severity progression
# ---------------------------------------------------------------------------

def build_mask_contact_sheet(tasks: list[tuple], val_metadata: pd.DataFrame, out_path: Path) -> None:
    chosen_emotions = ["anger", "happy", "neutral", "sad"]
    tasks_by_clip = {t[0]: t for t in tasks}
    clip_emotion = dict(zip(val_metadata["clip_id"], val_metadata["emotion"]))
    clip_actor = dict(zip(val_metadata["clip_id"], val_metadata["actor_id"]))

    chosen, seen_actors = [], set()
    for emo in chosen_emotions:
        for clip_id, (cid, frame_indices, landmarks) in tasks_by_clip.items():
            if clip_emotion.get(clip_id) == emo and clip_actor.get(clip_id) not in seen_actors:
                chosen.append(clip_id)
                seen_actors.add(clip_actor.get(clip_id))
                break

    tile = 150
    cols = ["clean", "sev=0.3", "sev=0.6", "sev=0.9"]
    sheet = Image.new("RGB", (tile * len(cols), tile * len(chosen) + 20), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for ci, col_name in enumerate(cols):
        draw.text((ci * tile + 5, 2), col_name, fill=(0, 0, 0))

    for ri, clip_id in enumerate(chosen):
        _, frame_indices, landmarks = tasks_by_clip[clip_id]
        pos = select_eval_frame_positions(len(frame_indices))[0]
        fi, lm = frame_indices[pos], landmarks[pos]
        crop_path = CROPS_DIR / clip_id / f"frame_{int(fi):03d}.jpg"
        crop = np.array(Image.open(crop_path))
        lm_valid = None if np.isnan(lm).any() else lm
        variants = [crop] + [apply_occlusion("mask", crop, lm_valid, s, clip_id, SEED) for s in SEVERITIES]
        for ci, var in enumerate(variants):
            img = Image.fromarray(var).resize((tile, tile))
            sheet.paste(img, (ci * tile, 20 + ri * tile))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    tasks, clip_true, val_metadata = load_val_tasks()
    print(f"val: {len(tasks)} clips")

    detection_df = pd.read_parquet(DETECTION_CACHE_PATH)
    det_lookup = detection_df.set_index(["clip_id", "frame_idx", "occlusion_type", "severity"])["detected"]

    # --- Check 2: coverage ---
    t0 = time.perf_counter()
    coverage_df = compute_coverage(tasks)
    coverage_df.to_csv(OUT_DIR / "phase4_mask_sunglasses_coverage.csv", index=False)
    print(f"\n[Check 2] coverage (computed in {time.perf_counter() - t0:.1f}s):")
    print(coverage_df.to_string(index=False))

    # --- Checks 1, 3, 4, 5: per-seed occlusion + clean detail ---
    per_seed_occ_scores = []   # rows: seed, occ_type, severity, scoring, macro_f1, accuracy, n_abstained, n_clips
    per_seed_clean_scores = []  # rows: seed, macro_f1, accuracy, mean_confidence
    per_seed_occ_confidence = []  # rows: seed, occ_type, severity, mean_confidence, accuracy
    seed0_detail_df = None
    seed0_clean_detail_df = None

    for seed in SEEDS:
        ckpt = CHECKPOINT_DIR / f"resnet18_clean_seed{seed}.pt"
        print(f"\n=== seed {seed} ===")

        t0 = time.perf_counter()
        occ_detail = evaluate_occlusion_detailed(tasks, clip_true, det_lookup, ckpt, device)
        print(f"  occlusion detailed eval: {(time.perf_counter() - t0) / 60:.1f} min")
        if seed == 0:
            seed0_detail_df = occ_detail
            occ_detail.to_csv(OUT_DIR / "phase4_occlusion_per_clip_seed0.csv", index=False)

        for occ_type, severity in CONDITIONS:
            sub = occ_detail[(occ_detail["occlusion_type"] == occ_type) & (occ_detail["severity"] == severity)]
            n_abstained = int(sub["abstained"].sum())
            for variant, exclude in [("uniform_argmax_counts", False), ("no_prediction_excluded", True)]:
                mf1, acc = score_variant(sub, exclude)
                per_seed_occ_scores.append({
                    "seed": seed, "occlusion_type": occ_type, "severity": severity, "scoring": variant,
                    "macro_f1": mf1, "accuracy": acc, "n_abstained": n_abstained, "n_clips": len(sub),
                })
            per_seed_occ_confidence.append({
                "seed": seed, "occlusion_type": occ_type, "severity": severity,
                "mean_confidence": sub["confidence"].mean(),
                "accuracy": (sub["y_true"] == sub["y_pred"]).mean(),
            })

        t0 = time.perf_counter()
        clean_detail = evaluate_clean_detailed(val_metadata, ckpt, device)
        print(f"  clean full-frame eval: {(time.perf_counter() - t0) / 60:.1f} min")
        clean_mf1 = f1_score(clean_detail["y_true"], clean_detail["y_pred"], labels=CLASS_ORDER, average="macro")
        clean_acc = (clean_detail["y_true"] == clean_detail["y_pred"]).mean()
        per_seed_clean_scores.append({"seed": seed, "macro_f1": clean_mf1, "accuracy": clean_acc, "mean_confidence": clean_detail["confidence"].mean()})
        print(f"  clean sanity check: macro_f1={clean_mf1:.4f} accuracy={clean_acc:.4f} mean_confidence={clean_detail['confidence'].mean():.4f}")
        if seed == 0:
            seed0_clean_detail_df = clean_detail

    occ_scores_df = pd.DataFrame(per_seed_occ_scores)
    clean_scores_df = pd.DataFrame(per_seed_clean_scores)
    occ_confidence_df = pd.DataFrame(per_seed_occ_confidence)

    occ_scores_df.to_csv(OUT_DIR / "phase4_occlusion_eval_by_seed.csv", index=False)
    clean_scores_df.to_csv(OUT_DIR / "phase4_clean_eval_by_seed.csv", index=False)
    occ_confidence_df.to_csv(OUT_DIR / "phase4_occlusion_confidence_by_seed.csv", index=False)

    # --- Check 1: both scoring variants, 3-seed mean+/-std ---
    print("\n[Check 1 + 5] occlusion table, both abstention-scoring variants, mean +/- std over 3 seeds:")
    agg = occ_scores_df.groupby(["occlusion_type", "severity", "scoring"]).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        n_abstained=("n_abstained", "first"), n_clips=("n_clips", "first"),
    ).reset_index()
    agg.to_csv(OUT_DIR / "phase4_occlusion_eval_table_mean_std_both_scoring.csv", index=False)
    print(agg.to_string(index=False))

    # --- Check 3: predicted-class distribution, mask vs clean (seed 0) ---
    dist_rows = []
    clean_dist = seed0_clean_detail_df["y_pred"].value_counts(normalize=True).reindex(CLASS_ORDER, fill_value=0.0)
    dist_rows.append({"condition": "clean", **clean_dist.to_dict()})
    for severity in SEVERITIES:
        sub = seed0_detail_df[(seed0_detail_df["occlusion_type"] == "mask") & (seed0_detail_df["severity"] == severity)]
        dist = sub["y_pred"].value_counts(normalize=True).reindex(CLASS_ORDER, fill_value=0.0)
        dist_rows.append({"condition": f"mask_sev{severity}", **dist.to_dict()})
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(OUT_DIR / "phase4_predicted_class_distribution_mask_vs_clean.csv", index=False)
    print("\n[Check 3] predicted-class distribution, mask vs. clean (seed 0):")
    print(dist_df.to_string(index=False))

    # --- Check 4: confidence vs accuracy, clean + every condition ---
    conf_rows = [{"condition": "clean", "mean_confidence": clean_scores_df["mean_confidence"].mean(), "accuracy": clean_scores_df["accuracy"].mean()}]
    for occ_type, severity in CONDITIONS:
        sub = occ_confidence_df[(occ_confidence_df["occlusion_type"] == occ_type) & (occ_confidence_df["severity"] == severity)]
        conf_rows.append({"condition": f"{occ_type}_sev{severity}", "mean_confidence": sub["mean_confidence"].mean(), "accuracy": sub["accuracy"].mean()})
    conf_df = pd.DataFrame(conf_rows)
    conf_df.to_csv(OUT_DIR / "phase4_confidence_vs_accuracy.csv", index=False)
    print("\n[Check 4] mean max-softmax confidence vs. accuracy (mean over 3 seeds; clean uses full-frame eval):")
    print(conf_df.to_string(index=False))

    # --- Contact sheet ---
    sheet_path = OUT_DIR / "phase4_mask_severity_contact_sheet.png"
    build_mask_contact_sheet(tasks, val_metadata, sheet_path)
    print(f"\nContact sheet saved to: {sheet_path}")


if __name__ == "__main__":
    main()
