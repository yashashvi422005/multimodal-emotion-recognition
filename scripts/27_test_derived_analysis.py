#!/usr/bin/env python
"""Derived analysis from the already-saved final test-set outputs --
NO model inference, NO re-run of evaluate.py --final, nothing about the
frozen run changes. Reads only:
    outputs/evaluate_test_by_seed.csv   -- per-seed, per-condition macro-F1
    outputs/evaluate_test_table.csv     -- 3-seed aggregates (cross-check)
    outputs/evaluate_test_per_clip.csv  -- per-clip predictions (for bootstrap CIs)

Produces:
  1. Fixed 50/50 fusion (aug_all/aug_no_mask/clean-trained + WavLM) on CLEAN
     faces: per-seed scores and mean +/- std.
  2. Per-seed scores for every face-only and fusion row already reported in
     RESULTS.md's "Final test results" (clean-condition face-only table,
     and the pooled-over-12-occlusion-conditions fusion table).
  3. Actor-level bootstrap 95% CIs (resample the 12 test actors, B=2000,
     seed-averaged per draw -- same method as P1-P3) for the ABSOLUTE
     macro-F1 of: tuned eGeMAPS, WavLM, the 3 face models on clean faces,
     and fixed fusion (aug_all + WavLM) on clean faces and pooled over the
     12 occlusion conditions.

Usage:
    python scripts/27_test_derived_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"
CLASS_ORDER = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
SEEDS = [0, 1, 2]
N_BOOTSTRAP = 2000
BOOT_SEED = 0
FACE_VARIANTS = ["clean", "aug_all", "aug_no_mask"]
AUDIO_BRANCHES = ["egemaps_tuned", "wavlm"]


def main() -> None:
    data_cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    sf = load_split(REPO_ROOT / data_cfg["paths"]["splits_dir"] / "split_v1.json")
    test_actors = sorted(sf.assignment["test"])
    metadata_all = pd.read_csv(REPO_ROOT / data_cfg["paths"]["outputs_dir"] / "metadata.csv", dtype={"actor_id": str})
    clip_to_actor = dict(zip(metadata_all["clip_id"], metadata_all["actor_id"]))

    by_seed = pd.read_csv(OUT_DIR / "evaluate_test_by_seed.csv")
    per_clip = pd.read_csv(OUT_DIR / "evaluate_test_per_clip.csv")

    def row(audio_branch, variant, method, occ_type=None):
        m = (by_seed["audio_branch"] == audio_branch) & (by_seed["model_variant"] == variant) & (by_seed["method"] == method)
        if occ_type is not None:
            m = m & (by_seed["occlusion_type"] == occ_type)
        else:
            m = m & (by_seed["occlusion_type"] != "clean")
        return by_seed[m]

    # ---------------------------------------------------------------
    # 1. Fixed 50/50 fusion on CLEAN faces, per face variant + WavLM
    # ---------------------------------------------------------------
    print("=" * 70)
    print("1. Fixed 50/50 fusion (WavLM) on CLEAN faces, per-seed + mean +/- std")
    print("=" * 70)
    clean_fusion_rows = []
    for variant in FACE_VARIANTS:
        sub = row("wavlm", variant, "fixed_5050", occ_type="clean")
        vals = [sub[sub["seed"] == s]["macro_f1"].iloc[0] for s in SEEDS]
        mean, std = float(np.mean(vals)), float(np.std(vals, ddof=1))
        print(f"  {variant:12s}: seeds={[f'{v:.4f}' for v in vals]}  mean={mean:.4f} +/- {std:.4f}")
        clean_fusion_rows.append({"model_variant": variant, "audio_branch": "wavlm", "method": "fixed_5050", "occlusion_type": "clean", "seed0": vals[0], "seed1": vals[1], "seed2": vals[2], "mean": mean, "std": std})
    pd.DataFrame(clean_fusion_rows).to_csv(OUT_DIR / "test_derived_clean_fusion_per_seed.csv", index=False)

    # ---------------------------------------------------------------
    # 2. Per-seed scores for every face-only and fusion row already
    #    reported (clean-condition face-only table; pooled-over-12-
    #    occlusion-conditions fusion table, all methods, both audio branches)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2a. Face-only, CLEAN condition -- per-seed + mean +/- std (cross-check vs RESULTS.md)")
    print("=" * 70)
    face_clean_rows = []
    for variant in FACE_VARIANTS:
        sub = row("wavlm", variant, "face_only", occ_type="clean")  # face_only doesn't depend on audio_branch
        vals = [sub[sub["seed"] == s]["macro_f1"].iloc[0] for s in SEEDS]
        mean, std = float(np.mean(vals)), float(np.std(vals, ddof=1))
        print(f"  {variant:12s}: seeds={[f'{v:.4f}' for v in vals]}  mean={mean:.4f} +/- {std:.4f}")
        face_clean_rows.append({"model_variant": variant, "occlusion_type": "clean", "method": "face_only", "seed0": vals[0], "seed1": vals[1], "seed2": vals[2], "mean": mean, "std": std})

    print("\n" + "=" * 70)
    print("2b. Pooled over 12 occlusion conditions -- per-seed + mean +/- std, all methods x variants x audio branches")
    print("=" * 70)
    pooled_rows = []
    methods = ["audio_only", "face_only", "fixed_5050", "conf_maxprob", "conf_negentropy", "learned_gate"]
    for audio_branch in AUDIO_BRANCHES:
        for variant in FACE_VARIANTS:
            for method in methods:
                sub = row(audio_branch, variant, method)  # occ_type=None -> pooled over the 12 non-clean conditions
                if sub.empty:
                    continue
                per_seed_vals = []
                for s in SEEDS:
                    seed_sub = sub[sub["seed"] == s]
                    per_seed_vals.append(float(seed_sub["macro_f1"].mean()))  # mean over the 12 conditions for this seed
                mean, std = float(np.mean(per_seed_vals)), float(np.std(per_seed_vals, ddof=1))
                print(f"  {audio_branch:13s} {variant:12s} {method:16s}: seeds={[f'{v:.4f}' for v in per_seed_vals]}  mean={mean:.4f} +/- {std:.4f}")
                pooled_rows.append({"audio_branch": audio_branch, "model_variant": variant, "method": method, "seed0": per_seed_vals[0], "seed1": per_seed_vals[1], "seed2": per_seed_vals[2], "mean": mean, "std": std})

    pd.DataFrame(face_clean_rows).to_csv(OUT_DIR / "test_derived_face_clean_per_seed.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(OUT_DIR / "test_derived_pooled_per_seed.csv", index=False)

    # ---------------------------------------------------------------
    # 3. Actor-level bootstrap 95% CIs for ABSOLUTE macro-F1
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. Actor-level bootstrap 95% CIs (B={}, seed-averaged per draw, same method as P1-P3)".format(N_BOOTSTRAP))
    print("=" * 70)

    def bootstrap_absolute(sub_by_seed: dict) -> tuple[float, float, float]:
        """sub_by_seed: {seed: dataframe with y_true, y_pred, clip_id}. Returns (point_mean, ci_lo, ci_hi)."""
        per_seed_data = {}
        for s, sub in sub_by_seed.items():
            y_true, y_pred = sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()
            actor = sub["clip_id"].map(clip_to_actor).to_numpy()
            idx = {a: np.where(actor == a)[0] for a in test_actors}
            per_seed_data[s] = (y_true, y_pred, idx)

        point = float(np.mean([f1_score(yt, yp, labels=CLASS_ORDER, average="macro") for yt, yp, _ in per_seed_data.values()]))

        rng = np.random.default_rng(BOOT_SEED)
        seeds_here = sorted(per_seed_data.keys())
        diffs = np.empty(N_BOOTSTRAP)
        for b in range(N_BOOTSTRAP):
            resampled = rng.choice(test_actors, size=len(test_actors), replace=True)
            seed_scores = np.empty(len(seeds_here))
            for i, s in enumerate(seeds_here):
                yt, yp, idx = per_seed_data[s]
                ia = np.concatenate([idx[a] for a in resampled])
                seed_scores[i] = f1_score(yt[ia], yp[ia], labels=CLASS_ORDER, average="macro")
            diffs[b] = seed_scores.mean()
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
        return point, float(ci_lo), float(ci_hi)

    ci_results = []

    # tuned eGeMAPS (clean condition; audio score is condition-invariant)
    sub_by_seed = {s: per_clip[(per_clip["audio_branch"] == "egemaps_tuned") & (per_clip["model_variant"] == "clean") & (per_clip["method"] == "audio_only") & (per_clip["occlusion_type"] == "clean") & (per_clip["seed"] == s)] for s in SEEDS}
    point, lo, hi = bootstrap_absolute(sub_by_seed)
    print(f"  eGeMAPS (tuned):                 point={point:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
    ci_results.append({"label": "egemaps_tuned", "point": point, "ci_lo": lo, "ci_hi": hi})

    # WavLM (clean condition)
    sub_by_seed = {s: per_clip[(per_clip["audio_branch"] == "wavlm") & (per_clip["model_variant"] == "clean") & (per_clip["method"] == "audio_only") & (per_clip["occlusion_type"] == "clean") & (per_clip["seed"] == s)] for s in SEEDS}
    point, lo, hi = bootstrap_absolute(sub_by_seed)
    print(f"  WavLM:                            point={point:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
    ci_results.append({"label": "wavlm", "point": point, "ci_lo": lo, "ci_hi": hi})

    # 3 face models, clean condition
    for variant in FACE_VARIANTS:
        sub_by_seed = {s: per_clip[(per_clip["audio_branch"] == "wavlm") & (per_clip["model_variant"] == variant) & (per_clip["method"] == "face_only") & (per_clip["occlusion_type"] == "clean") & (per_clip["seed"] == s)] for s in SEEDS}
        point, lo, hi = bootstrap_absolute(sub_by_seed)
        print(f"  face {variant:12s} (clean):      point={point:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
        ci_results.append({"label": f"face_{variant}_clean", "point": point, "ci_lo": lo, "ci_hi": hi})

    # fixed fusion (aug_all + WavLM) on clean faces
    sub_by_seed = {s: per_clip[(per_clip["audio_branch"] == "wavlm") & (per_clip["model_variant"] == "aug_all") & (per_clip["method"] == "fixed_5050") & (per_clip["occlusion_type"] == "clean") & (per_clip["seed"] == s)] for s in SEEDS}
    point, lo, hi = bootstrap_absolute(sub_by_seed)
    print(f"  fixed fusion (aug_all+WavLM, clean):      point={point:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
    ci_results.append({"label": "fixed_aug_all_wavlm_clean", "point": point, "ci_lo": lo, "ci_hi": hi})

    # fixed fusion (aug_all + WavLM) pooled over the 12 occlusion conditions
    sub_by_seed = {s: per_clip[(per_clip["audio_branch"] == "wavlm") & (per_clip["model_variant"] == "aug_all") & (per_clip["method"] == "fixed_5050") & (per_clip["occlusion_type"] != "clean") & (per_clip["seed"] == s)] for s in SEEDS}
    point, lo, hi = bootstrap_absolute(sub_by_seed)
    print(f"  fixed fusion (aug_all+WavLM, pooled 12):  point={point:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
    ci_results.append({"label": "fixed_aug_all_wavlm_pooled", "point": point, "ci_lo": lo, "ci_hi": hi})

    pd.DataFrame(ci_results).to_csv(OUT_DIR / "test_derived_bootstrap_ci.csv", index=False)


if __name__ == "__main__":
    main()
