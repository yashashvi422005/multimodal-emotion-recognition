#!/usr/bin/env python
"""Actor-level bootstrap 95% CIs on the demographic-CV out-of-fold
predictions (scripts/19), for two pre-specified group comparisons:
female-male and <=30 vs 46-60. Also computes the CI for the CHANGE in
the age gap from the clean-trained to the aug_all face model (face-only
and fusion), i.e. whether occlusion-augmented training widens or narrows
the age gap.

Resampling is actor-level and stratified within each group being
compared: each bootstrap draw resamples group A's actors (with
replacement, same N as the real group) independently from group B's
actors, then pools clips to compute macro-F1 for each group and takes
the difference. For the age-gap-change CI, the SAME bootstrap draw of
resampled actors is reused across both face-model variants, so the
within-draw clean vs. aug_all comparison is properly paired.

Training-free: operates entirely on outputs/phase_demographic_cv_predictions.csv.

Usage:
    python scripts/20_demographic_bootstrap_ci.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_ORDER = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
N_BOOT = 1000
RNG_SEED = 0
METHODS = ["audio_only", "face_only_clean", "face_only_aug_all", "fusion_clean", "fusion_aug_all"]


def load_data():
    preds = pd.read_csv(REPO_ROOT / "outputs" / "phase_demographic_cv_predictions.csv", dtype={"actor_id": str})
    meta = pd.read_csv(REPO_ROOT / "outputs" / "metadata.csv", dtype={"actor_id": str})
    actor_demo = meta.drop_duplicates("actor_id").set_index("actor_id")[["Age", "Sex", "Race"]]
    preds["Sex"] = preds["actor_id"].map(actor_demo["Sex"])
    preds["Age"] = preds["actor_id"].map(actor_demo["Age"])
    preds["AgeBand"] = pd.cut(preds["Age"], bins=[0, 30, 45, 60, 100], labels=["<=30", "31-45", "46-60", "60+"])
    return preds


def per_actor_arrays(df_method: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """actor_id -> (y_true array, y_pred array) for one method's rows."""
    out = {}
    for actor_id, sub in df_method.groupby("actor_id"):
        out[actor_id] = (sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
    return out


def macro_f1_for_actor_sample(arrays: dict, actor_sample: list[str]) -> float:
    y_true = np.concatenate([arrays[a][0] for a in actor_sample])
    y_pred = np.concatenate([arrays[a][1] for a in actor_sample])
    return f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro")


def bootstrap_group_diff(arrays: dict, actors_a: list[str], actors_b: list[str], n_boot: int, rng: np.random.Generator) -> np.ndarray:
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sample_a = rng.choice(actors_a, size=len(actors_a), replace=True)
        sample_b = rng.choice(actors_b, size=len(actors_b), replace=True)
        diffs[i] = macro_f1_for_actor_sample(arrays, sample_a) - macro_f1_for_actor_sample(arrays, sample_b)
    return diffs


def summarize_ci(point_a: float, point_b: float, diffs: np.ndarray) -> dict:
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "point_a": point_a, "point_b": point_b, "point_diff": point_a - point_b,
        "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0),
    }


def main() -> None:
    preds = load_data()
    rng = np.random.default_rng(RNG_SEED)

    sex_rows, age_rows, age_change_rows = [], [], []

    female_actors = sorted(preds[preds.Sex == "Female"]["actor_id"].unique())
    male_actors = sorted(preds[preds.Sex == "Male"]["actor_id"].unique())
    young_actors = sorted(preds[preds.AgeBand == "<=30"]["actor_id"].unique())
    midold_actors = sorted(preds[preds.AgeBand == "46-60"]["actor_id"].unique())
    print(f"female={len(female_actors)} male={len(male_actors)} <=30={len(young_actors)} 46-60={len(midold_actors)}")

    arrays_by_method = {m: per_actor_arrays(preds[preds.method == m]) for m in METHODS}

    print("\n=== female - male (per method) ===")
    for m in METHODS:
        arrays = arrays_by_method[m]
        point_f = macro_f1_for_actor_sample(arrays, female_actors)
        point_m = macro_f1_for_actor_sample(arrays, male_actors)
        diffs = bootstrap_group_diff(arrays, female_actors, male_actors, N_BOOT, rng)
        res = summarize_ci(point_f, point_m, diffs)
        res["method"] = m
        sex_rows.append(res)
        print(f"  {m:20s} female={point_f:.4f} male={point_m:.4f} diff={res['point_diff']:+.4f} "
              f"95%CI=[{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}] excludes_zero={res['excludes_zero']}")

    print("\n=== <=30 - 46-60 (per method) ===")
    for m in METHODS:
        arrays = arrays_by_method[m]
        point_y = macro_f1_for_actor_sample(arrays, young_actors)
        point_o = macro_f1_for_actor_sample(arrays, midold_actors)
        diffs = bootstrap_group_diff(arrays, young_actors, midold_actors, N_BOOT, rng)
        res = summarize_ci(point_y, point_o, diffs)
        res["method"] = m
        age_rows.append(res)
        print(f"  {m:20s} <=30={point_y:.4f} 46-60={point_o:.4f} diff={res['point_diff']:+.4f} "
              f"95%CI=[{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}] excludes_zero={res['excludes_zero']}")

    print("\n=== change in age gap, clean -> aug_all (paired bootstrap draws) ===")
    for branch, clean_method, aug_method in [("face", "face_only_clean", "face_only_aug_all"), ("fusion", "fusion_clean", "fusion_aug_all")]:
        arr_clean, arr_aug = arrays_by_method[clean_method], arrays_by_method[aug_method]
        gap_clean_point = macro_f1_for_actor_sample(arr_clean, young_actors) - macro_f1_for_actor_sample(arr_clean, midold_actors)
        gap_aug_point = macro_f1_for_actor_sample(arr_aug, young_actors) - macro_f1_for_actor_sample(arr_aug, midold_actors)
        change_point = gap_aug_point - gap_clean_point

        changes = np.empty(N_BOOT)
        for i in range(N_BOOT):
            sample_y = rng.choice(young_actors, size=len(young_actors), replace=True)
            sample_o = rng.choice(midold_actors, size=len(midold_actors), replace=True)
            gap_clean_i = macro_f1_for_actor_sample(arr_clean, sample_y) - macro_f1_for_actor_sample(arr_clean, sample_o)
            gap_aug_i = macro_f1_for_actor_sample(arr_aug, sample_y) - macro_f1_for_actor_sample(arr_aug, sample_o)
            changes[i] = gap_aug_i - gap_clean_i

        lo, hi = np.percentile(changes, [2.5, 97.5])
        excludes_zero = bool(lo > 0 or hi < 0)
        age_change_rows.append({
            "branch": branch, "gap_clean": gap_clean_point, "gap_aug_all": gap_aug_point,
            "change": change_point, "ci_lo": lo, "ci_hi": hi, "excludes_zero": excludes_zero,
        })
        print(f"  {branch:8s} gap_clean={gap_clean_point:+.4f} gap_aug_all={gap_aug_point:+.4f} "
              f"change={change_point:+.4f} 95%CI=[{lo:+.4f}, {hi:+.4f}] excludes_zero={excludes_zero}")

    pd.DataFrame(sex_rows).to_csv(REPO_ROOT / "outputs" / "phase_demographic_bootstrap_sex.csv", index=False)
    pd.DataFrame(age_rows).to_csv(REPO_ROOT / "outputs" / "phase_demographic_bootstrap_age.csv", index=False)
    pd.DataFrame(age_change_rows).to_csv(REPO_ROOT / "outputs" / "phase_demographic_bootstrap_age_change.csv", index=False)
    print("\nSaved: outputs/phase_demographic_bootstrap_{sex,age,age_change}.csv")


if __name__ == "__main__":
    main()
