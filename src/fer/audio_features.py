"""openSMILE eGeMAPS functional feature extraction, with per-clip failure
reporting (never assume extraction succeeds silently)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import opensmile
import pandas as pd
from tqdm import tqdm


def get_egemaps_extractor() -> opensmile.Smile:
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_egemaps_for_clips(
    metadata_df: pd.DataFrame,
    smile: opensmile.Smile | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """metadata_df must have columns clip_id, actor_id, audio_path, emotion.
    Returns (features_df, failures) where features_df has columns
    [clip_id, actor_id, emotion, <88 feature columns>] and failures is a
    list of {clip_id, audio_path, error} for clips openSMILE could not
    process."""
    smile = smile or get_egemaps_extractor()

    rows = []
    failures: list[dict] = []
    for _, r in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="extracting eGeMAPS functionals"):
        try:
            feats = smile.process_file(r["audio_path"])
        except Exception as e:  # noqa: BLE001 - report, don't crash the batch
            failures.append({"clip_id": r["clip_id"], "audio_path": r["audio_path"], "error": f"{type(e).__name__}: {e}"})
            continue
        row = {"clip_id": r["clip_id"], "actor_id": r["actor_id"], "emotion": r["emotion"]}
        row.update(feats.iloc[0].to_dict())
        rows.append(row)

    features_df = pd.DataFrame(rows)
    return features_df, failures


def save_features(features_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(path, index=False)


def load_features(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def feature_columns(features_df: pd.DataFrame) -> list[str]:
    non_feature = {"clip_id", "actor_id", "emotion"}
    return [c for c in features_df.columns if c not in non_feature]


def report_nan_inf(features_df: pd.DataFrame) -> dict:
    cols = feature_columns(features_df)
    arr = features_df[cols].to_numpy(dtype=np.float64)
    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    rows_with_nan = features_df.loc[nan_mask.any(axis=1), "clip_id"].tolist()
    rows_with_inf = features_df.loc[inf_mask.any(axis=1), "clip_id"].tolist()
    cols_with_nan = [cols[i] for i in np.where(nan_mask.any(axis=0))[0]]
    return {
        "n_rows_with_nan": len(rows_with_nan),
        "n_rows_with_inf": len(rows_with_inf),
        "clip_ids_with_nan": rows_with_nan,
        "clip_ids_with_inf": rows_with_inf,
        "columns_with_nan": cols_with_nan,
    }
