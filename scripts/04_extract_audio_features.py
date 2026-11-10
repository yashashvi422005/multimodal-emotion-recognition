#!/usr/bin/env python
"""Extract openSMILE eGeMAPSv02 functional features (88-dim) for the
canonical CREMA-D clips, cache to disk, and report any extraction
failures.

Scope: TRAIN + VAL actors only (not test). Feature extraction is
deterministic signal processing with no fitted parameters, but to keep
the test set genuinely untouched by anything outside evaluate.py
--final, test-actor clips are excluded here and will be extracted
separately only when that script exists.

Usage:
    python scripts/04_extract_audio_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.audio_features import extract_egemaps_for_clips, report_nan_inf, save_features
from fer.splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    paths = cfg["paths"]

    metadata_path = REPO_ROOT / paths["outputs_dir"] / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} not found — run scripts/01_verify_and_build_metadata.py first")
    metadata_df = pd.read_csv(metadata_path, dtype={"actor_id": str})

    split_path = REPO_ROOT / paths["splits_dir"] / "split_v1.json"
    sf = load_split(split_path)
    if sf.status != "frozen":
        raise RuntimeError(f"{split_path} is not frozen (status={sf.status}) — refusing to proceed")

    train_actors = set(sf.assignment["train"])
    val_actors = set(sf.assignment["val"])
    test_actors = set(sf.assignment["test"])
    train_val_actors = train_actors | val_actors

    clips = metadata_df[metadata_df["actor_id"].isin(train_val_actors)].copy()
    assert not set(clips["actor_id"]) & test_actors, "test actors leaked into extraction scope — refusing to proceed"

    print(f"Extracting eGeMAPSv02 functionals for {len(clips)} train+val clips ({len(train_val_actors)} actors).")
    print(f"Explicitly excluding {len(test_actors)} test actors ({(metadata_df['actor_id'].isin(test_actors)).sum()} clips).")

    features_df, failures = extract_egemaps_for_clips(clips)

    print(f"\nSucceeded: {len(features_df)} / {len(clips)}")
    print(f"Failed: {len(failures)}")
    for f in failures:
        print(f"  - {f['clip_id']}: {f['error']}")

    nan_report = report_nan_inf(features_df)
    print(f"\nRows with NaN in any feature: {nan_report['n_rows_with_nan']}")
    print(f"Rows with Inf in any feature: {nan_report['n_rows_with_inf']}")
    if nan_report["n_rows_with_nan"]:
        print(f"  clip_ids: {nan_report['clip_ids_with_nan']}")
        print(f"  affected columns: {nan_report['columns_with_nan']}")

    out_path = REPO_ROOT / "cache" / "features" / "egemaps_functionals_train_val.parquet"
    save_features(features_df, out_path)
    print(f"\nCached features to: {out_path}")

    failures_path = REPO_ROOT / "outputs" / "egemaps_extraction_failures.csv"
    pd.DataFrame(failures).to_csv(failures_path, index=False)
    print(f"Failure report written to: {failures_path}")


if __name__ == "__main__":
    main()
