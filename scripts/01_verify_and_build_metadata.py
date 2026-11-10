#!/usr/bin/env python
"""Phase 1, step 1: verify the raw CREMA-D files, parse filenames into a
metadata table, join demographics, and report class counts / demographic
breakdown / missing-extra-unreadable files.

Usage:
    python scripts/01_verify_and_build_metadata.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.demographics import load_demographics
from fer.envinfo import get_env_report
from fer.metadata import build_metadata_table

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    paths = cfg["paths"]
    video_dir = REPO_ROOT / paths["video_dir"]
    audio_dir = REPO_ROOT / paths["audio_dir"]
    demographics_csv = REPO_ROOT / paths["demographics_csv"]
    outputs_dir = REPO_ROOT / paths["outputs_dir"]
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)
    env = get_env_report(repo_root=str(REPO_ROOT))
    print(f"python: {env['python_version'].splitlines()[0]}")
    print(f"platform: {env['platform']}  machine: {env['machine']}")
    print(f"git commit: {env['git_commit']}")
    for pkg in ("pandas", "numpy", "av", "soundfile", "librosa"):
        print(f"  {pkg}: {env['package_versions'][pkg]}")

    print()
    print("=" * 70)
    print("LOADING DEMOGRAPHICS")
    print("=" * 70)
    demographics_df = load_demographics(demographics_csv)
    print(f"actors in demographics table: {len(demographics_df)}")
    print(demographics_df[["Age", "Sex", "Race", "Ethnicity"]].describe(include="all").to_string())

    print()
    print("=" * 70)
    print("SCANNING + PROBING RAW FILES (this reads real file headers, not filenames)")
    print("=" * 70)
    metadata_df, discovery, extra = build_metadata_table(video_dir, audio_dir, demographics_df, probe_media=True)

    print(f"audio files found in {audio_dir.name}/: {discovery.audio_total_files}")
    print(f"video files found in {video_dir.name}/: {discovery.video_total_files}")
    excluded = extra.get("excluded_known_bad_clips", {})
    print(f"clips with both a parseable .wav and .flv (before exclusions): {len(metadata_df) + len(excluded)}")
    print(f"excluded as known-bad (see below): {len(excluded)}")
    print(f"FINAL canonical clip count: {len(metadata_df)}")

    print(f"\nExcluded known-bad clips: {len(excluded)}")
    for clip_id, reason in excluded.items():
        print(f"  - {clip_id}: {reason}")

    def report_list(label: str, items: list[str]) -> None:
        print(f"\n{label}: {len(items)}")
        for item in items[:50]:
            print(f"  - {item}")
        if len(items) > 50:
            print(f"  ... and {len(items) - 50} more")

    report_list("Non-.flv files in VideoFlash/ (extra/anomalous)", discovery.video_non_flv_files)
    report_list("Unparseable audio filenames", discovery.audio_unparseable)
    report_list("Unparseable video filenames", discovery.video_unparseable)
    report_list("Duplicate audio stems", discovery.audio_duplicate_stems)
    report_list("Duplicate video stems", discovery.video_duplicate_stems)
    report_list("Clips with .wav but no matching .flv", discovery.clips_missing_video)
    report_list("Clips with .flv but no matching .wav", discovery.clips_missing_audio)
    report_list("Actors appearing in clips but not in VideoDemographics.csv", extra.get("actors_in_clips_not_in_demographics", []))
    report_list("Actors in VideoDemographics.csv with zero clips", extra.get("actors_in_demographics_with_no_clips", []))
    report_list("Unreadable video files (header could not be parsed)", extra.get("unreadable_video", []))
    report_list("Unreadable audio files (header could not be parsed)", extra.get("unreadable_audio", []))

    print()
    print("=" * 70)
    print("CLASS COUNTS (emotion, ground truth = filename-encoded label)")
    print("=" * 70)
    print(metadata_df["emotion"].value_counts().to_string())

    print()
    print("=" * 70)
    print("INTENSITY COUNTS")
    print("=" * 70)
    print(metadata_df["intensity"].value_counts().to_string())

    print()
    print("=" * 70)
    print("DEMOGRAPHIC BREAKDOWN (clip-level, i.e. weighted by #clips per actor)")
    print("=" * 70)
    print("By Sex:")
    print(metadata_df["Sex"].value_counts().to_string())
    print("\nBy Race:")
    print(metadata_df["Race"].value_counts().to_string())
    print("\nBy Ethnicity:")
    print(metadata_df["Ethnicity"].value_counts().to_string())
    print(f"\nAge stats (actor-level demographics, clip-weighted): \n{metadata_df['Age'].describe().to_string()}")

    print()
    print("=" * 70)
    print("REAL MEDIA PROPERTIES OBSERVED (do not assume — these are measured)")
    print("=" * 70)
    print("Video FPS value counts:")
    print(metadata_df["video_fps"].round(3).value_counts(dropna=False).to_string())
    print("\nVideo resolution (width x height) value counts:")
    res = metadata_df.dropna(subset=["video_width", "video_height"])
    print((res["video_width"].astype(int).astype(str) + "x" + res["video_height"].astype(int).astype(str)).value_counts().to_string())
    print("\nVideo codec value counts:")
    print(metadata_df["video_codec"].value_counts(dropna=False).to_string())
    print("\nAudio sample rate value counts:")
    print(metadata_df["audio_sample_rate"].value_counts(dropna=False).to_string())
    print("\nAudio channels value counts:")
    print(metadata_df["audio_channels"].value_counts(dropna=False).to_string())
    print("\nVideo duration (s) stats:")
    print(metadata_df["video_duration_s"].describe().to_string())
    print("\nAudio duration (s) stats:")
    print(metadata_df["audio_duration_s"].describe().to_string())

    out_csv = outputs_dir / "metadata.csv"
    metadata_df.to_csv(out_csv, index=False)
    print(f"\nFull per-clip metadata table written to: {out_csv}")


if __name__ == "__main__":
    main()
