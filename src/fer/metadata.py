"""Build the per-clip metadata table from the real files on disk.

This module does not assume the dataset is clean: it discovers what's
actually in VideoFlash/ and AudioWAV/, flags anomalies (missing, extra,
unreadable, or unparseable files) rather than silently dropping or
coercing them, and only then joins demographics and probes real media
properties for the canonical clip set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .filenames import FilenameParseError, parse_filename
from .known_issues import EXCLUDED_CLIPS
from .probe import probe_audio, probe_video


@dataclass
class DiscoveryReport:
    """Anomalies found while scanning the raw directories, before any
    metadata table is built. Every list here should be reported to the
    user verbatim — none of these are supposed to exist in a clean run,
    but CREMA-D is known to have at least one (the VideoFlash .mp4)."""

    audio_total_files: int = 0
    video_total_files: int = 0
    audio_unparseable: list[str] = field(default_factory=list)
    video_unparseable: list[str] = field(default_factory=list)
    video_non_flv_files: list[str] = field(default_factory=list)  # e.g. the stray .mp4
    audio_duplicate_stems: list[str] = field(default_factory=list)
    video_duplicate_stems: list[str] = field(default_factory=list)
    clips_missing_video: list[str] = field(default_factory=list)  # has .wav, no .flv
    clips_missing_audio: list[str] = field(default_factory=list)  # has .flv, no .wav


def _scan_dir(directory: Path) -> tuple[dict[str, Path], list[str], list[Path]]:
    """Returns (stem -> path for the expected primary extension's files,
    list of unparseable stems, list of ALL file paths found) for bookkeeping."""
    all_paths = sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith("."))
    return {}, [], all_paths


def discover_files(video_dir: Path, audio_dir: Path) -> tuple[DiscoveryReport, dict[str, Path], dict[str, Path]]:
    report = DiscoveryReport()

    _, _, audio_all = _scan_dir(audio_dir)
    _, _, video_all = _scan_dir(video_dir)
    report.audio_total_files = len(audio_all)
    report.video_total_files = len(video_all)

    audio_by_stem: dict[str, list[Path]] = {}
    for p in audio_all:
        if p.suffix.lower() != ".wav":
            report.audio_unparseable.append(p.name)
            continue
        try:
            parse_filename(p.stem)
        except FilenameParseError:
            report.audio_unparseable.append(p.name)
            continue
        audio_by_stem.setdefault(p.stem, []).append(p)

    video_by_stem: dict[str, list[Path]] = {}
    for p in video_all:
        if p.suffix.lower() != ".flv":
            report.video_non_flv_files.append(p.name)
            continue
        try:
            parse_filename(p.stem)
        except FilenameParseError:
            report.video_unparseable.append(p.name)
            continue
        video_by_stem.setdefault(p.stem, []).append(p)

    for stem, paths in audio_by_stem.items():
        if len(paths) > 1:
            report.audio_duplicate_stems.append(stem)
    for stem, paths in video_by_stem.items():
        if len(paths) > 1:
            report.video_duplicate_stems.append(stem)

    audio_map = {stem: paths[0] for stem, paths in audio_by_stem.items()}
    video_map = {stem: paths[0] for stem, paths in video_by_stem.items()}

    report.clips_missing_video = sorted(set(audio_map) - set(video_map))
    report.clips_missing_audio = sorted(set(video_map) - set(audio_map))

    return report, audio_map, video_map


def build_metadata_table(
    video_dir: Path,
    audio_dir: Path,
    demographics_df: pd.DataFrame,
    probe_media: bool = True,
) -> tuple[pd.DataFrame, DiscoveryReport, dict]:
    """Returns (metadata_df, discovery_report, extra_report) where
    extra_report covers demographic-join and probe-level anomalies
    (actors with no clips, clips with no demographics, unreadable files)."""

    report, audio_map, video_map = discover_files(video_dir, audio_dir)

    canonical_stems = sorted(set(audio_map) & set(video_map))
    excluded_stems = [s for s in canonical_stems if s in EXCLUDED_CLIPS]
    canonical_stems = [s for s in canonical_stems if s not in EXCLUDED_CLIPS]

    rows = []
    for stem in tqdm(canonical_stems, desc="parsing + probing clips", disable=not probe_media):
        parsed = parse_filename(stem)
        row = {
            "clip_id": parsed.clip_id,
            "actor_id": parsed.actor_id,
            "sentence_code": parsed.sentence_code,
            "sentence_text": parsed.sentence_text,
            "emotion_code": parsed.emotion_code,
            "emotion": parsed.emotion,
            "intensity_code": parsed.intensity_code,
            "intensity": parsed.intensity,
            "video_path": str(video_map[stem]),
            "audio_path": str(audio_map[stem]),
        }
        if probe_media:
            vp = probe_video(video_map[stem])
            ap = probe_audio(audio_map[stem])
            row.update(
                video_readable=vp.readable,
                video_error=vp.error,
                video_duration_s=vp.duration_s,
                video_fps=vp.fps,
                video_width=vp.width,
                video_height=vp.height,
                video_codec=vp.video_codec,
                video_has_audio_stream=vp.has_audio_stream,
                audio_readable=ap.readable,
                audio_error=ap.error,
                audio_duration_s=ap.duration_s,
                audio_sample_rate=ap.sample_rate,
                audio_channels=ap.channels,
                audio_subtype=ap.subtype,
            )
        rows.append(row)

    df = pd.DataFrame(rows)

    extra_report: dict = {"excluded_known_bad_clips": {s: EXCLUDED_CLIPS[s] for s in excluded_stems}}
    if not df.empty:
        metadata_actor_ids = set(df["actor_id"].unique())
        demo_actor_ids = set(demographics_df["ActorID"].unique())
        extra_report["actors_in_clips_not_in_demographics"] = sorted(metadata_actor_ids - demo_actor_ids)
        extra_report["actors_in_demographics_with_no_clips"] = sorted(demo_actor_ids - metadata_actor_ids)

        demo_cols = demographics_df[["ActorID", "Age", "Sex", "Race", "Ethnicity"]].reset_index(drop=True)
        df = df.merge(
            demo_cols,
            left_on="actor_id",
            right_on="ActorID",
            how="left",
        ).drop(columns=["ActorID"])

        if probe_media:
            extra_report["unreadable_video"] = df.loc[~df["video_readable"], "clip_id"].tolist()
            extra_report["unreadable_audio"] = df.loc[~df["audio_readable"], "clip_id"].tolist()

    return df, report, extra_report
