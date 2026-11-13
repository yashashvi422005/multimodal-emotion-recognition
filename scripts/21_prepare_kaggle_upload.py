#!/usr/bin/env python
"""Generates the small private-dataset payload for the Kaggle WavLM step:
splits/split_v1.json (copy), the excluded-clips list (the 4 clips
documented in README.md's "Known upstream data quirks" section), an MD5
checksum list of the local canonical 7,438-clip AudioWAV set, and a
supplementary_audio/ folder holding local copies of any canonical clip
known to be missing from the public ejlok1/cremad mirror (currently just
1040_ITH_SAD_XX, discovered when the notebook's verification step first
ran against that mirror). These let the Kaggle notebook verify its copy
of CREMA-D's audio matches this project's local files exactly -- using
the supplement for any clip the public mirror lacks -- before any
training happens.

Usage:
    python scripts/21_prepare_kaggle_upload.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "kaggle_upload"

# The 4 clips excluded from the canonical 7,442 -> 7,438 clip set, per
# README.md's "Known upstream data quirks" section (CHECKPOINT 1 decision).
EXCLUDED_CLIPS = [
    {"clip_id": "1064_IEO_DIS_MD", "reason": "compilation of all emotional displays for this sentence; the DIS label does not describe its actual content"},
    {"clip_id": "1064_TIE_SAD_XX", "reason": "degenerate/corrupt video (~0.57s); excluded project-wide for consistency even though audio is intact"},
    {"clip_id": "1076_MTI_NEU_XX", "reason": "degenerate/corrupt video (~0.07s); excluded project-wide for consistency even though audio is intact"},
    {"clip_id": "1076_MTI_SAD_XX", "reason": "\"Media offline\" placeholder slate, not real footage; failed face detection on all sampled frames"},
]

# Canonical clips confirmed (by running the Kaggle notebook's verification step)
# to be missing from the public ejlok1/cremad mirror. Supplied locally instead,
# via supplementary_audio/, and checksum-verified the same as every other file.
MIRROR_GAPS = ["1040_ITH_SAD_XX"]


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    data_cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    split_src = REPO_ROOT / data_cfg["paths"]["splits_dir"] / "split_v1.json"
    split_dst = OUT_DIR / "split_v1.json"
    shutil.copy(split_src, split_dst)
    print(f"copied {split_src} -> {split_dst}")

    excluded_path = OUT_DIR / "excluded_clips.json"
    excluded_path.write_text(json.dumps({
        "description": "Clips excluded from the canonical CREMA-D clip set used by this project (7442 raw -> 7438 canonical). See README.md 'Known upstream data quirks'.",
        "excluded_clips": EXCLUDED_CLIPS,
    }, indent=2))
    print(f"wrote {excluded_path} ({len(EXCLUDED_CLIPS)} excluded clips)")

    metadata_df = pd.read_csv(REPO_ROOT / data_cfg["paths"]["outputs_dir"] / "metadata.csv", dtype={"actor_id": str})
    excluded_ids = {c["clip_id"] for c in EXCLUDED_CLIPS}
    assert not (set(metadata_df["clip_id"]) & excluded_ids), "metadata.csv should already exclude these -- excluded_ids found inside it"
    print(f"canonical clip count in metadata.csv: {len(metadata_df)} (sanity check: should be 7438)")
    assert len(metadata_df) == 7438, f"expected 7438 canonical clips, found {len(metadata_df)}"

    audio_dir = REPO_ROOT / "data" / "CREMA-D" / "AudioWAV"
    rows = []
    missing = []
    for clip_id in sorted(metadata_df["clip_id"]):
        wav_path = audio_dir / f"{clip_id}.wav"
        if not wav_path.exists():
            missing.append(clip_id)
            continue
        md5 = hashlib.md5(wav_path.read_bytes()).hexdigest()
        rows.append({"clip_id": clip_id, "filename": wav_path.name, "md5": md5, "size_bytes": wav_path.stat().st_size})

    if missing:
        raise RuntimeError(f"{len(missing)} canonical clips have no local AudioWAV file: {missing[:5]}...")

    checksums_df = pd.DataFrame(rows)
    checksums_path = OUT_DIR / "audio_md5_checksums.csv"
    checksums_df.to_csv(checksums_path, index=False)
    print(f"wrote {checksums_path} ({len(checksums_df)} checksums)")

    supp_dir = OUT_DIR / "supplementary_audio"
    supp_dir.mkdir(exist_ok=True)
    for clip_id in MIRROR_GAPS:
        assert clip_id in set(metadata_df["clip_id"]), f"{clip_id} is in MIRROR_GAPS but not in the canonical clip set"
        src = audio_dir / f"{clip_id}.wav"
        assert src.exists(), f"MIRROR_GAPS clip {clip_id} has no local AudioWAV file at {src}"
        dst = supp_dir / f"{clip_id}.wav"
        shutil.copy(src, dst)
        print(f"copied supplementary audio: {src} -> {dst}")

    manifest = {
        "canonical_clip_count": len(metadata_df),
        "n_excluded_clips": len(EXCLUDED_CLIPS),
        "n_checksums": len(checksums_df),
        "mirror_gaps": MIRROR_GAPS,
        "files": ["split_v1.json", "excluded_clips.json", "audio_md5_checksums.csv", "supplementary_audio/"],
    }
    (OUT_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT_DIR / 'MANIFEST.json'}")
    print(f"\nkaggle_upload/ ready. Upload this folder's contents as a private Kaggle dataset.")


if __name__ == "__main__":
    main()
