"""Clips confirmed, by direct file inspection, to be unusable at their
filename-encoded label/content. Excluded from the canonical metadata
table (see fer.metadata.build_metadata_table). Each entry's reason is
independently confirmed against data/CREMA-D/README.md's own documented
known issues, not just our own duration probe.
"""

EXCLUDED_CLIPS: dict[str, str] = {
    "1064_IEO_DIS_MD": (
        "Video is ~61.8s and is actually a compilation of all emotional "
        "displays for this sentence, per README.md. The filename-encoded "
        "label 'disgust' (DIS) does not describe this clip's content — "
        "training on it would inject a mislabeled example."
    ),
    "1064_TIE_SAD_XX": (
        "Video duration is 0.567s — degenerate/near-empty video, per "
        "README.md ('file has no duration'). Audio track is intact."
    ),
    "1076_MTI_NEU_XX": (
        "Video duration is 0.066s — degenerate/near-empty video, per "
        "README.md ('very short, no audio' on the raw flv; audio track is "
        "fine in AudioWAV)."
    ),
}
