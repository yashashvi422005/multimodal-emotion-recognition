"""Parsing of CREMA-D filenames into structured metadata.

Filename convention (from data/CREMA-D/README.md, "Filename labeling
conventions"): ActorID_Sentence_Emotion_Intensity.ext, e.g. 1001_DFA_ANG_XX.wav
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FILENAME_RE = re.compile(r"^(?P<actor_id>\d{4})_(?P<sentence>[A-Z]{3})_(?P<emotion>[A-Z]{3})_(?P<intensity>[A-Z]{2})$")

SENTENCE_MAP = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

EMOTION_MAP = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

INTENSITY_MAP = {
    "LO": "low",
    "MD": "medium",
    "HI": "high",
    "XX": "unspecified",
}


class FilenameParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedFilename:
    clip_id: str
    actor_id: str
    sentence_code: str
    sentence_text: str
    emotion_code: str
    emotion: str
    intensity_code: str
    intensity: str


def parse_filename(stem: str) -> ParsedFilename:
    """Parse a CREMA-D filename stem (no extension) into its components.

    Raises FilenameParseError if the stem doesn't match the documented
    convention, or uses a sentence/emotion/intensity code not in the README.
    We do not guess at malformed names — unrecognized files must be reported,
    not silently coerced.
    """
    m = FILENAME_RE.match(stem)
    if not m:
        raise FilenameParseError(f"does not match ActorID_Sentence_Emotion_Intensity pattern: {stem!r}")

    actor_id = m.group("actor_id")
    sentence_code = m.group("sentence")
    emotion_code = m.group("emotion")
    intensity_code = m.group("intensity")

    if sentence_code not in SENTENCE_MAP:
        raise FilenameParseError(f"unknown sentence code {sentence_code!r} in {stem!r}")
    if emotion_code not in EMOTION_MAP:
        raise FilenameParseError(f"unknown emotion code {emotion_code!r} in {stem!r}")
    if intensity_code not in INTENSITY_MAP:
        raise FilenameParseError(f"unknown intensity code {intensity_code!r} in {stem!r}")

    return ParsedFilename(
        clip_id=stem,
        actor_id=actor_id,
        sentence_code=sentence_code,
        sentence_text=SENTENCE_MAP[sentence_code],
        emotion_code=emotion_code,
        emotion=EMOTION_MAP[emotion_code],
        intensity_code=intensity_code,
        intensity=INTENSITY_MAP[intensity_code],
    )
