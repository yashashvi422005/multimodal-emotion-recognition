"""Non-negotiable rule #1: no actor may appear in more than one of
train/val/test. This must pass before any model code is allowed to use
the split, and runs every time the split is loaded (see fer.splits.load_split),
not just here."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fer.demographics import load_demographics
from fer.splits import LeakageError, assert_no_leakage, content_hash, load_split

SPLIT_PATH = REPO_ROOT / "splits" / "split_v1.json"
DEMOGRAPHICS_CSV = REPO_ROOT / "data" / "CREMA-D" / "VideoDemographics.csv"


@pytest.fixture
def split_file():
    if not SPLIT_PATH.exists():
        pytest.skip(f"{SPLIT_PATH} does not exist yet — run scripts/02_make_split.py first")
    return load_split(SPLIT_PATH)


def test_load_split_passes_leakage_check(split_file):
    assert_no_leakage(split_file.assignment)


def test_no_actor_in_two_splits_directly():
    train = {"1001", "1002"}
    val = {"1003"}
    test = {"1004"}
    assert_no_leakage({"train": list(train), "val": list(val), "test": list(test)})

    with pytest.raises(LeakageError):
        assert_no_leakage({"train": ["1001", "1002"], "val": ["1002"], "test": ["1004"]})


def test_split_covers_every_actor_exactly_once(split_file):
    demographics_df = load_demographics(DEMOGRAPHICS_CSV)
    all_actors = set(demographics_df["ActorID"])
    assigned = sum(split_file.assignment.values(), [])
    assert len(assigned) == len(set(assigned)), "an actor id appears twice within the split file"
    assert set(assigned) == all_actors, "split does not cover exactly the demographics actor set"


def test_split_hash_matches_content(split_file):
    assert content_hash(split_file.assignment) == split_file.content_hash


def test_tampered_split_file_is_rejected(tmp_path, split_file):
    import json

    tampered = split_file.to_json_dict()
    # Move one actor from val to train without updating the stored hash —
    # this must be caught, not silently trusted.
    if tampered["assignment"]["val"]:
        moved = tampered["assignment"]["val"].pop()
        tampered["assignment"]["train"].append(moved)
    p = tmp_path / "tampered_split.json"
    p.write_text(json.dumps(tampered))

    with pytest.raises(LeakageError):
        load_split(p)
