"""Actor-level train/val/test split: computation, versioned persistence
with a content hash, and a leakage assertion that must run every time a
split is loaded (non-negotiable rule #1 — no actor in more than one split).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")


class LeakageError(AssertionError):
    pass


def _largest_remainder_targets(n_total: int, fracs: dict[str, float]) -> dict[str, int]:
    """Integer per-split counts summing exactly to n_total, as close to
    `fracs` as integer rounding allows (largest-remainder / Hare-Niemeyer
    method). Deterministic tie-break: train > val > test."""
    tie_order = {name: i for i, name in enumerate(SPLIT_NAMES)}
    raw = {name: n_total * fracs[name] for name in SPLIT_NAMES}
    floor = {name: int(np.floor(v)) for name, v in raw.items()}
    remainder = n_total - sum(floor.values())
    fracs_remainder = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(raw[name] - floor[name]), tie_order[name]),
    )
    for name in fracs_remainder[:remainder]:
        floor[name] += 1
    assert sum(floor.values()) == n_total
    return floor


def compute_actor_split(
    demographics_df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    stratify_on: list[str],
) -> dict[str, list[str]]:
    """Per-stratum proportional allocation of ActorIDs to train/val/test.

    Strata are formed by the cartesian product of `stratify_on` columns
    (e.g. Sex x Race). Within EACH stratum independently, actor counts are
    rounded to train/val/test targets via the largest-remainder method at
    the requested fractions. Doing the rounding per-stratum (rather than
    filling a single global train/val/test quota greedily) is what makes
    this "stratified": it guarantees each Sex/Race cell contributes to
    train/val/test in close to its own 70/15/15 proportion, so the overall
    Sex and Race marginals are preserved by construction. A global-quota
    greedy allocator was tried first and rejected: it let the single
    largest stratum get dumped almost entirely into train before any
    other split's quota mattered, skewing Sex balance badly (see split
    review history).

    A side effect: because each stratum rounds independently, the overall
    train/val/test totals land close to but not exactly at the requested
    fractions (e.g. 64/15/12 instead of an exact 64/14/13 for N=91) —
    stratification fidelity is prioritized over hitting the exact global
    count, per instructions to split "roughly 70/15/15 ... stratified as
    well as possible".

    Within a stratum, which specific actors land in which split is decided
    by a seeded shuffle — this `seed` is the split-construction seed, not
    one of the 3 training seeds used for model runs; the split is fixed
    once and reused everywhere.
    """
    fracs = {"train": train_frac, "val": val_frac, "test": test_frac}
    if abs(sum(fracs.values()) - 1.0) > 1e-9:
        raise ValueError(f"train/val/test fractions must sum to 1.0, got {fracs}")

    strata: dict[tuple, list[str]] = {}
    for actor_id, row in demographics_df.iterrows():
        key = tuple(row[col] for col in stratify_on)
        strata.setdefault(key, []).append(actor_id)

    rng = np.random.default_rng(seed)
    ordered_keys = sorted(strata.keys(), key=lambda k: (-len(strata[k]), k))

    assignment: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}

    for key in ordered_keys:
        actor_ids = list(strata[key])
        rng.shuffle(actor_ids)
        stratum_targets = _largest_remainder_targets(len(actor_ids), fracs)
        i = 0
        for name in SPLIT_NAMES:
            n = stratum_targets[name]
            assignment[name].extend(actor_ids[i : i + n])
            i += n
        assert i == len(actor_ids)

    for name in SPLIT_NAMES:
        assignment[name] = sorted(assignment[name])

    assert_no_leakage(assignment)
    all_assigned = sorted(sum(assignment.values(), []))
    all_actors = sorted(demographics_df["ActorID"].tolist())
    if all_assigned != all_actors:
        raise RuntimeError("split does not cover exactly the full actor set — this is a bug in compute_actor_split")

    return assignment


def assert_no_leakage(assignment: dict[str, list[str]]) -> None:
    """Must be called every time a split is loaded, not just in tests."""
    seen: dict[str, str] = {}
    for split_name, actor_ids in assignment.items():
        for actor_id in actor_ids:
            if actor_id in seen:
                raise LeakageError(
                    f"actor {actor_id} appears in both {seen[actor_id]!r} and {split_name!r} splits"
                )
            seen[actor_id] = split_name


def canonical_bytes(assignment: dict[str, list[str]]) -> bytes:
    canonical = {name: sorted(assignment[name]) for name in SPLIT_NAMES}
    return json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")


def content_hash(assignment: dict[str, list[str]]) -> str:
    return hashlib.sha256(canonical_bytes(assignment)).hexdigest()


@dataclass
class SplitFile:
    version: str
    status: str  # "proposed" | "frozen"
    created_at: str
    frozen_at: str | None
    fracs: dict[str, float]
    split_seed: int
    stratify_on: list[str]
    assignment: dict[str, list[str]]
    content_hash: str

    def to_json_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


def save_split(path: Path, assignment: dict[str, list[str]], version: str, fracs: dict, split_seed: int, stratify_on: list[str]) -> SplitFile:
    assert_no_leakage(assignment)
    sf = SplitFile(
        version=version,
        status="proposed",
        created_at=datetime.now(timezone.utc).isoformat(),
        frozen_at=None,
        fracs=fracs,
        split_seed=split_seed,
        stratify_on=stratify_on,
        assignment={name: sorted(assignment[name]) for name in SPLIT_NAMES},
        content_hash=content_hash(assignment),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sf.to_json_dict(), indent=2, sort_keys=True) + "\n")
    return sf


def load_split(path: Path) -> SplitFile:
    data = json.loads(path.read_text())
    sf = SplitFile(**data)
    assert_no_leakage(sf.assignment)
    recomputed = content_hash(sf.assignment)
    if recomputed != sf.content_hash:
        raise LeakageError(
            f"split file {path} has been modified: stored hash {sf.content_hash} != recomputed {recomputed}"
        )
    return sf


def freeze_split(path: Path) -> SplitFile:
    """Marks an existing proposed split file as frozen (reviewed and
    approved). Does NOT recompute or alter the actor assignment."""
    sf = load_split(path)
    if sf.status == "frozen":
        return sf
    sf.status = "frozen"
    sf.frozen_at = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(sf.to_json_dict(), indent=2, sort_keys=True) + "\n")
    return sf


AGE_BAND_EDGES = [20, 30, 45, 60, 75]  # [20,30) [30,45) [45,60) [60,75) — round-number decade-ish bins
AGE_BAND_LABELS = ["20-29", "30-44", "45-59", "60-74"]


def age_band(age: int) -> str:
    for lo, hi, label in zip(AGE_BAND_EDGES[:-1], AGE_BAND_EDGES[1:], AGE_BAND_LABELS):
        if lo <= age < hi:
            return label
    raise ValueError(f"age {age} outside expected CREMA-D range {AGE_BAND_EDGES[0]}-{AGE_BAND_EDGES[-1] - 1}")


def split_balance_table(assignment: dict[str, list[str]], demographics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name in SPLIT_NAMES:
        actor_ids = assignment[split_name]
        sub = demographics_df.loc[actor_ids]
        row = {
            "split": split_name,
            "n_actors": len(actor_ids),
            "pct_actors": 100 * len(actor_ids) / sum(len(a) for a in assignment.values()),
            "age_mean": sub["Age"].mean(),
            "age_std": sub["Age"].std(),
            "age_min": sub["Age"].min(),
            "age_max": sub["Age"].max(),
            "n_male": (sub["Sex"] == "Male").sum(),
            "n_female": (sub["Sex"] == "Female").sum(),
            "n_hispanic": (sub["Ethnicity"] == "Hispanic").sum(),
            "n_not_hispanic": (sub["Ethnicity"] == "Not Hispanic").sum(),
        }
        for race in sorted(demographics_df["Race"].unique()):
            row[f"n_race_{race.replace(' ', '_')}"] = (sub["Race"] == race).sum()
        for label in AGE_BAND_LABELS:
            row[f"n_age_{label}"] = sub["Age"].apply(age_band).eq(label).sum()
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")


def stratum_breakdown(
    assignment: dict[str, list[str]], demographics_df: pd.DataFrame, stratify_on: list[str]
) -> pd.DataFrame:
    """Per-stratum actor counts in each split — shows exactly how the
    independent per-stratum rounding produced the final totals, and
    flags strata with zero representation in val or test."""
    actor_to_split = {a: s for s, actors in assignment.items() for a in actors}
    df = demographics_df.copy()
    df["split"] = df["ActorID"].map(actor_to_split)
    grouped = df.groupby(stratify_on + ["split"]).size().unstack("split", fill_value=0)
    for name in SPLIT_NAMES:
        if name not in grouped.columns:
            grouped[name] = 0
    grouped = grouped[list(SPLIT_NAMES)]
    grouped["total"] = grouped.sum(axis=1)
    grouped["zero_in_val_or_test"] = (grouped["val"] == 0) | (grouped["test"] == 0)
    return grouped
