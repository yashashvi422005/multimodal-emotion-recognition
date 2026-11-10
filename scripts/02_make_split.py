#!/usr/bin/env python
"""Phase 1, step 2: propose an actor-level, demographically-stratified
train/val/test split and write it to splits/ as a versioned, hashed file
with status "proposed".

This does NOT freeze the split. Freezing happens only after explicit
review/approval, via scripts/03_freeze_split.py.

Usage:
    python scripts/02_make_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.demographics import load_demographics
from fer.splits import compute_actor_split, save_split, split_balance_table

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_VERSION = "v1"


def main() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    paths = cfg["paths"]
    split_cfg = cfg["split"]

    demographics_df = load_demographics(REPO_ROOT / paths["demographics_csv"])

    assignment = compute_actor_split(
        demographics_df,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
        seed=split_cfg["split_seed"],
        stratify_on=split_cfg["stratify_on"],
    )

    out_path = REPO_ROOT / paths["splits_dir"] / f"split_{SPLIT_VERSION}.json"
    sf = save_split(
        out_path,
        assignment,
        version=SPLIT_VERSION,
        fracs={"train": split_cfg["train_frac"], "val": split_cfg["val_frac"], "test": split_cfg["test_frac"]},
        split_seed=split_cfg["split_seed"],
        stratify_on=split_cfg["stratify_on"],
    )

    print(f"Wrote PROPOSED split to {out_path}")
    print(f"status: {sf.status}")
    print(f"content_hash: {sf.content_hash}")
    print(f"n_actors: train={len(sf.assignment['train'])} val={len(sf.assignment['val'])} test={len(sf.assignment['test'])}")

    print()
    print("=" * 70)
    print("DEMOGRAPHIC BALANCE TABLE (actor-level)")
    print("=" * 70)
    table = split_balance_table(assignment, demographics_df)
    print(table.to_string())

    print()
    print("=" * 70)
    print("PER-SPLIT ACTOR LISTS")
    print("=" * 70)
    for name in ("train", "val", "test"):
        print(f"{name} ({len(assignment[name])}): {assignment[name]}")


if __name__ == "__main__":
    main()
