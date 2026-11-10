"""Loading and validating VideoDemographics.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_demographics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"ActorID": str})
    df["ActorID"] = df["ActorID"].str.zfill(4)

    dupes = df["ActorID"][df["ActorID"].duplicated()].tolist()
    if dupes:
        raise ValueError(f"duplicate ActorID rows in {csv_path}: {dupes}")

    expected_cols = {"ActorID", "Age", "Sex", "Race", "Ethnicity"}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{csv_path} is missing expected columns: {missing_cols}")

    return df.set_index("ActorID", drop=False)
