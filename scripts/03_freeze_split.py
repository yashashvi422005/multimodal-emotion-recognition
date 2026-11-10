#!/usr/bin/env python
"""Phase 1, step 3: freeze a previously-proposed split after review and
explicit approval. Only flips status "proposed" -> "frozen" and records
the approval timestamp; does not touch the actor assignment or its hash.

Usage:
    python scripts/03_freeze_split.py splits/split_v1.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fer.splits import freeze_split


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python scripts/03_freeze_split.py <path-to-split.json>", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1])
    sf = freeze_split(path)
    print(f"{path}: status={sf.status} frozen_at={sf.frozen_at} content_hash={sf.content_hash}")


if __name__ == "__main__":
    main()
