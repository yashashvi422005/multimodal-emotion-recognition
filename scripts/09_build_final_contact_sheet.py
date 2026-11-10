#!/usr/bin/env python
"""Build the final Phase 2 Part B contact sheet: 4 clips from 4 different
actors x (original + 4 occlusion types x 3 severities), using the crops
already cached by scripts/08_phase2_full_run.py.

Usage:
    python scripts/09_build_final_contact_sheet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fer.face_pipeline import SEVERITIES

REPO_ROOT = Path(__file__).resolve().parents[1]
OCCLUSION_TYPES = ["mask", "sunglasses", "patch", "hand_over_mouth"]


def main() -> None:
    chosen = pd.read_csv("/tmp/contact_sheet_clips.csv", dtype={"actor_id": str})

    cols = ["original"] + [f"{t}\n{sev}" for t in OCCLUSION_TYPES for sev in SEVERITIES]
    cell = 130
    label_h = 40
    sheet = Image.new("RGB", (cell * len(cols), label_h + cell * len(chosen)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for col, label in enumerate(cols):
        for li, line in enumerate(label.split("\n")):
            draw.text((col * cell + 4, 4 + li * 11), line, fill=(0, 0, 0), font=font)

    for row_i, r in chosen.iterrows():
        clip_id = r["clip_id"]
        y0 = label_h + row_i * cell

        lm_npz = np.load(REPO_ROOT / "cache" / "landmarks" / f"{clip_id}.npz")
        frame_indices = lm_npz["frame_indices"]
        rep_idx = int(frame_indices[len(frame_indices) // 2])
        orig_path = REPO_ROOT / "cache" / "face_crops" / clip_id / f"frame_{rep_idx:03d}.jpg"
        orig_img = Image.open(orig_path).resize((cell - 6, cell - 6))
        sheet.paste(orig_img, (3, y0 + 3))
        draw.text((4, y0 + 4), f"{clip_id}\n({r['emotion']})", fill=(255, 0, 0), font=font)

        col = 1
        for occ_type in OCCLUSION_TYPES:
            for sev in SEVERITIES:
                occ_path = REPO_ROOT / "cache" / "occluded_crops" / clip_id / f"{occ_type}_{sev}.jpg"
                occ_img = Image.open(occ_path).resize((cell - 6, cell - 6))
                sheet.paste(occ_img, (col * cell + 3, y0 + 3))
                col += 1

    out_path = REPO_ROOT / "outputs" / "phase2_pilot" / "final_contact_sheet_part_b.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"Saved contact sheet to: {out_path}")
    print(f"Clips used: {chosen[['clip_id','actor_id','emotion']].to_dict('records')}")


if __name__ == "__main__":
    main()
