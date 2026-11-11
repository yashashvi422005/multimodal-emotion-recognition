"""Regression test for a reproducibility question raised after the first
occlusion-eval round: running the same checkpoint against the same
clips/condition must give the same number every time. Root cause was
MPS non-determinism in a GPU-side mean reduction over frame
probabilities (torch's `.mean(dim=0)` on an MPS tensor); fixed in
fer.occlusion_eval by moving the averaging to CPU (`.cpu().numpy().mean(...)`)
so only a deterministic, sequential reduction combines frame
probabilities. This test runs the same batch of clips/conditions twice
and asserts bit-identical output, so a future change that reintroduces
GPU-side reduction (or any other nondeterminism) fails loudly here
instead of silently drifting between runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fer.occlusion_eval import evaluate_occlusion_probs, load_resnet18_checkpoint
from fer.splits import load_split

FACE_CFG = yaml.safe_load((REPO_ROOT / "configs" / "face_model.yaml").read_text())
CLASS_ORDER = FACE_CFG["class_order"]
OCC_CFG = FACE_CFG["occlusion_eval"]
FRAMES_PER_CLIP_EVAL = OCC_CFG["frames_per_clip_eval"]
CHECKPOINT_PATH = REPO_ROOT / "models" / "face_resnet18" / "resnet18_clean_seed0.pt"
DETECTION_CACHE_PATH = REPO_ROOT / OCC_CFG["detection_cache_path"]
CROPS_DIR = REPO_ROOT / "cache" / "face_crops"
LANDMARKS_DIR = REPO_ROOT / "cache" / "landmarks"
N_TEST_CLIPS = 8


def _prereqs_available() -> bool:
    return CHECKPOINT_PATH.exists() and DETECTION_CACHE_PATH.exists()


@pytest.fixture
def small_task_set():
    if not _prereqs_available():
        pytest.skip("checkpoint or detection cache not present — run scripts 10/11 first")
    data_cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    sf = load_split(REPO_ROOT / data_cfg["paths"]["splits_dir"] / "split_v1.json")
    val_actors = set(sf.assignment["val"])
    metadata_df = pd.read_csv(REPO_ROOT / data_cfg["paths"]["outputs_dir"] / "metadata.csv", dtype={"actor_id": str})
    val_metadata = metadata_df[metadata_df["actor_id"].isin(val_actors)]

    tasks, clip_true = [], {}
    for _, r in val_metadata.iterrows():
        clip_id = r["clip_id"]
        npz_path = LANDMARKS_DIR / f"{clip_id}.npz"
        if not npz_path.exists():
            continue
        lm_data = np.load(npz_path)
        tasks.append((clip_id, lm_data["frame_indices"], lm_data["landmarks"]))
        clip_true[clip_id] = r["emotion"]
        if len(tasks) >= N_TEST_CLIPS:
            break
    return tasks, clip_true


def _run_once(tasks, clip_true, device):
    detection_df = pd.read_parquet(DETECTION_CACHE_PATH)
    relevant = {t[0] for t in tasks}
    detection_df = detection_df[detection_df["clip_id"].isin(relevant)]
    det_lookup = detection_df.set_index(["clip_id", "frame_idx", "occlusion_type", "severity"])["detected"]

    model = load_resnet18_checkpoint(CHECKPOINT_PATH, len(CLASS_ORDER), device)
    conditions = [("patch", 0.9), ("mask", 0.6)]  # the two conditions where confidence is lowest / most flip-prone
    return evaluate_occlusion_probs(
        tasks, clip_true, det_lookup, model, CLASS_ORDER, conditions, device,
        CROPS_DIR, FRAMES_PER_CLIP_EVAL, occlusion_seed=0,
    )


def test_same_checkpoint_same_condition_gives_identical_probabilities(small_task_set):
    tasks, clip_true = small_task_set
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    run_a = _run_once(tasks, clip_true, device)
    run_b = _run_once(tasks, clip_true, device)

    run_a = run_a.sort_values(["clip_id", "occlusion_type", "severity"]).reset_index(drop=True)
    run_b = run_b.sort_values(["clip_id", "occlusion_type", "severity"]).reset_index(drop=True)

    prob_cols = [f"prob_{c}" for c in CLASS_ORDER]
    np.testing.assert_array_equal(run_a[prob_cols].to_numpy(), run_b[prob_cols].to_numpy())
    assert (run_a["y_pred"] == run_b["y_pred"]).all()
    assert (run_a["abstained"] == run_b["abstained"]).all()
