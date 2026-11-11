"""Shared, importable (and testable) core for the Phase 4 face-model
occlusion evaluation. Centralizes logic that was previously duplicated
across scripts/11 and scripts/12, after a reproducibility question made
it clear this needed to be a single, test-covered source of truth rather
than copy-pasted per script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .face_model import IMAGENET_MEAN, IMAGENET_STD, FrameDataset, build_eval_rows
from .face_pipeline import apply_occlusion


def image_to_tensor(arr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)
    return (t - IMAGENET_MEAN) / IMAGENET_STD


def select_eval_frame_positions(n_frames: int, k: int) -> np.ndarray:
    """Evenly spaced positions into a clip's frame_indices/landmarks
    arrays. Depends only on n_frames — same positions for every occlusion
    type/severity/model."""
    if n_frames <= k:
        return np.arange(n_frames)
    return np.unique(np.round(np.linspace(0, n_frames - 1, k)).astype(int))


def load_resnet18_checkpoint(checkpoint_path: Path, n_classes: int, device: torch.device) -> torch.nn.Module:
    model = timm.create_model("resnet18", pretrained=False, num_classes=n_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def evaluate_occlusion_probs(
    tasks: list[tuple],
    clip_true: dict[str, str],
    det_lookup: "pd.Series",
    model: torch.nn.Module,
    class_order: list[str],
    conditions: list[tuple[str, float]],
    device: torch.device,
    crops_dir: Path,
    frames_per_clip_eval: int,
    occlusion_seed: int,
) -> pd.DataFrame:
    """Returns one row per (clip_id, occlusion_type, severity): full
    softmax probability vector (averaged over surviving frames), the
    predicted class, whether the clip abstained (zero surviving frames),
    and how many frames survived. Deterministic given the same model
    weights, detection cache, and inputs — see tests/test_occlusion_reproducibility.py."""
    records = []
    for occ_type, severity in conditions:
        for clip_id, frame_indices, landmarks in tasks:
            positions = select_eval_frame_positions(len(frame_indices), frames_per_clip_eval)
            batch_imgs = []
            for pos in positions:
                fi, lm = frame_indices[pos], landmarks[pos]
                key = (clip_id, int(fi), occ_type, severity)
                if key not in det_lookup.index or not det_lookup.loc[key]:
                    continue
                crop_path = crops_dir / clip_id / f"frame_{int(fi):03d}.jpg"
                crop = np.array(Image.open(crop_path))
                lm_valid = None if np.isnan(lm).any() else lm
                occluded = apply_occlusion(occ_type, crop, lm_valid, severity, clip_id, occlusion_seed)
                batch_imgs.append(image_to_tensor(occluded))

            if batch_imgs:
                with torch.no_grad():
                    x = torch.stack(batch_imgs).to(device)
                    probs = torch.softmax(model(x), dim=1).cpu().numpy().mean(axis=0)
                abstained, n_surv = False, len(batch_imgs)
            else:
                probs = np.ones(len(class_order)) / len(class_order)
                abstained, n_surv = True, 0

            pred_idx = int(np.argmax(probs))
            row = {
                "clip_id": clip_id, "occlusion_type": occ_type, "severity": severity,
                "y_true": clip_true[clip_id], "y_pred": class_order[pred_idx],
                "abstained": abstained, "n_surviving": n_surv,
            }
            for c, p in zip(class_order, probs):
                row[f"prob_{c}"] = float(p)
            records.append(row)
    return pd.DataFrame(records)


def evaluate_clean_probs(
    val_metadata: pd.DataFrame,
    model: torch.nn.Module,
    class_order: list[str],
    device: torch.device,
    crops_dir: Path,
) -> pd.DataFrame:
    """One row per val clip: full softmax probability vector averaged over
    ALL cached frames (clean condition — no subsampling, no occlusion)."""
    rows = build_eval_rows(val_metadata, crops_dir, class_order)
    ds = FrameDataset(rows, class_order)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    clip_prob_sums, clip_counts, clip_true = {}, {}, {}
    for images, labels, clip_ids in loader:
        images = images.to(device)
        with torch.no_grad():
            probs = torch.softmax(model(images), dim=1).cpu().numpy()
        for p, lbl, cid in zip(probs, labels.numpy(), clip_ids):
            clip_prob_sums.setdefault(cid, np.zeros(len(class_order)))
            clip_prob_sums[cid] += p
            clip_counts[cid] = clip_counts.get(cid, 0) + 1
            clip_true[cid] = int(lbl)

    records = []
    for cid in sorted(clip_prob_sums):
        mean_probs = clip_prob_sums[cid] / clip_counts[cid]
        pred_idx = int(np.argmax(mean_probs))
        row = {"clip_id": cid, "y_true": class_order[clip_true[cid]], "y_pred": class_order[pred_idx], "abstained": False, "n_surviving": clip_counts[cid]}
        for c, p in zip(class_order, mean_probs):
            row[f"prob_{c}"] = float(p)
        records.append(row)
    return pd.DataFrame(records)
