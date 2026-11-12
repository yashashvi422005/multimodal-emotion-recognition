"""Occlusion-augmented variant of Phase 4 training: each selected
training frame is independently left clean or occluded (one of the
allowed types, random severity), using a training-time RNG namespace
kept distinct from the evaluation occlusions (clip_seed's string-keyed
hashing means `f"{clip_id}_aug_{frame_idx}"` with the training seed never
collides with eval's `f"{clip_id}_{occ_type}"` with SEED=0). Everything
else — frame subsampling, clean val monitoring/early stopping, clean
final-checkpoint evaluation — is unchanged from fer.face_model, so the
clean-trained vs. augmented comparison isolates the effect of training
on occluded frames rather than any other training-loop difference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from .face_model import IMAGENET_MEAN, IMAGENET_STD, build_eval_rows, build_train_rows, compute_class_weights, list_cached_frames, predict_clip_level, select_train_frames
from .face_pipeline import SEVERITIES, apply_occlusion, clip_seed

AUGMENTATION_PROB = 0.5  # fraction of selected training frames that get an occlusion applied


def _load_landmarks_for_frame(landmarks_dir: Path, clip_id: str, frame_idx: int, cache: dict) -> np.ndarray | None:
    if clip_id not in cache:
        npz_path = landmarks_dir / f"{clip_id}.npz"
        if npz_path.exists():
            d = np.load(npz_path)
            cache[clip_id] = dict(zip(d["frame_indices"].tolist(), d["landmarks"]))
        else:
            cache[clip_id] = {}
    lm = cache[clip_id].get(frame_idx)
    if lm is None or np.isnan(lm).any():
        return None
    return lm


def build_augmented_train_rows(
    metadata_df: pd.DataFrame, crops_dir: Path, class_order: list[str], seed: int, frames_per_clip: int, allowed_types: list[str],
) -> list[tuple[Path, int, str, tuple | None]]:
    """Rows: (frame_path, label_idx, clip_id, occlusion_spec). occlusion_spec
    is None (clean) or (occ_type, severity), decided once per (clip, frame,
    seed+epoch) — deterministic given those, genuinely random across
    epochs/seeds like the frame subsample itself."""
    label_to_idx = {c: i for i, c in enumerate(class_order)}
    rows = []
    for _, r in metadata_df.iterrows():
        clip_id = r["clip_id"]
        frames = list_cached_frames(crops_dir, clip_id)
        if not frames:
            continue
        chosen = select_train_frames(frames, clip_id, seed, frames_per_clip)
        for p in chosen:
            frame_idx = int(p.stem.split("_")[1])
            rng = np.random.default_rng(clip_seed(f"{clip_id}_aug_{frame_idx}", seed))
            if rng.random() < AUGMENTATION_PROB:
                occ_type = allowed_types[int(rng.integers(len(allowed_types)))]
                severity = SEVERITIES[int(rng.integers(len(SEVERITIES)))]
                rows.append((p, label_to_idx[r["emotion"]], clip_id, (occ_type, severity)))
            else:
                rows.append((p, label_to_idx[r["emotion"]], clip_id, None))
    return rows


class AugmentedFrameDataset(Dataset):
    def __init__(self, rows: list[tuple[Path, int, str, tuple | None]], landmarks_dir: Path, seed: int):
        self.rows = rows
        self.landmarks_dir = landmarks_dir
        self.seed = seed
        self._lm_cache: dict = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        path, label_idx, clip_id, occ_spec = self.rows[i]
        img = np.array(Image.open(path).convert("RGB"))
        if occ_spec is not None:
            occ_type, severity = occ_spec
            frame_idx = int(Path(path).stem.split("_")[1])
            lm = _load_landmarks_for_frame(self.landmarks_dir, clip_id, frame_idx, self._lm_cache)
            img = apply_occlusion(occ_type, img, lm, severity, clip_id, self.seed)
        t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        return t, label_idx, clip_id


def time_one_augmented_epoch(train_metadata, crops_dir, landmarks_dir, class_order, seed, device, allowed_types, frames_per_clip=4, batch_size=64, lr=1e-4) -> float:
    """Builds one epoch's augmented rows and runs one forward+backward pass
    over them, returning wall-clock seconds. Used for the pre-training time
    estimate — no model state is kept."""
    import time

    class_weights = compute_class_weights(train_metadata, class_order).to(device)
    model = timm.create_model("resnet18", pretrained=True, num_classes=len(class_order)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    t0 = time.perf_counter()
    rows = build_augmented_train_rows(train_metadata, crops_dir, class_order, seed, frames_per_clip, allowed_types)
    ds = AugmentedFrameDataset(rows, landmarks_dir, seed)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen, num_workers=0)

    model.train()
    for images, labels, _clip_ids in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    return time.perf_counter() - t0


def train_one_seed_augmented(
    train_metadata: pd.DataFrame,
    val_metadata: pd.DataFrame,
    crops_dir: Path,
    landmarks_dir: Path,
    class_order: list[str],
    seed: int,
    device: torch.device,
    allowed_types: list[str],
    frames_per_clip: int = 4,
    monitor_frames_per_clip: int = 4,
    max_epochs: int = 6,
    patience: int = 2,
    batch_size: int = 64,
    lr: float = 1e-4,
    log_fn=print,
):
    """Mirrors fer.face_model.train_one_seed exactly, except the TRAIN
    loader draws from build_augmented_train_rows/AugmentedFrameDataset.
    Monitoring (early stopping) and the final reported metrics are on
    CLEAN val, same as the clean-trained model, so the comparison isolates
    the effect of training-time occlusion exposure."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    class_weights = compute_class_weights(train_metadata, class_order).to(device)
    model = timm.create_model("resnet18", pretrained=True, num_classes=len(class_order)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    val_actor_of_clip = dict(zip(val_metadata["clip_id"], val_metadata["actor_id"]))

    best_macro_f1, best_state, best_epoch, epochs_without_improve = -1.0, None, -1, 0
    history = []

    for epoch in range(max_epochs):
        train_rows = build_augmented_train_rows(train_metadata, crops_dir, class_order, seed * 1000 + epoch, frames_per_clip, allowed_types)
        train_ds = AugmentedFrameDataset(train_rows, landmarks_dir, seed * 1000 + epoch)
        gen = torch.Generator().manual_seed(seed * 1000 + epoch)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen, num_workers=0)

        model.train()
        total_loss = 0.0
        for images, labels, _clip_ids in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        train_loss = total_loss / len(train_ds)

        monitor_rows = build_train_rows(val_metadata, crops_dir, class_order, seed * 1000 + epoch, monitor_frames_per_clip)  # monitoring stays clean
        _, y_true, y_pred = predict_clip_level(model, monitor_rows, class_order, device)
        val_macro_f1 = f1_score(y_true, y_pred, labels=class_order, average="macro")
        val_acc = (y_true == y_pred).mean()
        history.append({"epoch": epoch, "train_loss": train_loss, "monitor_val_macro_f1": val_macro_f1, "monitor_val_acc": val_acc})
        log_fn(f"  seed={seed} epoch={epoch} train_loss={train_loss:.4f} monitor_val_macro_f1={val_macro_f1:.4f} monitor_val_acc={val_acc:.4f}")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1, best_epoch = val_macro_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                log_fn(f"  seed={seed}: early stopping at epoch {epoch} (best epoch {best_epoch}, best monitor_val_macro_f1={best_macro_f1:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    val_rows = build_eval_rows(val_metadata, crops_dir, class_order)
    clip_ids, y_true, y_pred = predict_clip_level(model, val_rows, class_order, device)
    actor_ids = np.array([val_actor_of_clip[c] for c in clip_ids])

    return {
        "model": model,
        "best_epoch": best_epoch,
        "history": history,
        "clip_ids": clip_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "actor_ids": actor_ids,
        "val_macro_f1": f1_score(y_true, y_pred, labels=class_order, average="macro"),
        "val_accuracy": (y_true == y_pred).mean(),
        "per_class_f1": dict(zip(class_order, f1_score(y_true, y_pred, labels=class_order, average=None))),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_order),
    }
