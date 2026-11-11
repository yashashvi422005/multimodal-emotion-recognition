"""Phase 4, first version: frame-level ResNet-18 fine-tuning on clean face
crops, with clip-level prediction by averaging frame-level softmax
probabilities. Trains on TRAIN actors' clips, evaluates on VAL actors'
clips only — test actors are never touched here.
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

from .face_pipeline import clip_seed

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_image_tensor(path: Path) -> torch.Tensor:
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1)
    return (t - IMAGENET_MEAN) / IMAGENET_STD


def list_cached_frames(crops_dir: Path, clip_id: str) -> list[Path]:
    d = crops_dir / clip_id
    if not d.exists():
        return []
    return sorted(d.glob("frame_*.jpg"))


def select_train_frames(frames: list[Path], clip_id: str, seed: int, k: int) -> list[Path]:
    """Deterministic given clip_id+seed, but genuinely different PER SEED —
    this is a real source of training-run variation (on top of weight
    init), distinct from the audio LR baseline's inert seed."""
    if len(frames) <= k:
        return frames
    rng = np.random.default_rng(clip_seed(f"{clip_id}_frametrain", seed))
    idx = rng.choice(len(frames), size=k, replace=False)
    return [frames[i] for i in sorted(idx)]


class FrameDataset(Dataset):
    def __init__(self, rows: list[tuple[Path, int, str]], class_order: list[str]):
        self.rows = rows  # (frame_path, label_idx, clip_id)
        self.class_order = class_order

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        path, label_idx, clip_id = self.rows[i]
        return load_image_tensor(path), label_idx, clip_id


def build_train_rows(metadata_df: pd.DataFrame, crops_dir: Path, class_order: list[str], seed: int, frames_per_clip: int) -> list[tuple[Path, int, str]]:
    label_to_idx = {c: i for i, c in enumerate(class_order)}
    rows = []
    for _, r in metadata_df.iterrows():
        frames = list_cached_frames(crops_dir, r["clip_id"])
        if not frames:
            continue
        chosen = select_train_frames(frames, r["clip_id"], seed, frames_per_clip)
        rows.extend((p, label_to_idx[r["emotion"]], r["clip_id"]) for p in chosen)
    return rows


def build_eval_rows(metadata_df: pd.DataFrame, crops_dir: Path, class_order: list[str]) -> list[tuple[Path, int, str]]:
    """All cached frames — eval is inference-only, so the full set is
    affordable and gives a more faithful clip-level aggregate."""
    label_to_idx = {c: i for i, c in enumerate(class_order)}
    rows = []
    for _, r in metadata_df.iterrows():
        frames = list_cached_frames(crops_dir, r["clip_id"])
        rows.extend((p, label_to_idx[r["emotion"]], r["clip_id"]) for p in frames)
    return rows


def compute_class_weights(metadata_df: pd.DataFrame, class_order: list[str]) -> torch.Tensor:
    counts = metadata_df["emotion"].value_counts()
    n_total, n_classes = len(metadata_df), len(class_order)
    weights = [n_total / (n_classes * counts.get(c, 1)) for c in class_order]
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict_clip_level(model: nn.Module, rows: list[tuple[Path, int, str]], class_order: list[str], device: torch.device, batch_size: int = 128):
    """Returns (clip_ids, y_true, y_pred) — one row per unique clip, by
    averaging per-frame softmax probabilities then taking argmax."""
    model.eval()
    ds = FrameDataset(rows, class_order)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    clip_prob_sums: dict[str, np.ndarray] = {}
    clip_true: dict[str, int] = {}
    for images, labels, clip_ids in loader:
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1).cpu().numpy()
        for p, lbl, cid in zip(probs, labels.numpy(), clip_ids):
            clip_prob_sums.setdefault(cid, np.zeros(len(class_order)))
            clip_prob_sums[cid] += p
            clip_true[cid] = int(lbl)

    clip_ids = sorted(clip_prob_sums.keys())
    y_true = np.array([class_order[clip_true[c]] for c in clip_ids])
    y_pred = np.array([class_order[int(np.argmax(clip_prob_sums[c]))] for c in clip_ids])
    return clip_ids, y_true, y_pred


def train_one_seed(
    train_metadata: pd.DataFrame,
    val_metadata: pd.DataFrame,
    crops_dir: Path,
    class_order: list[str],
    seed: int,
    device: torch.device,
    frames_per_clip: int = 4,
    monitor_frames_per_clip: int = 4,
    max_epochs: int = 6,
    patience: int = 2,
    batch_size: int = 64,
    lr: float = 1e-4,
    log_fn=print,
):
    """Benchmarked at ~110 img/s on this Mac's MPS with disk-loaded JPEGs
    (num_workers>0 made things SLOWER here — multiprocessing spawn
    overhead dominates at this dataset size — so num_workers=0
    throughout). At that throughput, full-frame val inference on every
    epoch would blow the ~30min/seed budget, so per-epoch early-stopping
    monitoring uses a cheap subsampled val set (same frames-per-clip
    scheme as training); only the FINAL best checkpoint is evaluated on
    the full cached val-frame set, which is what gets reported."""
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
        train_rows = build_train_rows(train_metadata, crops_dir, class_order, seed=seed * 1000 + epoch, frames_per_clip=frames_per_clip)
        train_ds = FrameDataset(train_rows, class_order)
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

        monitor_rows = build_train_rows(val_metadata, crops_dir, class_order, seed=seed * 1000 + epoch, frames_per_clip=monitor_frames_per_clip)
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
