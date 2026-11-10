"""Speaker-level uncertainty estimation for a fitted clip-level classifier.

Both methods here treat the ACTOR as the resampling/held-out unit, not the
clip — clips from one actor are acoustically/stylistically correlated, so
resampling clips directly would understate uncertainty (pseudo-replication).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def bootstrap_actor_level_macro_f1(
    actor_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_order: list[str],
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Cluster bootstrap over actors: resample the UNIQUE actor ids with
    replacement; if an actor is drawn k times, that actor's clips are
    included k times (duplicated) when pooling predictions for that
    replicate's macro-F1. This estimates how much macro-F1 would vary if
    we had evaluated on a different sample of speakers from the same
    population, given these fixed per-clip predictions (the model itself
    is not refit per replicate)."""
    unique_actors = np.unique(actor_ids)
    rng = np.random.default_rng(seed)

    actor_to_idx: dict[str, np.ndarray] = {a: np.where(actor_ids == a)[0] for a in unique_actors}

    scores = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        resampled_actors = rng.choice(unique_actors, size=len(unique_actors), replace=True)
        idx = np.concatenate([actor_to_idx[a] for a in resampled_actors])
        scores[b] = f1_score(y_true[idx], y_pred[idx], labels=class_order, average="macro")
    return scores


def actor_level_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    actor_ids: np.ndarray,
    class_order: list[str],
    n_splits: int,
    model_factory,
):
    """GroupKFold over actor_ids (deterministic, no shuffling — sklearn's
    GroupKFold assigns whole groups to folds without randomization). For
    each fold, a StandardScaler is fit on that fold's train actors only
    and a fresh model (from model_factory(), e.g. a LogisticRegression
    with fixed hyperparameters) is trained on that fold's train actors and
    evaluated on its held-out actors.

    Returns (per_fold_metrics: list[dict], oof_y_true, oof_y_pred, oof_actor_ids)
    where the out-of-fold (oof) arrays pool every actor's held-out-fold
    prediction exactly once, covering the full actor set passed in.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    gkf = GroupKFold(n_splits=n_splits)
    per_fold = []
    oof_true, oof_pred, oof_actors = [], [], []

    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=actor_ids)):
        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        model = model_factory()
        model.fit(X_train, y[train_idx])
        y_pred = model.predict(X_val)

        macro_f1 = f1_score(y[val_idx], y_pred, labels=class_order, average="macro")
        acc = (y_pred == y[val_idx]).mean()
        per_fold.append(
            {
                "fold": fold_i,
                "n_train_actors": len(np.unique(actor_ids[train_idx])),
                "n_val_actors": len(np.unique(actor_ids[val_idx])),
                "n_val_clips": len(val_idx),
                "macro_f1": macro_f1,
                "accuracy": acc,
            }
        )
        oof_true.append(y[val_idx])
        oof_pred.append(y_pred)
        oof_actors.append(actor_ids[val_idx])

    return (
        per_fold,
        np.concatenate(oof_true),
        np.concatenate(oof_pred),
        np.concatenate(oof_actors),
    )
