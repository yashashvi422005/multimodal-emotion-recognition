"""Shared WavLM audio-branch model: `microsoft/wavlm-base-plus` with its
CNN feature encoder frozen, a masked-mean pool over the transformer's
last hidden state, and a small 2-layer classification head. Identical
architecture to notebooks/wavlm_finetune_kaggle.ipynb (the Kaggle
training run) -- built from a downloaded config only (no pretrained
weights fetched), so every checkpoint load is exact and offline-capable
once the config is cached.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

TARGET_SR = 16000
MAX_SECONDS = 6.0


class WavLMClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        from transformers import WavLMConfig, WavLMModel
        config = WavLMConfig.from_pretrained("microsoft/wavlm-base-plus")
        self.backbone = WavLMModel(config)  # random init -- overwritten by a fine-tuned checkpoint
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden // 2, num_classes))

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_values=input_values, attention_mask=attention_mask)
        feat_mask = self.backbone._get_feature_vector_attention_mask(out.last_hidden_state.shape[1], attention_mask)
        feat_mask = feat_mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
        pooled = (out.last_hidden_state * feat_mask).sum(dim=1) / feat_mask.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


def get_feature_extractor():
    from transformers import Wav2Vec2FeatureExtractor
    return Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")


def truncate_to_max_seconds(wav: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    max_len = int(MAX_SECONDS * sr)
    return wav[:max_len] if len(wav) > max_len else wav
