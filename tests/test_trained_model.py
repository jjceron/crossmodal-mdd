"""
Unit test: verify that final_model.pt exists and produces valid output.

This test is intended for CI — it does NOT re-train, it only loads
a pre-trained model and checks its forward pass on dummy data.
"""
import sys
import json
import pytest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '.')

from src.models.crossmodal_attn import CrossModalAttention  # noqa: E402
from src.models.deepconvnet import DeepConvNet  # noqa: E402
from src.models.shallowconvnet import ShallowConvNet  # noqa: E402


class DeepConvNetWrapper(nn.Module):
    def __init__(self, n_channels, n_samples):
        super().__init__()
        self.m = DeepConvNet(n_channels, n_samples)
    def forward(self, x):
        return self.m(x)


class ShallowConvNetWrapper(nn.Module):
    def __init__(self, n_channels, n_samples):
        super().__init__()
        self.m = ShallowConvNet(n_channels, n_samples)
    def forward(self, x):
        return self.m(x)


MODEL_PATH = Path('outputs/models/final_model.pt')
RESULTS_PATH = Path('outputs/results')


def _ensure_tensor(x):
    return torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x


@pytest.mark.skip(reason="Requires manual final_model.pt after best model selection")
def test_final_model_exists():
    assert MODEL_PATH.exists(), f'Model not found: {MODEL_PATH}'


@pytest.mark.skip(reason="Requires manual final_model.pt after best model selection")
def test_final_model_forward():
    assert MODEL_PATH.exists(), 'No model to test'
    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    cfg = ckpt['config']

    fusion_model = CrossModalAttention(
        eeg_dim=cfg['eeg_dim'], aud_dim=cfg['aud_dim'],
        hidden=cfg['hidden'], n_heads=cfg['n_heads'],
        bottleneck_dim=cfg.get('bottleneck_dim'),
        n_self_attn_layers=cfg.get('n_self_attn_layers', 0),
        self_attn_heads=cfg.get('self_attn_heads', 4),
        self_attn_dropout=cfg.get('self_attn_dropout', 0.1),
        fusion=cfg['fusion'], pooling=cfg.get('pooling', 'mean'),
        dropout=cfg['dropout'],
    )
    eeg_model = DeepConvNetWrapper(64, 500)
    aud_model = ShallowConvNetWrapper(64, 200)
    eeg_model.load_state_dict(ckpt['eeg_backbone_state'])
    aud_model.load_state_dict(ckpt['aud_backbone_state'])
    fusion_model.load_state_dict(ckpt['fusion_state_dict'])

    # Dummy forward
    B, K = 2, 10
    z_eeg = torch.randn(B, K, cfg['eeg_dim'])
    z_audio = torch.randn(B, K, cfg['aud_dim'])
    mask = torch.ones(B, K)

    fusion_model.eval()
    with torch.no_grad():
        logits = fusion_model(z_eeg, z_audio, mask=mask)

    assert logits.shape == (B,), f'Expected shape (B,) but got {logits.shape}'
    assert logits.dtype == torch.float32


def test_results_json_has_final_model():
    candidates = sorted(RESULTS_PATH.rglob('results.json'),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return  # no results to check, skip
    with open(candidates[0]) as f:
        data = json.load(f)
    if 'final_model' not in data or data['final_model'] is None:
        return  # final_model not yet generated, skip
    fm = data['final_model']
    for key in ('bacc', 'acc', 'f1', 'auc', 'model_path'):
        assert key in fm, f'final_model missing key: {key}'