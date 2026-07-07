"""CAMP-Net: Cross-modal Attention Multimodal Physiological signals network.

Uses frozen pre-trained backbones (DeepConvNet for EEG, ShallowConvNet for audio)
with trainable cross-modal fusion + classifier head.

Modes:
  eeg_only      — EEG backbone → AttentionPool → Head
  audio_only    — Audio backbone → AttentionPool → Head
  early_fusion  — Pooled concat → Head
  late_fusion   — Independent logits → average
  cross_attn    — Bidirectional cross-MHA → AttentionPool → Head
"""
import torch, torch.nn as nn, math
from torch import Tensor

 
class RMSNorm(nn.Module): 
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.scale


class AttentionPool(nn.Module):
    """Learnable weighted sum over node embeddings."""
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim, 1))

    def forward(self, x):
        scores = x @ self.w / math.sqrt(x.shape[-1])
        weights = torch.softmax(scores, dim=1)
        return (x * weights).sum(dim=1)


class CrossModalFusion(nn.Module):
    """Bidirectional cross-attention: EEG ↔ Audio."""
    def __init__(self, dim, n_heads=2):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.scale = self.d_head ** -0.5

        self.e_q = nn.Linear(dim, dim, bias=False)
        self.e_o = nn.Linear(dim, dim, bias=False)
        self.a_q = nn.Linear(dim, dim, bias=False)
        self.a_o = nn.Linear(dim, dim, bias=False)

        self.norm_e = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)

    def _attend(self, q, k, v, o_proj):
        B = q.shape[0]
        q = q.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, -1, self.dim)
        return o_proj(out), attn.detach()

    def forward(self, eeg, audio):
        e_out, e_attn = self._attend(self.e_q(eeg), audio, audio, self.e_o)
        eeg = self.norm_e(eeg + e_out)

        a_out, a_attn = self._attend(self.a_q(audio), eeg, eeg, self.a_o)
        audio = self.norm_a(audio + a_out)

        return eeg, audio, (e_attn, a_attn)


class CrossModalDL(nn.Module):
    """CAMP-Net with frozen pre-trained backbones + trainable fusion."""
    def __init__(self, eeg_backbone, audio_backbone,
                 hidden=32, n_heads=2, dropout=0.5,
                 freeze_backbones=True, mode='cross_attn'):
        super().__init__()
        self._mode = mode
        self.eeg_bb = eeg_backbone
        self.aud_bb = audio_backbone

        if freeze_backbones:
            for p in self.eeg_bb.parameters():
                p.requires_grad = False
            for p in self.aud_bb.parameters():
                p.requires_grad = False

        self.eeg_proj = nn.Linear(32, hidden)
        self.aud_proj = nn.Linear(24, hidden)
        self.eeg_rms = RMSNorm(hidden)
        self.aud_rms = RMSNorm(hidden)

        self.fusion = CrossModalFusion(hidden, n_heads=n_heads)
        self.pool_eeg = AttentionPool(hidden)
        self.pool_aud = AttentionPool(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _encode_eeg(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.eeg_bb.block1(x)
        x = self.eeg_bb.block2(x)
        x = self.eeg_bb.block3(x)
        x = x.squeeze(2).permute(0, 2, 1)
        return self.eeg_rms(self.eeg_proj(x))

    def _encode_audio(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.aud_bb.temporal_conv(x)
        x = self.aud_bb.spatial_conv(x)
        x = self.aud_bb.bn(x)
        x = torch.square(x)
        x = self.aud_bb.pool(x)
        x = torch.log(torch.clamp(x, min=1e-7))
        x = self.aud_bb.dropout(x)
        x = x.squeeze(2).permute(0, 2, 1)
        return self.aud_rms(self.aud_proj(x))

    def _classify(self, e, a):
        return self.head(torch.cat([e, a], dim=-1))

    def forward(self, eeg_x=None, audio_x=None, mode=None):
        mode = mode or self._mode

        if mode == 'eeg_only':
            e = self.pool_eeg(self._encode_eeg(eeg_x))
            return self._classify(e, e).squeeze(-1), None

        if mode == 'audio_only':
            a = self.pool_aud(self._encode_audio(audio_x))
            return self._classify(a, a).squeeze(-1), None

        e = self._encode_eeg(eeg_x)
        a = self._encode_audio(audio_x)

        if mode == 'early_fusion':
            return self._classify(self.pool_eeg(e), self.pool_aud(a)).squeeze(-1), None

        if mode == 'late_fusion':
            pe, pa = self.pool_eeg(e), self.pool_aud(a)
            logit = (self._classify(pe, pa) + self._classify(pa, pe)) / 2
            return logit.squeeze(-1), None

        e_out, a_out, attn = self.fusion(e, a)
        return self._classify(self.pool_eeg(e_out), self.pool_aud(a_out)).squeeze(-1), attn


