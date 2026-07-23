"""Multi-modal fusion: project, fuse, self-attend, pool, classify.

Usage:
  model = CrossModalAttention(eeg_dim, aud_dim, fusion='cross_attn')
  logits = model(z_eeg, z_audio, mask)  # [B, K, dim] -> [B]
"""
import torch
import torch.nn as nn
import math 


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.scale


class WindowClassifier(nn.Module):
    """Auxiliary window-level classifier for multi-task learning."""
    def __init__(self, hidden):
        super().__init__()
        self.fc = nn.Linear(hidden * 2, hidden)
        self.head = nn.Linear(hidden, 1)
    def forward(self, e, a):
        # e, a: [B, K, hidden]
        B, K = e.shape[0], e.shape[1]
        h = torch.relu(self.fc(torch.cat([e, a], dim=-1)))
        return self.head(h.view(B * K, -1)).squeeze(-1)


class Adapter(nn.Module):
    """Bottleneck adapter with residual: dim → bottleneck → dim."""
    def __init__(self, dim, bottleneck=32):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
    def forward(self, x):
        return x + self.up(torch.relu(self.down(x)))


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
    """Bidirectional cross-attention: EEG ↔ Audio over window sequences."""
    def __init__(self, dim, n_heads=2):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.scale = self.d_head ** -0.5
        self.logit_scale = nn.Parameter(torch.ones(1) * math.log(10.0))

        self.e_q = nn.Linear(dim, dim, bias=False)
        self.e_k = nn.Linear(dim, dim, bias=False)
        self.e_v = nn.Linear(dim, dim, bias=False)
        self.e_o = nn.Linear(dim, dim, bias=False)
        self.a_q = nn.Linear(dim, dim, bias=False)
        self.a_k = nn.Linear(dim, dim, bias=False)
        self.a_v = nn.Linear(dim, dim, bias=False)
        self.a_o = nn.Linear(dim, dim, bias=False)

        self.norm_e = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)

    def _attend(self, q, k, v, o_proj, mask=None):
        B, L = q.shape[0], q.shape[1]
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        attn = q @ k.transpose(-2, -1) * self.scale * self.logit_scale.exp()
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, self.dim)
        return o_proj(out), attn.detach()

    def forward(self, eeg, audio, mask=None):
        e_out, e_attn = self._attend(self.e_q(eeg), self.a_k(audio), self.a_v(audio), self.e_o, mask)
        eeg = self.norm_e(eeg + e_out)
        a_out, a_attn = self._attend(self.a_q(audio), self.e_k(eeg), self.e_v(eeg), self.a_o, mask)
        audio = self.norm_a(audio + a_out)
        return eeg, audio, (e_attn, a_attn)


class SelfAttentionBlock(nn.Module):
    """Standard transformer encoder block (pre-norm)."""
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                          key_padding_mask=mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class CrossModalAttention(nn.Module):
    """Cross-modal fusion with optional self-attention over window sequence."""

    def __init__(self, eeg_dim, aud_dim, hidden=64, n_heads=2,
                 bottleneck_dim=None, n_self_attn_layers=0,
                 self_attn_heads=4, self_attn_dropout=0.1,
                 fusion='cross_attn', pooling='mean', dropout=0.5,
                 adapter_dim=None, window_aux=False,
                 feat_dropout=0.0):
        super().__init__()
        self.fusion = fusion
        self.pooling = pooling
        self.hidden = hidden
        self.bottleneck_dim = bottleneck_dim

        # Feature dropout (applied after projection to hidden, during training)
        self.feat_drop = nn.Dropout(feat_dropout) if feat_dropout > 0 else None

        # Adapter (parameter-efficient fine-tuning on raw backbone features)
        if adapter_dim is not None:
            self.eeg_adapter = Adapter(eeg_dim, adapter_dim)
            self.aud_adapter = Adapter(aud_dim, adapter_dim)

        # Optional bottleneck (compresses backbone output before projection)
        if bottleneck_dim is not None:
            self.eeg_bottleneck = nn.Linear(eeg_dim, bottleneck_dim)
            self.aud_bottleneck = nn.Linear(aud_dim, bottleneck_dim)

        if bottleneck_dim is not None:
            self.eeg_proj = nn.Linear(bottleneck_dim, hidden)
            self.aud_proj = nn.Linear(bottleneck_dim, hidden)
        else:
            self.eeg_proj = nn.Linear(eeg_dim, hidden)
            self.aud_proj = nn.Linear(aud_dim, hidden)

        self.eeg_rms = RMSNorm(hidden)
        self.aud_rms = RMSNorm(hidden)

        # Window-level auxiliary classifier
        if window_aux:
            self.win_cls = WindowClassifier(hidden)

        # Fusion
        if fusion == 'concat':
            self.concat_proj = nn.Linear(hidden * 2, hidden)
        elif fusion == 'gating':
            self.gate = nn.Linear(hidden * 2, hidden)
        elif fusion == 'cross_attn':
            self.cross = CrossModalFusion(hidden, n_heads)

        # Self-attention
        self.self_attn_layers = nn.ModuleList([
            SelfAttentionBlock(hidden, self_attn_heads, self_attn_dropout)
            for _ in range(n_self_attn_layers)
        ])

        # CLS token for pooling='cls'
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
            self.pool = AttentionPool(hidden)

        # Classifier head
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _encode(self, z_eeg, z_audio, mask=None):
        """Shared encoding: adapter → bottleneck → proj → fusion → self-attn.

        Returns:
            z: [B, K, hidden] — per-window features after self-attention
        """
        if hasattr(self, 'eeg_adapter'):
            z_eeg = self.eeg_adapter(z_eeg)
        if hasattr(self, 'aud_adapter'):
            z_audio = self.aud_adapter(z_audio)

        z_eeg_in = z_eeg
        z_audio_in = z_audio
        if self.bottleneck_dim is not None:
            z_eeg_in = torch.relu(self.eeg_bottleneck(z_eeg))
            z_audio_in = torch.relu(self.aud_bottleneck(z_audio))

        e = self.eeg_rms(self.eeg_proj(z_eeg_in))
        a = self.aud_rms(self.aud_proj(z_audio_in))
        if self.training and self.feat_drop is not None:
            e = self.feat_drop(e)
            a = self.feat_drop(a)

        if self.fusion == 'concat':
            z = self.concat_proj(torch.cat([e, a], dim=-1))
        elif self.fusion == 'gating':
            g = torch.sigmoid(self.gate(torch.cat([e, a], dim=-1)))
            z = g * e + (1 - g) * a
        elif self.fusion == 'cross_attn':
            e_out, a_out, self._attn_weights = self.cross(e, a, mask)
            z = (e_out + a_out) / 2

        attn_mask = (mask == 0) if mask is not None else None
        for layer in self.self_attn_layers:
            z = layer(z, mask=attn_mask)

        return z

    def forward_per_window(self, z_eeg, z_audio, mask=None):
        """Return per-window logits for majority voting.

        Args:
            z_eeg:   [B, K, eeg_dim]
            z_audio: [B, K, aud_dim]
            mask:    [B, K] — 1=valid window, 0=padding (or None)
        Returns:
            win_logits: [B, K] — logits per window (before pooling)
        """
        z = self._encode(z_eeg, z_audio, mask)  # [B, K, hidden]
        win_logits = self.head(z).squeeze(-1)    # [B, K]
        if mask is not None:
            win_logits = win_logits.masked_fill(mask == 0, float('-inf'))
        return win_logits

    def _win_logits_post(self, z, mask):
        return self.head(z).squeeze(-1)  # [B, K], raw logits (no -inf — masking handled by caller)

    def forward(self, z_eeg, z_audio, mask=None, return_window=False):
        """
        Args:
            z_eeg:   [B, K, eeg_dim]
            z_audio: [B, K, aud_dim]
            mask:    [B, K] — 1=valid window, 0=padding (or None)
            return_window: if True, also return window-level logits
        Returns:
            logits: [B]
            (or (logits, win_logits_pre, win_logits_post) if return_window)
        """
        B = z_eeg.shape[0]

        # Pre-fusion window logits (from raw projections, before cross-attn)
        win_logits_pre = None
        if return_window and hasattr(self, 'win_cls'):
            z_eeg_in = z_eeg
            z_audio_in = z_audio
            if self.bottleneck_dim is not None:
                z_eeg_in = torch.relu(self.eeg_bottleneck(z_eeg))
                z_audio_in = torch.relu(self.aud_bottleneck(z_audio))
            e_aux = self.eeg_rms(self.eeg_proj(z_eeg_in))
            a_aux = self.aud_rms(self.aud_proj(z_audio_in))
            win_logits_pre = self.win_cls(e_aux, a_aux)  # [B*K]

        # Shared encoding
        z = self._encode(z_eeg, z_audio, mask)  # [B, K, hidden]

        # Post-fusion window logits (after full pipeline, via head)
        win_logits_post = None
        if return_window:
            win_logits_post = self._win_logits_post(z, mask)  # [B, K]

        # Pooling
        if self.pooling == 'mean':
            if mask is not None:
                z_pooled = (z * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                z_pooled = z.mean(dim=1)
        elif self.pooling == 'cls':
            cls = self.cls_token.expand(B, -1, -1)
            z_pooled = torch.cat([cls, z], dim=1)
            z_pooled = self.pool(z_pooled)

        # Classifier
        logits = self.head(z_pooled).squeeze(-1)
        if return_window:
            return logits, win_logits_pre, win_logits_post
        return logits
