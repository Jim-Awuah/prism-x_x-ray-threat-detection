# BYOL – PRISM-X Stage 1  (Ahmed et al., IPM 2026, §4.2, Eq. 1, Fig. 3)
#
# Backbone: torchvision Swin Transformer V2 (Liu et al., 2022) — paper §5.1
#
# Two encoder branches on two augmented views of the same image:
#
#   v1 → Online encoder f_θ → Predictor q_θ → z1 ──┐
#                                                    ├─ L_BYOL = ||z1 - z2||²
#   v2 → Target encoder f_ξ → Projector  p_ξ → z2 ──┘  (SG on target)
#
# Paper Equation 1:
#   L_BYOL = || q_θ( f_θ(v1) ) - p_ξ( f_ξ(v2) ) ||²
#
# where ξ (target/momentum params) is updated as an EMA of θ (online params),
# and gradients are NOT propagated through the target network.
#
# Algorithm 1, lines 2-7:
#   v1, v2  ← augment(x)
#   z1 ← q_θ(f_θ(v1)),  z2 ← p_ξ(f_ξ(v2))
#   L_BYOL ← ||z1 - z2||²
#   backprop and update θ
#   ξ ← τ·ξ + (1-τ)·θ

from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    swin_v2_t, Swin_V2_T_Weights,
    swin_v2_s, Swin_V2_S_Weights,
    swin_v2_b, Swin_V2_B_Weights,
)

# ── Backbone registry ─────────────────────────────────────────────────────────

_SWIN = {
    "swin_v2_t": (swin_v2_t, Swin_V2_T_Weights.DEFAULT, 768),
    "swin_v2_s": (swin_v2_s, Swin_V2_S_Weights.DEFAULT, 768),
    "swin_v2_b": (swin_v2_b, Swin_V2_B_Weights.DEFAULT, 1024),
}


# ── Backbone wrapper f_θ / f_ξ ────────────────────────────────────────────────

class SwinBackbone(nn.Module):
    """
    Swin Transformer V2 feature extractor (Liu et al., 2022) — paper §5.1.

    This is f_θ (online) or f_ξ (target) in Eq. 1 — the encoder only,
    with no projector/predictor head attached. Those are separate
    modules in BYOL, matching the paper's q_θ ∘ f_θ composition.

    Pipeline:
        image (B, 3, H, W)
          → SwinV2 stages → (B, H', W', C)
          → LayerNorm
          → AdaptiveAvgPool2d(1,1)
          → flatten → (B, feat_dim)
    """

    def __init__(self, variant: str = "swin_v2_t", pretrained: bool = True) -> None:
        super().__init__()
        if variant not in _SWIN:
            raise ValueError(f"variant must be one of {list(_SWIN)}")

        factory, weights, feat_dim = _SWIN[variant]
        swin = factory(weights=weights if pretrained else None)

        # Keep feature extraction pipeline, discard the ImageNet classifier
        self.features = swin.features   # patch partition + 4 Swin stages
        self.norm     = swin.norm       # LayerNorm
        self.avgpool  = swin.avgpool    # AdaptiveAvgPool2d(1,1)
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, H, W)
        Returns:
            (B, feat_dim) — raw encoder features f(x)
        """
        x = self.norm(self.features(x))                            # (B, H', W', C)
        return torch.flatten(self.avgpool(x.permute(0, 3, 1, 2)), 1)  # (B, feat_dim)


# ── Projector / Predictor MLPs ────────────────────────────────────────────────

class MLPHead(nn.Module):
    """
    Two-layer MLP with BatchNorm + ReLU, used as both the projector
    (p_ξ on the target branch) and the predictor (q_θ on the online
    branch), matching the standard BYOL design.

    Pipeline:
        (B, in_dim) → Linear → BN → ReLU → Linear → (B, out_dim)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 4096, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── BYOL ──────────────────────────────────────────────────────────────────────

class BYOL(nn.Module):
    """
    BYOL as shown in Fig. 3 / Eq. 1 of the PRISM-X paper.

    Online branch  (trained by backprop):
        f_θ  (SwinBackbone)  →  p_θ (projector)  →  q_θ (predictor)
    Target branch  (updated by EMA only, no gradient):
        f_ξ  (SwinBackbone)  →  p_ξ (projector)

    Loss (Eq. 1):
        L_BYOL = || q_θ(p_θ(f_θ(v1))) - sg( p_ξ(f_ξ(v2)) ) ||²

    Call update_target() once after every optimizer.step().

    Args:
        backbone_variant    : "swin_v2_t" | "swin_v2_s" | "swin_v2_b"
        backbone_pretrained : load ImageNet weights
        projection_dim      : output dim of projector/predictor (paper-style: 256)
        hidden_dim           : hidden dim inside projector/predictor MLPs
        ema_decay            : τ (paper uses 0.996)
        num_classes          : OPTIONAL — kept for backward compatibility with
                                Stage 2/3 code that may still reference a
                                classification head; not used in the BYOL loss.
    """

    def __init__(
        self,
        backbone_variant:    str   = "swin_v2_t",
        backbone_pretrained: bool  = True,
        projection_dim:      int   = 256,
        hidden_dim:           int   = 4096,
        ema_decay:            float = 0.996,
        num_classes:          int | None = None,   # kept for API compatibility
    ) -> None:
        super().__init__()
        self.ema_decay = ema_decay

        # ── Online branch: f_θ → p_θ → q_θ ──────────────────────────────────
        self.online_encoder   = SwinBackbone(backbone_variant, backbone_pretrained)
        feat_dim              = self.online_encoder.feat_dim
        self.online_projector = MLPHead(feat_dim, hidden_dim, projection_dim)
        self.predictor        = MLPHead(projection_dim, hidden_dim, projection_dim)

        # ── Target branch: f_ξ → p_ξ  (deep-copied, frozen, EMA-only) ───────
        self.target_encoder   = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        # feat_dim exposed for Stage 2 (PseudoLabelGenerator) and Stage 3
        self.feat_dim = feat_dim

        # ── Backward-compatibility alias ─────────────────────────────────────
        # Older Stage 2/3 code may refer to `byol.student` for the encoder.
        # `student` now points at the online encoder directly so
        # `extract_features()` and any legacy `.student.*` access still works.
        self.student = self.online_encoder

    # ── EMA update (Algorithm 1, line 7: ξ ← τ·ξ + (1−τ)·θ) ───────────────────
    @torch.no_grad()
    def update_target(self) -> None:
        """Call once per step, after optimizer.step()."""
        for θ, ξ in zip(self.online_encoder.parameters(),
                        self.target_encoder.parameters()):
            ξ.data.mul_(self.ema_decay).add_(θ.data, alpha=1.0 - self.ema_decay)

        for θ, ξ in zip(self.online_projector.parameters(),
                        self.target_projector.parameters()):
            ξ.data.mul_(self.ema_decay).add_(θ.data, alpha=1.0 - self.ema_decay)

    # ── Forward (Eq. 1) ────────────────────────────────────────────────────────
    def forward(
        self, v1: torch.Tensor, v2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            v1 : (B, 3, H, W)  first augmented view  → online branch
            v2 : (B, 3, H, W)  second augmented view → target branch

        Returns:
            loss : scalar — squared L2 distance, Eq. 1
            z1   : (B, projection_dim) online prediction q_θ(p_θ(f_θ(v1)))
            z2   : (B, projection_dim) target projection  p_ξ(f_ξ(v2)) (detached)
        """
        # Online: f_θ → p_θ → q_θ
        z1 = self.predictor(self.online_projector(self.online_encoder(v1)))

        # Target: f_ξ → p_ξ   (stop-gradient — Eq. 1's sg(·))
        with torch.no_grad():
            z2 = self.target_projector(self.target_encoder(v2))

        # L_BYOL = || z1 - z2 ||²  on L2-normalised vectors (standard BYOL)
        z1n = F.normalize(z1, dim=-1)
        z2n = F.normalize(z2, dim=-1)
        loss = (z1n - z2n).pow(2).sum(dim=-1).mean()

        return loss, z1, z2

    # ── Inference ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Raw backbone features (B, feat_dim) from the online encoder f_θ.
        Used by Stage 2 PseudoLabelGenerator and Stage 3 FrozenVisionEncoder.

        Note: returns pre-projection backbone features, NOT z1 — this is
        the representation used for downstream classification/detection,
        consistent with standard BYOL practice (discard projector/predictor
        after pretraining, keep only f_θ).
        """
        return self.online_encoder(x)

    def trainable_parameters(self) -> list:
        """Online-branch parameters only, for the optimiser."""
        return (
            list(self.online_encoder.parameters())
            + list(self.online_projector.parameters())
            + list(self.predictor.parameters())
        )