# BYOL implementation for PRISM-X threat classification.
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


# ── Vision Encoder (backbone + Softmax head) ──────────────────────────────────

class VisionEncoder(nn.Module):
    """
    SwinV2 backbone with a linear projection head followed by Softmax.
    Outputs a probability distribution F of shape (B, num_classes).

    backbone: image (B,3,H,W) → avgpool → (B, feat_dim)
    head:     (B, feat_dim)   → Linear  → (B, num_classes)
    softmax:  (B, num_classes) → F
    """

    def __init__(
        self,
        num_classes: int,
        variant: str = "swin_v2_t",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if variant not in _SWIN:
            raise ValueError(f"variant must be one of {list(_SWIN)}")

        factory, weights, feat_dim = _SWIN[variant]
        swin = factory(weights=weights if pretrained else None)

        # Strip classifier head; keep feature-extraction pipeline
        self.features = swin.features
        self.norm     = swin.norm
        self.avgpool  = swin.avgpool
        self.head     = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        x = self.norm(self.features(x))                       # (B, H', W', C)
        x = torch.flatten(self.avgpool(x.permute(0, 3, 1, 2)), 1)  # (B, feat_dim)
        # Head + Softmax  →  F_S or F_T
        return F.softmax(self.head(x), dim=-1)                # (B, num_classes)


# ── BYOL ──────────────────────────────────────────────────────────────────────

class BYOL(nn.Module):
    """
    BYOL as shown in Fig. 3 of the PRISM-X paper.

    Student encoder  f_θ  is trained by backprop.
    Target encoder   f_ξ  is updated by EMA:  ξ ← τ·ξ + (1−τ)·θ

    Loss:  CE( F_S,  sg(F_T) )
      where F_S = softmax output of the student  (predictions)
            F_T = softmax output of the target   (soft labels, stop-gradient)

    Call update_target() once after every optimizer.step().

    Args:
        num_classes         : number of threat categories
        backbone_variant    : "swin_v2_t" | "swin_v2_s" | "swin_v2_b"
        backbone_pretrained : load ImageNet weights
        ema_decay           : τ — how slowly the target tracks the student
    """

    def __init__(
        self,
        num_classes: int,
        backbone_variant: str    = "swin_v2_t",
        backbone_pretrained: bool = True,
        ema_decay: float         = 0.996,
    ) -> None:
        super().__init__()
        self.ema_decay = ema_decay

        # Student (online) — updated by gradient descent
        self.student = VisionEncoder(num_classes, backbone_variant, backbone_pretrained)

        # Target — frozen copy, updated by EMA only
        self.target = copy.deepcopy(self.student)
        for p in self.target.parameters():
            p.requires_grad = False

    # ── EMA update:  ξ ← τ·ξ + (1−τ)·θ  ─────────────────────────────────────
    @torch.no_grad()
    def update_target(self) -> None:
        """Call once per step, after optimizer.step()."""
        for θ, ξ in zip(self.student.parameters(), self.target.parameters()):
            ξ.data.mul_(self.ema_decay).add_(θ.data, alpha=1.0 - self.ema_decay)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(
        self, v1: torch.Tensor, v2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            v1 : (B, 3, H, W)  first augmented view  → student
            v2 : (B, 3, H, W)  second augmented view → target

        Returns:
            loss : scalar CE loss
            F_S  : student softmax output  (B, num_classes)
            F_T  : target softmax output   (B, num_classes)
        """
        F_S = self.student(v1)                        # (B, num_classes)

        with torch.no_grad():                         # SG — no gradient to target
            F_T = self.target(v2)                     # (B, num_classes)

        # CE_Loss: treat F_T as soft labels for F_S
        # F_S is already softmax, so we use log for the student side
        loss = F.cross_entropy(
            torch.log(F_S + 1e-8),                   # log-probabilities for CE
            F_T,                                      # soft targets
        )

        return loss, F_S, F_T

    # ── Inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns softmax class probabilities from the student encoder.
        Shape: (B, num_classes)
        """
        return self.student(x)
    

if __name__ == "__main__":
   

    model = BYOL(num_classes=6)
    v1 = torch.randn(2, 3, 224, 224)   # batch of 2 fake images
    v2 = torch.randn(2, 3, 224, 224)
    loss, F_S, F_T = model(v1, v2)

    print("Loss:", loss.item())
    print("F_S shape:", F_S.shape)
    print("F_T shape:", F_T.shape)