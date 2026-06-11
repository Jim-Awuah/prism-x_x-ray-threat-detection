

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VLCLoss(nn.Module):
    """
    Vision-Language Contrastive loss (Eq. 3).

    Args:
        temperature : τ scaling factor (paper: 0.07)
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        visual_embeds: torch.Tensor,   # (B, K, D) — fused region embeddings ṽ
        text_embeds:   torch.Tensor,   # (B, K, D) — class text embeddings  t
    ) -> torch.Tensor:
        """
        Args:
            visual_embeds : (B, K, D)  batch of K fused region embeddings
            text_embeds   : (B, K, D)  corresponding text embeddings

        Returns:
            Scalar VLC loss averaged over the batch.
        """
        B, K, D = visual_embeds.shape

        # L2-normalise so that dot product = cosine similarity
        v = F.normalize(visual_embeds, dim=-1)   # (B, K, D)
        t = F.normalize(text_embeds,   dim=-1)   # (B, K, D)

        # Compute cosine similarity matrix: (B, K_v, K_t)
        # sim[b, i, j] = <ṽ_i, t_j> for image b
        sim = torch.bmm(v, t.transpose(1, 2)) / self.temperature  # (B, K, K)

        # Ground-truth: region i matches text i  →  diagonal is positive
        targets = torch.arange(K, device=visual_embeds.device)    # (K,)
        targets = targets.unsqueeze(0).expand(B, -1)               # (B, K)

        # Cross-entropy over the K text candidates for each visual region
        # Reshape to (B*K, K) so F.cross_entropy works straightforwardly
        loss = F.cross_entropy(
            sim.reshape(B * K, K),          # logits  (B*K, K)
            targets.reshape(B * K),         # labels  (B*K,)
        )
        return loss


class DetectionLoss(nn.Module):
    """
    Combined detection loss used in Stage 3.

    Components
    ----------
    L_cls   : cross-entropy classification loss
    L_bbox  : L1 regression loss on normalised box coordinates
    L_giou  : Generalised IoU loss for box quality
    L_vlc   : Vision-Language Contrastive loss (Eq. 3)

    Total loss = w_cls·L_cls + w_bbox·L_bbox + w_giou·L_giou + w_vlc·L_vlc

    Args:
        num_classes   : number of threat categories
        temperature   : τ for VLC loss
        w_cls / w_bbox / w_giou / w_vlc : loss weights
    """

    def __init__(
        self,
        num_classes: int,
        temperature: float = 0.07,
        w_cls:  float = 1.0,
        w_bbox: float = 5.0,
        w_giou: float = 2.0,
        w_vlc:  float = 1.0,
    ) -> None:
        super().__init__()
        self.w_cls  = w_cls
        self.w_bbox = w_bbox
        self.w_giou = w_giou
        self.w_vlc  = w_vlc

        self.vlc_loss = VLCLoss(temperature)

    # ── GIoU ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _giou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Generalised IoU loss for axis-aligned boxes.

        Args:
            pred   : (N, 4)  predicted boxes in [x1,y1,x2,y2] normalised [0,1]
            target : (N, 4)  target  boxes in [x1,y1,x2,y2] normalised [0,1]

        Returns:
            Scalar mean GIoU loss  (1 - GIoU), lower = better.
        """
        # Intersection
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter   = inter_w * inter_h

        # Union
        area_pred   = (pred[:, 2]   - pred[:, 0]) * (pred[:, 3]   - pred[:, 1])
        area_target = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = area_pred + area_target - inter + 1e-6

        iou = inter / union

        # Enclosing box
        enc_x1 = torch.min(pred[:, 0], target[:, 0])
        enc_y1 = torch.min(pred[:, 1], target[:, 1])
        enc_x2 = torch.max(pred[:, 2], target[:, 2])
        enc_y2 = torch.max(pred[:, 3], target[:, 3])
        enc    = (enc_x2 - enc_x1) * (enc_y2 - enc_y1) + 1e-6

        giou = iou - (enc - union) / enc
        return (1.0 - giou).mean()

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        pred_logits:    torch.Tensor,   # (B, Q, num_classes+1) — cls scores
        pred_boxes:     torch.Tensor,   # (B, Q, 4)             — predicted boxes
        target_labels:  torch.Tensor,   # (B, Q)                — gt class indices
        target_boxes:   torch.Tensor,   # (B, Q, 4)             — gt boxes [0,1]
        visual_embeds:  torch.Tensor,   # (B, K, D)             — fused ṽ
        text_embeds:    torch.Tensor,   # (B, K, D)             — text  t
        valid_mask:     torch.Tensor,   # (B, Q) bool           — which queries are valid
    ) -> dict[str, torch.Tensor]:
        """
        Compute all loss components and return them as a dict.

        The caller can sum total_loss for backprop and log the components
        individually for monitoring.

        Returns:
            {
                "total":  scalar,
                "cls":    scalar,
                "bbox":   scalar,
                "giou":   scalar,
                "vlc":    scalar,
            }
        """
        B, Q, _ = pred_logits.shape

        # ── Flatten valid predictions / targets ───────────────────────────────
        mask_flat    = valid_mask.reshape(-1)               # (B*Q,)
        logits_flat  = pred_logits.reshape(B * Q, -1)       # (B*Q, C+1)
        boxes_flat   = pred_boxes.reshape(B * Q, 4)         # (B*Q, 4)
        labels_flat  = target_labels.reshape(-1)            # (B*Q,)
        tgt_bx_flat  = target_boxes.reshape(B * Q, 4)       # (B*Q, 4)

        # Only compute detection losses over valid (non-padding) entries
        if mask_flat.sum() > 0:
            l_logits = logits_flat[mask_flat]
            l_labels = labels_flat[mask_flat]
            l_boxes  = boxes_flat[mask_flat]
            l_tgt_bx = tgt_bx_flat[mask_flat]

            l_cls  = F.cross_entropy(l_logits, l_labels)
            l_bbox = F.l1_loss(l_boxes, l_tgt_bx)
            l_giou = self._giou(l_boxes, l_tgt_bx)
        else:
            # Edge case: no valid queries in this batch
            l_cls  = pred_logits.sum() * 0.0
            l_bbox = pred_boxes.sum()  * 0.0
            l_giou = pred_boxes.sum()  * 0.0

        # ── Vision-Language Contrastive loss ──────────────────────────────────
        l_vlc = self.vlc_loss(visual_embeds, text_embeds)

        total = (
            self.w_cls  * l_cls  +
            self.w_bbox * l_bbox +
            self.w_giou * l_giou +
            self.w_vlc  * l_vlc
        )

        return {
            "total": total,
            "cls":   l_cls,
            "bbox":  l_bbox,
            "giou":  l_giou,
            "vlc":   l_vlc,
        }