# models/pseudo_labeler.py
#
# Implements Algorithm 1 lines 8-15 (paper §4.3):
#
#   for x_j in unlabeled set U:
#       z_j     = f_θ(x_j)           ← BYOL backbone features
#       y_hat_j = h_φ(z_j)           ← classification head prediction
#       B_hat_j = YOLOv12(x_j)       ← bounding box proposals
#       filter proposals by score threshold + NMS
#       store (x_j, y_hat_j, B_hat_j)
#
# Two components:
#   ClassificationHead  — lightweight linear layer trained on labeled features
#   PseudoLabelGenerator — combines BYOL + head + YOLOv12 into one workflow

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from Stage_1.byol import BYOL
from Stage_2.yolov12 import YOLOv12ProposalGenerator


# ── Classification head h_φ ───────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Single linear layer trained on labeled BYOL backbone features.

    Maps feat_dim → num_classes (raw logits).
    Trained on the small labeled set, then applied to unlabeled images
    to assign a class pseudo-label to each image (Algorithm 1, line 11).
    """

    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)   # (B, num_classes) raw logits


# ── Pseudo-label generator ────────────────────────────────────────────────────

class PseudoLabelGenerator:
    """
    Combines the BYOL encoder, a classification head, and YOLOv12 to
    produce pseudo-labeled training samples for unlabeled X-ray images.

    Workflow per image:
        1. Extract backbone feature vector z from BYOL encoder
        2. Predict class label y_hat from classification head
        3. Generate bounding box proposals from YOLOv12
        4. Filter proposals by confidence threshold (0.65)
        5. Return {img_path, pseudo_label, pseudo_conf, proposals}

    Args:
        byol_model       : trained BYOL model loaded from Stage 1
        num_classes      : number of threat categories
        proposal_gen     : YOLOv12ProposalGenerator (fine-tuned or COCO)
        conf_threshold   : minimum proposal confidence to keep (paper: 0.65)
        device           : "cuda" or "cpu"
    """

    def __init__(
        self,
        byol_model: BYOL,
        num_classes: int,
        proposal_gen: YOLOv12ProposalGenerator,
        conf_threshold: float = 0.65,
        device: Optional[str] = None,
    ) -> None:
        self.device         = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.num_classes    = num_classes
        self.proposal_gen   = proposal_gen

        # Move BYOL to device and freeze — features are fixed in Stage 2
        self.byol = byol_model.to(self.device)
        for p in self.byol.parameters():
            p.requires_grad = False

        # Feature dimension from the BYOL student backbone
        self.feat_dim = self.byol.student.head.in_features

        # Classification head — trained below in fit_head()
        self.head = ClassificationHead(self.feat_dim, num_classes).to(self.device)

    # ── Step 1: Train classification head on labeled set ─────────────────────

    def fit_head(self, labeled_dataset, epochs: int = 20, lr: float = 1e-3) -> None:
        """
        Train h_φ on labeled BYOL features.

        Extracts backbone features from all labeled images once (no grad),
        then trains the linear head for the given number of epochs.

        Args:
            labeled_dataset : dataset returning {"v1": tensor, "label": int}
            epochs          : training epochs for the head
            lr              : learning rate
        """
        loader    = DataLoader(labeled_dataset, batch_size=64, shuffle=True, num_workers=2)
        optimizer = torch.optim.Adam(self.head.parameters(), lr=lr)

        # Extract all labeled features in one pass (frozen encoder)
        all_z, all_y = [], []
        with torch.no_grad():
            for batch in loader:
                v = batch["v1"].to(self.device)
                y = batch["label"].to(self.device)
                z = self._backbone_features(v)
                all_z.append(z)
                all_y.append(y)

        all_z = torch.cat(all_z)   # (N, feat_dim)
        all_y = torch.cat(all_y)   # (N,)

        feat_loader = DataLoader(
            TensorDataset(all_z, all_y), batch_size=64, shuffle=True
        )

        self.head.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for z_batch, y_batch in feat_loader:
                optimizer.zero_grad()
                logits = self.head(z_batch)
                # Only train on valid labels (label >= 0)
                mask = y_batch >= 0
                if mask.sum() == 0:
                    continue
                loss = F.cross_entropy(logits[mask], y_batch[mask])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    # ── Backbone feature extraction ───────────────────────────────────────────

    @torch.no_grad()
    def _backbone_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Extract SwinV2 backbone features from a batch of images.

        Bypasses the softmax head — returns raw backbone vectors.

        Args:
            imgs : (B, 3, H, W)

        Returns:
            (B, feat_dim)
        """
        x = self.byol.student.features(imgs)
        x = self.byol.student.norm(x)
        x = torch.flatten(self.byol.student.avgpool(x.permute(0, 3, 1, 2)), 1)
        return x

    # ── Step 2: Generate pseudo-label for one image ───────────────────────────

    def generate(self, img_tensor: torch.Tensor, img_path: str) -> dict:
        """
        Generate a pseudo-label entry for a single unlabeled image.

        Args:
            img_tensor : preprocessed image tensor (3, H, W)
            img_path   : file path passed to YOLOv12 for proposals

        Returns:
            {
                img_path     : str,
                pseudo_label : int   (predicted class index),
                pseudo_conf  : float (classification confidence 0-1),
                proposals    : list of {bbox, score, label} dicts
            }
        """
        # Class prediction: BYOL features → head → softmax
        self.head.eval()
        z      = self._backbone_features(img_tensor.unsqueeze(0).to(self.device))
        logits = self.head(z)                      # (1, num_classes)
        probs  = F.softmax(logits, dim=-1)
        pseudo_conf, pseudo_label = probs.max(dim=-1)

        # Bounding box proposals from YOLOv12
        proposals = self.proposal_gen.propose(img_path)

        # Filter by confidence threshold (paper §4.3)
        proposals = [p for p in proposals if p["score"] >= self.conf_threshold]

        return {
            "img_path":     img_path,
            "pseudo_label": pseudo_label.item(),
            "pseudo_conf":  pseudo_conf.item(),
            "proposals":    proposals,
        }

    # ── Step 3: Run over entire unlabeled dataset ─────────────────────────────

    def generate_all(self, unlabeled_dataset) -> list[dict]:
        """
        Generate pseudo-labels for every image in the unlabeled dataset.

        Can be called every 10 epochs from Stage 3 to progressively
        refine pseudo-labels as the BYOL encoder improves (paper §4.3).

        Args:
            unlabeled_dataset : dataset returning {"v1": tensor, "img_path": str}

        Returns:
            List of pseudo-label dicts, one per image
        """
        loader  = DataLoader(
            unlabeled_dataset, batch_size=1, shuffle=False, num_workers=2
        )
        results = []
        for i, batch in enumerate(loader):
            img_tensor = batch["v1"].squeeze(0)
            img_path   = batch["img_path"][0]
            entry      = self.generate(img_tensor, img_path)
            results.append(entry)
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1} / {len(unlabeled_dataset)} images ...")
        return results