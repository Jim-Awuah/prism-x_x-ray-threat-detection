# Implements Algorithm 1 lines 8-15 (paper §4.3):
#
#   for x_j in unlabeled set U:
#       z_j     = f_θ(x_j)           ← BYOL backbone features
#       y_hat_j = h_φ(z_j)           ← classification head prediction
#       B_hat_j = YOLOv12(x_j)       ← bounding box proposals
#       filter proposals by score threshold
#       store (x_j, y_hat_j, B_hat_j)

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
    """

    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


# ── Pseudo-label generator ────────────────────────────────────────────────────

class PseudoLabelGenerator:
    """
    Combines the BYOL encoder, a classification head, and YOLOv12 to
    produce pseudo-labeled training samples for unlabeled X-ray images.

    Args:
        byol_model     : trained BYOL model from Stage 1
        num_classes    : number of threat categories
        proposal_gen   : YOLOv12ProposalGenerator
        conf_threshold : minimum proposal confidence (paper: 0.65)
        device         : "cuda" or "cpu"
    """

    def __init__(
        self,
        byol_model:     BYOL,
        num_classes:    int,
        proposal_gen:   YOLOv12ProposalGenerator,
        conf_threshold: float = 0.65,
        device:         Optional[str] = None,
    ) -> None:
        self.device         = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.num_classes    = num_classes
        self.proposal_gen   = proposal_gen

        # Freeze BYOL — features are fixed in Stage 2
        self.byol = byol_model.to(self.device)
        for p in self.byol.parameters():
            p.requires_grad = False

        # feat_dim from the student backbone
        self.feat_dim = self.byol.student.feat_dim

        # Classification head trained in fit_head()
        self.head = ClassificationHead(self.feat_dim, num_classes).to(self.device)

    # ── Train classification head on labeled set ──────────────────────────────

    def fit_head(self, labeled_dataset, epochs: int = 20, lr: float = 1e-3) -> None:
        """Train h_φ on labeled BYOL features (Algorithm 1, lines 7-8)."""
        loader    = DataLoader(labeled_dataset, batch_size=64,
                               shuffle=True, num_workers=2)
        optimizer = torch.optim.Adam(self.head.parameters(), lr=lr)

        # Extract all labeled features once using the public extract_features()
        all_z, all_y = [], []
        with torch.no_grad():
            for batch in loader:
                v = batch["v1"].to(self.device)
                y = batch["label"].to(self.device)
                z = self.byol.extract_features(v)   # (B, feat_dim) — clean API
                all_z.append(z)
                all_y.append(y)

        all_z = torch.cat(all_z)
        all_y = torch.cat(all_y)

        feat_loader = DataLoader(
            TensorDataset(all_z, all_y), batch_size=64, shuffle=True
        )

        self.head.train()
        for _ in range(epochs):
            for z_batch, y_batch in feat_loader:
                optimizer.zero_grad()
                logits = self.head(z_batch)
                mask   = y_batch >= 0          # skip unlabeled (-1) entries
                if mask.sum() == 0:
                    continue
                loss = F.cross_entropy(logits[mask], y_batch[mask])
                loss.backward()
                optimizer.step()

    # ── Generate pseudo-label for one image ───────────────────────────────────

    def generate(self, img_tensor: torch.Tensor, img_path: str) -> dict:
        """
        Generate a pseudo-label for a single unlabeled image.

        Returns:
            {img_path, pseudo_label, pseudo_conf, proposals}
        """
        self.head.eval()

        # Use public extract_features() — works correctly with torchvision SwinV2
        z      = self.byol.extract_features(
                     img_tensor.unsqueeze(0).to(self.device)
                 )                                    # (1, feat_dim)
        probs  = F.softmax(self.head(z), dim=-1)
        pseudo_conf, pseudo_label = probs.max(dim=-1)

        # Bounding box proposals from YOLOv12
        proposals = self.proposal_gen.propose(img_path)
        proposals = [p for p in proposals if p["score"] >= self.conf_threshold]

        return {
            "img_path":     img_path,
            "pseudo_label": pseudo_label.item(),
            "pseudo_conf":  pseudo_conf.item(),
            "proposals":    proposals,
        }

    # ── Generate for entire unlabeled dataset ─────────────────────────────────

    def generate_all(self, unlabeled_dataset) -> list[dict]:
        """
        Generate pseudo-labels for every image in the unlabeled dataset.
        Called every 10 epochs from Stage 3 for progressive refinement.
        """
        loader  = DataLoader(
            unlabeled_dataset, batch_size=1, shuffle=False, num_workers=2
        )
        results = []
        for i, batch in enumerate(loader):
            img_tensor = batch["v1"].squeeze(0)
            img_path   = batch["img_path"][0]
            results.append(self.generate(img_tensor, img_path))
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1} / {len(unlabeled_dataset)} images ...")
        return results