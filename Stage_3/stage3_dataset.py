# Stage_3/stage3_dataset.py
#
# PyTorch Dataset for Stage 3 training.
#
# Reads the pseudo-label JSON produced by Stage 2 (PseudoLabelGenerator)
# and returns batches suitable for VisionLanguageDetector.forward().
#
# Each sample:
#   image        : (3, img_size, img_size)  preprocessed X-ray scan
#   region_imgs  : (K, 3, crop_size, crop_size) cropped proposal regions
#   bbox_coords  : (K, 4)  normalised [x1,y1,x2,y2]  ∈ [0,1]
#   pseudo_labels: (K,)    int class index per proposal
#   gt_labels    : (K,)    ground-truth class index (-1 for unlabeled)
#   gt_boxes     : (K, 4)  ground-truth boxes (zeros if unlabeled)
#   valid_mask   : (K,)    bool — True when this proposal has a real GT box
#   img_path     : str     original file path (for debugging)
#
# Labeled samples (from the ground-truth set) are mixed in so that the
# detection head also trains on real annotations alongside pseudo-labels.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ── Default transforms ────────────────────────────────────────────────────────

def _default_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


def _crop_transform(crop_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


# ── Bounding box helpers ──────────────────────────────────────────────────────

def _clamp_box(box: list[float], W: int, H: int) -> list[float]:
    """Clamp pixel box to image dimensions."""
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(x1, W - 1)),
        max(0.0, min(y1, H - 1)),
        max(0.0, min(x2, W)),
        max(0.0, min(y2, H)),
    ]


def _normalise_box(box: list[float], W: int, H: int) -> list[float]:
    """Convert pixel coords to normalised [0,1] coords."""
    x1, y1, x2, y2 = box
    return [x1 / W, y1 / H, x2 / W, y2 / H]


# ── Stage3Dataset ─────────────────────────────────────────────────────────────

class Stage3Dataset(Dataset):
    """
    Dataset for Stage 3 vision-language detection training.

    Args:
        pseudo_label_file  : path to JSON file produced by Stage 2
                             (list of dicts: img_path, pseudo_label,
                              proposals: [{bbox, score, label}])
        labeled_annotations: optional path to a labeled annotation JSON
                             in the same format, used to mix in GT samples.
                             Format: [{img_path, label, boxes:[{bbox,label}]}]
        img_size           : resize all images to (img_size, img_size)
        crop_size          : resize all region crops to (crop_size, crop_size)
        max_proposals      : maximum K proposals kept per image
        min_proposals      : pad with zeros if fewer proposals exist
        return_crops       : if True, return cropped region images;
                             if False, caller can use whole-image features.
    """

    def __init__(
        self,
        pseudo_label_file:   str,
        labeled_annotations: Optional[str] = None,
        img_size:            int  = 224,
        crop_size:           int  = 224,
        max_proposals:       int  = 100,
        min_proposals:       int  = 1,
        return_crops:        bool = True,
    ) -> None:
        self.img_size       = img_size
        self.crop_size      = crop_size
        self.max_proposals  = max_proposals
        self.min_proposals  = min_proposals
        self.return_crops   = return_crops

        self.img_transform  = _default_transform(img_size)
        self.crop_transform = _crop_transform(crop_size)

        # ── Load pseudo-labeled samples ───────────────────────────────────────
        with open(pseudo_label_file) as f:
            pseudo_data = json.load(f)
        self.samples = self._convert_pseudo(pseudo_data)

        # ── Mix in labeled ground-truth samples ───────────────────────────────
        if labeled_annotations:
            with open(labeled_annotations) as f:
                gt_data = json.load(f)
            self.samples += self._convert_labeled(gt_data)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples loaded from {pseudo_label_file}. "
                "Check that Stage 2 completed successfully."
            )

    # ── Conversion helpers ────────────────────────────────────────────────────

    def _convert_pseudo(self, data: list[dict]) -> list[dict]:
        """Convert Stage 2 output format to internal sample format."""
        samples = []
        for item in data:
            if not item.get("proposals"):
                continue                      # skip images with no proposals
            samples.append({
                "img_path":      item["img_path"],
                "proposals":     item["proposals"],    # [{bbox, score, label}]
                "img_label":     item["pseudo_label"],
                "is_labeled":    False,
                "gt_boxes":      [],
                "gt_labels":     [],
            })
        return samples

    def _convert_labeled(self, data: list[dict]) -> list[dict]:
        """
        Convert a labeled annotation file to internal format.

        Expected JSON schema:
          [{"img_path": "...",
            "label": 2,
            "boxes": [{"bbox": [x1,y1,x2,y2], "label": 2}, ...]
           }, ...]
        """
        samples = []
        for item in data:
            if not item.get("boxes"):
                continue
            # Wrap GT boxes as proposals so the same pipeline applies
            proposals = [
                {"bbox": b["bbox"], "score": 1.0, "label": b["label"]}
                for b in item["boxes"]
            ]
            samples.append({
                "img_path":   item["img_path"],
                "proposals":  proposals,
                "img_label":  item.get("label", 0),
                "is_labeled": True,
                "gt_boxes":   [b["bbox"]  for b in item["boxes"]],
                "gt_labels":  [b["label"] for b in item["boxes"]],
            })
        return samples

    # ── __len__ / __getitem__ ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img    = Image.open(sample["img_path"]).convert("RGB")
        W, H   = img.size

        img_tensor = self.img_transform(img)    # (3, img_size, img_size)

        # ── Proposals ─────────────────────────────────────────────────────────
        props = sample["proposals"][: self.max_proposals]
        K     = max(len(props), self.min_proposals)

        bbox_coords   = torch.zeros(K, 4)
        pseudo_labels = torch.zeros(K, dtype=torch.long)
        gt_boxes      = torch.zeros(K, 4)
        gt_labels     = torch.full((K,), -1, dtype=torch.long)
        valid_mask    = torch.zeros(K, dtype=torch.bool)
        region_imgs   = []

        for i, p in enumerate(props):
            box_px = _clamp_box(p["bbox"], W, H)
            box_n  = _normalise_box(box_px, W, H)
            bbox_coords[i]   = torch.tensor(box_n)
            pseudo_labels[i] = int(p["label"])

            if self.return_crops:
                x1, y1, x2, y2 = [int(v) for v in box_px]
                # Guard against zero-area boxes
                x2 = max(x2, x1 + 1)
                y2 = max(y2, y1 + 1)
                crop = img.crop((x1, y1, x2, y2))
                region_imgs.append(self.crop_transform(crop))
            else:
                region_imgs.append(torch.zeros(3, self.crop_size, self.crop_size))

        # Fill in GT info for labeled samples
        if sample["is_labeled"]:
            for i, (gt_b, gt_l) in enumerate(
                zip(sample["gt_boxes"], sample["gt_labels"])
            ):
                if i >= K:
                    break
                gt_boxes[i]   = torch.tensor(_normalise_box(gt_b, W, H))
                gt_labels[i]  = int(gt_l)
                valid_mask[i] = True

        # Stack region images: (K, 3, crop_size, crop_size)
        region_stack = torch.stack(region_imgs) if region_imgs else \
            torch.zeros(K, 3, self.crop_size, self.crop_size)

        return {
            "image":         img_tensor,        # (3, H, W)
            "region_imgs":   region_stack,      # (K, 3, crop, crop)
            "bbox_coords":   bbox_coords,       # (K, 4)
            "pseudo_labels": pseudo_labels,     # (K,)
            "gt_boxes":      gt_boxes,          # (K, 4)
            "gt_labels":     gt_labels,         # (K,)
            "valid_mask":    valid_mask,         # (K,)
            "img_path":      sample["img_path"],
        }


# ── Collate function ──────────────────────────────────────────────────────────

def stage3_collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate that pads proposals to the same K across the batch.
    Needed because different images may have different proposal counts.
    """
    # Find the max K in this batch
    max_k = max(b["bbox_coords"].shape[0] for b in batch)

    def _pad(t: torch.Tensor, target_len: int, pad_val: float = 0.0) -> torch.Tensor:
        """Pad first dimension of t to target_len."""
        deficit = target_len - t.shape[0]
        if deficit <= 0:
            return t[:target_len]
        pad_shape = (deficit,) + t.shape[1:]
        return torch.cat([t, torch.full(pad_shape, pad_val, dtype=t.dtype)])

    images       = torch.stack([b["image"] for b in batch])
    bbox_coords  = torch.stack([_pad(b["bbox_coords"],   max_k) for b in batch])
    pseudo_lbls  = torch.stack([_pad(b["pseudo_labels"], max_k) for b in batch])
    gt_boxes     = torch.stack([_pad(b["gt_boxes"],      max_k) for b in batch])
    gt_labels    = torch.stack([_pad(b["gt_labels"],     max_k, -1) for b in batch])
    valid_masks  = torch.stack([_pad(b["valid_mask"].float(), max_k).bool()
                                for b in batch])

    # Region images: (B, K, 3, crop, crop)
    region_stack = torch.stack([
        _pad(b["region_imgs"], max_k)      # (K, 3, crop, crop)
        for b in batch
    ])

    return {
        "images":        images,
        "region_imgs":   region_stack,
        "bbox_coords":   bbox_coords,
        "pseudo_labels": pseudo_lbls,
        "gt_boxes":      gt_boxes,
        "gt_labels":     gt_labels,
        "valid_mask":    valid_masks,
        "img_paths":     [b["img_path"] for b in batch],
    }