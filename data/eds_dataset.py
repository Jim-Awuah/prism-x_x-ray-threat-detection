# data/eds_dataset.py
#
# EDS (Extended Domain Shift) dataset for PRISM-X cross-domain evaluation.
#
# Paper reference: Table 4 — Grounding accuracy under cross-domain settings.
# Trains on one domain (D1/D2/D3) and evaluates on a different domain.
#
# Folder structure:
#   EDS/
#     domain1/
#       image/   ← 00001.jpg, 00002.jpg ...
#       txt/     ← 00001.txt, 00002.txt ...
#     domain2/
#       image/
#       txt/
#     domain3/
#       image/
#       txt/
#
# Annotation format (one or more lines per txt file):
#   filename.jpg  class_name  x1  y1  x2  y2
#   e.g. 00001.jpg knife 231 194 402 431

from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from data.augmentations import StandardBYOLTransform, ThreatAwareTransform

# EDS class names — inferred from annotation format shown
# Update this list if your EDS dataset has different categories
EDS_CLASS_NAMES = [
    "device",
    "glassbottle",
    "knife",
    "laptop",
    "lighter",
    "plasticbottle",
    "powerbank",
    "pressure",
    "scissor",
    "umbrella",
]
EDS_CLASS_TO_IDX = {c: i for i, c in enumerate(EDS_CLASS_NAMES)}

# Domain folder names
DOMAIN_FOLDERS = {
    "D1": "domain1",
    "D2": "domain2",
    "D3": "domain3",
}


class EDSDataset(Dataset):
    """
    EDS dataset for cross-domain evaluation (paper Table 4).

    Used in two modes:
      - Training:   load one domain as source (labeled_only=False for BYOL,
                    labeled_only=True for Stage 2/3)
      - Evaluation: load a different domain as target

    Args:
        root         : EDS root directory containing domain1/, domain2/, domain3/
        domain       : which domain to load — "D1", "D2", or "D3"
        labeled_only : if True, only return images that have annotation files
        img_size     : spatial size for transforms
        augment      : optional custom transform override
    """

    CLASS_NAMES  = EDS_CLASS_NAMES
    CLASS_TO_IDX = EDS_CLASS_TO_IDX

    def __init__(
        self,
        root:         str,
        domain:       str = "D1",
        labeled_only: bool = False,
        img_size:     int = 224,
        augment:      Optional[Callable] = None,
    ) -> None:
        self.root   = Path(root)
        self.domain = domain

        if domain not in DOMAIN_FOLDERS:
            raise ValueError(
                f"domain must be one of {list(DOMAIN_FOLDERS)}. Got: {domain}"
            )

        domain_folder = self.root / DOMAIN_FOLDERS[domain]
        self.img_dir  = domain_folder / "image"
        self.txt_dir  = domain_folder / "txt"

        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"Image folder not found: {self.img_dir}\n"
                f"Check your --data_root path."
            )

        self.threat_transform   = ThreatAwareTransform(img_size=img_size)
        self.standard_transform = StandardBYOLTransform(img_size=img_size)
        if augment is not None:
            self.threat_transform   = augment
            self.standard_transform = augment

        self.samples = self._load_samples(labeled_only)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _parse_txt(self, txt_path: Path) -> list[dict]:
        """
        Parse an EDS annotation file.

        Format per line:
            filename.jpg  class_name  x1  y1  x2  y2

        Returns list of {class, class_idx, bbox: [x1,y1,x2,y2]}.
        """
        if not txt_path.exists():
            return []

        annotations = []
        for line in txt_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            # parts[0] = filename, parts[1] = class, parts[2:6] = bbox
            cls_name = parts[1].lower().strip()
            try:
                x1 = float(parts[2])
                y1 = float(parts[3])
                x2 = float(parts[4])
                y2 = float(parts[5])
            except ValueError:
                continue

            class_idx = self.CLASS_TO_IDX.get(cls_name, -1)
            if class_idx == -1:
                # Unknown class — still include with idx -1
                # so the sample is not silently dropped
                pass

            annotations.append({
                "class":     cls_name,
                "class_idx": class_idx,
                "bbox":      [x1, y1, x2, y2],
            })

        return annotations

    def _load_samples(self, labeled_only: bool) -> list[dict]:
        """Build sample list from image folder."""
        samples = []
        for img_path in sorted(self.img_dir.glob("*.jpg")):
            txt_path    = self.txt_dir / (img_path.stem + ".txt")
            annotations = self._parse_txt(txt_path)
            is_labeled  = len(annotations) > 0

            if labeled_only and not is_labeled:
                continue

            samples.append({
                "img_path":    img_path,
                "annotations": annotations,
                "is_labeled":  is_labeled,
            })

        if not samples:
            raise RuntimeError(
                f"No images found in {self.img_dir}.\n"
                f"Check that images have .jpg extension."
            )

        return samples

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img    = Image.open(sample["img_path"]).convert("RGB")

        if sample["is_labeled"] and sample["annotations"]:
            ann    = sample["annotations"][0]
            v1, v2 = self.threat_transform(img, bbox=ann["bbox"])
            label  = ann["class_idx"]
            bbox   = torch.tensor(ann["bbox"], dtype=torch.float32)
        else:
            v1, v2 = self.standard_transform(img)
            label  = -1
            bbox   = torch.zeros(4)

        return {
            "v1":        v1,
            "v2":        v2,
            "label":     torch.tensor(label, dtype=torch.long),
            "bbox":      bbox,
            "is_labeled": sample["is_labeled"],
            "img_path":  str(sample["img_path"]),
        }