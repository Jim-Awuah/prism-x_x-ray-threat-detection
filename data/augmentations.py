"""
Threat-aware data augmentations for PRISM-X Stage 1.

Key design principle (Section 4.2):
  "We introduce a targeted augmentation strategy where partial occlusions
   are applied specifically to threat regions in the labeled images."

Two transform pipelines are provided:
  1. ThreatAwareTransform  – used for *labeled* images; applies partial
                             occlusion specifically inside bounding boxes.
  2. StandardBYOLTransform – used for *unlabeled* images; standard BYOL
                             color-jitter + blur augmentations.

Both return two views (v1, v2) ready for BYOL.
"""

import random
from typing import Optional

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


# Helpers

class RandomPartialOcclusion:
    """
    Randomly erase a rectangular patch *inside* a provided bounding box to
    simulate realistic concealment in baggage scans.

    Args:
        occlusion_ratio : fraction of bbox area to occlude (0-1)
        p               : probability of applying the transform
    """

    def __init__(self, occlusion_ratio: float = 0.4, p: float = 0.5):
        self.occlusion_ratio = occlusion_ratio
        self.p = p

    def __call__(
        self,
        img: torch.Tensor,
        bbox: Optional[list] = None,   # [x1, y1, x2, y2] in pixel coords
    ) -> torch.Tensor:
        if bbox is None or random.random() > self.p:
            return img

        _, H, W = img.shape
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Clamp to image bounds
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)

        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return img

        # Occluded patch area = occlusion_ratio * bbox area
        patch_area = self.occlusion_ratio * bw * bh
        patch_h = int((patch_area / max(bw / bh, 1e-6)) ** 0.5)
        patch_w = int(patch_h * (bw / max(bh, 1)))
        patch_h, patch_w = max(1, patch_h), max(1, patch_w)

        # Random position inside bbox
        ox = random.randint(x1, max(x1, x2 - patch_w))
        oy = random.randint(y1, max(y1, y2 - patch_h))

        img = img.clone()
        img[:, oy : oy + patch_h, ox : ox + patch_w] = 0.0
        return img


# Base BYOL augmentation (used for unlabeled data)
class StandardBYOLTransform:
    """
    Classic BYOL augmentation pipeline applied twice to produce views v1, v2.

    Args:
        img_size : spatial resolution fed to the encoder
    """

    def __init__(self, img_size: int = 224):
        self.transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([
                T.GaussianBlur(kernel_size=img_size // 10 * 2 + 1, sigma=(0.1, 2.0))
            ], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(img), self.transform(img)


# Threat-aware augmentation (used for labeled data)
class ThreatAwareTransform:
    """
    Extends StandardBYOLTransform with targeted partial occlusion applied
    inside ground-truth bounding boxes (Section 4.2).

    Usage::

        transform = ThreatAwareTransform(img_size=224)
        v1, v2 = transform(pil_image, bbox=[x1, y1, x2, y2])

    Args:
        img_size        : spatial resolution fed to encoder
        occlusion_ratio : fraction of bbox area to occlude
        occ_p           : probability of applying occlusion
    """

    def __init__(
        self,
        img_size: int = 224,
        occlusion_ratio: float = 0.4,
        occ_p: float = 0.5,
    ):
        self.img_size = img_size
        self.partial_occlude = RandomPartialOcclusion(
            occlusion_ratio=occlusion_ratio, p=occ_p
        )
        self.base_transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([
                T.GaussianBlur(img_size // 10 * 2 + 1, sigma=(0.1, 2.0))
            ], p=0.5),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __call__(
        self,
        img: Image.Image,
        bbox: Optional[list] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v1 = self.base_transform(img)
        v2 = self.base_transform(img)

        if bbox is not None:
            # Scale bbox to the cropped/resized image — approximate scaling
            scale = self.img_size / max(img.width, img.height)
            scaled_bbox = [int(c * scale) for c in bbox]
            v1 = self.partial_occlude(v1, scaled_bbox)
            v2 = self.partial_occlude(v2, scaled_bbox)

        return v1, v2