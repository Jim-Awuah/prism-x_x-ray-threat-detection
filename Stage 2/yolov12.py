from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO as _YOLO
    _ULTRALYTICS_OK = True
except ImportError:
    _ULTRALYTICS_OK = False


class YOLOv12ProposalGenerator:
    """
    YOLOv12 region proposal generator for PRISM-X Stage 2.

    Combines two responsibilities:
      1. Fine-tuning  — adapts COCO-pretrained weights to X-ray imagery
                        using a two-phase frozen/unfrozen strategy.
      2. Inference    — runs the (optionally fine-tuned) model on X-ray
                        scans and returns filtered bounding box proposals.

    The model is always used inference-only during Stage 2 — it is never
    further trained on SIXray/CLCXray labels. Its role is to suggest
    *where* objects might be; the BYOL encoder decides *what* they are.

    Args:
        weights_path   : starting weights — COCO pretrained (.pt filename)
                         or path to already fine-tuned weights
        conf_threshold : minimum detection confidence to keep (paper: 0.65)
        iou_threshold  : NMS IoU threshold for removing duplicate boxes
        img_size       : inference resolution (must match fine-tuning size)
        device         : "cuda", "cpu", or None (auto-detect)
        max_proposals  : hard cap on boxes returned per image
        output_dir     : where fine-tuning checkpoints are saved
    """

    # Default fine-tuning hyperparameters — mirror finetune_config.py
    # but kept here so the class works standalone without importing configs.
    _DEFAULT_FT = {
        "epochs":        100,
        "freeze_epochs": 10,
        "freeze_layers": 10,
        "batch_size":    16,
        "num_workers":   4,
        "optimizer":     "AdamW",
        "lr0":           1e-4,
        "lrf":           0.01,
        "momentum":      0.937,
        "weight_decay":  5e-4,
        "warmup_epochs": 3,
        "mosaic":        1.0,
        "mixup":         0.0,
        "copy_paste":    0.1,
        "hsv_h":         0.0,   # no hue shift — X-rays are pseudo-colour
        "hsv_s":         0.2,
        "hsv_v":         0.4,
        "fliplr":        0.5,
        "flipud":        0.0,
        "scale":         0.5,
        "translate":     0.1,
        "save_period":   10,
        "patience":      20,
    }

    def __init__(
        self,
        weights_path: str = "yolov12n.pt",
        conf_threshold: float = 0.65,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: Optional[str] = None,
        max_proposals: int = 100,
        output_dir: str = "outputs/finetune",
    ) -> None:
        if not _ULTRALYTICS_OK:
            raise ImportError(
                "ultralytics is required. Install: pip install ultralytics"
            )

        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.img_size       = img_size
        self.max_proposals  = max_proposals
        self.output_dir     = output_dir
        self.device         = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Track which weights are currently loaded
        self._weights_path  = weights_path
        self._finetuned     = False

        # Load the starting model
        self.model = _YOLO(weights_path)
        self.model.to(self.device)
        logger.info("Loaded weights: %s  (device=%s)", weights_path, self.device)

    # ── Fine-tuning ───────────────────────────────────────────────────────────

    def finetune(
        self,
        data_yaml: str,
        epochs: Optional[int] = None,
        freeze_epochs: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Fine-tune the model on an X-ray dataset in two phases.

        Phase 1 — frozen backbone:
            Only the detection head trains. Lets the head adapt to
            X-ray threat categories before the backbone starts moving.

        Phase 2 — full fine-tuning:
            All layers unfreeze. The backbone adapts to X-ray domain
            features at a low learning rate.

        After fine-tuning completes, the class automatically switches to
        the best fine-tuned weights for all subsequent propose() calls.

        Args:
            data_yaml     : path to the dataset YAML (e.g. opixray.yaml)
            epochs        : total epochs (overrides default 100)
            freeze_epochs : frozen-backbone epochs (overrides default 10)
            **kwargs      : override any other fine-tuning hyperparameter

        Returns:
            Path to best.pt fine-tuned weights file
        """
        cfg = dict(self._DEFAULT_FT)
        cfg["img_size"] = self.img_size
        if epochs        is not None: cfg["epochs"]        = epochs
        if freeze_epochs is not None: cfg["freeze_epochs"] = freeze_epochs
        cfg.update(kwargs)

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("Starting fine-tuning on: %s", data_yaml)
        logger.info("Epochs: %d  (freeze=%d)", cfg["epochs"], cfg["freeze_epochs"])

        # ── Phase 1: frozen backbone ──────────────────────────────────────
        if cfg["freeze_epochs"] > 0:
            logger.info("Phase 1: backbone frozen for %d epochs ...",
                        cfg["freeze_epochs"])
            self.model.train(
                data          = data_yaml,
                epochs        = cfg["freeze_epochs"],
                imgsz         = cfg["img_size"],
                batch         = cfg["batch_size"],
                workers       = cfg["num_workers"],
                optimizer     = cfg["optimizer"],
                lr0           = cfg["lr0"],
                lrf           = cfg["lrf"],
                momentum      = cfg["momentum"],
                weight_decay  = cfg["weight_decay"],
                warmup_epochs = cfg["warmup_epochs"],
                freeze        = cfg["freeze_layers"],
                mosaic        = cfg["mosaic"],
                mixup         = cfg["mixup"],
                copy_paste    = cfg["copy_paste"],
                hsv_h         = cfg["hsv_h"],
                hsv_s         = cfg["hsv_s"],
                hsv_v         = cfg["hsv_v"],
                fliplr        = cfg["fliplr"],
                flipud        = cfg["flipud"],
                scale         = cfg["scale"],
                translate     = cfg["translate"],
                project       = self.output_dir,
                name          = "phase1",
                save_period   = cfg["save_period"],
                exist_ok      = True,
                verbose       = True,
            )
            # Reload best phase-1 checkpoint before phase 2
            phase1_best = (
                Path(self.output_dir) / "phase1" / "weights" / "best.pt"
            )
            if phase1_best.exists():
                self.model = _YOLO(str(phase1_best))
                self.model.to(self.device)
                logger.info("Phase 1 done — loaded %s", phase1_best)

        # ── Phase 2: full fine-tuning ─────────────────────────────────────
        remaining = cfg["epochs"] - cfg["freeze_epochs"]
        logger.info("Phase 2: full fine-tuning for %d epochs ...", remaining)

        self.model.train(
            data          = data_yaml,
            epochs        = remaining,
            imgsz         = cfg["img_size"],
            batch         = cfg["batch_size"],
            workers       = cfg["num_workers"],
            optimizer     = cfg["optimizer"],
            lr0           = cfg["lr0"],
            lrf           = cfg["lrf"],
            momentum      = cfg["momentum"],
            weight_decay  = cfg["weight_decay"],
            warmup_epochs = 0,              # no warmup — resuming from phase 1
            freeze        = 0,              # unfreeze all layers
            mosaic        = cfg["mosaic"],
            mixup         = cfg["mixup"],
            copy_paste    = cfg["copy_paste"],
            hsv_h         = cfg["hsv_h"],
            hsv_s         = cfg["hsv_s"],
            hsv_v         = cfg["hsv_v"],
            fliplr        = cfg["fliplr"],
            flipud        = cfg["flipud"],
            scale         = cfg["scale"],
            translate     = cfg["translate"],
            project       = self.output_dir,
            name          = "phase2",
            patience      = cfg["patience"],
            save_period   = cfg["save_period"],
            exist_ok      = True,
            verbose       = True,
        )

        # Switch the active model to fine-tuned weights
        best_pt = Path(self.output_dir) / "phase2" / "weights" / "best.pt"
        if best_pt.exists():
            self.load_weights(str(best_pt))
            logger.info("Fine-tuning complete — now using %s", best_pt)
        else:
            logger.warning("best.pt not found at %s — keeping current weights",
                           best_pt)

        return str(best_pt)

    # ── Weight management ─────────────────────────────────────────────────────

    def load_weights(self, weights_path: str) -> None:
        """
        Load a different set of weights into the model.

        Use this to swap between COCO weights and fine-tuned weights,
        or to load a checkpoint from a previous fine-tuning run.

        Args:
            weights_path : path to a .pt weights file
        """
        self.model = _YOLO(weights_path)
        self.model.to(self.device)
        self._weights_path = weights_path
        self._finetuned    = "phase2" in weights_path or "best" in weights_path
        logger.info("Loaded weights: %s", weights_path)

    @property
    def is_finetuned(self) -> bool:
        """True if the model is currently running fine-tuned weights."""
        return self._finetuned

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, data_yaml: str) -> dict:
        """
        Run validation and return mAP metrics.

        Args:
            data_yaml : path to the dataset YAML used for fine-tuning

        Returns:
            dict with mAP50, mAP50-95, precision, recall
        """
        metrics = self.model.val(
            data=data_yaml, imgsz=self.img_size, verbose=True
        )
        results = {
            "mAP50":     metrics.box.map50,
            "mAP50-95":  metrics.box.map,
            "precision": metrics.box.mp,
            "recall":    metrics.box.mr,
        }
        logger.info("mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f",
                    results["mAP50"], results["mAP50-95"],
                    results["precision"], results["recall"])
        return results

    # ── Inference ─────────────────────────────────────────────────────────────

    def propose(self, image_path: str) -> list[dict]:
        """
        Run YOLOv12 on a single image and return filtered proposals.

        Uses fine-tuned weights if available (after calling finetune()),
        otherwise falls back to the weights loaded at construction time.

        Args:
            image_path : path to the X-ray image file

        Returns:
            List of proposal dicts, each containing:
                bbox  : [x1, y1, x2, y2] in pixel coordinates
                score : confidence score (float)
                label : predicted class index (int)
        """
        results = self.model.predict(
            source  = image_path,
            conf    = self.conf_threshold,
            iou     = self.iou_threshold,
            imgsz   = self.img_size,
            verbose = False,
            device  = self.device,
        )

        proposals = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes  = result.boxes.xyxy.cpu().tolist()        # [x1, y1, x2, y2]
            scores = result.boxes.conf.cpu().tolist()        # confidence scores
            labels = result.boxes.cls.cpu().int().tolist()   # class indices

            for bbox, score, label in zip(boxes, scores, labels):
                proposals.append({
                    "bbox":  bbox,
                    "score": score,
                    "label": label,
                })

        # Sort highest confidence first, then cap
        proposals.sort(key=lambda p: p["score"], reverse=True)
        return proposals[: self.max_proposals]

    def propose_batch(self, image_paths: list[str]) -> list[list[dict]]:
        """
        Run proposals on a list of images.

        Args:
            image_paths : list of image file paths

        Returns:
            List of proposal lists, one per image (same order as input)
        """
        return [self.propose(p) for p in image_paths]

    # ── String representation ─────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"YOLOv12ProposalGenerator(\n"
            f"  weights      = {self._weights_path}\n"
            f"  finetuned    = {self._finetuned}\n"
            f"  conf         = {self.conf_threshold}\n"
            f"  iou          = {self.iou_threshold}\n"
            f"  img_size     = {self.img_size}\n"
            f"  max_proposals= {self.max_proposals}\n"
            f"  device       = {self.device}\n"
            f")"
        )