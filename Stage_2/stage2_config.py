STAGE2_CONFIG = {
    # ── YOLOv12 proposal generator ────────────────────────────────────────
   
    # If fine-tuningis skipped, "yolov12n.pt" for COCO weights.
    "yolo_weights":   "outputs/finetune/phase2/weights/best.pt",
    "conf_threshold": 0.65,    # paper §4.3: keep proposals above 0.65
    "iou_threshold":  0.45,    # NMS — merge boxes overlapping more than 45%
    "img_size":       640,     # YOLOv12 inference resolution
    "max_proposals":  100,     # hard cap on boxes per image

    # ── Classification head h_φ ───────────────────────────────────────────
    # Lightweight linear layer trained on labeled BYOL features.
    "head_epochs":    20,
    "head_lr":        1e-3,

    # ── Shared ────────────────────────────────────────────────────────────
    "num_classes":       6,          # updated by main.py per dataset
    "backbone_variant":  "swin_v2_t",
}