# Stages:
#   prepare  — convert OPIXray to YOLO format
#   finetune — fine-tune YOLOv12 on OPIXray
#   1        — BYOL self-supervised pre-training
#   2        — pseudo-label generation
#   3        — vision-language detection training
#   all      — finetune -> 1 -> 2 -> 3  end-to-end
#
# Quick start:
#   Step 0:  python main.py --stage prepare --opixray_src "/path/to/OPI Xray"
#   Step 1:  python main.py --stage finetune --data_yaml opixray_yolo/opixray.yaml
#   Step 2:  python main.py --stage 1 --data_root "/path/to/SIXray"
#   Step 3:  python main.py --stage 2 --data_root "/path/to/SIXray"
#   Step 4:  python main.py --stage 3 --data_root "/path/to/SIXray"
#   All:     python main.py --stage all \
#                --opixray_src "/path/to/OPI Xray" \
#                --data_root   "/path/to/SIXray"

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Dataset registry ──────────────────────────────────────────────────────────
# Text prompts use "An X-ray of a <class>" — best format per Table 8 (paper §6.2.7)

DATASETS = {
    "sixray": {
        "num_classes": 6,
        "class_names": [
            "An X-ray of a gun",
            "An X-ray of a knife",
            "An X-ray of a wrench",
            "An X-ray of a pliers",
            "An X-ray of a scissors",
            "An X-ray of a hammer",
        ],
    },
    "clcxray": {
        "num_classes": 12,
        "class_names": [
            "An X-ray of a scissors",
            "An X-ray of a knife",
            "An X-ray of a dagger",
            "An X-ray of a blade",
            "An X-ray of a swiss army knife",
            "An X-ray of a spray cans",
            "An X-ray of a vacuum cup",
            "An X-ray of a plastic bottle",
            "An X-ray of a glass bottle",
            "An X-ray of a carton drinks",
            "An X-ray of a cans",
            "An X-ray of a tin",
        ],
    },
}


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PRISM-X pipeline")

    p.add_argument("--stage", default="all",
                   choices=["prepare", "finetune", "1", "2", "3", "all"])
    p.add_argument("--dataset",   choices=list(DATASETS), default="sixray")
    p.add_argument("--data_root", default=None)
    p.add_argument("--backbone",  default="swin_v2_t",
                   choices=["swin_v2_t", "swin_v2_s", "swin_v2_b"])

    # OPIXray preparation
    p.add_argument("--opixray_src", default=None)
    p.add_argument("--opixray_dst", default="opixray_yolo")

    # Fine-tuning
    p.add_argument("--data_yaml",       default="opixray_yolo/opixray.yaml")
    p.add_argument("--yolo_variant",    default="n",
                   choices=["n", "s", "m", "l", "x"])
    p.add_argument("--finetune_dir",    default="outputs/finetune")
    p.add_argument("--finetune_epochs", type=int, default=None)

    # Stage 1
    p.add_argument("--stage1_dir",  default="outputs/stage1")
    p.add_argument("--epochs",      type=int, default=None)
    p.add_argument("--batch_size",  type=int, default=None)
    p.add_argument("--no_pretrain", action="store_true")
    p.add_argument("--no_resume",   action="store_true")

    # Stage 2
    p.add_argument("--stage2_dir",   default="outputs/stage2")
    p.add_argument("--yolo_weights", default=None)

    # Stage 3
    p.add_argument("--stage3_dir",          default="outputs/stage3")
    p.add_argument("--stage3_epochs",       type=int,   default=50)
    p.add_argument("--stage3_batch_size",   type=int,   default=8)
    p.add_argument("--stage3_lr",           type=float, default=1e-4)
    p.add_argument("--labeled_annotations", default=None)
    p.add_argument("--fusion_dim",          type=int,   default=256)
    p.add_argument("--num_decoder_layers",  type=int,   default=3)
    p.add_argument("--pseudo_label_refresh",type=int,   default=10,
                   help="Regenerate pseudo-labels every N epochs (paper: 10)")

    p.add_argument("--eval", action="store_true")
    return p.parse_args()


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_prepare(args):
    if not args.opixray_src:
        raise ValueError("--opixray_src is required for --stage prepare")
    from finetune.data.prepare_opixray import prepare
    prepare(args.opixray_src, args.opixray_dst)


def run_finetune(args) -> str:
    from Stage_2.yolov12 import YOLOv12ProposalGenerator
    gen = YOLOv12ProposalGenerator(
        weights_path  = f"yolo12{args.yolo_variant}.pt",
        output_dir   = args.finetune_dir,
    )
    best_pt = gen.finetune(
        data_yaml = args.data_yaml,
        epochs    = args.finetune_epochs,
    )
    if args.eval:
        gen.evaluate(args.data_yaml)
    return best_pt


def run_stage1(args, num_classes):
    if not args.data_root:
        raise ValueError("--data_root is required for stage 1")

    from data.datasets import SIXrayDataset, CLCXrayDataset
    from Stage_1.byol import train_byol

    cfg = {
        "num_classes":         num_classes,
        "backbone_variant":    args.backbone,
        "backbone_pretrained": not args.no_pretrain,
        "resume":              not args.no_resume,
        "epochs":              args.epochs     or 50,
        "batch_size":          args.batch_size or 32,
        "num_workers":         4,
        "lr":                  1e-4,
        "lr_min":              1e-6,
        "weight_decay":        1e-4,
        "grad_clip":           1.0,
        "ema_decay":           0.996,
        "img_size":            224,
    }

    DatasetCls   = SIXrayDataset if args.dataset == "sixray" else CLCXrayDataset
    labeled_ds   = DatasetCls(root=args.data_root, labeled_only=True,  img_size=224)
    unlabeled_ds = DatasetCls(root=args.data_root, labeled_only=False, img_size=224)
    logger.info("Labeled: %d  |  Unlabeled: %d", len(labeled_ds), len(unlabeled_ds))

    train_byol(labeled_ds, unlabeled_ds, cfg, output_dir=args.stage1_dir)


def run_stage2(args, num_classes):
    if not args.data_root:
        raise ValueError("--data_root is required for stage 2")

    from Stage_2.stage2_config import STAGE2_CONFIG
    from data.datasets import SIXrayDataset, CLCXrayDataset
    from Stage_2.pseudo_labeler import run_stage2 as _run

    DatasetCls = SIXrayDataset if args.dataset == "sixray" else CLCXrayDataset
    cfg = dict(STAGE2_CONFIG)
    cfg["num_classes"]      = num_classes
    cfg["backbone_variant"] = args.backbone
    cfg["yolo_weights"]     = (
        args.yolo_weights
        or f"{args.finetune_dir}/finetune/weights/best.pt"
    )

    labeled_ds   = DatasetCls(root=args.data_root, labeled_only=True,  img_size=224)
    unlabeled_ds = DatasetCls(root=args.data_root, labeled_only=False, img_size=224)

    _run(labeled_ds, unlabeled_ds, cfg,
         stage1_dir = args.stage1_dir,
         output_dir = args.stage2_dir)


def run_stage3(args, num_classes, class_names):
    """
    Stage 3 — Vision-Language Detection training.

    Pseudo-labels are regenerated every pseudo_label_refresh epochs
    (default 10) using the updated BYOL encoder — paper §4.3.
    """
    import os
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from pathlib import Path

    from Stage_3.vision_language_detector import VisionLanguageDetector
    from Stage_3.vlc import DetectionLoss
    from Stage_3.stage3_dataset import Stage3Dataset, stage3_collate_fn

    os.makedirs(args.stage3_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Stage 3 — Vision-Language Detection  (device: %s)", device)

    pseudo_label_file = str(Path(args.stage2_dir) / "pseudo_labels.json")
    stage1_ckpt       = str(Path(args.stage1_dir) / "best.pth")

    if not Path(pseudo_label_file).exists():
        raise FileNotFoundError(
            f"pseudo_labels.json not found at {pseudo_label_file}.\n"
            "Run Stage 2 first:  python main.py --stage 2 --data_root ..."
        )
    if not Path(stage1_ckpt).exists():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found at {stage1_ckpt}.\n"
            "Run Stage 1 first:  python main.py --stage 1 --data_root ..."
        )

    # ── Helper to build DataLoader from current pseudo_labels.json ───────────
    def _build_loader():
        ds = Stage3Dataset(
            pseudo_label_file   = pseudo_label_file,
            labeled_annotations = args.labeled_annotations,
            img_size            = 224,
            crop_size           = 224,
            max_proposals       = 100,
            return_crops        = True,
        )
        return DataLoader(
            ds,
            batch_size  = args.stage3_batch_size,
            shuffle     = True,
            num_workers = 4,
            collate_fn  = stage3_collate_fn,
            pin_memory  = True,
        )

    loader = _build_loader()
    logger.info("Stage 3 dataset: %d samples", len(loader.dataset))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = VisionLanguageDetector(
        num_classes        = num_classes,
        backbone_variant   = args.backbone,
        fusion_dim         = args.fusion_dim,
        num_decoder_layers = args.num_decoder_layers,
        stage1_ckpt        = stage1_ckpt,
        class_names        = class_names,   # "An X-ray of a <class>" prompts
    ).to(device)

    criterion = DetectionLoss(num_classes=num_classes)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.stage3_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.stage3_epochs, eta_min=1e-6
    )

    best_loss = float("inf")

    for epoch in range(args.stage3_epochs):

        # ── Progressive pseudo-label refinement (paper §4.3) ─────────────────
        # Regenerate every pseudo_label_refresh epochs (default 10)
        if epoch > 0 and epoch % args.pseudo_label_refresh == 0:
            logger.info(
                "Epoch %d — regenerating pseudo-labels (paper §4.3) ...",
                epoch + 1
            )
            run_stage2(args, num_classes)   # overwrites pseudo_labels.json
            loader = _build_loader()         # reload dataset with fresh labels
            logger.info("Pseudo-labels refreshed. Continuing training ...")

        # ── Training step ─────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in loader:
            images        = batch["images"].to(device)
            region_imgs   = batch["region_imgs"].to(device)
            bbox_coords   = batch["bbox_coords"].to(device)
            pseudo_labels = batch["pseudo_labels"].to(device)
            gt_boxes      = batch["gt_boxes"].to(device)
            gt_labels     = batch["gt_labels"].to(device)
            valid_mask    = batch["valid_mask"].to(device)

            B, K, C, h, w = region_imgs.shape
            optimizer.zero_grad()

            out = model(
                images          = images,
                bbox_proposals  = bbox_coords,
                pseudo_labels   = pseudo_labels,
                region_images   = region_imgs.view(B * K, C, h, w),
            )

            losses = criterion(
                pred_logits   = out["class_logits"],
                pred_boxes    = out["pred_boxes"],
                target_labels = gt_labels,
                target_boxes  = gt_boxes,
                visual_embeds = out["visual_embeds"],
                text_embeds   = out["text_embeds"],
                valid_mask    = valid_mask,
            )

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            epoch_loss += losses["total"].item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        logger.info(
            "Epoch [%d/%d]  loss=%.4f  cls=%.4f  bbox=%.4f  giou=%.4f  vlc=%.4f",
            epoch + 1, args.stage3_epochs, avg_loss,
            losses["cls"].item(), losses["bbox"].item(),
            losses["giou"].item(), losses["vlc"].item(),
        )

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_path  = str(Path(args.stage3_dir) / "best.pth")
            torch.save({
                "epoch":      epoch + 1,
                "loss":       avg_loss,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
            }, best_path)
            logger.info("  → saved best checkpoint (loss=%.4f)", avg_loss)

    logger.info("Stage 3 complete. Best loss: %.4f", best_loss)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args        = parse_args()
    meta        = DATASETS[args.dataset]
    num_classes = meta["num_classes"]
    class_names = meta["class_names"]

    if args.stage == "prepare":
        run_prepare(args)
    elif args.stage == "finetune":
        run_finetune(args)
    elif args.stage == "1":
        run_stage1(args, num_classes)
    elif args.stage == "2":
        run_stage2(args, num_classes)
    elif args.stage == "3":
        run_stage3(args, num_classes, class_names)
    elif args.stage == "all":
        if args.opixray_src:
            run_prepare(args)
        best_pt = run_finetune(args)
        args.yolo_weights = best_pt
        run_stage1(args, num_classes)
        run_stage2(args, num_classes)
        run_stage3(args, num_classes, class_names)


if __name__ == "__main__":
    main()