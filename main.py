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
#   Step 2:  python main.py --stage 1 \ --dataset sixray \ --data_root "/path/to/SIXray" \ --stage1_dir outputs/sixray/stage1
#   Step 3:  python main.py --stage 2 \ --dataset sixray \ --data_root "/path/to/SIXray" \ --stage1_dir outputs/sixray/stage1 \ --stage2_dir outputs/sixray/stage2
#   Step 4:  python main.py --stage 3 \ --dataset sixray \ --data_root "/path/to/SIXray" \ --stage1_dir outputs/sixray/stage1 \ --stage2_dir outputs/sixray/stage2 \ --stage3_dir outputs/sixray/stage3
#   All:     python main.py --stage all \
#    --opixray_src "/path/to/OPI Xray" \
#    --data_root   "/path/to/SIXray" \
#    --stage1_dir  outputs/sixray/stage1 \
#    --stage2_dir  outputs/sixray/stage2 \
#    --stage3_dir  outputs/sixray/stage3

import argparse
import logging
import os
import sys
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
 
# Text prompts use "An X-ray of a <class>" — best format per Table 8 (paper §6.2.7)
DATASETS = {
    "sixray": {
        "num_classes": 5,
        "class_names": [
            "An X-ray of a gun",
            "An X-ray of a knife",
            "An X-ray of a wrench",
            "An X-ray of a pliers",
            "An X-ray of a scissors",
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
    p.add_argument("--subset",    default="SIXray10",
                   choices=["SIXray10", "SIXray100", "SIXray1000", "all"],
                   help="SIXray subset to use (ignored for CLCXray)")
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
    p.add_argument("--stage3_dir",           default="outputs/stage3")
    p.add_argument("--stage3_epochs",        type=int,   default=50)
    p.add_argument("--stage3_batch_size",    type=int,   default=8)
    p.add_argument("--stage3_lr",            type=float, default=1e-4)
    p.add_argument("--labeled_annotations",  default=None)
    p.add_argument("--fusion_dim",           type=int,   default=256)
    p.add_argument("--num_decoder_layers",   type=int,   default=3)
    p.add_argument("--pseudo_label_refresh", type=int,   default=10)
 
    p.add_argument("--eval", action="store_true")
    return p.parse_args()
 
 
# ── Helper: build datasets ────────────────────────────────────────────────────
 
def _build_datasets(args, labeled_only_flag=None):
    """Build labeled and unlabeled datasets matching your folder structure."""
    from data.datasets import SIXrayDataset, CLCXrayDataset
 
    if args.dataset == "sixray":
        labeled_ds   = SIXrayDataset(root=args.data_root, subset=args.subset,
                                     labeled_only=True,  img_size=224)
        unlabeled_ds = SIXrayDataset(root=args.data_root, subset=args.subset,
                                     labeled_only=False, img_size=224)
    else:
        labeled_ds   = CLCXrayDataset(root=args.data_root, labeled_only=True,  img_size=224)
        unlabeled_ds = CLCXrayDataset(root=args.data_root, labeled_only=False, img_size=224)
 
    return labeled_ds, unlabeled_ds
 
 
# ── Stage runners ─────────────────────────────────────────────────────────────
 
def run_prepare(args):
    if not args.opixray_src:
        raise ValueError("--opixray_src is required for --stage prepare")
    from finetune.data.prepare_opixray import prepare
    prepare(args.opixray_src, args.opixray_dst)
 
 
def run_finetune(args) -> str:
    from Stage_2.yolov12 import YOLOv12ProposalGenerator
    gen = YOLOv12ProposalGenerator(
        weights_path = f"yolov12{args.yolo_variant}.pt",
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
 
    from Stage_1.byol import BYOL
    from data.datasets import SIXrayDataset, CLCXrayDataset
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, ConcatDataset
    from utils.checkpoint import save_checkpoint, load_checkpoint
    from utils.metrics import AverageMeter
    from pathlib import Path
 
    os.makedirs(args.stage1_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Stage 1 — BYOL pre-training on %s", device)
 
    labeled_ds, unlabeled_ds = _build_datasets(args)
    logger.info("Labeled: %d  |  Unlabeled: %d", len(labeled_ds), len(unlabeled_ds))
 
    loader = DataLoader(
        ConcatDataset([labeled_ds, unlabeled_ds]),
        batch_size  = args.batch_size or 32,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
        drop_last   = True,
    )
 
    model = BYOL(
        backbone_variant    = args.backbone,
        backbone_pretrained = not args.no_pretrain,
        ema_decay           = 0.996,
    ).to(device)
 
    # Resume from checkpoint if available
    start_epoch = 0
    ckpt_path   = Path(args.stage1_dir) / "last.pth"
    if ckpt_path.exists() and not args.no_resume:
        start_epoch = load_checkpoint(model, str(ckpt_path))
        logger.info("Resumed from epoch %d", start_epoch)
 
    epochs    = args.epochs or 50
    optimizer = optim.AdamW(model.trainable_parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
 
    # Mixed-precision training (paper §5.1)
    use_amp = torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
 
    best_loss = float("inf")
    for epoch in range(start_epoch, epochs):
        model.train()
        meter = AverageMeter()
 
        for batch in loader:
            v1 = batch["v1"].to(device, non_blocking=True)
            v2 = batch["v2"].to(device, non_blocking=True)
            optimizer.zero_grad()
 
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, _, _ = model(v1, v2)
 
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            model.update_target()
            meter.update(loss.item(), v1.size(0))
 
        scheduler.step()
        logger.info("Epoch [%d/%d]  loss=%.4f  lr=%.6f",
                    epoch + 1, epochs, meter.avg, scheduler.get_last_lr()[0])
 
        is_best   = meter.avg < best_loss
        best_loss = min(meter.avg, best_loss)
        save_checkpoint(model, optimizer, epoch + 1, meter.avg, args.stage1_dir, is_best)
 
    logger.info("Stage 1 done. Best loss: %.4f", best_loss)
 
 
def run_stage2(args, num_classes):
    if not args.data_root:
        raise ValueError("--data_root is required for stage 2")
 
    import json
    import torch
    from pathlib import Path
    from Stage_1.byol import BYOL
    from Stage_2.yolov12 import YOLOv12ProposalGenerator
    from Stage_2.pseudo_labeler import PseudoLabelGenerator
    from Stage_2.stage2_config import STAGE2_CONFIG
    from utils.checkpoint import load_checkpoint
 
    os.makedirs(args.stage2_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Stage 2 — Pseudo-label generation  (device: %s)", device)
 
    # Load BYOL from Stage 1 checkpoint
    byol = BYOL(
        backbone_variant    = args.backbone,
        backbone_pretrained = False,
    )
    stage1_ckpt = str(Path(args.stage1_dir) / "best.pth")
    if not Path(stage1_ckpt).exists():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found at {stage1_ckpt}.\n"
            "Run stage 1 first: python main.py --stage 1 --data_root ..."
        )
    load_checkpoint(byol, stage1_ckpt)
    logger.info("Loaded Stage 1 encoder from %s", stage1_ckpt)
 
    # Load YOLOv12
    cfg          = dict(STAGE2_CONFIG)
    yolo_weights = (
        args.yolo_weights
        or f"{args.finetune_dir}/finetune/weights/best.pt"
    )
    if not Path(yolo_weights).exists():
        logger.warning("YOLOv12 weights not found — falling back to yolov12n.pt")
        yolo_weights = "yolov12n.pt"
 
    proposal_gen = YOLOv12ProposalGenerator(
        weights_path   = yolo_weights,
        conf_threshold = cfg.get("conf_threshold", 0.65),
        iou_threshold  = cfg.get("iou_threshold",  0.45),
        img_size       = cfg.get("img_size",        640),
        device         = device,
        max_proposals  = cfg.get("max_proposals",   100),
    )
 
    # Build pseudo-label generator
    generator = PseudoLabelGenerator(
        byol_model     = byol,
        num_classes    = num_classes,
        proposal_gen   = proposal_gen,
        conf_threshold = cfg.get("conf_threshold", 0.65),
        device         = device,
    )
 
    labeled_ds, unlabeled_ds = _build_datasets(args)
 
    # Train classification head on labeled set
    logger.info("Training classification head on %d labeled samples ...",
                len(labeled_ds))
    generator.fit_head(labeled_ds,
                       epochs=cfg.get("head_epochs", 20),
                       lr=cfg.get("head_lr", 1e-3))
 
    torch.save(generator.head.state_dict(),
               os.path.join(args.stage2_dir, "head.pth"))
 
    # Generate pseudo-labels for all unlabeled images
    logger.info("Generating pseudo-labels for %d unlabeled images ...",
                len(unlabeled_ds))
    pseudo_labels = generator.generate_all(unlabeled_ds)
 
    json_path = os.path.join(args.stage2_dir, "pseudo_labels.json")
    with open(json_path, "w") as f:
        json.dump(pseudo_labels, f, indent=2)
    logger.info("Saved %d pseudo-labels to %s", len(pseudo_labels), json_path)
 
 
def run_stage3(args, num_classes, class_names):
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
            "Run Stage 2 first: python main.py --stage 2 --data_root ..."
        )
    if not Path(stage1_ckpt).exists():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found at {stage1_ckpt}.\n"
            "Run Stage 1 first: python main.py --stage 1 --data_root ..."
        )
 
    def _build_loader():
        ds = Stage3Dataset(
            pseudo_label_file   = pseudo_label_file,
            labeled_annotations = args.labeled_annotations,
            img_size            = 224,
            crop_size           = 224,
            max_proposals       = 100,
            return_crops        = True,
        )
        return DataLoader(ds, batch_size=args.stage3_batch_size,
                          shuffle=True, num_workers=4,
                          collate_fn=stage3_collate_fn, pin_memory=True)
 
    loader = _build_loader()
    logger.info("Stage 3 dataset: %d samples", len(loader.dataset))
 
    model = VisionLanguageDetector(
        num_classes        = num_classes,
        backbone_variant   = args.backbone,
        fusion_dim         = args.fusion_dim,
        num_decoder_layers = args.num_decoder_layers,
        stage1_ckpt        = stage1_ckpt,
        class_names        = class_names,
    ).to(device)
 
    criterion = DetectionLoss(num_classes=num_classes)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.stage3_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.stage3_epochs, eta_min=1e-6
    )
 
    # Mixed-precision training (paper §5.1)
    use_amp = torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
 
    best_loss = float("inf")
    for epoch in range(args.stage3_epochs):
 
        # Regenerate pseudo-labels every N epochs (paper §4.3)
        if epoch > 0 and epoch % args.pseudo_label_refresh == 0:
            logger.info("Epoch %d — regenerating pseudo-labels ...", epoch + 1)
            run_stage2(args, num_classes)
            loader = _build_loader()
 
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
 
            with torch.cuda.amp.autocast(enabled=use_amp):
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
 
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
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
 
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch":      epoch + 1,
                "loss":       avg_loss,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
            }, str(Path(args.stage3_dir) / "best.pth"))
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
 