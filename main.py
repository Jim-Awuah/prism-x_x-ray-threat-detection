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
#   Step 4:  python main.py --stage 3 \ --dataset sixray \ --data_root "/Users/dersunscheinyn/SIXray_dataset/OpenDataLab___SIXray/raw/SIXray" \ --stage1_dir outputs/sixray/stage1 \ --stage2_dir outputs/sixray/stage2 \ --stage3_dir outputs/sixray/stage3
#   All:     python main.py --stage all \
#    --opixray_src "/path/to/OPI Xray" \
#    --data_root   "/path/to/SIXray" \
#    --stage1_dir  outputs/sixray/stage1 \
#    --stage2_dir  outputs/sixray/stage2 \
#    --stage3_dir  outputs/sixray/stage3
#
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

# Stage 3 defaults are sourced from Stage_3/stage3_config.py so the CLI and
# the documented config can never drift apart again (this is what caused
# --stage3_batch_size to silently default to 8 instead of the paper's 32).
# Falls back to paper-matching hardcoded values only if the config module
# can't be imported (e.g. running main.py outside the project root).
try:
    from Stage_3.stage3_config import STAGE3_CONFIG
except Exception as _cfg_err:
    STAGE3_CONFIG = {}
    logging.getLogger(__name__).warning(
        "Could not import Stage_3.stage3_config (%s) — falling back to "
        "hardcoded CLI defaults for Stage 3. Run main.py from the project "
        "root so config values stay the single source of truth.", _cfg_err
    )

 
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
    # Order MUST match EDSDataset.CLASS_NAMES in data/datasets.py and
    # EDS_CLASS_NAMES in data/prepare_labeled_annotations.py index-for-index:
    #   0 device        1 glassbottle   2 knife     3 laptop    4 lighter
    #   5 plasticbottle 6 powerbank     7 pressure  8 scissor   9 umbrella
    "eds": {
        "num_classes": 10,
        "class_names": [
            "An X-ray of a device",
            "An X-ray of a glass bottle",
            "An X-ray of a knife",
            "An X-ray of a laptop",
            "An X-ray of a lighter",
            "An X-ray of a plastic bottle",
            "An X-ray of a powerbank",
            "An X-ray of a pressure",
            "An X-ray of a scissor",
            "An X-ray of an umbrella",
        ],
    },
}
 
 
# Argument parser 
 
def parse_args():
    p = argparse.ArgumentParser(description="PRISM-X pipeline")
 
    p.add_argument("--stage", default="all",
                   choices=["prepare", "finetune", "1", "2", "3", "all"])
    p.add_argument("--dataset",   choices=list(DATASETS), default="sixray")
    p.add_argument("--data_root", default=None)
    p.add_argument("--subset",    default="SIXray10",
                   choices=["SIXray10", "SIXray100", "SIXray1000", "all"],
                   help="SIXray subset to use (ignored for CLCXray and EDS)")
    p.add_argument("--domain",    default="all",
                   choices=["domain1", "domain2", "domain3", "all"],
                   help="EDS acquisition domain (ignored for SIXray and "
                        "CLCXray). EDS ships as three domains rather than "
                        "train/test splits — use a single domain to train "
                        "under the domain-generalisation protocol, or 'all' "
                        "to pool every domain.")
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
    p.add_argument("--stage3_epochs",        type=int,   default=STAGE3_CONFIG.get("epochs", 50))
    p.add_argument("--stage3_batch_size",    type=int,   default=STAGE3_CONFIG.get("batch_size", 32),
                   help="Paper §5.1 uses batch_size=32. Previously defaulted to 8 here — "
                        "now sourced from stage3_config.py so it can't silently drift again.")
    p.add_argument("--stage3_lr",            type=float, default=STAGE3_CONFIG.get("lr", 1e-4))
    p.add_argument("--labeled_annotations",  default=None)
    p.add_argument("--eval_labeled_annotations", default=None,
                   help="Labeled annotations to EVALUATE on, when different "
                        "from the training set. This is what the EDS "
                        "cross-domain protocol (paper Table 4) needs: train "
                        "on a source domain, evaluate on a different target "
                        "domain. Defaults to --labeled_annotations "
                        "(in-domain evaluation, as used for SIXray/CLCXray "
                        "in Tables 2 and 3).")
    p.add_argument("--eval_pseudo_labels", default=None,
                   help="pseudo_labels.json for the EVAL domain. Region "
                        "proposals must come from the images being "
                        "evaluated, so cross-domain eval needs Stage 2 run "
                        "on the target domain too. Defaults to the training "
                        "domain's <stage2_dir>/pseudo_labels.json.")
    p.add_argument("--fusion_dim",           type=int,   default=STAGE3_CONFIG.get("fusion_dim", 256))
    p.add_argument("--num_decoder_layers",   type=int,   default=STAGE3_CONFIG.get("num_decoder_layers", 3))
    p.add_argument("--pseudo_label_refresh", type=int,   default=STAGE3_CONFIG.get("pseudo_refresh_every", 10),
                   help="Regenerate pseudo-labels every N epochs (paper: 10). "
                        "On MPS each refresh takes hours (full Stage 2 re-run "
                        "on 74k images). Use 25 or 50 to reduce frequency, "
                        "or --no_pseudo_refresh to disable entirely.")
    p.add_argument("--no_pseudo_refresh", action="store_true", default=False,
                   help="Disable mid-training pseudo-label regeneration. "
                        "Fastest option for MPS: use the Stage 2 labels "
                        "generated before training and never re-run Stage 2. "
                        "Slightly reduces final accuracy vs the paper's "
                        "every-10-epoch refresh, but avoids multi-hour "
                        "Stage 2 re-runs on MPS mid-training.")
 
    p.add_argument("--eval", action="store_true")
    return p.parse_args()
 
 
# Helper: build datasets 
def _build_datasets(args, labeled_only_flag=None):
    """Build labeled and unlabeled datasets matching your folder structure."""
    from data.datasets import SIXrayDataset, CLCXrayDataset

    if args.dataset == "sixray":
        labeled_ds   = SIXrayDataset(root=args.data_root, subset=args.subset,
                                     labeled_only=True,  img_size=224)
        unlabeled_ds = SIXrayDataset(root=args.data_root, subset=args.subset,
                                     labeled_only=False, img_size=224)
    elif args.dataset == "clcxray":
        labeled_ds   = CLCXrayDataset(root=args.data_root, labeled_only=True,  img_size=224)
        unlabeled_ds = CLCXrayDataset(root=args.data_root, labeled_only=False, img_size=224)
    elif args.dataset == "eds":
        try:
            from data.datasets import EDSDataset
        except ImportError:
            raise ImportError(
                "EDSDataset not found in data/datasets.py. "
                "Copy the updated datasets.py file:\n"
                "  cp ~/Downloads/datasets.py data/datasets.py"
            )
        labeled_ds   = EDSDataset(root=args.data_root, domain=args.domain,
                                  labeled_only=True,  img_size=224)
        unlabeled_ds = EDSDataset(root=args.data_root, domain=args.domain,
                                  labeled_only=False, img_size=224)
        logger.info("EDS domain(s): %s  -  %d labeled / %d total images",
                    args.domain, len(labeled_ds), len(unlabeled_ds))
    else:
        raise ValueError(
            f"Unknown dataset {args.dataset!r}. Known datasets: "
            f"{list(DATASETS)}"
        )

    return labeled_ds, unlabeled_ds
 
 
# Stage runners 
 
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
    device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
    )
    print(f"Using device: {device}")
    logger.info("Stage 1 — BYOL pre-training on %s", device)
 
    labeled_ds, unlabeled_ds = _build_datasets(args)
    logger.info("Labeled: %d  |  Unlabeled: %d", len(labeled_ds), len(unlabeled_ds))
 
    loader = DataLoader(
        ConcatDataset([labeled_ds, unlabeled_ds]),
        batch_size  = args.batch_size or 32,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = torch.cuda.is_available(),
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

    # Mixed-precision training (paper §5.1). Only well-supported on CUDA —
    # MPS's autocast op coverage is still incomplete/experimental in
    # mainstream PyTorch, so on MPS/CPU we intentionally train in full FP32
    # rather than risk silent numerical issues from partial autocast support.
    use_amp = (device == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger.info(
        "Precision: %s  (paper uses mixed-precision on CUDA; %s)",
        "mixed (fp16 autocast)" if use_amp else "full fp32",
        "matches paper" if use_amp else
        f"device is '{device}', not CUDA — falling back to fp32 as the stable option",
    )
 
    best_loss = float("inf")
    for epoch in range(start_epoch, epochs):
        model.train()
        meter = AverageMeter()
 
        for batch in loader:
            v1 = batch["v1"].to(device, non_blocking=True)
            v2 = batch["v2"].to(device, non_blocking=True)
            optimizer.zero_grad()
 
            with torch.amp.autocast("cuda", enabled=use_amp):
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
    device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
    )
    print(f"Using device: {device}")
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
    YOLO_WEIGHTS_PATH = (
        "/Users/dersunscheinyn/Desktop/prism-x_x-ray-threat-detection"
        "/runs/detect/outputs/finetune/finetune/weights/best.pt"
    )
    yolo_weights = args.yolo_weights or YOLO_WEIGHTS_PATH
    if not Path(yolo_weights).exists():
        logger.warning("YOLOv12 weights not found at %s — falling back to yolo12n.pt",
                       yolo_weights)
        yolo_weights = "yolo12n.pt"
 
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
    device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
    )
    print(f"Using device: {device}")
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
        # num_workers=0 on MPS: multiple worker processes cause allocator
        # contention on Apple Silicon and trigger MallocStackLogging spam.
        # On CUDA, 4 workers are fine.
        nw = 0 if (device == "mps") else 4
        return DataLoader(ds, batch_size=args.stage3_batch_size,
                          shuffle=True, num_workers=nw,
                          collate_fn=stage3_collate_fn,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=False)
 
    loader = _build_loader()
    logger.info("Stage 3 dataset: %d samples", len(loader.dataset))

    # Report how many labeled samples are actually feeding training, and
    # whether that matches the paper's fixed-1000-sample protocol (§5.1).
    # This is informational only — it does not change what was already
    # written by prepare_labeled_annotations.py — but makes any resulting
    # numbers self-documenting rather than silently non-comparable.
    if args.labeled_annotations and Path(args.labeled_annotations).exists():
        try:
            import json as _json
            with open(args.labeled_annotations) as _f:
                _n_labeled = len(_json.load(_f))
            _paper_n = STAGE3_CONFIG.get("num_labeled_samples", 1000)
            logger.info(
                "Labeled samples: %d  (paper §5.1 protocol: %d fixed labeled "
                "samples; %s)",
                _n_labeled, _paper_n,
                "matches paper" if _n_labeled == _paper_n else
                "DIFFERS from paper — re-run prepare_labeled_annotations.py "
                f"with --num_labeled {_paper_n} to match, or note the "
                "deviation when comparing results",
            )
        except Exception:
            pass  # purely informational logging; never block training on it
 
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
 
    # Mixed-precision training (paper §5.1). Only well-supported on CUDA —
    # MPS's autocast op coverage is still incomplete/experimental in
    # mainstream PyTorch, so on MPS/CPU we intentionally train in full FP32
    # rather than risk silent numerical issues from partial autocast support.
    use_amp = (device == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger.info(
        "Precision: %s  (paper uses mixed-precision on CUDA; %s)",
        "mixed (fp16 autocast)" if use_amp else "full fp32",
        "matches paper" if use_amp else
        f"device is '{device}', not CUDA — falling back to fp32 as the stable option",
    )
    logger.info(
        "Stage 3 batch_size=%d  (paper §5.1 uses 32; source: %s)",
        args.stage3_batch_size,
        "STAGE3_CONFIG" if "batch_size" in STAGE3_CONFIG else "hardcoded fallback",
    )
 
    best_loss = float("inf")
    for epoch in range(args.stage3_epochs):
 
        # Regenerate pseudo-labels every N epochs (paper §4.3)
        do_refresh = (
            not getattr(args, "no_pseudo_refresh", False)
            and epoch > 0
            and args.pseudo_label_refresh > 0
            and epoch % args.pseudo_label_refresh == 0
        )
        if do_refresh:
            logger.info(
                "Epoch %d — regenerating pseudo-labels "
                "(suppress with --no_pseudo_refresh, or increase interval "
                "with --pseudo_label_refresh N) ...", epoch + 1
            )
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
 
            with torch.amp.autocast("cuda", enabled=use_amp):
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
            # Cache scalar values immediately so no tensor/graph reference
            # survives past this batch step. On MPS, holding graph refs
            # across batches accumulates memory and causes the progressive
            # slowdown (each epoch takes longer than the last).
            last_loss = {k: v.item() for k, v in losses.items()}
 
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        # Flush MPS allocator cache. Without this, the MPS memory allocator
        # accumulates fragmented buffers across epochs — allocation gets
        # progressively slower (visible as each epoch taking longer than
        # the last). This is the MPS equivalent of torch.cuda.empty_cache().
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        logger.info(
            "Epoch [%d/%d]  loss=%.4f  cls=%.4f  bbox=%.4f  giou=%.4f  vlc=%.4f",
            epoch + 1, args.stage3_epochs, avg_loss,
            last_loss.get("cls", 0.0), last_loss.get("bbox", 0.0),
            last_loss.get("giou", 0.0), last_loss.get("vlc", 0.0),
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

    # ── Final evaluation (paper §5.1 — mAP, AP50, AP75, Grounding Accuracy) ──
    #
    # IMPORTANT: mAP/AP50/AP75 and Grounding Accuracy are DELIBERATELY two
    # separate computations that must never be merged into one:
    #
    #   - mAP/AP50/AP75 : standard COCO-style detection metrics. ALL real
    #     (non-padded) proposals per image are treated as candidate
    #     detections, ranked by confidence, and matched against ALL GT
    #     boxes for that image via torchmetrics' MeanAveragePrecision
    #     (which wraps pycocotools under the hood). This rewards a model
    #     that ranks correct detections above false positives.
    #
    #   - Grounding Accuracy : a per-OBJECT recall metric. For each
    #     individual ground-truth box, we ask "does *any* real proposal
    #     predict the correct class with IoU >= 0.5 against THIS box?" —
    #     independent of confidence ranking or how many false positives
    #     the model also produced elsewhere in the image.
    #
    #   These measure different things and WILL differ numerically (see
    #   the reference paper: AP50=87.4% vs Grounding Acc=48.3% — a ~39pt
    #   gap). If a future edit makes these two come out identical again,
    #   that is a strong signal the two computations have been accidentally
    #   collapsed into one — check for a shared top-1-only prediction loop
    #   before trusting the numbers.
    logger.info("Running final evaluation on best checkpoint ...")

    # Load best checkpoint
    best_ckpt = str(Path(args.stage3_dir) / "best.pth")
    if Path(best_ckpt).exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        logger.info("Loaded best checkpoint from epoch %d", ckpt["epoch"])
    model.eval()

    # Build test dataset using labeled annotations only.
    #
    # For SIXray/CLCXray (paper Tables 2, 3) evaluation is in-domain, so
    # eval_ann falls back to the training annotations. For the EDS
    # cross-domain benchmark (paper Table 4) the model trains on a source
    # domain and is evaluated on a DIFFERENT target domain, which is what
    # --eval_labeled_annotations / --eval_pseudo_labels are for. Note that
    # Table 4 reports grounding accuracy only — mAP/AP50/AP75 are not
    # reported for EDS in the paper.
    eval_ann    = args.eval_labeled_annotations or args.labeled_annotations
    eval_pseudo = args.eval_pseudo_labels or pseudo_label_file

    if eval_ann and Path(eval_ann).exists():
        if args.eval_labeled_annotations:
            logger.info(
                "CROSS-DOMAIN evaluation: trained on %s, evaluating on %s",
                args.labeled_annotations, eval_ann,
            )
            if not args.eval_pseudo_labels:
                logger.warning(
                    "--eval_labeled_annotations was set but "
                    "--eval_pseudo_labels was not. Region proposals will "
                    "come from the TRAINING domain's pseudo_labels.json, "
                    "which does not contain the target domain's images — "
                    "most eval images will have no proposals and the "
                    "reported numbers will be wrong. Run Stage 2 on the "
                    "target domain and pass its pseudo_labels.json."
                )
        from Stage_3.stage3_dataset import Stage3Dataset, stage3_collate_fn
        eval_ds = Stage3Dataset(
            pseudo_label_file   = eval_pseudo,
            labeled_annotations = eval_ann,
            img_size            = 224,
            crop_size           = 224,
            max_proposals       = 100,
            return_crops        = True,
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size  = args.stage3_batch_size,
            shuffle     = False,
            num_workers = 4,
            collate_fn  = stage3_collate_fn,
            pin_memory  = torch.cuda.is_available(),
        )

        # Verify the metrics library is available BEFORE running the (slow)
        # evaluation loop, so a missing dependency fails fast with a clear,
        # actionable message instead of a cryptic error after minutes of
        # inference, or — worse — silently skipping the report entirely.
        try:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision
        except ModuleNotFoundError as e:
            logger.error(
                "Evaluation requires 'torchmetrics' and 'pycocotools' for "
                "correct COCO-style mAP computation, but they are not "
                "installed (%s).\n"
                "Install them with:\n"
                "    pip install torchmetrics pycocotools\n"
                "Evaluation was skipped — no eval_results.json was written.",
                str(e),
            )
            return

        from torchvision.ops import box_iou

        # ── Metric accumulators ──────────────────────────────────────────
        # Per-image dicts consumed by torchmetrics' MeanAveragePrecision.
        map_preds:   list[dict] = []
        map_targets: list[dict] = []

        # Per-object counters for grounding accuracy (see note above).
        grounding_correct = 0
        grounding_total   = 0

        with torch.no_grad():
            for batch in eval_loader:
                images        = batch["images"].to(device)
                region_imgs   = batch["region_imgs"].to(device)
                bbox_coords   = batch["bbox_coords"].to(device)
                pseudo_labels = batch["pseudo_labels"].to(device)
                gt_boxes      = batch["gt_boxes"]
                gt_labels     = batch["gt_labels"]
                valid_mask    = batch["valid_mask"]

                B, K, C, h, w = region_imgs.shape

                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(
                        images         = images,
                        bbox_proposals = bbox_coords,
                        pseudo_labels  = pseudo_labels,
                        region_images  = region_imgs.view(B * K, C, h, w),
                    )

                pred_logits     = out["class_logits"].cpu()   # (B, Q, C+1)
                pred_boxes      = out["pred_boxes"].cpu()     # (B, Q, 4)
                bbox_coords_cpu = bbox_coords.cpu()

                probs  = torch.softmax(pred_logits, dim=-1)[..., :-1]  # exclude background
                scores, labels = probs.max(dim=-1)                     # (B, Q)

                for b in range(B):
                    gt_mask = valid_mask[b]                    # (K,) bool — real GT slots
                    if gt_mask.sum() == 0:
                        continue                                # nothing to evaluate against
                    gidx     = gt_mask.nonzero(as_tuple=True)[0]
                    g_boxes  = gt_boxes[b][gidx]                # (M, 4)
                    g_labels = gt_labels[b][gidx]               # (M,)

                    # Real (non-padded) proposals only. stage3_collate_fn
                    # pads every field with zeros, so a padded slot is an
                    # exact [0,0,0,0] box (area == 0) — this reliably tells
                    # real Stage-2 proposals apart from batch padding
                    # without needing an extra field in the dataset.
                    props     = bbox_coords_cpu[b]
                    prop_area = (props[:, 2] - props[:, 0]) * (props[:, 3] - props[:, 1])
                    prop_mask = prop_area > 0

                    if prop_mask.sum() == 0:
                        # No real proposals at all for this image: contribute
                        # an empty prediction set (every GT box is a miss).
                        map_preds.append({
                            "boxes":  torch.zeros((0, 4)),
                            "scores": torch.zeros((0,)),
                            "labels": torch.zeros((0,), dtype=torch.long),
                        })
                        map_targets.append({"boxes": g_boxes, "labels": g_labels})
                        grounding_total += g_boxes.shape[0]
                        continue

                    p_boxes  = pred_boxes[b][prop_mask]         # (P, 4)
                    p_scores = scores[b][prop_mask]             # (P,)
                    p_labels = labels[b][prop_mask]             # (P,)

                    map_preds.append({
                        "boxes":  p_boxes,
                        "scores": p_scores,
                        "labels": p_labels,
                    })
                    map_targets.append({"boxes": g_boxes, "labels": g_labels})

                    # ── Grounding accuracy: per-GT-object recall @ IoU 0.5 ──
                    # For each GT box, look only at same-class proposals and
                    # take the best IoU among them. This is independent of
                    # confidence ranking and independent of how many other
                    # (possibly wrong) predictions exist elsewhere in the
                    # image — unlike mAP/AP50 above.
                    iou         = box_iou(g_boxes, p_boxes)              # (M, P)
                    same_class  = g_labels.unsqueeze(1) == p_labels.unsqueeze(0)  # (M, P)
                    iou_matched = iou.masked_fill(~same_class, 0.0)
                    best_iou, _ = iou_matched.max(dim=1)                 # (M,)
                    grounding_correct += int((best_iou >= 0.5).sum().item())
                    grounding_total   += g_boxes.shape[0]

        # ── Compute & report metrics ───────────────────────────────────────
        try:
            if len(map_preds) == 0:
                raise RuntimeError(
                    "No images with valid ground truth were found during "
                    "evaluation. Check that --labeled_annotations points to "
                    "a non-empty file and that Stage3Dataset is actually "
                    "mixing labeled samples into the eval set."
                )

            map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
            map_metric.update(map_preds, map_targets)
            map_result = map_metric.compute()

            # torchmetrics returns -1 for an undefined metric (e.g. a class
            # with no predictions at all in the whole eval set) — clamp so
            # we never report a nonsensical negative percentage.
            mAP  = max(float(map_result["map"]),    0.0)
            AP50 = max(float(map_result["map_50"]), 0.0)
            AP75 = max(float(map_result["map_75"]), 0.0)

            grounding = grounding_correct / max(grounding_total, 1)

            logger.info("=" * 60)
            logger.info("FINAL EVALUATION RESULTS")
            logger.info("=" * 60)
            logger.info("  mAP (0.5:0.95) : %.4f  (%.1f%%)", mAP,  mAP  * 100)
            logger.info("  AP50           : %.4f  (%.1f%%)", AP50, AP50 * 100)
            logger.info("  AP75           : %.4f  (%.1f%%)", AP75, AP75 * 100)
            logger.info("  Grounding Acc  : %.4f  (%.1f%%)  [%d/%d objects]",
                         grounding, grounding * 100, grounding_correct, grounding_total)
            logger.info("=" * 60)

            # Save results to file
            import json as _json
            results = {
                "mAP":           round(mAP   * 100, 2),
                "AP50":          round(AP50  * 100, 2),
                "AP75":          round(AP75  * 100, 2),
                "grounding_acc": round(grounding * 100, 2),
                "grounding_correct": grounding_correct,
                "grounding_total":   grounding_total,
                "epoch":         ckpt.get("epoch", 0) if Path(best_ckpt).exists() else 0,
            }
            results_path = str(Path(args.stage3_dir) / "eval_results.json")
            with open(results_path, "w") as f:
                _json.dump(results, f, indent=2)
            logger.info("Results saved to %s", results_path)

        except Exception:
            # Log the FULL traceback rather than swallowing it into a one-
            # line warning — a silently-skipped, misleading eval report is
            # worse than a loud failure here.
            logger.exception(
                "Evaluation failed with an unexpected error — see traceback "
                "above. No eval_results.json was written."
            )

    else:
        logger.warning(
            "No labeled_annotations provided — skipping final evaluation.\n"
            "Re-run with --labeled_annotations outputs/sixray/labeled_annotations.json"
        )
 
 
# Entry point
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