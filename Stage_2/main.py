#Stages:
#   prepare  — convert OPIXray to YOLO format
#   finetune — fine-tune YOLOv12 on OPIXray
#   1        — BYOL self-supervised pre-training on SIXray/CLCXray
#   2        — pseudo-label generation (requires stage 1 + finetune outputs)
#   all      — finetune -> stage 1 -> stage 2  end-to-end
#
# Quick start:
#   Step 0:  python main.py --stage prepare --opixray_src "/path/to/OPI Xray"
#   Step 1:  python main.py --stage finetune --data_yaml opixray_yolo/opixray.yaml
#   Step 2:  python main.py --stage 1 --data_root "/path/to/SIXray"
#   Step 3:  python main.py --stage 2 --data_root "/path/to/SIXray"

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATASETS = {
    "sixray":  {"num_classes": 6},
    "clcxray": {"num_classes": 12},
}


def parse_args():
    p = argparse.ArgumentParser(description="PRISM-X pipeline")
    p.add_argument("--stage", default="all",
                   choices=["prepare", "finetune", "1", "2", "all"])
    p.add_argument("--dataset",   choices=list(DATASETS), default="sixray")
    p.add_argument("--data_root", default=None)
    p.add_argument("--backbone",  default="swin_v2_t",
                   choices=["swin_v2_t", "swin_v2_s", "swin_v2_b"])
    p.add_argument("--opixray_src", default=None)
    p.add_argument("--opixray_dst", default="opixray_yolo")
    p.add_argument("--data_yaml",     default="opixray_yolo/opixray.yaml")
    p.add_argument("--yolo_variant",  default="n",
                   choices=["n", "s", "m", "l", "x"])
    p.add_argument("--finetune_dir",  default="outputs/finetune")
    p.add_argument("--finetune_epochs", type=int, default=None)
    p.add_argument("--stage1_dir",  default="outputs/stage1")
    p.add_argument("--epochs",      type=int, default=None)
    p.add_argument("--batch_size",  type=int, default=None)
    p.add_argument("--no_pretrain", action="store_true")
    p.add_argument("--no_resume",   action="store_true")
    p.add_argument("--stage2_dir",  default="outputs/stage2")
    p.add_argument("--yolo_weights", default=None)
    p.add_argument("--eval", action="store_true")
    return p.parse_args()


def run_prepare(args):
    if not args.opixray_src:
        raise ValueError("--opixray_src is required for --stage prepare")
    # prepare_opixray.py lives in data/
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
        epochs    = args.finetune_epochs,   # None uses _DEFAULT_FT value
    )
    if args.eval:
        gen.evaluate(args.data_yaml)
    return best_pt


def run_stage1(args, num_classes):
    if not args.data_root:
        raise ValueError("--data_root is required for stage 1")

    from data.datasets import SIXrayDataset, CLCXrayDataset
    from Stage_1.byol import train_byol

    # BYOL hyperparameters
    cfg = {
        "num_classes":         num_classes,
        "backbone_variant":    args.backbone,
        "backbone_pretrained": not args.no_pretrain,
        "resume":              not args.no_resume,
        "epochs":              args.epochs      or 50,
        "batch_size":          args.batch_size  or 32,
        "num_workers":         4,
        "lr":                  1e-4,
        "lr_min":              1e-6,
        "weight_decay":        1e-4,
        "grad_clip":           1.0,
        "ema_decay":           0.996,
        "img_size":            224,
    }

    DatasetCls   = SIXrayDataset if args.dataset == "sixray" else CLCXrayDataset
    labeled_ds   = DatasetCls(root=args.data_root, labeled_only=True,  img_size=cfg["img_size"])
    unlabeled_ds = DatasetCls(root=args.data_root, labeled_only=False, img_size=cfg["img_size"])

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
         stage1_dir=args.stage1_dir,
         output_dir=args.stage2_dir)


def main():
    args        = parse_args()
    num_classes = DATASETS[args.dataset]["num_classes"]
    if args.stage == "prepare":
        run_prepare(args)
    elif args.stage == "finetune":
        run_finetune(args)
    elif args.stage == "1":
        run_stage1(args, num_classes)
    elif args.stage == "2":
        run_stage2(args, num_classes)
    elif args.stage == "all":
        if args.opixray_src:
            run_prepare(args)
        best_pt = run_finetune(args)
        args.yolo_weights = best_pt
        run_stage1(args, num_classes)
        run_stage2(args, num_classes)


if __name__ == "__main__":
    main()