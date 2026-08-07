# data/prepare_labeled_annotations.py
#
# Converts SIXray ground truth XML annotations into the JSON format
# that Stage3Dataset._convert_labeled() expects.
#
# Run once before Stage 3:
#   python data/prepare_labeled_annotations.py \
#       --data_root "/path/to/SIXray" \
#       --subset    SIXray10 \
#       --split     train \
#       --out       outputs/sixray/labeled_annotations.json
#
# Paper §5.1: "For all semi-supervised experiments, we use a fixed set of
# 1000 labeled samples to ensure consistent and fair comparison across
# baselines." By default this script now subsamples down to that many
# samples (deterministically, via --seed) after collecting all valid
# P-prefix images with parseable XML annotations. Pass --num_labeled 0 to
# use ALL available labeled samples instead (the old, pre-fix behavior) —
# note this is a different experimental protocol than the paper's and
# results won't be directly comparable to it.

import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_NAMES  = ["gun", "knife", "wrench", "pliers", "scissors"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

SUBSET_MAP = {"SIXray10": "10", "SIXray100": "100", "SIXray1000": "1000"}

# Paper §5.1 default. Kept in sync with Stage_3/stage3_config.py's
# "num_labeled_samples" — if you change one, change the other.
PAPER_NUM_LABELED = 1000
PAPER_SEED        = 42


def parse_xml(xml_path: Path) -> list[dict]:
    if not xml_path.exists():
        return []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return []

    boxes = []
    for obj in root.findall("object"):
        name_tag = obj.find("name")
        if name_tag is None:
            continue
        cls_name = name_tag.text.strip().lower()
        if cls_name not in CLASS_TO_IDX:
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            x1 = float(bndbox.find("xmin").text)
            y1 = float(bndbox.find("ymin").text)
            x2 = float(bndbox.find("xmax").text)
            y2 = float(bndbox.find("ymax").text)
        except (TypeError, ValueError):
            continue
        boxes.append({"bbox": [x1, y1, x2, y2], "label": CLASS_TO_IDX[cls_name]})
    return boxes


def prepare(data_root: str, subset: str, split: str, out: str,
            num_labeled: int = PAPER_NUM_LABELED, seed: int = PAPER_SEED):
    root     = Path(data_root)
    ann_dir  = root / "Annotation"
    sdir     = SUBSET_MAP.get(subset, "10")
    csv_path = Path("/Users/dersunscheinyn/SIXray_dataset/OpenDataLab___SIXray/raw/SIXray/ImageSet/10/train.csv")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Build stem → image path map
    skip = {"Annotation", "ImageSet", ".cache", ".DS_Store"}
    stem_to_path = {}
    for folder in root.iterdir():
        if not folder.is_dir() or folder.name in skip:
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img in folder.glob(ext):
                stem_to_path[img.stem] = img

    # Read CSV — only P-prefix stems have annotations
    with open(csv_path) as f:
        lines = f.read().strip().splitlines()

    samples = []
    for line in lines[1:]:   # skip header
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        stem   = parts[0].strip()
        labels = [int(x) for x in parts[1:]]

        # Only process threat images (P-prefix) with at least one class present
        if not stem.startswith("P"):
            continue
        if not any(l == 1 for l in labels):
            continue

        img_path = stem_to_path.get(stem)
        if img_path is None:
            continue

        boxes = parse_xml(ann_dir / (stem + ".xml"))
        if not boxes:
            continue

        # Primary label = first present class
        primary = next((i for i, l in enumerate(labels) if l == 1), 0)

        samples.append({
            "img_path": str(img_path),
            "label":    primary,
            "boxes":    boxes,
        })

    n_available = len(samples)

    # ── Paper §5.1 protocol: fixed labeled-sample subset ────────────────
    # num_labeled <= 0 means "use everything" (explicit opt-out of the
    # paper's protocol — not the default, so it can't happen by accident).
    if num_labeled > 0 and n_available > num_labeled:
        rng = random.Random(seed)
        samples = rng.sample(samples, num_labeled)
        print(f"Subsampled {n_available} available labeled images down to "
              f"{num_labeled} (seed={seed}), matching paper §5.1's fixed "
              f"labeled-set protocol.")
    elif num_labeled > 0 and n_available < num_labeled:
        print(f"WARNING: only {n_available} labeled images available, "
              f"fewer than the requested --num_labeled {num_labeled}. "
              f"Using all {n_available} — this run will NOT match the "
              f"paper's fixed-1000-sample protocol.")
    elif num_labeled <= 0:
        print(f"--num_labeled <= 0: using all {n_available} available "
              f"labeled images. NOTE: this differs from the paper's "
              f"fixed-1000-sample protocol (§5.1) — results trained on this "
              f"file won't be directly comparable to the paper's numbers.")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} labeled samples to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--subset",    default="SIXray10",
                   choices=["SIXray10", "SIXray100", "SIXray1000"])
    p.add_argument("--split",     default="train",
                   choices=["train", "test"])
    p.add_argument("--out",       default="outputs/sixray/labeled_annotations.json")
    p.add_argument("--num_labeled", type=int, default=PAPER_NUM_LABELED,
                   help=f"Fixed labeled-sample budget (paper §5.1 uses "
                        f"{PAPER_NUM_LABELED}). Pass 0 to use ALL available "
                        f"labeled samples instead (non-paper protocol).")
    p.add_argument("--seed", type=int, default=PAPER_SEED,
                   help="Random seed for the fixed-sample subsampling, for "
                        "reproducibility.")
    args = p.parse_args()
    prepare(args.data_root, args.subset, args.split, args.out,
            num_labeled=args.num_labeled, seed=args.seed)