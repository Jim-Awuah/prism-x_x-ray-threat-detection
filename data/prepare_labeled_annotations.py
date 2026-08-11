# data/prepare_labeled_annotations.py
#
# Converts ground truth annotations into the JSON format that
# Stage3Dataset._convert_labeled() expects, for either SIXray (Pascal VOC
# XML + CSV) or CLCXray (COCO-format JSON).
#
# Run once before Stage 3:
#   python data/prepare_labeled_annotations.py \
#       --dataset   sixray \
#       --data_root "/path/to/SIXray" \
#       --subset    SIXray10 \
#       --split     train \
#       --out       outputs/sixray/labeled_annotations.json
#
#   python data/prepare_labeled_annotations.py \
#       --dataset   clcxray \
#       --data_root "/path/to/CLCXray" \
#       --split     train \
#       --out       outputs/clcxray/labeled_annotations.json
#
# Paper §5.1: "For all semi-supervised experiments, we use a fixed set of
# 1000 labeled samples to ensure consistent and fair comparison across
# baselines." By default this script subsamples down to that many samples
# (deterministically, via --seed) after collecting all valid labeled
# images. Pass --num_labeled 0 to use ALL available labeled samples
# instead (the old, pre-fix behavior) — note this is a different
# experimental protocol than the paper's and results won't be directly
# comparable to it.

import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ── SIXray ──────────────────────────────────────────────────────────────
SIXRAY_CLASS_NAMES  = ["gun", "knife", "wrench", "pliers", "scissors"]
SIXRAY_CLASS_TO_IDX = {c: i for i, c in enumerate(SIXRAY_CLASS_NAMES)}
SUBSET_MAP = {"SIXray10": "10", "SIXray100": "100", "SIXray1000": "1000"}

# ── CLCXray ─────────────────────────────────────────────────────────────
# MUST stay index-for-index identical to CLCXrayDataset.CLASS_NAMES in
# data/datasets.py, and to DATASETS["clcxray"]["class_names"] in main.py —
# all three are the same 12 classes in the same order. If you add/reorder
# classes in one place, update all three or class labels will silently
# train against the wrong text prompts.
CLCXRAY_CLASS_NAMES = [
    "scissors", "knife", "dagger", "blade", "swiss_army_knife",
    "spray_cans", "vacuum_cup", "plastic_bottle", "glass_bottle",
    "carton_drinks", "cans", "tin",
]
CLCXRAY_CLASS_TO_IDX = {c: i for i, c in enumerate(CLCXRAY_CLASS_NAMES)}

# ── EDS ─────────────────────────────────────────────────────────────────
# MUST stay index-for-index identical to EDSDataset.CLASS_NAMES in
# data/datasets.py and DATASETS["eds"]["class_names"] in main.py.
EDS_CLASS_NAMES = [
    "device", "glassbottle", "knife", "laptop", "lighter",
    "plasticbottle", "powerbank", "pressure", "scissor", "umbrella",
]
EDS_CLASS_TO_IDX = {c: i for i, c in enumerate(EDS_CLASS_NAMES)}
EDS_DOMAINS = ["domain1", "domain2", "domain3"]

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
        if cls_name not in SIXRAY_CLASS_TO_IDX:
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
        boxes.append({"bbox": [x1, y1, x2, y2], "label": SIXRAY_CLASS_TO_IDX[cls_name]})
    return boxes


def collect_sixray_samples(data_root: str, subset: str, split: str) -> list[dict]:
    root     = Path(data_root)
    ann_dir  = root / "Annotation"
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

    return samples


def collect_clcxray_samples(data_root: str, split: str) -> list[dict]:
    """
    Mirrors CLCXrayDataset._load_coco() in data/datasets.py — same
    category-name normalisation, same bbox conversion (COCO [x,y,w,h] ->
    [x1,y1,x2,y2]), same class index mapping — so the labeled subset used
    for Stage 3 is consistent with what Stage 1/2 already saw for CLCXray.
    """
    root      = Path(data_root)
    json_path = root / "annotations" / f"instances_{split}2017.json"
    img_dir   = root / f"{split}2017"

    if not json_path.exists():
        raise FileNotFoundError(f"CLCXray annotation not found:\n  {json_path}")

    with open(json_path) as f:
        coco = json.load(f)

    id_to_class = {
        cat["id"]: cat["name"].lower().strip()
        for cat in coco.get("categories", [])
    }

    img_to_anns: dict[int, list] = {}
    for ann in coco.get("annotations", []):
        img_id   = ann["image_id"]
        cls_name = id_to_class.get(ann["category_id"], "")
        if cls_name not in CLCXRAY_CLASS_TO_IDX:
            continue
        x, y, w, h = ann["bbox"]   # COCO: [x, y, width, height]
        img_to_anns.setdefault(img_id, []).append({
            "bbox":  [x, y, x + w, y + h],
            "label": CLCXRAY_CLASS_TO_IDX[cls_name],
        })

    samples = []
    n_missing = 0
    n_fallback_used = 0
    for img_info in coco.get("images", []):
        img_id   = img_info["id"]
        img_path = img_dir / img_info["file_name"]

        # Defensive fallback: COCO-format annotation exports frequently
        # list file_name with a stale/wrong extension (commonly ".jpg")
        # even when the actual images were converted to a different
        # format (e.g. ".png"). If the exact path from the JSON doesn't
        # exist, try the same stem with common image extensions before
        # giving up on that image, so a bulk extension mismatch doesn't
        # silently empty out the entire dataset.
        if not img_path.exists():
            stem = Path(img_info["file_name"]).stem
            fallback = None
            for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    fallback = candidate
                    break
            if fallback is not None:
                img_path = fallback
                n_fallback_used += 1
            else:
                n_missing += 1
                continue

        boxes = img_to_anns.get(img_id, [])
        if not boxes:
            continue   # only labeled images — mirrors SIXray's boxes-required filter

        primary = boxes[0]["label"]
        samples.append({
            "img_path": str(img_path),
            "label":    primary,
            "boxes":    boxes,
        })

    if n_fallback_used:
        print(f"NOTE: {n_fallback_used} images resolved via extension "
              f"fallback (JSON file_name didn't match on-disk extension "
              f"exactly). Consider fixing file_name in the JSON if this "
              f"number is large.")
    if n_missing:
        print(f"WARNING: {n_missing} images listed in the JSON could not "
              f"be found on disk under any common extension (checked "
              f".png/.jpg/.jpeg/.bmp) and were skipped.")

    return samples


def collect_eds_samples(data_root: str, domain: str) -> list[dict]:
    """
    Mirrors EDSDataset._parse_txt() / _load_domain() in data/datasets.py —
    same class mapping, same box ordering fixes — so the Stage 3 labeled
    subset is consistent with what Stage 1/2 saw.

    EDS annotation lines are:
        <image_name> <class_name> <xmin> <ymin> <xmax> <ymax>
    with one object per line, so a file yields multiple boxes across
    multiple classes.
    """
    root    = Path(data_root)
    domains = EDS_DOMAINS if domain == "all" else [domain]

    samples = []
    n_bad_class = 0
    n_bad_line  = 0
    for dom in domains:
        img_dir = root / dom / "images"
        txt_dir = root / dom / "txt"
        if not img_dir.is_dir():
            raise FileNotFoundError(
                f"EDS image directory not found:\n  {img_dir}\n"
                f"Expected layout: {root}/<domain>/images and /txt"
            )

        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in sorted(img_dir.glob(ext)):
                txt_path = txt_dir / (img_path.stem + ".txt")
                if not txt_path.exists():
                    continue

                boxes = []
                with open(txt_path) as f:
                    for line in f.read().splitlines():
                        parts = line.split()
                        if len(parts) < 6:
                            if line.strip():
                                n_bad_line += 1
                            continue
                        cls_name = parts[1].strip().lower()
                        if cls_name not in EDS_CLASS_TO_IDX:
                            n_bad_class += 1
                            continue
                        try:
                            x1, y1, x2, y2 = (float(v) for v in parts[2:6])
                        except ValueError:
                            n_bad_line += 1
                            continue
                        if x2 < x1:
                            x1, x2 = x2, x1
                        if y2 < y1:
                            y1, y2 = y2, y1
                        if x2 <= x1 or y2 <= y1:
                            continue
                        boxes.append({
                            "bbox":  [x1, y1, x2, y2],
                            "label": EDS_CLASS_TO_IDX[cls_name],
                        })

                if not boxes:
                    continue

                samples.append({
                    "img_path": str(img_path),
                    "label":    boxes[0]["label"],
                    "boxes":    boxes,
                })

    if n_bad_class:
        print(f"NOTE: skipped {n_bad_class} objects whose class name is not "
              f"one of the {len(EDS_CLASS_NAMES)} canonical EDS classes.")
    if n_bad_line:
        print(f"NOTE: skipped {n_bad_line} malformed annotation lines.")

    return samples


def prepare(dataset: str, data_root: str, subset: str, split: str, out: str,
            num_labeled: int = PAPER_NUM_LABELED, seed: int = PAPER_SEED,
            domain: str = "all"):
    if dataset == "sixray":
        samples = collect_sixray_samples(data_root, subset, split)
    elif dataset == "clcxray":
        samples = collect_clcxray_samples(data_root, split)
    elif dataset == "eds":
        samples = collect_eds_samples(data_root, domain)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

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
    p.add_argument("--dataset",   default="sixray",
                   choices=["sixray", "clcxray", "eds"])
    p.add_argument("--domain",    default="all",
                   choices=["domain1", "domain2", "domain3", "all"],
                   help="EDS-only; ignored for sixray/clcxray")
    p.add_argument("--data_root", required=True)
    p.add_argument("--subset",    default="SIXray10",
                   choices=["SIXray10", "SIXray100", "SIXray1000"],
                   help="SIXray-only; ignored for --dataset clcxray")
    p.add_argument("--split",     default="train",
                   choices=["train", "test", "val"],
                   help="SIXray uses train/test; CLCXray uses train/val/test")
    p.add_argument("--out",       default="outputs/sixray/labeled_annotations.json")
    p.add_argument("--num_labeled", type=int, default=PAPER_NUM_LABELED,
                   help=f"Fixed labeled-sample budget (paper §5.1 uses "
                        f"{PAPER_NUM_LABELED}). Pass 0 to use ALL available "
                        f"labeled samples instead (non-paper protocol).")
    p.add_argument("--seed", type=int, default=PAPER_SEED,
                   help="Random seed for the fixed-sample subsampling, for "
                        "reproducibility.")
    args = p.parse_args()

    if args.dataset == "sixray" and args.split == "val":
        raise ValueError("SIXray only supports --split train/test, not val.")

    prepare(args.dataset, args.data_root, args.subset, args.split, args.out,
            num_labeled=args.num_labeled, seed=args.seed, domain=args.domain)

