# data/prepare_eds_annotations.py
#
# Converts EDS domain annotations into the JSON format that
# Stage3Dataset._convert_labeled() expects.
#
# Run once per domain before Stage 3 cross-domain evaluation:
#   python data/prepare_eds_annotations.py \
#       --data_root "/path/to/EDS" \
#       --domain D1 \
#       --out outputs/eds/labeled_annotations_D1.json

import argparse
import json
from pathlib import Path

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
CLASS_TO_IDX = {c: i for i, c in enumerate(EDS_CLASS_NAMES)}
DOMAIN_FOLDERS = {"D1": "domain1", "D2": "domain2", "D3": "domain3"}


def parse_txt(txt_path: Path) -> list[dict]:
    if not txt_path.exists():
        return []
    annotations = []
    for line in txt_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        cls_name = parts[1].lower().strip()
        try:
            x1, y1, x2, y2 = float(parts[2]), float(parts[3]), \
                              float(parts[4]), float(parts[5])
        except ValueError:
            continue
        annotations.append({
            "bbox":  [x1, y1, x2, y2],
            "label": CLASS_TO_IDX.get(cls_name, -1),
        })
    return [a for a in annotations if a["label"] >= 0]


def prepare(data_root: str, domain: str, out: str):
    root          = Path(data_root)
    domain_folder = root / DOMAIN_FOLDERS[domain]
    img_dir       = domain_folder / "image"
    txt_dir       = domain_folder / "txt"

    if not img_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {img_dir}")

    samples = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        txt_path = txt_dir / (img_path.stem + ".txt")
        boxes    = parse_txt(txt_path)
        if not boxes:
            continue
        samples.append({
            "img_path": str(img_path),
            "label":    boxes[0]["label"],
            "boxes":    boxes,
        })

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} labeled samples → {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--domain", default="D1",
                   choices=["D1", "D2", "D3"])
    p.add_argument("--out", default="outputs/eds/labeled_annotations_D1.json")
    args = p.parse_args()
    prepare(args.data_root, args.domain, args.out)