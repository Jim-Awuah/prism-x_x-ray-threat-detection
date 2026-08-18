"""
Dataset classes for PRISM-X.

SIXray structure (verified from actual files):

    SIXray/
      0/ 1/ ... 20/       ← image folders (N-prefix = negative, P-prefix by folder)
      Annotation/          ← Pascal VOC XML files (P00001.xml etc.)
      ImageSet/
        10/
          train.csv        ← columns: name, Gun, Knife, Wrench, Pliers, Scissors
          test.csv
        100/
        1000/

CSV format:
    name,Gun,Knife,Wrench,Pliers,Scissors
    P03198,1,-1,1,-1,-1        ← threat image: Gun + Wrench present
    N0210755,-1,-1,-1,-1,-1   ← negative image: no threats

    1  = class present
   -1  = class absent

P-prefix stems → have XML annotations in Annotation/
N-prefix stems → negative images, no XML annotation (all labels -1)

CLCXray structure (verified from actual files):

    CLCXray/
      annotations/
        instances_train2017.json   ← COCO-format
        instances_val2017.json
        instances_test2017.json
      train2017/
      val2017/
      test2017/
"""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from data.augmentations import StandardBYOLTransform, ThreatAwareTransform


#  SIXray 

class SIXrayDataset(Dataset):
    """
    SIXray dataset.

    The CSV files are the ground truth for which images belong to which
    split and what labels they carry. The XML files provide bounding
    boxes for P-prefix (threat) images.

    N-prefix images are always unlabeled (all class columns are -1).
    P-prefix images are labeled — their class columns show which threats
    are present, and their XML files provide bounding box coordinates.

    Args:
        root         : SIXray root directory
        subset       : "SIXray10" | "SIXray100" | "SIXray1000" | "all"
        split        : "train" | "test" | "all"
        labeled_only : if True, only return P-prefix images with annotations
        img_size     : spatial size for transforms
        augment      : optional custom transform override
    """

    # Column order in the CSV header matches these class names
    CLASS_NAMES  = ["gun", "knife", "wrench", "pliers", "scissors"]
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

    # The CSV has a 6th column for Hammer in some versions — kept for safety
    # Note: SIXray10 CSV only has 5 threat columns (no Hammer)
    CSV_COLUMNS  = ["gun", "knife", "wrench", "pliers", "scissors"]

    SUBSET_MAP = {
        "SIXray10":   "10",
        "SIXray100":  "100",
        "SIXray1000": "1000",
    }

    def __init__(
        self,
        root: str,
        subset: str = "SIXray10",
        split: str = "train",
        labeled_only: bool = False,
        img_size: int = 224,
        augment: Optional[Callable] = None,
    ) -> None:
        self.root         = Path(root)
        self.ann_dir      = self.root / "Annotation"
        self.labeled_only = labeled_only

        self.threat_transform   = ThreatAwareTransform(img_size=img_size)
        self.standard_transform = StandardBYOLTransform(img_size=img_size)
        if augment is not None:
            self.threat_transform   = augment
            self.standard_transform = augment

        # Build a map of all image stems → full paths by scanning folders
        self._stem_to_path = self._scan_image_folders()

        # Load samples from the CSV
        self.samples = self._load_from_csv(subset, split)

    #  Folder scanning 

    def _scan_image_folders(self) -> dict[str, Path]:
        """
        Scan all numbered class folders and build stem → image path map.
        Handles both N-prefix and P-prefix filenames.
        """
        stem_to_path: dict[str, Path] = {}
        skip = {"Annotation", "ImageSet", ".cache", ".DS_Store"}
        for folder in self.root.iterdir():
            if not folder.is_dir() or folder.name in skip:
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in folder.glob(ext):
                    stem_to_path[img_path.stem] = img_path
        return stem_to_path

    #  CSV loading 

    def _load_from_csv(self, subset: str, split: str) -> list[dict]:
        """
        Read the CSV file(s) and build the sample list.

        CSV columns: name, Gun, Knife, Wrench, Pliers, Scissors
        Values:  1 = present,  -1 = absent

        For P-prefix stems: look up XML to get bounding boxes.
        For N-prefix stems: no XML, all labels -1, treated as unlabeled.
        """
        # Determine which CSV files to read
        if subset == "all":
            subset_dirs = list(self.SUBSET_MAP.values())
        else:
            if subset not in self.SUBSET_MAP:
                raise ValueError(
                    f"subset must be one of {list(self.SUBSET_MAP)} or 'all'"
                )
            subset_dirs = [self.SUBSET_MAP[subset]]

        split_files = {
            "train": ["train.csv"],
            "test":  ["test.csv"],
            "all":   ["train.csv", "test.csv"],
        }
        csv_names = split_files.get(split, ["train.csv"])

        # Collect all rows from all matching CSVs
        all_rows: dict[str, list[int]] = {}   # stem → label list
        for sdir in subset_dirs:
            for csv_name in csv_names:
                csv_path = self.root / "ImageSet" / sdir / csv_name
                if not csv_path.exists():
                    continue
                with open(csv_path) as f:
                    lines = f.read().strip().splitlines()

                # First line is the header: name,Gun,Knife,Wrench,Pliers,Scissors
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if len(parts) < 2:
                        continue
                    stem   = parts[0].strip()
                    labels = [int(x) for x in parts[1:]]
                    all_rows[stem] = labels   # deduplicates across subsets

        # Build sample list
        samples = []
        for stem, labels in all_rows.items():
            img_path = self._stem_to_path.get(stem)
            if img_path is None:
                continue   # image file not found on disk

            is_threat = stem.startswith("P")
            is_labeled = is_threat and any(l == 1 for l in labels)

            if self.labeled_only and not is_labeled:
                continue

            # For threat images, load XML for bounding boxes
            annotations = []
            if is_threat:
                annotations = self._parse_xml(
                    self.ann_dir / (stem + ".xml"), labels
                )

            # Determine primary class label for Stage 2 classification head
            # Use the first present class (leftmost column with value 1)
            primary_label = -1
            for i, lval in enumerate(labels):
                if lval == 1 and i < len(self.CLASS_NAMES):
                    primary_label = i
                    break

            samples.append({
                "img_path":     img_path,
                "stem":         stem,
                "labels":       labels,
                "primary_label": primary_label,
                "annotations":  annotations,
                "is_labeled":   is_labeled,
            })

        return samples

    #  XML parsing 

    def _parse_xml(self, xml_path: Path, csv_labels: list[int]) -> list[dict]:
        """
        Parse Pascal VOC XML for bounding boxes.

        Uses the XML class names (e.g. 'Knife') to identify which threat
        each box belongs to. Falls back to CSV labels if XML name is unclear.

        Returns list of {class, class_idx, bbox: [x1,y1,x2,y2]}.
        """
        if not xml_path.exists():
            return []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            return []

        annotations = []
        for obj in root.findall("object"):
            name_tag = obj.find("name")
            if name_tag is None:
                continue
            cls_name = name_tag.text.strip().lower()
            bndbox   = obj.find("bndbox")
            if bndbox is None:
                continue
            try:
                x1 = float(bndbox.find("xmin").text)
                y1 = float(bndbox.find("ymin").text)
                x2 = float(bndbox.find("xmax").text)
                y2 = float(bndbox.find("ymax").text)
            except (TypeError, ValueError):
                continue

            class_idx = self.CLASS_TO_IDX.get(cls_name, -1)
            if class_idx == -1:
                continue

            annotations.append({
                "class":     cls_name,
                "class_idx": class_idx,
                "bbox":      [x1, y1, x2, y2],
            })

        return annotations

    #  Dataset interface 

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img    = Image.open(sample["img_path"]).convert("RGB")

        if sample["is_labeled"] and sample["annotations"]:
            ann    = sample["annotations"][0]
            v1, v2 = self.threat_transform(img, bbox=ann["bbox"])
            label  = ann["class_idx"]
            bbox   = torch.tensor(ann["bbox"], dtype=torch.float32)
        else:
            v1, v2 = self.standard_transform(img)
            label  = sample["primary_label"]  # may still be -1 for negatives
            bbox   = torch.zeros(4)

        return {
            "v1":        v1,
            "v2":        v2,
            "label":     torch.tensor(label, dtype=torch.long),
            "bbox":      bbox,
            "is_labeled": sample["is_labeled"],
            "img_path":  str(sample["img_path"]),
        }


# CLCXray 

class CLCXrayDataset(Dataset):
    """
    CLCXray dataset — COCO JSON format.

    Folder structure:
        root/
          annotations/
            instances_train2017.json
            instances_val2017.json
            instances_test2017.json
          train2017/
          val2017/
          test2017/

    Args:
        root         : CLCXray root directory
        split        : "train" | "val" | "test"
        labeled_only : if True, only return images with bounding box annotations
        img_size     : spatial size for transforms
        augment      : optional custom transform override
    """

    CLASS_NAMES = [
        "scissors", "knife", "dagger", "blade", "swiss_army_knife",
        "spray_cans", "vacuum_cup", "plastic_bottle", "glass_bottle",
        "carton_drinks", "cans", "tin",
    ]
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

    def __init__(
        self,
        root: str,
        split: str = "train",
        labeled_only: bool = False,
        img_size: int = 224,
        augment: Optional[Callable] = None,
    ) -> None:
        self.root         = Path(root)
        self.labeled_only = labeled_only

        self.threat_transform   = ThreatAwareTransform(img_size=img_size)
        self.standard_transform = StandardBYOLTransform(img_size=img_size)
        if augment is not None:
            self.threat_transform   = augment
            self.standard_transform = augment

        self.samples = self._load_coco(split)

    def _load_coco(self, split: str) -> list[dict]:
        json_path = self.root / "annotations" / f"instances_{split}2017.json"
        img_dir   = self.root / f"{split}2017"

        if not json_path.exists():
            raise FileNotFoundError(
                f"CLCXray annotation not found:\n  {json_path}"
            )

        with open(json_path) as f:
            coco = json.load(f)

        # category id → lowercase class name
        id_to_class = {
            cat["id"]: cat["name"].lower().strip()
            for cat in coco.get("categories", [])
        }

        # image id → list of annotation dicts
        img_to_anns: dict[int, list] = {}
        for ann in coco.get("annotations", []):
            img_id   = ann["image_id"]
            cls_name = id_to_class.get(ann["category_id"], "")
            if cls_name not in self.CLASS_TO_IDX:
                continue
            x, y, w, h = ann["bbox"]   # COCO: [x, y, width, height]
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append({
                "class":     cls_name,
                "class_idx": self.CLASS_TO_IDX[cls_name],
                "bbox":      [x, y, x + w, y + h],
            })

        samples = []
        for img_info in coco.get("images", []):
            img_id   = img_info["id"]
            img_path = img_dir / img_info["file_name"]

            if not img_path.exists():
                continue

            annotations = img_to_anns.get(img_id, [])
            is_labeled  = len(annotations) > 0

            if self.labeled_only and not is_labeled:
                continue

            samples.append({
                "img_path":    img_path,
                "annotations": annotations,
                "is_labeled":  is_labeled,
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img    = Image.open(sample["img_path"]).convert("RGB")

        if sample["is_labeled"] and sample["annotations"]:
            ann    = sample["annotations"][0]
            v1, v2 = self.threat_transform(img, bbox=ann["bbox"])
            label  = ann["class_idx"]
            bbox   = torch.tensor(ann["bbox"], dtype=torch.float32)
        else:
            v1, v2 = self.standard_transform(img)
            label  = -1
            bbox   = torch.zeros(4)

        return {
            "v1":        v1,
            "v2":        v2,
            "label":     torch.tensor(label, dtype=torch.long),
            "bbox":      bbox,
            "is_labeled": sample["is_labeled"],
            "img_path":  str(sample["img_path"]),
        }

# EDS 

class EDSDataset(Dataset):
    """
    EDS (Endoscopy/X-ray Domain Shift) dataset — plain-text annotations,
    split across three acquisition domains.

    Folder structure:
        root/
          domain1/
            images/          ← .jpg files
            txt/             ← one .txt per image, same stem
          domain2/
            images/
            txt/
          domain3/
            images/
            txt/

    Annotation format — one object per line, multiple lines per file:

        00001.jpg knife         231 194 402 431
        00001.jpg lighter       371 438 475 511
        00001.jpg lighter       386 454 490 524
        00001.jpg device        245 100 491 202
        00001.jpg plasticbottle 264 231 556 405

    i.e.  <image_name> <class_name> <xmin> <ymin> <xmax> <ymax>

    Boxes are kept in absolute pixel [x1, y1, x2, y2] form — the SAME
    convention SIXrayDataset and CLCXrayDataset return, and the form
    Stage3Dataset._convert_labeled() expects before it normalises. Do NOT
    convert to YOLO cxcywh here.

    Unlike SIXray (which has a train/test CSV split) and CLCXray (which
    has train/val/test folders), EDS ships as three *domains*, not splits.
    Domains are acquisition conditions, not train/test partitions, so this
    class takes `domain` rather than `split`. If you need a held-out set,
    hold out a whole domain (the standard domain-generalisation protocol
    for this dataset) via --domain, or split indices downstream.

    Args:
        root         : EDS_Dataset root directory
        domain       : "domain1" | "domain2" | "domain3" | "all"
        labeled_only : if True, only return images that have >=1 parseable box
        img_size     : spatial size for transforms
        augment      : optional custom transform override
    """

    # MUST stay index-for-index identical to DATASETS["eds"]["class_names"]
    # in main.py, and to EDS_CLASS_NAMES in
    # data/prepare_labeled_annotations.py. If you add or reorder a class in
    # one place, change all three — otherwise boxes silently train against
    # the wrong text prompt and nothing errors.
    CLASS_NAMES = [
        "device",         # 0
        "glassbottle",    # 1
        "knife",          # 2
        "laptop",         # 3
        "lighter",        # 4
        "plasticbottle",  # 5
        "powerbank",      # 6
        "pressure",       # 7
        "scissor",        # 8
        "umbrella",       # 9
    ]
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

    DOMAINS = ["domain1", "domain2", "domain3"]

    def __init__(
        self,
        root: str,
        domain: str = "all",
        labeled_only: bool = False,
        img_size: int = 224,
        augment: Optional[Callable] = None,
    ) -> None:
        self.root         = Path(root)
        self.labeled_only = labeled_only

        self.threat_transform   = ThreatAwareTransform(img_size=img_size)
        self.standard_transform = StandardBYOLTransform(img_size=img_size)
        if augment is not None:
            self.threat_transform   = augment
            self.standard_transform = augment

        if domain == "all":
            domains = self.DOMAINS
        elif domain in self.DOMAINS:
            domains = [domain]
        else:
            raise ValueError(
                f"domain must be one of {self.DOMAINS} or 'all', got {domain!r}"
            )
        self.domains = domains

        # One flat sample list across domains rather than a ConcatDataset.
        # ConcatDataset would hide .samples / .CLASS_NAMES behind a wrapper,
        # and Stage 1/2 code reaches for those attributes directly — a flat
        # list keeps EDSDataset a drop-in for SIXrayDataset/CLCXrayDataset.
        self.samples: list[dict] = []
        for d in domains:
            self.samples.extend(self._load_domain(d))

    #  Annotation parsing 

    def _parse_txt(self, txt_path: Path) -> list[dict]:
        """
        Parse one EDS annotation file into a list of box dicts.

        Every line is an independent object, so a single file yields
        multiple boxes and may span multiple classes. Malformed lines and
        unknown class names are skipped individually rather than
        discarding the whole file.
        """
        if not txt_path.exists():
            return []

        annotations = []
        try:
            with open(txt_path) as f:
                lines = f.read().splitlines()
        except OSError:
            return []

        for line in lines:
            parts = line.split()
            if len(parts) < 6:
                continue                      # blank or truncated line

            # parts[0] is the image name, which we already know from the
            # filename — it is not re-checked, since some EDS copies have
            # a stale name in-file after images were renamed.
            cls_name = parts[1].strip().lower()
            if cls_name not in self.CLASS_TO_IDX:
                continue                      # class outside the 10 canonical ones

            try:
                x1 = float(parts[2])
                y1 = float(parts[3])
                x2 = float(parts[4])
                y2 = float(parts[5])
            except ValueError:
                continue                      # non-numeric coordinates

            # Guard against inverted boxes in the raw annotations. A box
            # with x2 < x1 produces a negative area and blows up IoU/GIoU
            # downstream, so normalise the ordering here at the source.
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            if x2 <= x1 or y2 <= y1:
                continue                      # zero-area box, unusable

            annotations.append({
                "class":     cls_name,
                "class_idx": self.CLASS_TO_IDX[cls_name],
                "bbox":      [x1, y1, x2, y2],
            })

        return annotations

    #  Domain loading 

    def _load_domain(self, domain: str) -> list[dict]:
        """
        Build the sample list for one domain.

        Annotations are matched to images by *stem* (00001.jpg <-> 00001.txt).
        Matching by sorted-glob position instead would silently mis-pair
        every image whose annotation file is missing.
        """
        dom_dir = self.root / domain
        img_dir = dom_dir / "image"   # EDS uses "image" not "images"
        txt_dir = dom_dir / "txt"

        if not img_dir.is_dir():
            raise FileNotFoundError(
                f"EDS image directory not found:\n  {img_dir}\n"
                f"Expected layout: {self.root}/<domain>/image and /txt"
            )

        samples = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in sorted(img_dir.glob(ext)):
                annotations = self._parse_txt(txt_dir / (img_path.stem + ".txt"))
                is_labeled  = len(annotations) > 0

                if self.labeled_only and not is_labeled:
                    continue

                samples.append({
                    "img_path":    img_path,
                    "stem":        img_path.stem,
                    "domain":      domain,
                    "annotations": annotations,
                    "is_labeled":  is_labeled,
                })

        return samples

    #  Dataset interface 

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img    = Image.open(sample["img_path"]).convert("RGB")

        # Mirrors SIXrayDataset/CLCXrayDataset exactly: BYOL consumes the
        # two views, and the first annotation supplies the crop hint plus
        # the single label/bbox the Stage 1/2 heads expect. The full
        # multi-object annotation list is preserved in self.samples and is
        # what data/prepare_labeled_annotations.py reads for Stage 3, so no
        # boxes are lost — Stage 3 gets every object.
        if sample["is_labeled"] and sample["annotations"]:
            ann    = sample["annotations"][0]
            v1, v2 = self.threat_transform(img, bbox=ann["bbox"])
            label  = ann["class_idx"]
            bbox   = torch.tensor(ann["bbox"], dtype=torch.float32)
        else:
            v1, v2 = self.standard_transform(img)
            label  = -1
            bbox   = torch.zeros(4)

        return {
            "v1":        v1,
            "v2":        v2,
            "label":     torch.tensor(label, dtype=torch.long),
            "bbox":      bbox,
            "is_labeled": sample["is_labeled"],
            "img_path":  str(sample["img_path"]),
        }