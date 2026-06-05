import json
import os
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

from doclayout_yolo import YOLOv10
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


CLASS_COLORS = {
    "title": "#EE1F30",
    "plain text": "#1B8136",
    "abandon": "#F176FF",
    "figure": "#2300EB",
    "figure_caption": "#FFD504",
    "table": "#F4A261",
    "table_caption": "#264653",
    "table_footnote": "#BC6C25",
    "isolate_formula": "#9B5DE5",
    "formula_caption": "#00BBF9"
}


def compute_iou(boxA: Dict[str, float], boxB: Dict[str, float]) -> float:
    """Calculate Intersection over Union (IoU) between two boxes."""
    xA = max(boxA["x1"], boxB["x1"])
    yA = max(boxA["y1"], boxB["y1"])
    xB = min(boxA["x2"], boxB["x2"])
    yB = min(boxA["y2"], boxB["y2"])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA["x2"] - boxA["x1"]) * (boxA["y2"] - boxA["y1"])
    boxB_area = (boxB["x2"] - boxB["x1"]) * (boxB["y2"] - boxB["y1"])

    if boxA_area + boxB_area == 0:
        return 0.0
    return inter_area / (boxA_area + boxB_area - inter_area)


def is_contained(
    small: Dict,
    large: Dict,
    iou_threshold: float = 0.85,
    margin: float = 0.04
) -> bool:
    """Check if small box is almost completely inside large box."""
    if compute_iou(small, large) >= iou_threshold:
        return True
    w = large["x2"] - large["x1"]
    h = large["y2"] - large["y1"]
    return (
        small["x1"] >= large["x1"] - margin * w and
        small["x2"] <= large["x2"] + margin * w and
        small["y1"] >= large["y1"] - margin * h and
        small["y2"] <= large["y2"] + margin * h
    )


def subtract_vertical_portion(
    child_box: Dict[str, float],
    parent_box: Dict[str, float]
) -> List[Dict[str, float]]:
    """Subtract parent from child vertically, returning 0-2 boxes."""
    if child_box["y2"] <= parent_box["y1"] or child_box["y1"] >= parent_box["y2"]:
        return [child_box.copy()]

    if child_box["y1"] >= parent_box["y1"] and child_box["y2"] <= parent_box["y2"]:
        return []

    new_boxes = []
    if child_box["y1"] < parent_box["y1"]:
        top = child_box.copy()
        top["y2"] = parent_box["y1"]
        if top["y2"] - top["y1"] > 2:
            new_boxes.append(top)

    if child_box["y2"] > parent_box["y2"]:
        bottom = child_box.copy()
        bottom["y1"] = parent_box["y2"]
        if bottom["y2"] - bottom["y1"] > 2:
            new_boxes.append(bottom)

    return new_boxes


def merge_boxes(group: List[Dict]) -> Dict:
    """Merge a group of boxes into a single bounding box."""
    if not group:
        return {}
    return {
        "box": {
            "x1": min(b["box"]["x1"] for b in group),
            "y1": min(b["box"]["y1"] for b in group),
            "x2": max(b["box"]["x2"] for b in group),
            "y2": max(b["box"]["y2"] for b in group)
        }
    }


def should_merge(box1: Dict[str, float], box2: Dict[str, float]) -> bool:
    """Criterion for merging uncertain blocks."""
    iou = compute_iou(box1, box2)
    cont1 = is_contained(box1, box2)
    cont2 = is_contained(box2, box1)
    return iou >= 0.3 or cont1 or cont2


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        px, py = self.find(x), self.find(y)
        if px != py:
            if self.rank[px] < self.rank[py]:
                self.parent[px] = py
            elif self.rank[px] > self.rank[py]:
                self.parent[py] = px
            else:
                self.parent[py] = px
                self.rank[px] += 1


def split_low_against_highs(low: Dict, high_conf: List[Dict]) -> List[Dict]:
    """
    Process a single low-confidence block (or merged block) against all
    high-confidence reference blocks. Applies vertical subtraction on any
    overlap. Works correctly even when multiple high blocks are inside one low.
    """
    current = [low.copy()]

    for high in high_conf:
        new_current = []
        high_box = high["box"]

        for part in current:
            part_box = part["box"]
            iou = compute_iou(part_box, high_box)
            part_in_high = is_contained(part_box, high_box)
            high_in_part = is_contained(high_box, part_box)

            if part_in_high:
                continue

            if iou < 0.3 and not high_in_part:
                new_current.append(part)
                continue

            splits = subtract_vertical_portion(part_box, high_box)
            for sp in splits:
                new_part = part.copy()
                new_part["box"] = sp
                new_current.append(new_part)

        current = new_current
        if not current:
            return []

    return current


def filter_detections(detections: List[Dict]) -> List[Dict]:
    """
    Full filtering with two passes against high_conf:
    1. Split raw low-confidence blocks
    2. Merge using DSU (Disjoint Set Union)
    3. Second pass after merging to catch cases where reference blocks
       ended up inside merged low-confidence blocks
    """
    high_conf = [d for d in detections if d.get("confidence", 0) >= 0.7]
    low_conf = [d for d in detections if d.get("confidence", 0) < 0.7]

    final_boxes = high_conf.copy()

    remaining_pieces = []
    for low in low_conf:
        processed = split_low_against_highs(low, high_conf)
        remaining_pieces.extend(processed)

    if not remaining_pieces:
        return final_boxes

    n = len(remaining_pieces)
    dsu = DSU(n)
    for i in range(n):
        for j in range(i + 1, n):
            if should_merge(remaining_pieces[i]["box"], remaining_pieces[j]["box"]):
                dsu.union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[dsu.find(i)].append(remaining_pieces[i])

    candidate_lows = []
    for group in groups.values():
        if len(group) == 1:
            candidate_lows.append(group[0])
        else:
            merged = merge_boxes(group)
            best = max(group, key=lambda x: x.get("confidence", 0))
            merged["name"] = best["name"]
            merged["confidence"] = best["confidence"]
            candidate_lows.append(merged)

    final_lows = []
    for cand in candidate_lows:
        processed = split_low_against_highs(cand, high_conf)
        final_lows.extend(processed)

    final_boxes.extend(final_lows)
    return final_boxes


def get_yolo_boxes(image_dir: str, model_weights: str = 'soviet_yolo.pt') -> List[Dict]:
    """Run YOLO detection on all images in the specified directory."""
    image_files = [f for f in os.listdir(image_dir)]
    image_files.sort()

    script_dir = Path(__file__).resolve().parent
    weights_path = script_dir / model_weights

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model = YOLOv10(weights_path)

    frames = []
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        if not os.path.isfile(img_path) or not (
            img_path.lower().endswith('.jpg') or
            img_path.lower().endswith('.jpeg')
        ):
            print(f"Skipping file: {img_path} (not a .jpg/.jpeg)")
            continue

        results = model.predict(img_path, imgsz=1024, conf=0.15, device="cpu")
        detections = json.loads(results[0].tojson(normalize=True))
        filtered_detections = filter_detections(detections)

        labels = []
        for idx, det in enumerate(filtered_detections):
            box = det["box"]
            label = {
                "index": idx,
                "box_type": det["name"],
                "box_coord": {
                    "x1": float(box["x1"]),
                    "y1": float(box["y1"]),
                    "x2": float(box["x2"]),
                    "y2": float(box["y2"])
                }
            }
            labels.append(label)

        frame = {
            "img_path": img_path,
            "labels": labels
        }
        frames.append(frame)

    return frames


def visualize_boxes(yolo_result: Dict) -> None:
    """Visualize YOLO detections on the image."""
    detections = yolo_result['labels']
    image_path = yolo_result['img_path']

    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
        return

    width, height = img.size

    fig, ax = plt.subplots(1, figsize=(16, 16))
    ax.imshow(img)
    ax.axis('off')

    for det in detections:
        class_name = det['box_type']
        box = det['box_coord']

        x1 = box['x1'] * width
        y1 = box['y1'] * height
        x2 = box['x2'] * width
        y2 = box['y2'] * height

        box_width = x2 - x1
        box_height = y2 - y1

        color = CLASS_COLORS.get(class_name, "#000000")

        rect = patches.Rectangle(
            (x1, y1),
            box_width,
            box_height,
            linewidth=1,
            edgecolor=color,
            facecolor='none',
            linestyle='-'
        )
        ax.add_patch(rect)

        label_text = f"{class_name}"

        ax.text(
            x1,
            max(0, y1 - 10),
            label_text,
            color=color,
            fontsize=8,
            fontweight='bold',
            bbox=dict(
                facecolor='white',
                alpha=0.85,
                edgecolor=color,
                linewidth=1,
                pad=3
            )
        )

    plt.tight_layout()
    plt.show()
