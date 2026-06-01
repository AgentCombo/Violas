"""
COCO锛歀oad image paths and captions from a JSON list, bucketed by COCO-80 categories predicted by CLIP (same CLIP KeyPredictor as flower).
"""
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from violas.core.clip_features import KeyPredictor, extract_feature

# Consistent with coco_semantic_multimodal_bench
COCO_80_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def extract_all_features_from_coco_json(
    root_path: str,
    json_file: str = "coco_dataset_40.json",
    max_items: Optional[int] = None,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[Dict[str, Any]]]]:
    """
    return with flower ``extract_all_features_from_folders`` 锛?all_vectors[category] = [vec, ...]
    all_descriptions[category] = [{'img_path': str, 'idx': int}, ...]
    """
    json_path = os.path.join(root_path, json_file)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"COCO json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("COCO json Expected to be a list of objects")

    predictor = KeyPredictor.from_key_lists(sorted(COCO_80_CATEGORIES))

    all_vectors: Dict[str, List[np.ndarray]] = {}
    all_descriptions: Dict[str, List[Dict[str, Any]]] = {}
    global_idx = 0

    n = len(dataset) if max_items is None else min(len(dataset), max_items)
    print(f"from {json_path} load {n} Articles (Total {len(dataset)}锛夛紝Press CLIP鈫扖OCO-80 Bucketing...", flush=True)

    for item in tqdm(dataset[:n], desc="coco clip"):
        rel = item.get("path") or item.get("file_name")
        if not rel:
            continue
        full_path = os.path.join(root_path, rel)
        if not os.path.isfile(full_path):
            continue
        try:
            vec = extract_feature(full_path)
        except Exception as e:
            print(f"skip {full_path}: {e}", flush=True)
            continue
        folder, _ = predictor.predict_from_vector(vec, top_k=1)
        if folder not in all_vectors:
            all_vectors[folder] = []
            all_descriptions[folder] = []
        all_vectors[folder].append(vec)
        all_descriptions[folder].append({"img_path": full_path, "idx": global_idx})
        global_idx += 1

    if not all_vectors:
        raise ValueError("COCO: No vectors were extracted")
    print(f"Built {len(all_vectors)} predicted category buckets with {global_idx} images", flush=True)
    return all_vectors, all_descriptions
