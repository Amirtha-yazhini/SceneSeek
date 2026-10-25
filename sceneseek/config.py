"""Central settings for SceneSeek.

Every value can be overridden with an environment variable named
SCENESEEK_<FIELD_NAME_IN_CAPS>, e.g. SCENESEEK_DEVICE=cuda.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # --- paths -----------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    videos_dir: Path = PROJECT_ROOT / "data" / "videos"
    annotations_dir: Path = PROJECT_ROOT / "data" / "annotations"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    thumbnails_dir: Path = PROJECT_ROOT / "data" / "thumbnails"
    clips_dir: Path = PROJECT_ROOT / "data" / "clips"
    ground_truth_path: Path = PROJECT_ROOT / "data" / "ground_truth.json"
    eval_dir: Path = PROJECT_ROOT / "data" / "eval"

    # --- video processing ------------------------------------------------
    sample_fps: float = 8.0          # frames per second read from each video
    segment_seconds: float = 2.0     # length of one searchable segment
    segment_stride: float = 1.0      # step between segment starts (overlap if < segment_seconds)
    clip_frames_per_segment: int = 3 # frames per segment sent to CLIP
    action_frames: int = 16          # frames per segment sent to the action model

    # --- models ----------------------------------------------------------
    device: str = "auto"             # auto | cpu | cuda
    yolo_model: str = "yolo11n.pt"
    yolo_conf: float = 0.35
    yolo_classes: tuple = (0, 1, 2, 3, 5, 7, 24, 26, 28)  # person, bicycle, car, motorcycle, bus, truck, backpack, handbag, suitcase
    max_crops_per_segment: int = 4   # object crops embedded with CLIP per segment
    min_crop_pixels: int = 24        # skip detections smaller than this (either side)
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    action_model: str = "r3d_18"     # torchvision video model pretrained on Kinetics-400
    action_top_k: int = 5
    use_action_model: bool = True

    # --- search ----------------------------------------------------------
    # score = visual + weight_action * (action - mean action score for this query) + object boost
    # "visual" = best CLIP image-text similarity over the segment's frames and object crops.
    weight_action: float = 0.25
    object_boost: float = 0.01       # per object class named in the query and detected in the segment
    score_threshold: float = 0.33    # below this -> "No matching event found" (tune with `evaluate --sweep`)
    merge_gap: float = 1.0           # seconds; matching segments closer than this are merged
    merge_drop: float = 0.03         # neighbours join a range only if within this of the range's peak score
    max_range_seconds: float = 30.0  # never merge into ranges longer than this
    candidate_pool: int = 400        # vectors pulled from FAISS before re-ranking
    default_top_k: int = 5

    # --- storage ---------------------------------------------------------
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "sceneseek"
    use_mongo: bool = False          # falls back to JSON files if Mongo is unreachable or False

    # --- evaluation ------------------------------------------------------
    iou_threshold: float = 0.3       # a prediction counts as correct at or above this temporal IoU
    recall_ks: tuple = (1, 5, 10)

    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for f in fields(self):
            env = os.environ.get(f"SCENESEEK_{f.name.upper()}")
            if env is None:
                continue
            current = getattr(self, f.name)
            setattr(self, f.name, _coerce(env, current))
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name.endswith(("_dir", "_path")) and not isinstance(v, Path):
                setattr(self, f.name, Path(v))

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.videos_dir, self.annotations_dir, self.index_dir,
                  self.thumbnails_dir, self.clips_dir, self.eval_dir):
            d.mkdir(parents=True, exist_ok=True)


def _coerce(raw: str, current):
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, tuple):
        return tuple(int(x) if x.strip().lstrip("-").isdigit() else x.strip() for x in raw.split(","))
    if isinstance(current, Path):
        return Path(raw)
    return raw
