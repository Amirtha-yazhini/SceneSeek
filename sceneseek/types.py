"""Plain data objects shared across the pipeline (and handed to the dashboard)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class VideoInfo:
    video_id: str
    path: str
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2) in pixels

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 4),
                "box": [int(v) for v in self.box]}


@dataclass
class Segment:
    segment_id: str          # "<video_id>:<index>"
    video_id: str
    index: int
    start: float             # seconds
    end: float               # seconds
    detections: list = field(default_factory=list)   # list[dict] from Detection.to_dict()
    object_counts: dict = field(default_factory=dict) # {"person": 2, "car": 1}
    actions: list = field(default_factory=list)       # [{"label": "walking", "prob": 0.31}, ...]
    thumbnail: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearchResult:
    """One merged time range in one video that matches the query."""
    video_id: str
    video_path: str
    start: float
    end: float
    score: float                     # fused similarity in roughly [0, 1]
    visual_score: float
    action_score: float
    segment_ids: list
    thumbnail: Optional[str] = None
    objects: dict = field(default_factory=dict)
    actions: list = field(default_factory=list)

    @property
    def timestamp_range(self) -> str:
        return f"{format_ts(self.start)} - {format_ts(self.end)}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp_range"] = self.timestamp_range
        return d


@dataclass
class SearchResponse:
    query: str
    results: list                    # list[SearchResult], best first, all above threshold
    latency_ms: float
    threshold: float
    best_score: float                # best score seen, even if below threshold
    top_k: int

    @property
    def no_match(self) -> bool:
        return not self.results

    @property
    def message(self) -> str:
        return "No matching event found" if self.no_match else f"{len(self.results)} matching event(s)"

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "latency_ms": round(self.latency_ms, 2),
            "threshold": self.threshold,
            "best_score": round(self.best_score, 4),
            "top_k": self.top_k,
            "no_match": self.no_match,
            "message": self.message,
        }


def format_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    frac = seconds - int(seconds)
    base = f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
    return f"{base}.{int(frac * 10)}"
