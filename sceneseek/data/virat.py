"""Turn VIRAT annotations into ground-truth timestamps for evaluation.

Two annotation formats are supported:

1. KPF YAML from gitlab.kitware.com/viratdata/viratannotations
   (files like VIRAT_S_000001.activities.yml, frame numbers in `tsr0`).
2. The original VIRAT 2.0 release text files from viratdata.org
   (files like VIRAT_S_000001.viratdata.events.txt).

Output (data/ground_truth.json):

    {"queries": [
        {"query_id": "entering", "query": "a person entering a building",
         "activity": "Entering",
         "instances": [{"video_id": "VIRAT_S_000001", "start": 12.3, "end": 15.1,
                        "start_frame": 369, "end_frame": 453}, ...]},
        ...]}

You can hand-edit this file: change query wording, add new queries with
manually written timestamps, or delete activities you do not care about.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)

# VIRAT 2.0 release (events.txt) numeric event types
VIRAT2_EVENT_TYPES = {
    1: "loading_object_vehicle",
    2: "unloading_object_vehicle",
    3: "opening_trunk",
    4: "closing_trunk",
    5: "getting_into_vehicle",
    6: "getting_out_of_vehicle",
    7: "gesturing",
    8: "digging",
    9: "carrying_object",
    10: "running",
    11: "entering_facility",
    12: "exiting_facility",
}

# Natural-language query for each activity (keys are normalised: lowercase, no punctuation).
ACTIVITY_QUERIES = {
    # VIRAT 2.0 names
    "loading_object_vehicle": "a person loading an object into a vehicle",
    "unloading_object_vehicle": "a person unloading an object from a car",
    "opening_trunk": "a person opening the trunk of a car",
    "closing_trunk": "a person closing the trunk of a car",
    "getting_into_vehicle": "a person getting into a car",
    "getting_out_of_vehicle": "a person getting out of a car",
    "gesturing": "a person gesturing with their hands",
    "digging": "a person digging",
    "carrying_object": "a person carrying an object",
    "running": "a person running",
    "entering_facility": "a person entering a building through a door",
    "exiting_facility": "a person walking out of a building door",
    # KPF / viratannotations names
    "entering": "a person entering a building through a door",
    "exiting": "a person walking out of a building door",
    "loading": "a person loading an object into a vehicle",
    "unloading": "a person unloading an object from a car",
    "open_trunk": "a person opening the trunk of a car",
    "closing_trunk": "a person closing the trunk of a car",
    "opening": "a person opening a car door",
    "closing": "a person closing a car door",
    "riding": "a person riding a bicycle",
    "talking": "two people standing and talking to each other",
    "pull": "a person pulling a cart or object",
    "push": "a person pushing a cart or object",
    "interacts": "two people interacting",
    "transport_heavycarry": "a person carrying a heavy object",
    "activity_carrying": "a person carrying an object",
    "specialized_talking_phone": "a person talking on a phone",
    "specialized_texting_phone": "a person texting on a phone",
    "vehicle_turning_left": "a car turning left",
    "vehicle_turning_right": "a car turning right",
    "vehicle_u_turn": "a car making a u-turn",
}


def normalise_activity(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def query_for(activity: str) -> str:
    key = normalise_activity(activity)
    if key in ACTIVITY_QUERIES:
        return ACTIVITY_QUERIES[key]
    words = key.replace("activity_", "").replace("specialized_", "").replace("_", " ")
    return f"a person {words}"


# ---------------------------------------------------------------- parsers
def parse_kpf_activities(path: Path) -> list[dict]:
    """Return [{"activity", "start_frame", "end_frame"}] from a *.activities.yml file."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    out = []
    for item in data:
        act = item.get("act") if isinstance(item, dict) else None
        if not act:
            continue
        if act.get("src") not in (None, "truth"):
            continue
        labels = act.get("act2") or act.get("activity") or {}
        if isinstance(labels, str):
            labels = {labels: 1.0}
        if not labels:
            continue
        name = max(labels.items(), key=lambda kv: kv[1])[0]
        for span in act.get("timespan", []) or []:
            tsr = span.get("tsr0") if isinstance(span, dict) else None
            if tsr and len(tsr) == 2:
                out.append({"activity": str(name), "start_frame": int(tsr[0]), "end_frame": int(tsr[1])})
    return out


def parse_virat2_events(path: Path) -> list[dict]:
    """Return [{"activity", "start_frame", "end_frame"}] from a *.viratdata.events.txt file."""
    events: dict[int, dict] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            eid, etype, _dur, s, e = (int(float(x)) for x in parts[:5])
        except ValueError:
            continue
        events.setdefault(eid, {"activity": VIRAT2_EVENT_TYPES.get(etype, f"event_{etype}"),
                                "start_frame": s, "end_frame": e})
    return list(events.values())


def _video_id_from_annotation(path: Path) -> str:
    name = Path(path).name
    for suffix in (".activities.yml", ".viratdata.events.txt", ".events.txt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem.split(".")[0]


def find_annotation_files(folder: Path) -> list[Path]:
    folder = Path(folder)
    return sorted(list(folder.rglob("*.activities.yml")) + list(folder.rglob("*.events.txt")))


# ---------------------------------------------------------------- builder
def merge_intervals(items: list[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(items):
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def build_ground_truth(annotations_dir: Path, fps_by_video: Optional[dict[str, float]] = None,
                       video_ids: Optional[Iterable[str]] = None, activities: Optional[Iterable[str]] = None,
                       default_fps: float = 30.0, merge_overlaps: bool = True,
                       min_instances: int = 1) -> dict:
    """Collect activity instances per activity -> one query each.

    fps_by_video: real frame rate per video (frames -> seconds). Missing -> default_fps.
    video_ids:    only keep these videos (e.g. the ones you downloaded/indexed).
    activities:   only keep these activity names (normalised match).
    merge_overlaps: overlapping instances of the same activity in the same video become one
                  interval (search returns time ranges, not individual people).
    """
    fps_by_video = fps_by_video or {}
    keep_videos = set(video_ids) if video_ids is not None else None
    keep_acts = {normalise_activity(a) for a in activities} if activities else None

    per_act: dict[str, dict[str, list[tuple[float, float, int, int]]]] = defaultdict(lambda: defaultdict(list))
    raw_name: dict[str, str] = {}
    files = find_annotation_files(annotations_dir)
    for f in files:
        vid = _video_id_from_annotation(f)
        if keep_videos is not None and vid not in keep_videos:
            continue
        parsed = parse_kpf_activities(f) if f.name.endswith(".yml") else parse_virat2_events(f)
        fps = fps_by_video.get(vid, default_fps)
        for ev in parsed:
            key = normalise_activity(ev["activity"])
            if keep_acts is not None and key not in keep_acts:
                continue
            raw_name.setdefault(key, ev["activity"])
            per_act[key][vid].append((ev["start_frame"] / fps, ev["end_frame"] / fps,
                                      ev["start_frame"], ev["end_frame"]))
    log.info("Parsed %d annotation files, %d activity types", len(files), len(per_act))

    queries = []
    for key in sorted(per_act):
        instances = []
        for vid, evs in sorted(per_act[key].items()):
            fps = fps_by_video.get(vid, default_fps)
            spans = merge_intervals([(s, e) for s, e, *_ in evs]) if merge_overlaps else [(s, e) for s, e, *_ in evs]
            for s, e in spans:
                instances.append({"video_id": vid, "start": round(s, 3), "end": round(e, 3),
                                  "start_frame": int(round(s * fps)), "end_frame": int(round(e * fps))})
        if len(instances) < min_instances:
            continue
        queries.append({"query_id": key, "query": query_for(raw_name[key]), "activity": raw_name[key],
                        "instances": instances})
    return {"source": str(annotations_dir), "default_fps": default_fps, "queries": queries}


def save_ground_truth(gt: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(gt, indent=2), encoding="utf-8")


def load_ground_truth(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
