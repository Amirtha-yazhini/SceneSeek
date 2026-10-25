"""Fast tests that do not need torch/CLIP/YOLO weights."""
import json
from pathlib import Path

import numpy as np
import pytest

from sceneseek.config import Config
from sceneseek.data.virat import (build_ground_truth, merge_intervals, parse_kpf_activities,
                                  parse_virat2_events, query_for)
from sceneseek.eval.metrics import aggregate, query_metrics, temporal_iou
from sceneseek.index.metadata import JsonStore
from sceneseek.index.vector_index import VectorIndex
from sceneseek.pipeline.search import SceneSeek, objects_in_query
from sceneseek.types import format_ts
from sceneseek.video.frames import segment_windows


# ------------------------------------------------------------------ metrics
def test_temporal_iou():
    assert temporal_iou((0, 10), (0, 10)) == 1.0
    assert temporal_iou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    assert temporal_iou((0, 1), (2, 3)) == 0.0


def test_query_metrics_basic():
    gts = [{"video_id": "a", "start": 10, "end": 20}, {"video_id": "a", "start": 50, "end": 60}]
    preds = [{"video_id": "a", "start": 11, "end": 19},   # hit gt0
             {"video_id": "b", "start": 50, "end": 60},   # wrong video
             {"video_id": "a", "start": 48, "end": 62}]   # hit gt1
    m = query_metrics(preds, gts, iou_threshold=0.5, ks=(1, 2, 3))
    assert m["true_positives"] == 2
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == 1.0
    assert m["recall@1"] == 0.5 and m["recall@3"] == 1.0
    assert m["hit@1"] == 1.0
    agg = aggregate([m], [5.0])
    assert agg["micro"]["true_positives"] == 2


def test_no_double_counting():
    gts = [{"video_id": "a", "start": 0, "end": 10}]
    preds = [{"video_id": "a", "start": 0, "end": 10}, {"video_id": "a", "start": 1, "end": 9}]
    m = query_metrics(preds, gts, 0.3, ks=(1,))
    assert m["true_positives"] == 1 and m["precision"] == 0.5


# ------------------------------------------------------------------ VIRAT parsing
KPF = """
- { meta: "test" }
- { act: { act2: { Entering: 1.0 }, id2: 1, timespan: [{tsr0: [300, 450]}], src: truth,
           actors: [{id1: 2, timespan: [{tsr0: [300, 450]}]}] } }
- { act: { act2: { Entering: 1.0 }, id2: 2, timespan: [{tsr0: [400, 600]}], src: truth } }
- { act: { act2: { Talking: 1.0 }, id2: 3, timespan: [{tsr0: [900, 1200]}], src: truth } }
"""

EVENTS = """1 11 120 300 420 300 10 10 20 40
1 11 120 300 420 301 10 10 20 40
2 5 60 1000 1060 1000 5 5 20 40
"""


def test_parsers_and_builder(tmp_path):
    (tmp_path / "VIRAT_S_000001.activities.yml").write_text(KPF)
    (tmp_path / "VIRAT_S_000002.viratdata.events.txt").write_text(EVENTS)
    kpf = parse_kpf_activities(tmp_path / "VIRAT_S_000001.activities.yml")
    assert len(kpf) == 3 and kpf[0]["activity"] == "Entering"
    ev = parse_virat2_events(tmp_path / "VIRAT_S_000002.viratdata.events.txt")
    assert {e["activity"] for e in ev} == {"entering_facility", "getting_into_vehicle"}

    gt = build_ground_truth(tmp_path, fps_by_video={"VIRAT_S_000001": 30.0})
    by_id = {q["query_id"]: q for q in gt["queries"]}
    # the two overlapping "Entering" events are merged into one interval: frames 300-600
    ent = by_id["entering"]["instances"]
    assert len(ent) == 1 and ent[0]["start"] == 10.0 and ent[0]["end"] == 20.0
    assert "door" in by_id["entering"]["query"]
    gt2 = build_ground_truth(tmp_path, video_ids=["VIRAT_S_000002"])
    assert {q["query_id"] for q in gt2["queries"]} == {"entering_facility", "getting_into_vehicle"}


def test_query_for_unknown():
    assert query_for("Something_Weird") == "a person something weird"


def test_merge_intervals():
    assert merge_intervals([(5, 6), (0, 2), (1, 3)]) == [(0, 3), (5, 6)]


# ------------------------------------------------------------------ video helpers
def test_segment_windows():
    w = segment_windows(5.0, 2.0, 1.0)
    assert w[0] == (0.0, 2.0) and w[-1][1] == 5.0
    assert segment_windows(1.0, 2.0, 1.0) == [(0.0, 1.0)]


def test_format_ts():
    assert format_ts(65.5) == "01:05.5"
    assert format_ts(3725) == "1:02:05.0"


def test_objects_in_query():
    assert objects_in_query("Find a person with a backpack getting into a car") == {"person", "backpack", "car"}


# ------------------------------------------------------------------ index + search (fake CLIP)
class FakeClip:
    """Maps words to fixed axes so we know what should match."""
    dim = 8
    vocab = ["person", "car", "door", "running", "dog"]

    def encode_query(self, q):
        v = np.zeros(self.dim, dtype=np.float32)
        for i, w in enumerate(self.vocab):
            if w in q.lower():
                v[i] = 1
        v[-1] = 0.1
        return v / np.linalg.norm(v)


def _unit(*axes, dim=8):
    v = np.zeros(dim, dtype=np.float32)
    for a in axes:
        v[a] = 1
    v[-1] = 0.1
    return v / np.linalg.norm(v)


def _build(tmp_path):
    cfg = Config(data_dir=tmp_path, index_dir=tmp_path / "index", videos_dir=tmp_path / "v",
                 annotations_dir=tmp_path / "a", thumbnails_dir=tmp_path / "t", clips_dir=tmp_path / "c",
                 eval_dir=tmp_path / "e", ground_truth_path=tmp_path / "gt.json", use_mongo=False,
                 score_threshold=0.5, merge_gap=1.0)
    cfg.ensure_dirs()
    store = JsonStore(cfg.index_dir)
    vis, act = VectorIndex(8), VectorIndex(8)
    segs = []
    # video "cam1": 10 segments of 2s with stride 1s; a person at a door in segments 3-5
    for i in range(10):
        sid = f"cam1:{i}"
        seg_vec = _unit(0, 2) if 3 <= i <= 5 else _unit(1)
        vis.add(seg_vec[None], [sid], ["frame"])
        act.add(_unit(3)[None], [sid], ["action"])
        segs.append({"segment_id": sid, "video_id": "cam1", "index": i, "start": float(i), "end": float(i + 2),
                     "object_counts": {"person": 1}, "actions": [], "thumbnail": None, "detections": []})
    vis.save(cfg.index_dir, "visual"); act.save(cfg.index_dir, "action")
    store.upsert_video({"video_id": "cam1", "path": "cam1.mp4", "duration": 11.0, "fps": 30.0,
                        "frame_count": 330, "width": 10, "height": 10, "num_segments": 10})
    store.upsert_segments(segs)
    store.save()
    return cfg


def test_search_merges_adjacent_segments(tmp_path):
    cfg = _build(tmp_path)
    engine = SceneSeek(cfg, clip=FakeClip(), store=JsonStore(cfg.index_dir))
    resp = engine.search("person at a door", top_k=3)
    assert not resp.no_match
    top = resp.results[0]
    assert top.video_id == "cam1"
    assert (top.start, top.end) == (3.0, 7.0)           # segments 3,4,5 merged
    assert top.segment_ids == ["cam1:3", "cam1:4", "cam1:5"]
    d = resp.to_dict()
    json.dumps(d)                                         # dashboard needs JSON-able output
    assert d["results"][0]["timestamp_range"] == "00:03.0 - 00:07.0"
    assert engine.history(1)[0]["query"] == "person at a door"


def test_search_no_match(tmp_path):
    cfg = _build(tmp_path)
    engine = SceneSeek(cfg, clip=FakeClip(), store=JsonStore(cfg.index_dir))
    resp = engine.search("a dog", top_k=3)
    assert resp.no_match and resp.message == "No matching event found"


def test_vector_index_roundtrip_and_remove(tmp_path):
    idx = VectorIndex(8)
    idx.add(np.stack([_unit(0), _unit(1), _unit(2)]), ["a:0", "a:1", "b:0"], ["frame"] * 3)
    idx.save(tmp_path, "x")
    idx2 = VectorIndex.load(tmp_path, "x")
    assert idx2.search(_unit(1), 1)[0][0] == 1
    assert idx2.remove_segments("a:") == 2
    assert idx2.row_segment == ["b:0"] and idx2.search(_unit(2), 5)[0][0] == 0


def test_evaluate_end_to_end(tmp_path):
    from sceneseek.eval.evaluate import evaluate

    cfg = _build(tmp_path)
    gt = {"queries": [{"query_id": "door", "query": "person at a door",
                       "instances": [{"video_id": "cam1", "start": 3.5, "end": 7.0}]}]}
    cfg.ground_truth_path.write_text(json.dumps(gt))
    engine = SceneSeek(cfg, clip=FakeClip(), store=JsonStore(cfg.index_dir))
    rep = evaluate(engine, cfg.ground_truth_path, sweep=True)
    assert rep["summary"]["macro"]["recall"] == 1.0
    assert rep["summary"]["macro"]["recall@1"] == 1.0
    assert (cfg.eval_dir / "latest.json").exists() and (cfg.eval_dir / "latest_per_query.csv").exists()
    assert "best" in rep["threshold_sweep"]
