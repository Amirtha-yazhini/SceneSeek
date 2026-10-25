"""Integration tests for SceneSeek dashboard APIs and local JSON metadata."""
from pathlib import Path
import pytest
import numpy as np

from sceneseek.config import Config
from sceneseek.index.metadata import JsonStore, open_store
from sceneseek.index.vector_index import VectorIndex
from sceneseek.pipeline.search import SceneSeek
from sceneseek.types import SearchResponse, SearchResult, format_ts
from sceneseek.eval.evaluate import load_latest_report


class DummyClip:
    dim = 8
    def encode_query(self, q):
        v = np.ones(self.dim, dtype=np.float32)
        return v / np.linalg.norm(v)


def test_format_ts_display():
    assert format_ts(0.0) == "00:00.0"
    assert format_ts(12.5) == "00:12.5"
    assert format_ts(65.0) == "01:05.0"
    assert format_ts(3600.0) == "1:00:00.0"


def test_json_store_persistence(tmp_path):
    index_dir = tmp_path / "index"
    store = JsonStore(index_dir)
    assert store.backend == "json"

    # Add video and segments
    store.upsert_video({
        "video_id": "test_cam",
        "path": "test_cam.mp4",
        "duration": 60.0,
        "fps": 30.0,
        "width": 1280,
        "height": 720,
        "num_segments": 2
    })
    store.upsert_segments([
        {
            "segment_id": "test_cam:0",
            "video_id": "test_cam",
            "index": 0,
            "start": 0.0,
            "end": 2.0,
            "object_counts": {"person": 1},
            "actions": [{"label": "walking", "prob": 0.5}],
            "thumbnail": "thumb0.jpg"
        },
        {
            "segment_id": "test_cam:1",
            "video_id": "test_cam",
            "index": 1,
            "start": 1.0,
            "end": 3.0,
            "object_counts": {"car": 1},
            "actions": [],
            "thumbnail": "thumb1.jpg"
        }
    ])
    store.save()

    # Re-open in new store instance to verify disk persistence
    store2 = JsonStore(index_dir)
    vids = store2.list_videos()
    assert len(vids) == 1
    assert vids[0]["video_id"] == "test_cam"

    segs = store2.segments_for_video("test_cam")
    assert len(segs) == 2
    assert segs[0]["segment_id"] == "test_cam:0"

    # Search history logging & persistence
    store2.log_search({
        "ts": 123456789.0,
        "query": "person walking",
        "latency_ms": 42.5,
        "num_results": 1,
        "best_score": 0.88,
        "top": []
    })

    store3 = JsonStore(index_dir)
    history = store3.search_history(10)
    assert len(history) == 1
    assert history[0]["query"] == "person walking"


def test_engine_apis_mocked(tmp_path):
    cfg = Config(
        data_dir=tmp_path,
        index_dir=tmp_path / "index",
        videos_dir=tmp_path / "v",
        thumbnails_dir=tmp_path / "t",
        clips_dir=tmp_path / "c",
        eval_dir=tmp_path / "e",
        use_mongo=False,
    )
    cfg.ensure_dirs()

    # Create dummy vector index
    vis = VectorIndex(8)
    vec = np.ones((1, 8), dtype=np.float32) / np.sqrt(8)
    vis.add(vec, ["test_cam:0"], ["frame"])
    vis.save(cfg.index_dir, "visual")

    act = VectorIndex(8)
    act.add(vec, ["test_cam:0"], ["action"])
    act.save(cfg.index_dir, "action")

    store = JsonStore(cfg.index_dir)
    store.upsert_video({
        "video_id": "test_cam",
        "path": "test_cam.mp4",
        "duration": 10.0,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "num_segments": 1
    })
    store.upsert_segments([{
        "segment_id": "test_cam:0",
        "video_id": "test_cam",
        "index": 0,
        "start": 0.0,
        "end": 2.0,
        "object_counts": {"person": 1},
        "actions": [],
        "thumbnail": None
    }])
    store.save()

    engine = SceneSeek(cfg, clip=DummyClip(), store=store)

    # Test list_videos
    vids = engine.list_videos()
    assert len(vids) == 1
    assert vids[0]["video_id"] == "test_cam"

    # Test segments
    segs = engine.segments("test_cam")
    assert len(segs) == 1

    # Test stats
    st = engine.stats()
    assert st["videos"] == 1
    assert st["metadata_backend"] == "json"

    # Test search
    resp = engine.search("person", top_k=5, threshold=0.1)
    assert not resp.no_match
    assert len(resp.results) == 1
    r = resp.results[0]
    assert r.video_id == "test_cam"
    assert r.timestamp_range == "00:00.0 - 00:02.0"

    # Test history
    hist = engine.history(limit=5)
    assert len(hist) >= 1
    assert hist[0]["query"] == "person"
