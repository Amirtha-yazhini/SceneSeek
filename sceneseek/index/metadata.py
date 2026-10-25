"""Metadata storage: videos, segments and search history.

MongoDB is used when reachable; otherwise everything is kept in JSON files
under data/index/ so the project still runs on a laptop without Mongo.
Both stores have the same methods, so callers never need to care.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class MetadataStore:
    backend = "base"

    def upsert_video(self, video: dict) -> None: ...
    def get_video(self, video_id: str) -> Optional[dict]: ...
    def list_videos(self) -> list[dict]: ...
    def delete_video(self, video_id: str) -> None: ...
    def upsert_segments(self, segments: list[dict]) -> None: ...
    def get_segments(self, segment_ids: list[str]) -> dict[str, dict]: ...
    def segments_for_video(self, video_id: str) -> list[dict]: ...
    def log_search(self, entry: dict) -> None: ...
    def search_history(self, limit: int = 50) -> list[dict]: ...
    def save(self) -> None: ...


class MongoStore(MetadataStore):
    backend = "mongodb"

    def __init__(self, uri: str, db_name: str, timeout_ms: int = 1500):
        from pymongo import ASCENDING, MongoClient, UpdateOne

        self._UpdateOne = UpdateOne
        self.client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        self.client.admin.command("ping")  # raises if unreachable
        self.db = self.client[db_name]
        self.db.videos.create_index([("video_id", ASCENDING)], unique=True)
        self.db.segments.create_index([("segment_id", ASCENDING)], unique=True)
        self.db.segments.create_index([("video_id", ASCENDING), ("start", ASCENDING)])
        self.db.searches.create_index([("ts", ASCENDING)])

    def upsert_video(self, video):
        self.db.videos.replace_one({"video_id": video["video_id"]}, video, upsert=True)

    def get_video(self, video_id):
        return self.db.videos.find_one({"video_id": video_id}, {"_id": 0})

    def list_videos(self):
        return list(self.db.videos.find({}, {"_id": 0}).sort("video_id", 1))

    def delete_video(self, video_id):
        self.db.videos.delete_one({"video_id": video_id})
        self.db.segments.delete_many({"video_id": video_id})

    def upsert_segments(self, segments):
        if segments:
            ops = [self._UpdateOne({"segment_id": s["segment_id"]}, {"$set": s}, upsert=True) for s in segments]
            self.db.segments.bulk_write(ops, ordered=False)

    def get_segments(self, segment_ids):
        cur = self.db.segments.find({"segment_id": {"$in": list(segment_ids)}}, {"_id": 0})
        return {d["segment_id"]: d for d in cur}

    def segments_for_video(self, video_id):
        return list(self.db.segments.find({"video_id": video_id}, {"_id": 0}).sort("start", 1))

    def log_search(self, entry):
        self.db.searches.insert_one(dict(entry))

    def search_history(self, limit=50):
        return list(self.db.searches.find({}, {"_id": 0}).sort("ts", -1).limit(limit))


class JsonStore(MetadataStore):
    backend = "json"

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.videos: dict[str, dict] = self._read("videos.json", {})
        self.segments: dict[str, dict] = self._read("segments.json", {})
        self.searches: list[dict] = self._read("searches.json", [])

    def _read(self, name, default):
        p = self.folder / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

    def _write(self, name, data):
        tmp = self.folder / f"{name}.tmp"
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.folder / name)

    def upsert_video(self, video):
        self.videos[video["video_id"]] = dict(video)

    def get_video(self, video_id):
        return self.videos.get(video_id)

    def list_videos(self):
        return [self.videos[k] for k in sorted(self.videos)]

    def delete_video(self, video_id):
        self.videos.pop(video_id, None)
        self.segments = {k: v for k, v in self.segments.items() if v["video_id"] != video_id}

    def upsert_segments(self, segments):
        for s in segments:
            self.segments[s["segment_id"]] = dict(s)

    def get_segments(self, segment_ids):
        return {i: self.segments[i] for i in segment_ids if i in self.segments}

    def segments_for_video(self, video_id):
        return sorted((s for s in self.segments.values() if s["video_id"] == video_id), key=lambda s: s["start"])

    def log_search(self, entry):
        self.searches.append(dict(entry))
        self.searches = self.searches[-1000:]
        self._write("searches.json", self.searches)

    def search_history(self, limit=50):
        return list(reversed(self.searches[-limit:]))

    def save(self):
        self._write("videos.json", self.videos)
        self._write("segments.json", self.segments)
        self._write("searches.json", self.searches)


def open_store(cfg) -> MetadataStore:
    if cfg.use_mongo:
        try:
            store = MongoStore(cfg.mongo_uri, cfg.mongo_db)
            log.info("Metadata: MongoDB at %s/%s", cfg.mongo_uri, cfg.mongo_db)
            return store
        except Exception as e:  # pymongo missing or server down
            log.warning("MongoDB unavailable (%s); using JSON files in %s", type(e).__name__, cfg.index_dir)
    return JsonStore(cfg.index_dir)


def now() -> float:
    return time.time()
