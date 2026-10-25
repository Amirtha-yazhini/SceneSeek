"""Stage 4: text query -> ranked, merged timestamp ranges.

This module is the main entry point for the Streamlit dashboard:

    from sceneseek import SceneSeek
    engine = SceneSeek.load()
    response = engine.search("person wearing a black hoodie entering a building")
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..config import Config
from ..index.metadata import JsonStore, MetadataStore, open_store
from ..index.vector_index import VectorIndex
from ..types import SearchResponse, SearchResult
from .ingest import ACTION_INDEX, VISUAL_INDEX

log = logging.getLogger(__name__)

# words in a query -> YOLO/COCO class names we detect
OBJECT_WORDS = {
    "person": ["person", "people", "man", "men", "woman", "women", "pedestrian", "someone", "guy", "girl",
               "boy", "child", "kid", "human", "individual"],
    "car": ["car", "cars", "sedan", "suv", "vehicle", "vehicles"],
    "truck": ["truck", "trucks", "lorry", "pickup", "van"],
    "bus": ["bus", "buses"],
    "bicycle": ["bicycle", "bike", "bikes", "cyclist", "bicycles"],
    "motorcycle": ["motorcycle", "motorbike", "scooter"],
    "backpack": ["backpack", "rucksack"],
    "handbag": ["handbag", "purse", "bag", "bags"],
    "suitcase": ["suitcase", "luggage"],
}


def objects_in_query(query: str) -> set[str]:
    words = set(re.findall(r"[a-z]+", query.lower()))
    return {cls for cls, syns in OBJECT_WORDS.items() if words.intersection(syns)}


class SceneSeek:
    def __init__(self, cfg: Optional[Config] = None, clip=None, store: Optional[MetadataStore] = None):
        self.cfg = cfg or Config()
        self.cfg.ensure_dirs()
        if clip is None:
            from ..models.clip_encoder import ClipEncoder

            clip = ClipEncoder(self.cfg.clip_model, self.cfg.clip_pretrained, self.cfg.resolved_device())
        self.clip = clip
        self.store = store or open_store(self.cfg)
        self.reload()

    @classmethod
    def load(cls, cfg: Optional[Config] = None) -> "SceneSeek":
        return cls(cfg)

    def reload(self) -> None:
        """Re-read the index from disk (call after new videos are ingested)."""
        d = self.cfg.index_dir
        self.visual = VectorIndex.load(d, VISUAL_INDEX) if VectorIndex.exists(d, VISUAL_INDEX) else VectorIndex(self.clip.dim)
        self.actions = VectorIndex.load(d, ACTION_INDEX) if VectorIndex.exists(d, ACTION_INDEX) else VectorIndex(self.clip.dim)
        self._vis_rows = self.visual.rows_by_segment()
        self._act_row = {s: r for r, s in enumerate(self.actions.row_segment)}
        # segment order per video, to find neighbours when merging
        self._seg_order: dict[str, list[str]] = {}
        for sid in self._vis_rows:
            vid = sid.rsplit(":", 1)[0]
            self._seg_order.setdefault(vid, []).append(sid)
        for vid in self._seg_order:
            self._seg_order[vid].sort(key=lambda s: int(s.rsplit(":", 1)[1]))
        self._seg_pos = {s: i for lst in self._seg_order.values() for i, s in enumerate(lst)}
        if isinstance(self.store, JsonStore):  # re-read files another process may have written
            self.store = JsonStore(self.cfg.index_dir)
        self._seg_times: dict[str, tuple[float, float]] = {}
        for vid in self._seg_order:
            for doc in self.store.segments_for_video(vid):
                self._seg_times[doc["segment_id"]] = (doc["start"], doc["end"])

    # ------------------------------------------------------------------ search
    def search(self, query: str, top_k: Optional[int] = None, threshold: Optional[float] = None,
               video_ids: Optional[Iterable[str]] = None, log_history: bool = True) -> SearchResponse:
        cfg = self.cfg
        top_k = top_k or cfg.default_top_k
        threshold = cfg.score_threshold if threshold is None else threshold
        t0 = time.perf_counter()
        allowed = set(video_ids) if video_ids else None

        if not query.strip() or not len(self.visual):
            return SearchResponse(query, [], (time.perf_counter() - t0) * 1000, threshold, 0.0, top_k)

        q = self.clip.encode_query(query)
        wanted_objects = objects_in_query(query)

        # mean action similarity for this query, so the action term is a +/- adjustment
        act_all = self.actions.vectors @ q if len(self.actions) else np.zeros(0)
        act_mean = float(act_all.mean()) if len(act_all) else 0.0

        # 1) candidates from FAISS (visual and action indexes)
        cands: set[str] = set()
        for row, _ in self.visual.search(q, cfg.candidate_pool):
            cands.add(self.visual.row_segment[row])
        for row, _ in self.actions.search(q, max(20, cfg.candidate_pool // 4)):
            cands.add(self.actions.row_segment[row])
        if allowed is not None:
            cands = {c for c in cands if c.rsplit(":", 1)[0] in allowed}

        # 2) exact scores (cached, also used for neighbours while merging)
        cache: dict[str, tuple[float, float, float]] = {}

        def score(sid: str) -> tuple[float, float, float]:
            if sid in cache:
                return cache[sid]
            rows = self._vis_rows.get(sid, [])
            vis = float(self.visual.scores_for_rows(q, rows).max()) if rows else 0.0
            ar = self._act_row.get(sid)
            act = float(act_all[ar]) if ar is not None else act_mean
            total = vis + cfg.weight_action * (act - act_mean)
            if wanted_objects:
                kinds = {self.visual.row_kind[r].split(":", 1)[1] for r in rows
                         if self.visual.row_kind[r].startswith("crop:")}
                total += cfg.object_boost * len(wanted_objects & kinds)
            cache[sid] = (total, vis, act)
            return cache[sid]

        ranked = sorted(cands, key=lambda s: -score(s)[0])
        best = score(ranked[0])[0] if ranked else 0.0

        # 3) grow a time range around each peak, best peaks first
        used: set[str] = set()
        ranges: list[tuple[float, list[str]]] = []
        for sid in ranked:
            peak = score(sid)[0]
            if peak < threshold:
                break
            if sid in used:
                continue
            members = self._grow(sid, peak, threshold, score, used)
            used.update(members)
            # segments overlap in time, so a weaker peak right next to an existing range is the
            # same event: fold it into that range instead of returning a near-duplicate
            span = self._span(members)
            vid = sid.rsplit(":", 1)[0]
            for i, (p, mem) in enumerate(ranges):
                if mem[0].rsplit(":", 1)[0] == vid and self._overlaps(span, self._span(mem)):
                    ranges[i] = (p, sorted(mem + members, key=lambda s: self._seg_pos.get(s, 0)))
                    break
            else:
                ranges.append((peak, members))
            if len(ranges) >= top_k:
                break

        results = self._to_results(ranges, score)
        latency = (time.perf_counter() - t0) * 1000
        resp = SearchResponse(query, results, latency, threshold, best, top_k)
        if log_history:
            try:
                self.store.log_search({"ts": time.time(), "query": query, "latency_ms": round(latency, 2),
                                       "num_results": len(results), "best_score": round(best, 4),
                                       "top": [r.to_dict() for r in results[:3]]})
            except Exception as e:  # history is best-effort
                log.debug("could not log search: %s", e)
        return resp

    def _grow(self, sid, peak, threshold, score, used) -> list[str]:
        cfg = self.cfg
        vid = sid.rsplit(":", 1)[0]
        order = self._seg_order.get(vid, [sid])
        pos = self._seg_pos.get(sid, 0)
        floor = max(threshold, peak - cfg.merge_drop)
        members = [sid]
        seg_times = self._times
        start, end = seg_times(sid)
        for step in (-1, 1):
            i = pos
            while True:
                j = i + step
                if j < 0 or j >= len(order):
                    break
                nb = order[j]
                if nb in used:
                    break
                ns, ne = seg_times(nb)
                gap = (start - ne) if step < 0 else (ns - end)
                if gap > cfg.merge_gap:
                    break
                if max(end, ne) - min(start, ns) > cfg.max_range_seconds:
                    break
                if score(nb)[0] < floor:
                    break
                members.append(nb)
                start, end = min(start, ns), max(end, ne)
                i = j
        return sorted(members, key=lambda s: self._seg_pos.get(s, 0))

    def _span(self, members: list[str]) -> tuple[float, float]:
        times = [self._times(m) for m in members]
        return min(t[0] for t in times), max(t[1] for t in times)

    @staticmethod
    def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
        return min(a[1], b[1]) - max(a[0], b[0]) > 0

    def _times(self, sid: str) -> tuple[float, float]:
        return self._seg_times.get(sid, (0.0, 0.0))

    def _to_results(self, ranges, score) -> list[SearchResult]:
        all_ids = [s for _, m in ranges for s in m]
        docs = self.store.get_segments(all_ids) if all_ids else {}
        videos: dict[str, dict] = {}
        out = []
        for peak, members in ranges:
            vid = members[0].rsplit(":", 1)[0]
            if vid not in videos:
                videos[vid] = self.store.get_video(vid) or {}
            peak_sid = max(members, key=lambda s: score(s)[0])
            pdoc = docs.get(peak_sid, {})
            starts = [docs[s]["start"] for s in members if s in docs]
            ends = [docs[s]["end"] for s in members if s in docs]
            _, vis, act = score(peak_sid)
            out.append(SearchResult(
                video_id=vid, video_path=videos[vid].get("path", ""),
                start=min(starts) if starts else 0.0, end=max(ends) if ends else 0.0,
                score=round(peak, 4), visual_score=round(vis, 4), action_score=round(act, 4),
                segment_ids=members, thumbnail=pdoc.get("thumbnail"),
                objects=pdoc.get("object_counts", {}), actions=pdoc.get("actions", [])[:3],
            ))
        return out

    # ------------------------------------------------------------- helpers
    def list_videos(self) -> list[dict]:
        return self.store.list_videos()

    def segments(self, video_id: str) -> list[dict]:
        return self.store.segments_for_video(video_id)

    def history(self, limit: int = 50) -> list[dict]:
        return self.store.search_history(limit)

    def stats(self) -> dict:
        vids = self.list_videos()
        return {
            "videos": len(vids),
            "segments": len(self._vis_rows),
            "visual_vectors": len(self.visual),
            "action_vectors": len(self.actions),
            "hours_indexed": round(sum(v.get("duration", 0) for v in vids) / 3600, 3),
            "metadata_backend": self.store.backend,
            "clip_model": f"{self.cfg.clip_model}/{self.cfg.clip_pretrained}",
            "threshold": self.cfg.score_threshold,
        }

    def export_clip(self, result: SearchResult, pad: float = 1.0) -> Path:
        """Write a short mp4 of a result (for previews/downloads)."""
        from ..video.frames import export_clip

        out = self.cfg.clips_dir / f"{result.video_id}_{result.start:.1f}_{result.end:.1f}.mp4"
        if not out.exists():
            export_clip(Path(result.video_path), result.start, result.end, out, pad=pad)
        return out
