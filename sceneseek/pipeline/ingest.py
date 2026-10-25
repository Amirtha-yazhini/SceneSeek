"""Stages 1-3: video -> segments -> detections / embeddings / actions -> FAISS + metadata."""
from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..config import Config
from ..index.metadata import MetadataStore, open_store
from ..index.vector_index import VectorIndex
from ..types import Segment
from ..video.frames import iter_segments, list_videos, pick_evenly, probe, save_thumbnail

log = logging.getLogger(__name__)

VISUAL_INDEX = "visual"   # whole-frame and object-crop CLIP image embeddings
ACTION_INDEX = "action"   # CLIP text embedding of recognised action labels


class Models:
    """Loads each model once and shares it between ingest and search."""

    def __init__(self, cfg: Config, need_detector: bool = True, need_action: bool = True):
        from ..models.clip_encoder import ClipEncoder

        self.device = cfg.resolved_device()
        log.info("Loading OpenCLIP %s (%s) on %s", cfg.clip_model, cfg.clip_pretrained, self.device)
        self.clip = ClipEncoder(cfg.clip_model, cfg.clip_pretrained, self.device)
        self.detector = None
        self.action = None
        self._label_emb: dict[str, np.ndarray] | None = None
        if need_detector:
            from ..models.detector import ObjectDetector

            log.info("Loading YOLO %s", cfg.yolo_model)
            self.detector = ObjectDetector(cfg.yolo_model, cfg.yolo_conf, cfg.yolo_classes, self.device)
        if need_action and cfg.use_action_model:
            from ..models.action import ActionRecognizer

            log.info("Loading action model %s", cfg.action_model)
            self.action = ActionRecognizer(cfg.action_model, self.device, cfg.action_frames)

    def action_label_embeddings(self) -> dict[str, np.ndarray]:
        if self._label_emb is None and self.action is not None:
            labels = self.action.labels
            embs = self.clip.encode_texts([f"a person {l}" if not l.startswith(("a ", "the ")) else l
                                           for l in labels])
            self._label_emb = dict(zip(labels, embs))
        return self._label_emb or {}


class Indexer:
    def __init__(self, cfg: Optional[Config] = None, models: Optional[Models] = None,
                 store: Optional[MetadataStore] = None):
        self.cfg = cfg or Config()
        self.cfg.ensure_dirs()
        self.models = models or Models(self.cfg)
        self.store = store or open_store(self.cfg)
        dim = self.models.clip.dim
        self.visual = self._load_or_new(VISUAL_INDEX, dim)
        self.actions = self._load_or_new(ACTION_INDEX, dim)

    def _load_or_new(self, name: str, dim: int) -> VectorIndex:
        if VectorIndex.exists(self.cfg.index_dir, name):
            idx = VectorIndex.load(self.cfg.index_dir, name)
            if idx.dim == dim:
                return idx
            log.warning("Existing %s index has dim %d, model has %d; starting fresh", name, idx.dim, dim)
        return VectorIndex(dim)

    def ingest_folder(self, folder: Optional[Path] = None, reindex: bool = False, limit: Optional[int] = None,
                      progress: Optional[Callable[[str, float], None]] = None) -> list[str]:
        videos = list_videos(folder or self.cfg.videos_dir)
        if limit:
            videos = videos[:limit]
        done = []
        for i, path in enumerate(videos):
            log.info("[%d/%d] %s", i + 1, len(videos), path.name)
            if self.ingest_video(path, reindex=reindex, progress=progress):
                done.append(path.stem)
        return done

    def ingest_video(self, path: Path, reindex: bool = False,
                     progress: Optional[Callable[[str, float], None]] = None) -> bool:
        """Process one video. Returns False if it was already indexed and reindex is False."""
        cfg, m = self.cfg, self.models
        info = probe(Path(path))
        vid = info.video_id
        if self.store.get_video(vid) and not reindex:
            log.info("  already indexed, skipping (use --reindex to redo)")
            return False
        # remove any earlier version of this video
        self.visual.remove_segments(f"{vid}:")
        self.actions.remove_segments(f"{vid}:")
        self.store.delete_video(vid)

        label_emb = m.action_label_embeddings()
        t0 = time.time()
        seg_docs: list[dict] = []
        vis_vecs, vis_ids, vis_kinds = [], [], []
        act_vecs, act_ids = [], []

        for idx, start, end, frames in iter_segments(path, cfg.sample_fps, cfg.segment_seconds,
                                                     cfg.segment_stride, info.duration or None):
            sid = f"{vid}:{idx}"
            clip_frames = pick_evenly(frames, cfg.clip_frames_per_segment)
            mid = clip_frames[len(clip_frames) // 2]

            # --- object detection on the CLIP frames
            detections, crops, crop_labels = [], [], []
            if m.detector is not None:
                per_frame = m.detector.detect([f.rgb for f in clip_frames])
                for f, dets in zip(clip_frames, per_frame):
                    for d in dets:
                        detections.append((f, d))
                # embed the most confident, reasonably sized objects
                from ..models.detector import crop as crop_fn

                ranked = sorted(detections, key=lambda fd: -fd[1].confidence)
                for f, d in ranked:
                    x1, y1, x2, y2 = d.box
                    if min(x2 - x1, y2 - y1) < cfg.min_crop_pixels:
                        continue
                    crops.append(crop_fn(f.rgb, d.box))
                    crop_labels.append(d.label)
                    if len(crops) >= cfg.max_crops_per_segment:
                        break

            # --- CLIP: frames + crops in one batch
            embs = m.clip.encode_images([f.rgb for f in clip_frames] + crops)
            frame_emb = embs[: len(clip_frames)].mean(axis=0)
            frame_emb /= max(np.linalg.norm(frame_emb), 1e-8)
            vis_vecs.append(frame_emb); vis_ids.append(sid); vis_kinds.append("frame")
            for e, lab in zip(embs[len(clip_frames):], crop_labels):
                vis_vecs.append(e); vis_ids.append(sid); vis_kinds.append(f"crop:{lab}")

            # --- action recognition over the whole segment
            actions = []
            if m.action is not None:
                actions = m.action.predict([f.rgb for f in pick_evenly(frames, cfg.action_frames)],
                                           cfg.action_top_k)
                from ..models.action import action_text_embedding

                act_vecs.append(action_text_embedding(actions, label_emb, m.clip.dim)); act_ids.append(sid)

            # per-segment object counts = max count seen in any single frame
            counts: Counter = Counter()
            per_frame_counts: dict[int, Counter] = {}
            for f, d in detections:
                per_frame_counts.setdefault(f.frame_index, Counter())[d.label] += 1
            for c in per_frame_counts.values():
                for k, v in c.items():
                    counts[k] = max(counts[k], v)

            thumb = save_thumbnail(mid.rgb, cfg.thumbnails_dir / vid / f"{idx:06d}.jpg")
            seg = Segment(
                segment_id=sid, video_id=vid, index=idx, start=start, end=end,
                detections=[d.to_dict() | {"t": round(f.time, 2)} for f, d in detections if f is mid],
                object_counts=dict(counts),
                actions=[{"label": l, "prob": round(p, 4)} for l, p in actions],
                thumbnail=thumb,
            )
            seg_docs.append(seg.to_dict())
            if progress and info.duration:
                progress(vid, min(1.0, end / info.duration))

        if vis_vecs:
            self.visual.add(np.stack(vis_vecs), vis_ids, vis_kinds)
        if act_vecs:
            self.actions.add(np.stack(act_vecs), act_ids, ["action"] * len(act_ids))
        doc = info.to_dict() | {"num_segments": len(seg_docs), "indexed_at": time.time(),
                                "processing_seconds": round(time.time() - t0, 2)}
        self.store.upsert_video(doc)
        self.store.upsert_segments(seg_docs)
        self.save()
        log.info("  %d segments in %.1fs", len(seg_docs), time.time() - t0)
        return True

    def save(self) -> None:
        self.visual.save(self.cfg.index_dir, VISUAL_INDEX)
        self.actions.save(self.cfg.index_dir, ACTION_INDEX)
        self.store.save()
