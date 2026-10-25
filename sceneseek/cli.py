"""Command line: python -m sceneseek <command> ...

    ingest     index every video in data/videos
    build-gt   turn VIRAT annotations into data/ground_truth.json
    search     run a text query
    evaluate   score search results against the ground truth
    stats      show what is indexed
    clip       cut a short mp4 for a time range
    reset      delete the index and metadata
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Config


def _cmd_ingest(args, cfg: Config) -> None:
    from .pipeline.ingest import Indexer

    if args.no_action:
        cfg.use_action_model = False
    indexer = Indexer(cfg)
    folder = Path(args.videos) if args.videos else cfg.videos_dir
    done = indexer.ingest_folder(folder, reindex=args.reindex, limit=args.limit)
    print(f"Indexed {len(done)} video(s). Metadata backend: {indexer.store.backend}. Index: {cfg.index_dir}")


def _cmd_build_gt(args, cfg: Config) -> None:
    from .data.virat import build_ground_truth, save_ground_truth
    from .video.frames import list_videos, probe

    videos_dir = Path(args.videos) if args.videos else cfg.videos_dir
    vids = list_videos(videos_dir)
    fps = {}
    for v in vids:
        try:
            fps[v.stem] = probe(v).fps
        except IOError:
            pass
    video_ids = None if args.all_videos else list(fps) or None
    if video_ids is None and not args.all_videos:
        print(f"No videos found in {videos_dir}; keeping annotations for all videos.")
    gt = build_ground_truth(Path(args.annotations) if args.annotations else cfg.annotations_dir,
                            fps_by_video=fps, video_ids=video_ids, activities=args.activities,
                            min_instances=args.min_instances)
    out = Path(args.out) if args.out else cfg.ground_truth_path
    save_ground_truth(gt, out)
    n = sum(len(q["instances"]) for q in gt["queries"])
    print(f"Wrote {len(gt['queries'])} queries / {n} instances to {out}")
    for q in gt["queries"]:
        print(f"  {q['query_id']:<30} {len(q['instances']):>4}  \"{q['query']}\"")


def _cmd_search(args, cfg: Config) -> None:
    from .pipeline.search import SceneSeek

    engine = SceneSeek(cfg)
    resp = engine.search(args.query, top_k=args.top_k, threshold=args.threshold)
    if args.json:
        print(json.dumps(resp.to_dict(), indent=2))
        return
    print(f'"{resp.query}"  ->  {resp.message}  ({resp.latency_ms:.0f} ms, best score {resp.best_score:.3f}, '
          f"threshold {resp.threshold:.3f})")
    for i, r in enumerate(resp.results, 1):
        acts = ", ".join(a["label"] for a in r.actions)
        objs = ", ".join(f"{k}x{v}" for k, v in r.objects.items())
        print(f"{i:>2}. {r.video_id}  {r.timestamp_range}  score={r.score:.3f} "
              f"(visual {r.visual_score:.3f}, action {r.action_score:.3f})  objects[{objs}]  actions[{acts}]")


def _cmd_evaluate(args, cfg: Config) -> None:
    from .eval.evaluate import evaluate
    from .pipeline.search import SceneSeek

    if args.threshold is not None:
        cfg.score_threshold = args.threshold
    engine = SceneSeek(cfg)
    rep = evaluate(engine, Path(args.gt) if args.gt else cfg.ground_truth_path,
                   iou_threshold=args.iou, sweep=args.sweep)
    s = rep["summary"]
    print(json.dumps({"macro": s["macro"], "micro": s["micro"], "latency_ms": s["latency_ms"]}, indent=2))
    if "threshold_sweep" in rep:
        print("Best threshold:", rep["threshold_sweep"]["best"])


def _cmd_stats(args, cfg: Config) -> None:
    from .pipeline.search import SceneSeek

    engine = SceneSeek(cfg)
    print(json.dumps(engine.stats(), indent=2))
    for v in engine.list_videos():
        print(f"  {v['video_id']:<32} {v['duration']:>8.1f}s  {v.get('num_segments', 0):>5} segments")


def _cmd_clip(args, cfg: Config) -> None:
    from .video.frames import export_clip

    out = Path(args.out) if args.out else cfg.clips_dir / f"{Path(args.video).stem}_{args.start}_{args.end}.mp4"
    print(export_clip(Path(args.video), args.start, args.end, out, pad=args.pad))


def _cmd_reset(args, cfg: Config) -> None:
    import shutil

    from .index.metadata import MongoStore, open_store

    if not args.yes:
        print("This deletes the FAISS index, thumbnails, clips and metadata. Re-run with --yes to confirm.")
        return
    store = open_store(cfg)
    if isinstance(store, MongoStore):
        store.client.drop_database(cfg.mongo_db)
    for d in (cfg.index_dir, cfg.thumbnails_dir, cfg.clips_dir):
        shutil.rmtree(d, ignore_errors=True)
    print("Index and metadata cleared. Videos, annotations and ground truth were kept.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sceneseek", description="Natural-language search over video.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="index videos")
    s.add_argument("--videos", help="folder of videos (default data/videos)")
    s.add_argument("--reindex", action="store_true", help="re-process videos already indexed")
    s.add_argument("--limit", type=int, help="only the first N videos")
    s.add_argument("--no-action", action="store_true", help="skip the action-recognition model (faster)")

    s = sub.add_parser("build-gt", help="build ground truth from VIRAT annotations")
    s.add_argument("--annotations", help="annotation folder (default data/annotations)")
    s.add_argument("--videos", help="video folder used for fps + filtering (default data/videos)")
    s.add_argument("--out", help="output json (default data/ground_truth.json)")
    s.add_argument("--activities", nargs="*", help="only these activity names")
    s.add_argument("--all-videos", action="store_true", help="keep annotations for videos you do not have")
    s.add_argument("--min-instances", type=int, default=1)

    s = sub.add_parser("search", help="search with a text query")
    s.add_argument("query")
    s.add_argument("--top-k", type=int)
    s.add_argument("--threshold", type=float)
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("evaluate", help="evaluate against ground truth")
    s.add_argument("--gt", help="ground truth json (default data/ground_truth.json)")
    s.add_argument("--iou", type=float, help="temporal IoU needed to count as correct (default 0.3)")
    s.add_argument("--threshold", type=float, help="score threshold to evaluate at")
    s.add_argument("--sweep", action="store_true", help="also try many thresholds and report the best")

    sub.add_parser("stats", help="show index contents")

    s = sub.add_parser("clip", help="export a short mp4 for a time range")
    s.add_argument("video"); s.add_argument("start", type=float); s.add_argument("end", type=float)
    s.add_argument("--out"); s.add_argument("--pad", type=float, default=1.0)

    s = sub.add_parser("reset", help="delete the index and metadata (keeps videos)")
    s.add_argument("--yes", action="store_true")

    args = p.parse_args(argv)
    # our own messages at INFO; third-party libraries (open_clip, httpx, faiss...) only warnings
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("sceneseek").setLevel(logging.DEBUG if args.verbose else logging.INFO)
    cfg = Config()
    handler = {"ingest": _cmd_ingest, "build-gt": _cmd_build_gt, "search": _cmd_search,
               "evaluate": _cmd_evaluate, "stats": _cmd_stats, "clip": _cmd_clip,
               "reset": _cmd_reset}[args.cmd]
    handler(args, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
