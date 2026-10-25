"""Stage 5: run every ground-truth query through the search engine and score it."""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..data.virat import load_ground_truth
from .metrics import aggregate, query_metrics

log = logging.getLogger(__name__)


def evaluate(engine, gt_path: Path, out_dir: Optional[Path] = None, iou_threshold: Optional[float] = None,
             top_k: Optional[int] = None, sweep: bool = False) -> dict:
    """Evaluate `engine` (a SceneSeek) against the ground-truth file.

    Only videos that are both indexed and annotated are used. With sweep=True,
    also reports metrics for a range of score thresholds and the one with best F1
    (filtering predictions by score after search, which is a close approximation
    of re-running search at each threshold).
    """
    cfg = engine.cfg
    iou_threshold = cfg.iou_threshold if iou_threshold is None else iou_threshold
    ks = tuple(cfg.recall_ks)
    top_k = top_k or max(ks)
    out_dir = Path(out_dir or cfg.eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(gt_path)
    indexed = {v["video_id"] for v in engine.list_videos()}
    if not indexed:
        raise RuntimeError("Nothing is indexed yet - run `python -m sceneseek ingest` first.")

    per_query, latencies, raw = [], [], []
    for q in gt["queries"]:
        gts = [g for g in q["instances"] if g["video_id"] in indexed]
        if not gts:
            continue
        # search without a threshold so the sweep can apply any threshold later
        resp = engine.search(q["query"], top_k=top_k, threshold=-1.0, video_ids=indexed, log_history=False)
        latencies.append(resp.latency_ms)
        preds = [r.to_dict() for r in resp.results]
        raw.append((q, gts, preds))
        kept = [p for p in preds if p["score"] >= cfg.score_threshold]
        m = query_metrics(kept, gts, iou_threshold, ks)
        m.update({"query_id": q["query_id"], "query": q["query"], "latency_ms": resp.latency_ms,
                  "best_score": resp.best_score})
        per_query.append(m)
        log.info("%-28s P=%.2f R=%.2f F1=%.2f R@%d=%.2f tIoU=%.2f  (%d GT, %.0f ms)", q["query_id"][:28],
                 m["precision"], m["recall"], m["f1"], ks[-1], m[f"recall@{ks[-1]}"], m["mean_tiou"],
                 len(gts), resp.latency_ms)

    if not per_query:
        raise RuntimeError("No ground-truth instances fall in the indexed videos. "
                           "Check video ids in ground_truth.json match the indexed file names.")

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ground_truth": str(gt_path),
        "settings": {"iou_threshold": iou_threshold, "score_threshold": cfg.score_threshold,
                     "top_k": top_k, "clip_model": f"{cfg.clip_model}/{cfg.clip_pretrained}",
                     "action_model": cfg.action_model if cfg.use_action_model else None,
                     "segment_seconds": cfg.segment_seconds, "segment_stride": cfg.segment_stride,
                     "indexed_videos": len(indexed)},
        "summary": aggregate(per_query, latencies),
        "per_query": per_query,
    }
    # metrics at several IoU thresholds (common in temporal localisation papers)
    report["by_iou"] = {}
    for t in (0.1, 0.3, 0.5, 0.7):
        rows = [query_metrics([p for p in preds if p["score"] >= cfg.score_threshold], gts, t, ks)
                for _, gts, preds in raw]
        report["by_iou"][str(t)] = aggregate(rows, latencies)["macro"]

    if sweep:
        all_scores = [p["score"] for _, _, preds in raw for p in preds]
        lo, hi = (min(all_scores), max(all_scores)) if all_scores else (0.0, 1.0)
        sweep_rows = []
        for t in np.linspace(lo, hi, 25):
            rows = [query_metrics([p for p in preds if p["score"] >= t], gts, iou_threshold, ks)
                    for _, gts, preds in raw]
            agg = aggregate(rows, latencies)
            sweep_rows.append({"threshold": round(float(t), 4), **{k: agg["macro"][k] for k in
                               ("precision", "recall", "f1")}, "micro_f1": agg["micro"]["f1"]})
        best = max(sweep_rows, key=lambda r: r["f1"])
        report["threshold_sweep"] = {"rows": sweep_rows, "best": best}
        log.info("Best threshold by macro F1: %.4f (F1=%.3f). Set SCENESEEK_SCORE_THRESHOLD to use it.",
                 best["threshold"], best["f1"])

    stamp = time.strftime("%Y%m%d_%H%M%S")
    for name in (f"report_{stamp}.json", "latest.json"):
        (out_dir / name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(out_dir / "latest_per_query.csv", "w", newline="", encoding="utf-8") as fh:
        cols = ["query_id", "query", "num_ground_truth", "num_predictions", "true_positives", "precision",
                "recall", "f1", *[f"recall@{k}" for k in ks], *[f"hit@{k}" for k in ks], "mean_tiou",
                "mean_best_tiou_per_gt", "best_score", "latency_ms"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_query)
    log.info("Report written to %s", out_dir / "latest.json")
    return report


def load_latest_report(eval_dir: Path) -> Optional[dict]:
    p = Path(eval_dir) / "latest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
