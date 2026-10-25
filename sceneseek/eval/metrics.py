"""Evaluation metrics: Precision, Recall, F1, Recall@K, temporal IoU, latency.

Definitions (per query):
- A prediction is a (video_id, start, end) range, ranked best first.
- A prediction is *correct* if its temporal IoU with a not-yet-matched
  ground-truth interval in the same video is >= iou_threshold
  (greedy one-to-one matching in rank order).
- Precision = correct predictions / predictions
- Recall    = matched ground-truth intervals / ground-truth intervals
- F1        = harmonic mean of the two
- Recall@K  = matched ground-truth intervals using only the top K predictions / ground-truth intervals
- Hit@K     = 1 if any of the top K predictions is correct, else 0
- mean tIoU = average IoU over matched pairs (how tight the timestamps are)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


@dataclass
class Match:
    pred_index: int
    gt_index: int
    iou: float


def match(preds: list[dict], gts: list[dict], iou_threshold: float) -> list[Match]:
    """Greedy matching in prediction rank order. Items need video_id/start/end."""
    used: set[int] = set()
    matches = []
    for pi, p in enumerate(preds):
        best_j, best_iou = -1, 0.0
        for gj, g in enumerate(gts):
            if gj in used or g["video_id"] != p["video_id"]:
                continue
            iou = temporal_iou((p["start"], p["end"]), (g["start"], g["end"]))
            if iou > best_iou:
                best_j, best_iou = gj, iou
        if best_j >= 0 and best_iou >= iou_threshold:
            used.add(best_j)
            matches.append(Match(pi, best_j, best_iou))
    return matches


def query_metrics(preds: list[dict], gts: list[dict], iou_threshold: float = 0.3,
                  ks: tuple = (1, 5, 10)) -> dict:
    m = match(preds, gts, iou_threshold)
    tp = len(m)
    precision = tp / len(preds) if preds else (1.0 if not gts else 0.0)
    recall = tp / len(gts) if gts else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {
        "num_predictions": len(preds),
        "num_ground_truth": len(gts),
        "true_positives": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_tiou": float(np.mean([x.iou for x in m])) if m else 0.0,
        # best IoU each GT interval gets from any prediction (no threshold) - shows near misses
        "mean_best_tiou_per_gt": float(np.mean([
            max((temporal_iou((p["start"], p["end"]), (g["start"], g["end"]))
                 for p in preds if p["video_id"] == g["video_id"]), default=0.0) for g in gts
        ])) if gts else 0.0,
    }
    for k in ks:
        mk = match(preds[:k], gts, iou_threshold)
        out[f"recall@{k}"] = len(mk) / len(gts) if gts else 1.0
        out[f"hit@{k}"] = 1.0 if mk else 0.0
    return out


def aggregate(per_query: list[dict], latencies_ms: list[float]) -> dict:
    """Macro average over queries, plus micro P/R/F1 over all predictions and GT."""
    if not per_query:
        return {}
    keys = [k for k in per_query[0] if isinstance(per_query[0][k], float)]
    macro = {k: float(np.mean([q[k] for q in per_query])) for k in keys}
    tp = sum(q["true_positives"] for q in per_query)
    npred = sum(q["num_predictions"] for q in per_query)
    ngt = sum(q["num_ground_truth"] for q in per_query)
    mp = tp / npred if npred else 0.0
    mr = tp / ngt if ngt else 0.0
    lat = np.asarray(latencies_ms) if latencies_ms else np.zeros(1)
    return {
        "macro": macro,
        "micro": {"precision": mp, "recall": mr, "f1": 2 * mp * mr / (mp + mr) if mp + mr else 0.0,
                  "true_positives": tp, "predictions": npred, "ground_truth": ngt},
        "latency_ms": {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)),
                       "p95": float(np.percentile(lat, 95)), "max": float(lat.max())},
        "num_queries": len(per_query),
    }
