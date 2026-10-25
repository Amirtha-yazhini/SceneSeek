# SceneSeek

Natural-language search over surveillance video. Type a description ("a person getting out of a car"), get back the matching videos and timestamp ranges.

Pipeline: **video → segments → YOLO objects + OpenCLIP embeddings + action recognition → FAISS index (+ MongoDB metadata) → text query → merged timestamp ranges**, with an evaluation harness against VIRAT ground truth.

The Streamlit dashboard is not in this repo yet. See [HANDOFF.md](HANDOFF.md).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r requirements.txt
```

Optional:
- **MongoDB** on `mongodb://localhost:27017`. If it is not running, metadata is stored as JSON files in `data/index/` automatically.
- **FFmpeg** on PATH, used to cut browser-playable preview clips. Without it, clips are written with OpenCV, and some browsers won't play those inline.

The first run downloads model weights: OpenCLIP ViT-B-32 (~600 MB), R3D-18 (~130 MB) and YOLO11n (~6 MB).

## Data

```
data/
  videos/        put VIRAT .mp4 files here
  annotations/   git clone https://gitlab.kitware.com/viratdata/viratannotations.git data/annotations
```

Videos: https://viratdata.org (VIRAT Ground 2.0). File names must match the annotation names, e.g. `VIRAT_S_000001.mp4` ↔ `VIRAT_S_000001.activities.yml`. The original release's `*.viratdata.events.txt` files are also supported.

## Usage

```bash
python -m sceneseek ingest                 # index everything in data/videos (skips already indexed)
python -m sceneseek build-gt               # VIRAT annotations -> data/ground_truth.json
python -m sceneseek search "a person entering a building through a door"
python -m sceneseek evaluate --sweep       # metrics + best score threshold -> data/eval/latest.json
python -m sceneseek stats
python -m sceneseek clip data/videos/X.mp4 12.0 18.5
python -m sceneseek reset --yes            # wipe index + metadata (keeps videos)
```

Useful flags: `ingest --limit 3`, `ingest --no-action` (faster, skips the action model), `ingest --reindex`, `search --json`, `evaluate --iou 0.5`.

No VIRAT data yet? `python scripts/get_sample_videos.py` downloads 7 short public surveillance-style videos (about 7 minutes in total: parking lot, hallway, store aisle, warehouse, pedestrians). `data/ground_truth.json` is hand-labelled for those videos, so `ingest` then `evaluate` works straight away. `python scripts/make_demo_video.py` writes a tiny synthetic clip instead.

## How it works

| Stage | Code | What happens |
|---|---|---|
| Frames | [sceneseek/video/frames.py](sceneseek/video/frames.py) | OpenCV decodes each video once at 8 fps and slices it into 2 s windows every 1 s. |
| Objects | [sceneseek/models/detector.py](sceneseek/models/detector.py) | YOLO11n finds people, vehicles and bags on 3 frames per segment. |
| Visual-text embedding | [sceneseek/models/clip_encoder.py](sceneseek/models/clip_encoder.py) | OpenCLIP embeds the frames (averaged) **and** up to 4 object crops per segment. In wide VIRAT shots people are small, so the crops are what let "black hoodie" match. |
| Actions | [sceneseek/models/action.py](sceneseek/models/action.py) | R3D-18 (Kinetics-400) classifies 16 frames per segment. Its top-5 labels become one CLIP text embedding, so queries can be compared to actions without fine-tuning. |
| Index | [sceneseek/index/](sceneseek/index/) | Two FAISS inner-product indexes (`visual`, `action`) plus metadata (videos, segments, search history) in MongoDB or JSON. |
| Search | [sceneseek/pipeline/search.py](sceneseek/pipeline/search.py) | Embeds the query (prompt-ensembled), pulls candidates from FAISS and re-scores them: `score = best visual similarity + 0.25 × (action similarity − query mean) + small object-name boost`. Then it grows a time range around each peak over neighbouring segments within 0.03 of the peak (max 30 s). Results below the threshold → "No matching event found". |
| Ground truth | [sceneseek/data/virat.py](sceneseek/data/virat.py) | VIRAT activity → natural-language query + list of (video, start, end). Overlapping instances are merged. Hand-edit `data/ground_truth.json` freely. |
| Evaluation | [sceneseek/eval/](sceneseek/eval/) | Precision, Recall, F1, Recall@K / Hit@K (K=1,5,10), temporal IoU, latency (mean/p50/p95). A prediction counts as correct at tIoU ≥ 0.3 (also reported at 0.1/0.5/0.7). `--sweep` finds the best score threshold. |

All settings are in [sceneseek/config.py](sceneseek/config.py) and can be overridden with environment variables, e.g. `SCENESEEK_SCORE_THRESHOLD=0.24`, `SCENESEEK_DEVICE=cuda`, `SCENESEEK_USE_MONGO=false`.

## Tests

```bash
python -m pytest -q
```

The tests use a fake CLIP model, so they run in seconds without weights or MongoDB.

## Not done yet

- Streamlit dashboard: see [HANDOFF.md](HANDOFF.md).
- Fine-tuning: the plan says "only if needed". Run `evaluate` on VIRAT first and see which activities score poorly.
- No results are measured on VIRAT yet.

## Results on the sample videos

7 videos, 411 segments, 13 hand-labelled queries (labels are approximate, ±1 s), threshold 0.33, CPU only. Indexing took ~13.5 min for ~7 min of video. Search takes ~0.3 s.

| Metric (tIoU ≥ 0.3) | Value |
|---|---|
| Hit@1: the top result is a correct event | 0.54 (7 of 13 queries) |
| Recall@5 | 0.79 |
| Recall (top 10) | 0.82 |
| Precision (top 10) | 0.25 (expected to be low: up to 10 results are returned per query, and most queries have 1 true event) |
| Mean temporal IoU of correct results | 0.58 |

What works: the right video is almost always ranked first, and people, cars, bicycles and clothing colours are found. Off-topic queries ("a dog running on grass") return "No matching event found".

What doesn't: CLIP is weak at **counting** ("two workers" vs one) and at telling **similar colours** apart on small cars (silver vs white; that query scored 0). The Kinetics action labels are mostly noise on this footage ("hopscotch", "sweeping floor"), so they add little. That fits the plan's "fine-tune only if needed" step.
