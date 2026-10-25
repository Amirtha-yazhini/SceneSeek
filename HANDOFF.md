# Handoff: building the SceneSeek Streamlit dashboard

Everything in the project plan except stage 6 (the Streamlit app) is built and smoke-tested. Your job is the dashboard from slide 8. The backend already gives you everything the mockup shows, through one Python object. You shouldn't need to touch the model code.

## 1. Get it running (≈15 min, mostly downloads)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install streamlit

python scripts/get_sample_videos.py   # 7 short real sample videos (~50 MB), matching data/ground_truth.json
python -m sceneseek ingest            # ~15 min on CPU; first run also downloads ~750 MB of model weights
python -m sceneseek evaluate --sweep  # fills data/eval/latest.json for the metrics panel
python -m sceneseek search "a person walking"
python -m pytest -q                   # 13 tests, a few seconds
```

MongoDB is optional. If nothing is listening on `localhost:27017`, metadata goes to JSON files in `data/index/` automatically and the API stays the same.

Once real VIRAT videos are in `data/videos/`, run `python -m sceneseek reset --yes` to drop the samples, then `ingest`, `build-gt` and `evaluate --sweep` (see README).

## 2. The API you build against

Create a new file such as `app.py` at the repo root, next to the `sceneseek/` folder.

```python
import streamlit as st
from sceneseek import SceneSeek

@st.cache_resource          # loading CLIP takes ~5-15 s; do it once per server
def get_engine():
    return SceneSeek.load()

engine = get_engine()
resp = engine.search(query, top_k=5)              # optional: threshold=0.25, video_ids=[...]
```

### `engine.search(...)` → `SearchResponse`

| Field | Type | Use it for |
|---|---|---|
| `resp.results` | `list[SearchResult]` | "Top results" list, best first. Only results above the threshold are included. |
| `resp.no_match` | `bool` | Show **"No matching event found."** when `True` |
| `resp.message` | `str` | `"No matching event found"` or `"3 matching event(s)"` |
| `resp.latency_ms` | `float` | "Search time" |
| `resp.best_score` | `float` | Best score even when nothing passed the threshold. Good for "closest was 0.19" hints. |
| `resp.threshold` | `float` | Threshold that was applied |
| `resp.to_dict()` | `dict` | JSON-safe version of all of the above |

### Each `SearchResult`

| Field | Example | Use it for |
|---|---|---|
| `video_id` | `"VIRAT_S_000001"` | label |
| `video_path` | absolute path to the .mp4 | `st.video(...)` |
| `start`, `end` | `12.0`, `18.0` (seconds) | seek the player |
| `timestamp_range` | `"00:12.0 - 00:18.0"` | "Timestamp range" row |
| `score` | `0.263` | "Similarity score" (fused; this is the one to show) |
| `visual_score`, `action_score` | `0.25`, `0.71` | optional "why it matched" detail; these are on different scales, so don't compare them to each other |
| `thumbnail` | path to a .jpg | result cards (`st.image`) |
| `objects` | `{"person": 2, "car": 1}` | chips/tags |
| `actions` | `[{"label": "walking the dog", "prob": 0.31}, ...]` | chips/tags (Kinetics labels; often noisy on surveillance video) |
| `segment_ids` | `["VIRAT_S_000001:12", ...]` | rarely needed |

### Other calls

| Call | Returns | Mockup section |
|---|---|---|
| `engine.list_videos()` | list of dicts: `video_id, path, duration, fps, width, height, num_segments, indexed_at` | **Library** |
| `engine.segments(video_id)` | every 2 s segment with `start, end, thumbnail, object_counts, actions` | Library detail / timeline |
| `engine.history(limit=50)` | past searches: `ts, query, latency_ms, num_results, best_score, top` | **History** (every search is logged automatically; pass `log_history=False` to skip) |
| `engine.stats()` | counts, hours indexed, backend, model, threshold | **Dashboard** home tiles |
| `engine.export_clip(result)` | path to a short .mp4 around the result | download button / preview |
| `engine.reload()` | — | call after new videos are ingested while the app is running |
| `engine.cfg` | `Config` object | **Settings** (e.g. `engine.cfg.score_threshold`) |
| `sceneseek.eval.evaluate.load_latest_report(engine.cfg.eval_dir)` | dict or `None` | **Evaluation metrics** panel |

Evaluation report shape (`data/eval/latest.json`, produced by `python -m sceneseek evaluate --sweep`):

```
summary.macro       -> precision, recall, f1, recall@1/5/10, hit@1/5/10, mean_tiou
summary.micro       -> precision, recall, f1
summary.latency_ms  -> mean, p50, p95, max
per_query[]         -> same metrics per query, plus query text
by_iou["0.5"]       -> macro metrics at stricter IoU
threshold_sweep     -> rows[] of {threshold, precision, recall, f1} + best   (nice line chart)
```

There's also `data/eval/latest_per_query.csv` if a table is easier.

## 3. Minimal working page

```python
import streamlit as st
from sceneseek import SceneSeek

st.set_page_config(page_title="SceneSeek", layout="wide")

@st.cache_resource
def get_engine():
    return SceneSeek.load()

engine = get_engine()
st.title("Video Search")
query = st.text_input("Describe an event")
if st.button("Search") and query:
    resp = engine.search(query, top_k=5)
    st.caption(f"Search time: {resp.latency_ms:.0f} ms")
    if resp.no_match:
        st.info("No matching event found. Try refining your description.")
    else:
        top = resp.results[0]
        left, right = st.columns([3, 2])
        with left:
            st.video(top.video_path, start_time=int(top.start), end_time=int(top.end) + 1)
            st.write(f"**Timestamp range:** {top.timestamp_range}")
            st.write(f"**Similarity score:** {top.score:.3f}")
        with right:
            for r in resp.results:
                st.image(r.thumbnail, width=160)
                st.write(f"{r.video_id} · {r.timestamp_range} · {r.score:.3f}")
```

## 4. Things to know

- **Model load time.** Without `@st.cache_resource`, every rerun reloads CLIP (5–15 s). One search takes ~0.3 s on CPU.
- **Preview playback.** VIRAT .mp4 files are H.264 and play in `st.video` directly. `export_clip` makes browser-playable clips only if FFmpeg is installed; without it, some browsers won't play the clip inline. FFmpeg isn't installed on the machine this was built on.
- **Threshold.** The default is `0.33`, tuned on the sample videos: real matches score ~0.37–0.43, off-topic queries ≤ 0.31. Re-check with `evaluate --sweep` once VIRAT is indexed. Exposing it as a slider in Settings would be useful (`engine.search(q, threshold=slider_value)`).
- **Scores are small numbers.** CLIP cosine similarities usually sit between 0.15 and 0.35, so a "good" match may only be 0.27. If you show a percentage or bar, rescale it, e.g. relative to the threshold.
- **Good demo queries on the samples:** "a red car driving in a parking lot", "a person riding a bicycle", "a woman in a white dress walking in a hallway", "three people walking together towards the camera". For the no-match state, try "a dog running on grass".
- **Paths are absolute and local.** `video_path` and `thumbnail` point into `data/` on the machine that ran `ingest`. Run `ingest` on your own machine; don't copy someone else's `data/index`.
- **Ingest from the UI.** If you want an upload page: save the file into `data/videos/`, then `Indexer().ingest_video(path, progress=callback)` (from `sceneseek.pipeline.ingest`), then `engine.reload()`. The callback receives `(video_id, fraction_done)` for `st.progress`. It's slow on CPU (~1.5× real time with all models), so keep it optional.
- **Not built on the backend:** "Saved Searches" from the mockup. History exists; saved searches would just be a list of queries you store (session state, a JSON file, or a `saved` collection in Mongo).

## 5. Where things are

```
sceneseek/
  config.py             all settings (env vars SCENESEEK_*)
  types.py              SearchResponse / SearchResult definitions
  pipeline/search.py    SceneSeek class  <- your entry point
  pipeline/ingest.py    Indexer (video -> index)
  eval/                 metrics + evaluate()
  data/virat.py         VIRAT annotations -> ground_truth.json
  video/frames.py       frame sampling, thumbnails, clip export
  models/               YOLO, OpenCLIP, action-recognition wrappers
tests/test_core.py      fast tests (fake CLIP); also a good usage example
```

If you hit something the API doesn't give you, it's probably a small addition in `pipeline/search.py` rather than in the dashboard.
