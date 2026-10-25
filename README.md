# SceneSeek

**Natural-Language Semantic Search for Surveillance Videos**

SceneSeek allows users to search surveillance and CCTV footage using natural-language queries (e.g., *"a person riding a bicycle"*, *"a red car driving in a parking lot"*). The system returns ranked matching videos with precise timestamp ranges, similarity scores, object/action detections, and inline playback.

Pipeline: **Video → 2.0s Segments → YOLO11n Objects + OpenCLIP (LAION-2B) Embeddings + R3D-18 Action Recognition → Dual FAISS Index + Local JSON Metadata Storage → Text Query (Prompt Ensembled) → Merged Timestamp Ranges → Streamlit Web Dashboard**.

---

## Key Features

- **Natural-Language Video Search:** Search by activity, person description, clothing, or vehicle appearance without manual video tagging.
- **Temporal Event Localization:** Merges contiguous segment detections into cohesive timestamp ranges with inline playback seeked to event onset.
- **Interactive Streamlit Dashboard:** Complete, modern multi-page interface featuring:
  - **Dashboard:** Real-time statistics, search box, recent searches, and quick-query triggers.
  - **Video Search:** Full semantic search, similarity threshold controls, top-K selection, video filter, and split-view preview/cards.
  - **Video Library:** Browse all indexed videos, inspect resolution/fps, view keyframe timelines, and upload new videos.
  - **Search History:** Query history persisted across restarts in JSON with one-click re-run.
  - **Model Evaluation:** Quantitative metrics (Precision, Recall, F1, Recall@K, Hit@K, Temporal IoU, Latency, and Threshold Sweep chart).
  - **Settings:** Active storage backend display, vector index statistics, model configuration, and MongoDB Atlas configuration guide.
- **Local JSON Metadata Storage:** Completely functional on a laptop without requiring MongoDB or any external database server. All data, segment indices, and search histories persist across application restarts.
- **MongoDB Atlas Ready:** The backend `MetadataStore` interface is decoupled so that when cloud credentials arrive, setting `SCENESEEK_USE_MONGO=true` immediately switches storage without modifying the dashboard.

---

## Windows PowerShell Setup Instructions

### 1. Create and Activate Virtual Environment

```powershell
# Navigate to the project root
cd c:\Users\valli\Documents\SceneSeek

# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

*(If script execution is disabled in PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once).*

### 2. Install Dependencies

Install PyTorch CPU build first, then the remaining project requirements:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Download the 7 Demonstration Videos

```powershell
python scripts/get_sample_videos.py
```

Downloads the 7 short surveillance sample videos (~50 MB total) into `data/videos/`.

### 4. Ingest & Index Demonstration Videos

```powershell
python -m sceneseek ingest
```

Extracts frames at 8 fps, runs YOLO11n object detection, extracts OpenCLIP visual embeddings for frames & object crops, computes R3D-18 action embeddings, builds FAISS index files in `data/index/`, and saves metadata to JSON.

### 5. Run Evaluation Benchmark (Optional)

```powershell
python -m sceneseek evaluate --sweep
```

Runs the 13 ground-truth queries against the 7 sample videos and generates `data/eval/latest.json`.

---

## Running the Streamlit Dashboard

Launch the web dashboard with:

```powershell
streamlit run app.py
```

Open your browser to: **`http://localhost:8501`**

---

## Dashboard Pages

| Page | Description |
|---|---|
| **Dashboard** | System overview metrics (indexed videos, segments, total hours, search counts), direct search bar, quick benchmark queries, and recent search activity. |
| **Video Search** | Primary search interface. Type natural-language queries, adjust threshold and Top-K, preview matching video clips with seeked timestamps, and export mp4 event clips. |
| **Video Library** | Overview of all indexed videos with duration, resolution, frame rate, segment counts, keyframe galleries, and an upload interface for new surveillance footage. |
| **Search History** | Persistent log of past queries with execution timestamps, result counts, and latency, with one-click re-run functionality. |
| **Evaluation** | Quantitative benchmarks on the sample videos: Precision, Recall, F1, Recall@1/5/10, Hit@1/5/10, Temporal IoU, and interactive similarity threshold sweep chart. |
| **Settings** | Active metadata backend (Local JSON), FAISS vector counts, model weights details, and future MongoDB Atlas integration configuration. |

---

## Local JSON Metadata Storage & MongoDB Atlas

SceneSeek uses a modular storage abstraction defined in `sceneseek/index/metadata.py`:

- **Active Backend (Local JSON):**
  - Stored in `data/index/` (`videos.json`, `segments.json`, `searches.json`).
  - No database server or connection URI is required.
  - Automatically persists all video data and search logs across application restarts.
  - Eliminates duplicate entries upon re-indexing.
- **MongoDB Atlas Integration (Future):**
  - MongoDB integration code (`MongoStore`) is fully preserved.
  - When MongoDB Atlas credentials become available, activate it by setting environment variables without changing any dashboard code:
    ```powershell
    $env:SCENESEEK_USE_MONGO="true"
    $env:SCENESEEK_MONGO_URI="mongodb+srv://<username>:<password>@cluster.mongodb.net/sceneseek"
    ```

---

## CLI Usage

```powershell
# Search from CLI
python -m sceneseek search "a person riding a bicycle"

# Output search result as JSON
python -m sceneseek search "a red car driving in a parking lot" --json

# View index statistics
python -m sceneseek stats

# Export a video clip for a specific timestamp range
python -m sceneseek clip data/videos/car-detection.mp4 15.0 19.5

# Re-index all videos
python -m sceneseek ingest --reindex

# Clear index and metadata (videos are preserved)
python -m sceneseek reset --yes
```

---

## Running Tests

Run the full automated test suite:

```powershell
python -m pytest -v
```

All 16 unit and integration tests execute and pass without requiring external services or GPU hardware.
