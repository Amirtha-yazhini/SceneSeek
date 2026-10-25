"""SceneSeek — Natural-Language Semantic Video Search Engine.

A clean, production-grade interface for neural surveillance video retrieval.
"""
from __future__ import annotations

import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from sceneseek import SceneSeek
from sceneseek.config import Config
from sceneseek.eval.evaluate import evaluate as run_eval, load_latest_report
from sceneseek.pipeline.ingest import Indexer
from sceneseek.types import SearchResponse, SearchResult, format_ts

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SceneSeek | Video Semantic Search",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Sophisticated Dark Theme CSS
# -----------------------------------------------------------------------------
MODERN_DARK_CSS = """
<style>
/* Global resets & typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #E6EDF3;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1440px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0D1117;
    border-right: 1px solid #21262D;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Brand header */
.brand-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #F0F6FC;
    letter-spacing: -0.025em;
    margin: 0;
    line-height: 1.2;
}
.brand-subtitle {
    font-size: 0.85rem;
    color: #8B949E;
    margin-top: 0.35rem;
    margin-bottom: 1.5rem;
    line-height: 1.4;
}

/* Navigation items in sidebar */
div[data-testid="stRadio"] > div {
    gap: 0.25rem;
}
div[data-testid="stRadio"] label {
    background: transparent;
    border: 1px solid transparent;
    padding: 0.5rem 0.85rem;
    border-radius: 6px;
    font-size: 0.875rem;
    font-weight: 500;
    color: #8B949E;
    transition: all 0.15s ease;
}
div[data-testid="stRadio"] label:hover {
    color: #F0F6FC;
    background: #161B22;
}
div[data-testid="stRadio"] label[data-checked="true"] {
    background: #161B22;
    border-color: #30363D;
    color: #F0F6FC;
}

/* Minimal Metric Cards */
.metric-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 1.15rem 1.25rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 104px;
}
.metric-card .label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #8B949E;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-size: 1.75rem;
    font-weight: 600;
    color: #F0F6FC;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.metric-card .subtext {
    font-size: 0.75rem;
    color: #6E7681;
    margin-top: 0.45rem;
}

/* Section Header */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #F0F6FC;
    margin-bottom: 0.75rem;
    letter-spacing: -0.01em;
}
.section-subtitle {
    font-size: 0.82rem;
    color: #8B949E;
    margin-top: -0.5rem;
    margin-bottom: 1rem;
}

/* Badges & Tags */
.badge-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
}
.badge-cyan {
    background: rgba(56, 189, 248, 0.1);
    color: #38BDF8;
    border: 1px solid rgba(56, 189, 248, 0.25);
}
.badge-emerald {
    background: rgba(46, 160, 67, 0.12);
    color: #3FB950;
    border: 1px solid rgba(46, 160, 67, 0.25);
}
.badge-muted {
    background: #21262D;
    color: #8B949E;
    border: 1px solid #30363D;
}
.chip-item {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    background: #161B22;
    color: #C9D1D9;
    border: 1px solid #30363D;
    border-radius: 4px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
}

/* Status summary bar */
.query-status-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 1.25rem;
    font-size: 0.82rem;
    color: #8B949E;
    background: #161B22;
    border: 1px solid #30363D;
    padding: 0.6rem 1rem;
    border-radius: 6px;
    margin-bottom: 1.25rem;
}
.query-status-item {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}
.query-status-item strong {
    color: #F0F6FC;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

/* Result Cards */
.result-card-box {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.65rem;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}
.result-card-box:hover {
    border-color: #484F58;
    background: #1C2128;
}
.result-card-selected {
    border-color: #38BDF8 !important;
    background: #141A22 !important;
}

/* Callout / Alert Panels */
.notice-panel {
    background: #161B22;
    border: 1px solid #30363D;
    border-left: 3px solid #38BDF8;
    border-radius: 6px;
    padding: 0.85rem 1.15rem;
    font-size: 0.85rem;
    color: #8B949E;
    margin-bottom: 1.25rem;
}
.notice-panel strong {
    color: #F0F6FC;
}
.notice-panel-warning {
    border-left-color: #D29922;
}

/* Spec Row in Video Details */
.spec-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.75rem;
    background: #0D1117;
    border: 1px solid #21262D;
    border-radius: 6px;
    padding: 0.85rem 1rem;
    margin-top: 0.75rem;
    margin-bottom: 1rem;
}
.spec-item .spec-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #6E7681;
    margin-bottom: 0.2rem;
}
.spec-item .spec-value {
    font-size: 0.88rem;
    font-weight: 600;
    color: #F0F6FC;
    font-family: 'JetBrains Mono', monospace;
}

/* Status dot */
.dot-indicator {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: #3FB950;
    display: inline-block;
    margin-right: 0.35rem;
}
</style>
"""
st.markdown(MODERN_DARK_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Engine Access & Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing SceneSeek engine and models...")
def get_engine() -> SceneSeek:
    """Load the SceneSeek backend once and cache across user interactions."""
    cfg = Config()
    cfg.ensure_dirs()
    return SceneSeek.load(cfg)


try:
    engine = get_engine()
except Exception as e:
    st.error(f"Failed to initialize SceneSeek search engine: {e}")
    st.info("Verify model weights and local data dependencies are installed.")
    st.stop()


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "last_search_response" not in st.session_state:
    st.session_state.last_search_response = None
if "selected_result_idx" not in st.session_state:
    st.session_state.selected_result_idx = 0


# -----------------------------------------------------------------------------
# Sidebar Navigation & System Status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand-title">SceneSeek</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Semantic Video Search Engine</div>', unsafe_allow_html=True)

    pages = [
        "Dashboard",
        "Video Search",
        "Video Library",
        "Search History",
        "Evaluation",
        "Settings",
    ]

    selected_page = st.radio(
        "Navigation",
        pages,
        index=pages.index(st.session_state.current_page)
        if st.session_state.current_page in pages
        else 0,
        label_visibility="collapsed",
    )
    st.session_state.current_page = selected_page

    st.write("")
    st.divider()

    # System Status Panel
    st.markdown('<div class="section-title" style="font-size: 0.85rem;">System Overview</div>', unsafe_allow_html=True)
    stats = engine.stats()
    num_vids = stats.get("videos", 0)
    num_segs = stats.get("segments", 0)
    device_label = engine.cfg.resolved_device().upper()

    st.markdown(
        f"""
        <div style="font-size: 0.8rem; line-height: 1.8; color: #8B949E;">
            <div>Backend: <span class="badge-pill badge-emerald">Local JSON</span></div>
            <div>Indexed: <b style="color: #E6EDF3;">{num_vids}</b> videos ({num_segs} segments)</div>
            <div>Accelerator: <b style="color: #E6EDF3;">{device_label}</b></div>
            <div>Status: <span><span class="dot-indicator"></span>Ready</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    if st.button("Reload Index", use_container_width=True):
        engine.reload()
        st.cache_resource.clear()
        st.success("Index refreshed.")
        st.rerun()


# =============================================================================
# PAGE 1: DASHBOARD
# =============================================================================
if st.session_state.current_page == "Dashboard":
    st.markdown('<h2 style="font-weight: 700; color: #F0F6FC; margin: 0 0 0.25rem 0;">Overview</h2>', unsafe_allow_html=True)
    st.markdown('<p class="brand-subtitle">Natural-language semantic video retrieval over surveillance archives</p>', unsafe_allow_html=True)

    # Search Bar on Dashboard
    dash_search_col1, dash_search_col2 = st.columns([5, 1])
    with dash_search_col1:
        dash_query = st.text_input(
            "Quick Search",
            value="",
            placeholder="Search events, e.g., 'a red car driving in a parking lot' or 'a person riding a bicycle'...",
            label_visibility="collapsed",
            key="dash_search_input",
        )
    with dash_search_col2:
        dash_btn = st.button("Search Archive", type="primary", use_container_width=True)

    if dash_btn and dash_query.strip():
        st.session_state.search_query = dash_query.strip()
        st.session_state.current_page = "Video Search"
        st.rerun()

    st.write("")

    # Summary Statistics Tiles
    stats = engine.stats()
    history_entries = engine.history(limit=100)
    total_searches = len(history_entries)
    total_duration_hours = stats.get("hours_indexed", 0.0)
    total_duration_mins = total_duration_hours * 60

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Indexed Videos</div>
                <div class="value">{stats.get('videos', 0)}</div>
                <div class="subtext">Processed surveillance recordings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Indexed Segments</div>
                <div class="value">{stats.get('segments', 0)}</div>
                <div class="subtext">2.0s windows with 1.0s stride</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        dur_display = (
            f"{total_duration_mins:.1f}m"
            if total_duration_mins < 60
            else f"{total_duration_hours:.2f}h"
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Archived Duration</div>
                <div class="value">{dur_display}</div>
                <div class="subtext">Available for semantic search</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Executed Searches</div>
                <div class="value">{total_searches}</div>
                <div class="subtext">Persisted in local audit log</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.divider()

    # Split View: Recent Activity & Quick Benchmark Queries
    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.markdown('<div class="section-title">Recent Query Activity</div>', unsafe_allow_html=True)
        recent_history = engine.history(limit=5)
        if recent_history:
            for item in recent_history:
                ts_val = item.get("ts", time.time())
                ts_str = datetime.fromtimestamp(ts_val).strftime("%b %d, %H:%M:%S") if ts_val else "Recent"
                q_text = item.get("query", "")
                num_res = item.get("num_results", 0)
                lat = item.get("latency_ms", 0.0)

                col_q, col_act = st.columns([4, 1])
                with col_q:
                    st.markdown(f"**\"{q_text}\"**")
                    st.caption(f"{ts_str} · {num_res} match(es) · {lat:.0f} ms")
                with col_act:
                    if st.button("Run Query", key=f"dash_run_{ts_val}_{q_text}"):
                        st.session_state.search_query = q_text
                        st.session_state.current_page = "Video Search"
                        st.rerun()
                st.write("---")
        else:
            st.info("No queries executed yet. Use the search input above to begin.")

    with c_right:
        st.markdown('<div class="section-title">Benchmark Queries</div>', unsafe_allow_html=True)
        st.caption("Select a validated query from the test ground-truth dataset:")

        sample_queries = [
            "a red car driving in a parking lot",
            "a person riding a bicycle",
            "a woman in a white dress walking in a hallway",
            "three people walking together towards the camera",
            "a dog running on grass",
        ]

        for sq in sample_queries:
            if st.button(f"\"{sq}\"", key=f"quick_q_{sq}", use_container_width=True):
                st.session_state.search_query = sq
                st.session_state.current_page = "Video Search"
                st.rerun()

        st.write("")
        st.markdown('<div class="section-title">Pipeline Architecture</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size: 0.82rem; color: #8B949E; line-height: 1.7;">
                • <b>Multimodal Model:</b> OpenCLIP ViT-B-32 (LAION-2B)<br>
                • <b>Object Detector:</b> YOLO11n (COCO classes)<br>
                • <b>Action Recognition:</b> R3D-18 (Kinetics-400)<br>
                • <b>Similarity Index:</b> Dual FAISS FlatIP (Visual & Action)<br>
                • <b>Storage Engine:</b> Local JSON (MongoDB Atlas ready)
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE 2: VIDEO SEARCH
# =============================================================================
elif st.session_state.current_page == "Video Search":
    st.markdown('<h2 style="font-weight: 700; color: #F0F6FC; margin: 0 0 0.25rem 0;">Video Search</h2>', unsafe_allow_html=True)
    st.markdown('<p class="brand-subtitle">Natural-language query with temporal localization and timestamped playback</p>', unsafe_allow_html=True)

    # Search Bar
    s_col1, s_col2 = st.columns([5, 1])
    with s_col1:
        query = st.text_input(
            "Natural-language query",
            value=st.session_state.search_query,
            placeholder="Describe an event (e.g., 'a red car driving in a parking lot' or 'a person riding a bicycle')...",
            label_visibility="collapsed",
        )
    with s_col2:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    # Search Parameters Expander
    with st.expander("Search Parameters & Filters", expanded=False):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            top_k = st.slider("Top-K Results", min_value=1, max_value=20, value=engine.cfg.default_top_k)
        with f_col2:
            threshold = st.slider(
                "Similarity Threshold",
                min_value=0.10,
                max_value=0.60,
                value=float(engine.cfg.score_threshold),
                step=0.01,
                help="Candidates scoring below this threshold are filtered out.",
            )
        with f_col3:
            all_vids = [v["video_id"] for v in engine.list_videos()]
            selected_vids = st.multiselect("Filter Video Scope", options=all_vids, default=[])

    # Sample query chips
    st.markdown(
        """
        <div style="font-size: 0.78rem; font-weight: 500; color: #8B949E; margin-bottom: 0.5rem;">
            SAMPLE QUERIES:
        </div>
        """,
        unsafe_allow_html=True,
    )
    chip_cols = st.columns(5)
    demo_chips = [
        "a red car driving in a parking lot",
        "a person riding a bicycle",
        "a woman in a white dress walking in a hallway",
        "three people walking together towards the camera",
        "a dog running on grass",
    ]
    for i, chip_text in enumerate(demo_chips):
        with chip_cols[i]:
            if st.button(chip_text, key=f"chip_btn_{i}", use_container_width=True):
                st.session_state.search_query = chip_text
                st.rerun()

    # Execute Search
    if (search_clicked or (query and st.session_state.last_search_response is None)) and query.strip():
        st.session_state.search_query = query.strip()
        v_filter = selected_vids if selected_vids else None

        with st.spinner("Executing similarity search..."):
            resp: SearchResponse = engine.search(
                query.strip(),
                top_k=top_k,
                threshold=threshold,
                video_ids=v_filter,
                log_history=True,
            )
            st.session_state.last_search_response = resp
            st.session_state.selected_result_idx = 0

    # Display Search Results
    resp: Optional[SearchResponse] = st.session_state.last_search_response
    if resp is not None and resp.query:
        st.divider()

        # Metrics status bar
        st.markdown(
            f"""
            <div class="query-status-bar">
                <span class="query-status-item">Latency: <strong>{resp.latency_ms:.1f} ms</strong></span>
                <span>·</span>
                <span class="query-status-item">Applied Threshold: <strong>{resp.threshold:.2f}</strong></span>
                <span>·</span>
                <span class="query-status-item">Top Score: <strong>{resp.best_score:.3f}</strong></span>
                <span>·</span>
                <span class="query-status-item">Matches: <strong>{len(resp.results)}</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if resp.no_match:
            st.markdown(
                f"""
                <div class="notice-panel notice-panel-warning">
                    <strong>No matching event found.</strong><br>
                    No video segments met the similarity threshold of <b>{resp.threshold:.2f}</b>.
                    The top candidate scored <b>{resp.best_score:.3f}</b>.
                    Consider refining the query description or reducing the threshold parameter.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            results = resp.results
            curr_idx = min(st.session_state.selected_result_idx, len(results) - 1)
            active_res: SearchResult = results[curr_idx]

            left_col, right_col = st.columns([3, 2])

            with left_col:
                st.markdown(f'<div class="section-title">Video Playback — {active_res.video_id}</div>', unsafe_allow_html=True)

                vid_path = Path(active_res.video_path)
                if vid_path.exists():
                    st.video(
                        str(vid_path),
                        start_time=int(max(0.0, active_res.start - 0.5)),
                    )
                else:
                    st.warning(f"File missing at path: {active_res.video_path}")

                # Structured event specification panel
                st.markdown(
                    f"""
                    <div class="spec-grid">
                        <div class="spec-item">
                            <div class="spec-label">Timestamp Range</div>
                            <div class="spec-value">{active_res.timestamp_range}</div>
                        </div>
                        <div class="spec-item">
                            <div class="spec-label">Combined Score</div>
                            <div class="spec-value" style="color: #38BDF8;">{active_res.score:.3f}</div>
                        </div>
                        <div class="spec-item">
                            <div class="spec-label">Duration</div>
                            <div class="spec-value">{active_res.end - active_res.start:.1f}s</div>
                        </div>
                        <div class="spec-item">
                            <div class="spec-label">Sub-segments Merged</div>
                            <div class="spec-value">{len(active_res.segment_ids)}</div>
                        </div>
                        <div class="spec-item">
                            <div class="spec-label">Visual Score (CLIP)</div>
                            <div class="spec-value">{active_res.visual_score:.3f}</div>
                        </div>
                        <div class="spec-item">
                            <div class="spec-label">Action Score (R3D)</div>
                            <div class="spec-value">{active_res.action_score:.3f}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Detections
                c_obj, c_act = st.columns(2)
                with c_obj:
                    st.markdown('<div class="spec-label" style="margin-bottom: 0.35rem;">Detected Objects</div>', unsafe_allow_html=True)
                    if active_res.objects:
                        obj_html = "".join(
                            f'<span class="chip-item">{k}: {v}</span>' for k, v in active_res.objects.items()
                        )
                        st.markdown(obj_html, unsafe_allow_html=True)
                    else:
                        st.caption("No prominent objects detected.")
                with c_act:
                    st.markdown('<div class="spec-label" style="margin-bottom: 0.35rem;">Action Predictions</div>', unsafe_allow_html=True)
                    if active_res.actions:
                        act_html = "".join(
                            f'<span class="chip-item" style="border-color: #384457;">{a["label"]} ({a["prob"]*100:.0f}%)</span>'
                            for a in active_res.actions
                        )
                        st.markdown(act_html, unsafe_allow_html=True)
                    else:
                        st.caption("No action classification available.")

                st.write("")
                if st.button("Export Event Clip (.mp4)", key="export_clip_btn"):
                    with st.spinner("Extracting segment clip..."):
                        try:
                            clip_p = engine.export_clip(active_res)
                            st.success(f"Generated clip: {clip_p.name}")
                            with open(clip_p, "rb") as fh:
                                st.download_button(
                                    label="Download Clip",
                                    data=fh.read(),
                                    file_name=clip_p.name,
                                    mime="video/mp4",
                                )
                        except Exception as ex:
                            st.error(f"Clip extraction failed: {ex}")

            with right_col:
                st.markdown(f'<div class="section-title">Matching Segments ({len(results)})</div>', unsafe_allow_html=True)
                st.caption("Select a segment card to update playback:")

                for idx, r in enumerate(results):
                    is_active = idx == curr_idx
                    active_class = "result-card-selected" if is_active else ""

                    with st.container():
                        st.markdown(
                            f"""
                            <div class="result-card-box {active_class}">
                                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                                    <div>
                                        <div style="font-weight: 600; font-size: 0.9rem; color: #F0F6FC;">{r.video_id}</div>
                                        <div style="font-size: 0.8rem; color: #8B949E; font-family: 'JetBrains Mono', monospace; margin-top: 0.2rem;">
                                            {r.timestamp_range} ({r.end - r.start:.1f}s)
                                        </div>
                                    </div>
                                    <span class="badge-pill badge-cyan">
                                        Score: {r.score:.3f}
                                    </span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        col_th, col_btn = st.columns([2, 3])
                        with col_th:
                            if r.thumbnail and Path(r.thumbnail).exists():
                                st.image(r.thumbnail, use_container_width=True)
                            else:
                                st.caption("No thumbnail")
                        with col_btn:
                            btn_label = "Active Selection" if is_active else "Inspect Segment"
                            if st.button(
                                btn_label,
                                key=f"sel_res_{idx}",
                                use_container_width=True,
                                disabled=is_active,
                            ):
                                st.session_state.selected_result_idx = idx
                                st.rerun()

                        st.write("---")


# =============================================================================
# PAGE 3: VIDEO LIBRARY
# =============================================================================
elif st.session_state.current_page == "Video Library":
    st.markdown('<h2 style="font-weight: 700; color: #F0F6FC; margin: 0 0 0.25rem 0;">Video Library</h2>', unsafe_allow_html=True)
    st.markdown('<p class="brand-subtitle">Catalog of indexed video recordings, keyframe timelines, and video upload</p>', unsafe_allow_html=True)

    videos = engine.list_videos()

    # Video Upload Expander
    with st.expander("Upload & Ingest Video Footage", expanded=False):
        st.write("Upload an MP4, AVI, or MOV file to extract frames, compute neural embeddings, and add to the FAISS index.")
        uploaded_file = st.file_uploader("Select video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            upload_btn = st.button("Process & Index Video", type="primary")
            if upload_btn:
                save_dest = engine.cfg.videos_dir / uploaded_file.name
                with open(save_dest, "wb") as out_f:
                    out_f.write(uploaded_file.getbuffer())

                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_prog(v_id: str, frac: float):
                    progress_bar.progress(frac)
                    status_text.text(f"Processing '{v_id}': {frac*100:.0f}%")

                try:
                    indexer = Indexer(engine.cfg)
                    status_text.text(f"Running detection and feature extraction on '{uploaded_file.name}'...")
                    indexer.ingest_video(save_dest, reindex=True, progress=update_prog)
                    engine.reload()
                    progress_bar.progress(1.0)
                    st.success(f"Successfully indexed '{uploaded_file.name}'. Footage is now searchable.")
                    time.sleep(1)
                    st.rerun()
                except Exception as ex:
                    st.error(f"Ingestion failed: {ex}")

    st.write("")
    st.markdown(f'<div class="section-title">Indexed Video Archives ({len(videos)})</div>', unsafe_allow_html=True)

    if not videos:
        st.warning("No video files indexed yet.")
    else:
        for v in videos:
            vid_id = v.get("video_id", "Unknown")
            dur = v.get("duration", 0.0)
            fps = v.get("fps", 0.0)
            res = f"{v.get('width', 0)} x {v.get('height', 0)}"
            segs_count = v.get("num_segments", 0)
            vpath = v.get("path", "")

            with st.container():
                st.markdown(
                    f"""
                    <div style="background: #161B22; border: 1px solid #30363D; border-radius: 6px; padding: 1rem 1.25rem; margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.4rem;">
                            <span style="font-weight: 600; font-size: 0.95rem; color: #F0F6FC;">{vid_id}</span>
                            <span class="badge-pill badge-muted">{segs_count} Segments</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #8B949E; font-family: 'JetBrains Mono', monospace;">
                            Duration: {format_ts(dur)} ({dur:.1f}s) · Resolution: {res} · Framerate: {fps:.1f} fps
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_v, col_info = st.columns([3, 2])
                with col_v:
                    if Path(vpath).exists():
                        st.video(str(vpath))
                    else:
                        st.caption(f"File missing at: {vpath}")
                with col_info:
                    st.markdown('<div class="spec-label">Segment Keyframes</div>', unsafe_allow_html=True)
                    segs = engine.segments(vid_id)
                    st.caption(f"Displaying sample keyframes across {len(segs)} windows:")

                    thumb_cols = st.columns(3)
                    for idx, s in enumerate(segs[:6]):
                        th_p = s.get("thumbnail")
                        with thumb_cols[idx % 3]:
                            if th_p and Path(th_p).exists():
                                st.image(th_p, caption=f"{format_ts(s['start'])}", use_container_width=True)

                st.write("---")


# =============================================================================
# PAGE 4: SEARCH HISTORY
# =============================================================================
elif st.session_state.current_page == "Search History":
    st.markdown('<h2 style="font-weight: 700; color: #F0F6FC; margin: 0 0 0.25rem 0;">Search History</h2>', unsafe_allow_html=True)
    st.markdown('<p class="brand-subtitle">Persistent audit log of executed natural-language queries</p>', unsafe_allow_html=True)

    history = engine.history(limit=100)

    if not history:
        st.info("No search queries logged yet.")
    else:
        avg_latency = sum(h.get("latency_ms", 0.0) for h in history) / max(len(history), 1)
        h1, h2, h3 = st.columns(3)
        with h1:
            st.metric("Total Executions", len(history))
        with h2:
            st.metric("Mean Latency", f"{avg_latency:.1f} ms")
        with h3:
            st.metric("Storage Layer", "Local JSON Storage")

        st.write("")
        st.markdown('<div class="section-title">Execution Log</div>', unsafe_allow_html=True)

        for idx, entry in enumerate(history):
            q_text = entry.get("query", "")
            ts = entry.get("ts", time.time())
            ts_dt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            num_res = entry.get("num_results", 0)
            lat = entry.get("latency_ms", 0.0)
            best_sc = entry.get("best_score", 0.0)

            c_h1, c_h2, c_h3 = st.columns([4, 2, 1])
            with c_h1:
                st.markdown(f"**\"{q_text}\"**")
                st.caption(f"{ts_dt} · Top candidate score: {best_sc:.3f}")
            with c_h2:
                st.markdown(f"<span class='badge-pill badge-cyan'>{num_res} match(es)</span>", unsafe_allow_html=True)
                st.caption(f"Latency: {lat:.1f} ms")
            with c_h3:
                if st.button("Rerun", key=f"hist_rerun_{idx}"):
                    st.session_state.search_query = q_text
                    st.session_state.current_page = "Video Search"
                    st.rerun()

            st.write("---")


# =============================================================================
# PAGE 5: EVALUATION
# =============================================================================
elif st.session_state.current_page == "Evaluation":
    st.markdown('<h2 style="font-weight: 700; color: #F0F6FC; margin: 0 0 0.25rem 0;">Model Evaluation</h2>', unsafe_allow_html=True)
    st.markdown('<p class="brand-subtitle">Quantitative performance benchmark against ground-truth surveillance annotations</p>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="notice-panel">
            <strong>Benchmark Scope:</strong> Quantitative metrics evaluated on the <b>7 sample demonstration videos</b>
            against hand-labelled annotations (<code>data/ground_truth.json</code>).
            Temporal IoU threshold is fixed at <b>0.30</b> as standard for action detection.
        </div>
        """,
        unsafe_allow_html=True,
    )

    report = load_latest_report(engine.cfg.eval_dir)

    c_run1, c_run2 = st.columns([4, 1])
    with c_run1:
        if report:
            created_at = report.get("created_at", "Unknown")
            st.caption(f"Report generated at: **{created_at}**")
    with c_run2:
        if st.button("Run Sweep", type="secondary", use_container_width=True):
            with st.spinner("Computing evaluation sweep..."):
                try:
                    report = run_eval(engine, engine.cfg.ground_truth_path, sweep=True)
                    st.success("Evaluation sweep complete.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Evaluation failed: {ex}")

    if not report:
        st.warning("No evaluation report found in `data/eval/latest.json`. Click 'Run Sweep' above.")
    else:
        summary = report.get("summary", {})
        macro = summary.get("macro", {})
        latency_info = summary.get("latency_ms", {})

        # Primary Metric Cards
        e1, e2, e3, e4, e5 = st.columns(5)
        with e1:
            p_val = macro.get("precision")
            st.metric("Macro Precision", f"{p_val:.2f}" if p_val is not None else "Not evaluated")
        with e2:
            r_val = macro.get("recall")
            st.metric("Macro Recall", f"{r_val:.2f}" if r_val is not None else "Not evaluated")
        with e3:
            f1_val = macro.get("f1")
            st.metric("Macro F1", f"{f1_val:.2f}" if f1_val is not None else "Not evaluated")
        with e4:
            tiou_val = macro.get("mean_tiou")
            st.metric("Mean tIoU", f"{tiou_val:.2f}" if tiou_val is not None else "Not evaluated")
        with e5:
            lat_mean = latency_info.get("mean")
            st.metric("Mean Latency", f"{lat_mean:.0f} ms" if lat_mean is not None else "Not evaluated")

        st.write("")
        st.divider()

        # Ranking Performance
        st.markdown('<div class="section-title">Ranking Performance</div>', unsafe_allow_html=True)
        r_cols = st.columns(3)
        with r_cols[0]:
            h1 = macro.get("hit@1")
            r1 = macro.get("recall@1")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">Top-1 Accuracy</div>
                    <div class="value">{h1 if h1 is not None else 'N/A'}</div>
                    <div class="subtext">Hit@1: Top result correct (Recall@1: {r1})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r_cols[1]:
            h5 = macro.get("hit@5")
            r5 = macro.get("recall@5")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">Top-5 Recall</div>
                    <div class="value">{r5 if r5 is not None else 'N/A'}</div>
                    <div class="subtext">Hit@5: {h5}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r_cols[2]:
            h10 = macro.get("hit@10")
            r10 = macro.get("recall@10")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">Top-10 Recall</div>
                    <div class="value">{r10 if r10 is not None else 'N/A'}</div>
                    <div class="subtext">Hit@10: {h10}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # Threshold Sweep Chart
        if "threshold_sweep" in report:
            st.markdown('<div class="section-title">Similarity Threshold Sweep</div>', unsafe_allow_html=True)
            sweep_data = report["threshold_sweep"].get("rows", [])
            best_sweep = report["threshold_sweep"].get("best", {})

            if sweep_data:
                df_sweep = pd.DataFrame(sweep_data)
                df_sweep = df_sweep.set_index("threshold")[["precision", "recall", "f1"]]
                st.line_chart(df_sweep)
                if best_sweep:
                    st.markdown(
                        f"""
                        <div class="notice-panel">
                            <b>Optimal Operating Threshold:</b> <code>{best_sweep.get('threshold', 0.33):.4f}</code> 
                            (F1: <code>{best_sweep.get('f1', 0):.3f}</code>, Precision: <code>{best_sweep.get('precision', 0):.2f}</code>, Recall: <code>{best_sweep.get('recall', 0):.2f}</code>)
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Per-Query Breakdown Table
        st.write("")
        st.markdown('<div class="section-title">Per-Query Benchmark Table</div>', unsafe_allow_html=True)
        per_query = report.get("per_query", [])
        if per_query:
            df_pq = pd.DataFrame(per_query)
            cols_to_show = [
                "query",
                "precision",
                "recall",
                "f1",
                "mean_tiou",
                "latency_ms",
            ]
            cols_existing = [c for c in cols_to_show if c in df_pq.columns]
            st.dataframe(
                df_pq[cols_existing].rename(
                    columns={
                        "query": "Query Description",
                        "precision": "Precision",
                        "recall": "Recall",
                        "f1": "F1",
                        "mean_tiou": "Mean tIoU",
                        "latency_ms": "Latency (ms)",
                    }
                ),
                use_container_width=True,
            )


# =============================================================================
# PAGE 6: SETTINGS
# =============================================================================
elif st.session_state.current_page == "Settings":
    st.markdown('<h2 style="font-weight: 700; color: #F0F6FC; margin: 0 0 0.25rem 0;">Settings</h2>', unsafe_allow_html=True)
    st.markdown('<p class="brand-subtitle">Metadata storage configuration, vector index status, and model runtimes</p>', unsafe_allow_html=True)

    # Active Storage Backend
    st.markdown('<div class="section-title">Metadata Storage</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: #161B22; border: 1px solid #30363D; border-radius: 6px; padding: 1.15rem; margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div>
                    <div style="font-weight: 600; color: #F0F6FC; font-size: 0.95rem;">Active Backend: Local JSON Storage</div>
                    <div style="font-size: 0.8rem; color: #8B949E; margin-top: 0.25rem;">
                        Metadata, segments, and queries persist locally in <code>data/index/</code> without external database dependencies.
                    </div>
                </div>
                <span class="badge-pill badge-emerald">Local JSON Storage</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # MongoDB Atlas Staging
    st.markdown('<div class="section-title">MongoDB Atlas Integration</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: #161B22; border: 1px dashed #484F58; border-radius: 6px; padding: 1.15rem; margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div style="font-weight: 600; color: #F0F6FC; font-size: 0.95rem;">MongoDB Atlas integration — To be configured later</div>
                <span class="badge-pill badge-muted">Staged for Deployment</span>
            </div>
            <p style="font-size: 0.82rem; color: #8B949E; margin: 0.5rem 0 0.75rem 0; line-height: 1.6;">
                The metadata layer implements a decoupled <code>MetadataStore</code> interface. When Atlas cluster access is provided, setting the environment variables:
            </p>
            <pre style="margin: 0; font-size: 0.78rem; background: #0D1117; color: #E6EDF3; padding: 0.75rem; border-radius: 4px; border: 1px solid #30363D; font-family: 'JetBrains Mono', monospace;">SCENESEEK_USE_MONGO=true\nSCENESEEK_MONGO_URI="mongodb+srv://&lt;username&gt;:&lt;password&gt;@cluster.mongodb.net/sceneseek"</pre>
            <p style="font-size: 0.8rem; color: #6E7681; margin: 0.5rem 0 0 0;">
                switches storage to MongoDB Atlas without requiring any modifications to the dashboard code.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Vector Index Metrics
    st.markdown('<div class="section-title">Vector Index Status</div>', unsafe_allow_html=True)
    stats = engine.stats()
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Visual Vectors</div>
                <div class="value">{stats.get('visual_vectors', 0)}</div>
                <div class="subtext">Frames and object crops</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Action Vectors</div>
                <div class="value">{stats.get('action_vectors', 0)}</div>
                <div class="subtext">Kinetics action embeddings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Index Implementation</div>
                <div class="value" style="font-size: 1.25rem;">FAISS FlatIP</div>
                <div class="subtext">Exact inner-product cosine search</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Model Runtimes
    st.markdown('<div class="section-title">Model Specifications</div>', unsafe_allow_html=True)
    m_df = pd.DataFrame(
        [
            {"Component": "Text & Image Encoder", "Model": engine.cfg.clip_model, "Details": engine.cfg.clip_pretrained},
            {"Component": "Object Detector", "Model": engine.cfg.yolo_model, "Details": "COCO classes (people, vehicles, bags)"},
            {"Component": "Action Recognizer", "Model": engine.cfg.action_model, "Details": "Kinetics-400 (16-frame windows)"},
            {"Component": "Hardware Accelerator", "Model": engine.cfg.resolved_device().upper(), "Details": "Laptop CPU/GPU execution"},
            {"Component": "Storage Directory", "Model": "Videos", "Details": str(engine.cfg.videos_dir)},
            {"Component": "Index Directory", "Model": "FAISS & JSON", "Details": str(engine.cfg.index_dir)},
        ]
    )
    st.dataframe(m_df, use_container_width=True, hide_index=True)

    # Search Hyperparameters
    st.write("")
    st.markdown('<div class="section-title">Default Search Parameters</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        st.slider("Default Score Threshold", min_value=0.10, max_value=0.60, value=float(engine.cfg.score_threshold), disabled=True)
    with p2:
        st.number_input("Default Top-K", value=int(engine.cfg.default_top_k), disabled=True)
