"""Reading videos, sampling frames, cutting them into segments (OpenCV + FFmpeg)."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..types import VideoInfo

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".webm"}


def list_videos(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def video_id_for(path: Path) -> str:
    return Path(path).stem


def probe(path: Path) -> VideoInfo:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1 or fps > 240:  # some containers report junk
        fps = 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return VideoInfo(video_id=video_id_for(path), path=str(Path(path).resolve()), fps=float(fps),
                     frame_count=n, duration=n / fps if n else 0.0, width=w, height=h)


@dataclass
class SampledFrame:
    time: float          # seconds
    frame_index: int     # index in the original video
    rgb: np.ndarray      # HxWx3 uint8, RGB


def sample_frames(path: Path, sample_fps: float, max_side: int = 640) -> Iterator[SampledFrame]:
    """Yield frames at roughly `sample_fps`, decoded sequentially (fast for long files).

    Frames are resized so the longest side is at most `max_side` to keep models fast.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1 or fps > 240:
        fps = 30.0
    step = max(fps / sample_fps, 1.0)
    next_pick = 0.0
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx + 1e-6 >= next_pick:
                ok, bgr = cap.retrieve()
                if ok and bgr is not None:
                    h, w = bgr.shape[:2]
                    scale = max_side / max(h, w)
                    if scale < 1:
                        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    yield SampledFrame(time=idx / fps, frame_index=idx,
                                       rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                next_pick += step
            idx += 1
    finally:
        cap.release()


def segment_windows(duration: float, length: float, stride: float) -> list[tuple[float, float]]:
    """Start/end times of fixed-length windows covering [0, duration]."""
    if duration <= 0:
        return []
    if duration <= length:
        return [(0.0, duration)]
    windows = []
    t = 0.0
    while t < duration - 1e-6:
        end = min(t + length, duration)
        windows.append((round(t, 3), round(end, 3)))
        if end >= duration:
            break
        t += stride
    return windows


def iter_segments(path: Path, sample_fps: float, length: float, stride: float,
                  duration: float | None = None) -> Iterator[tuple[int, float, float, list[SampledFrame]]]:
    """Read the video once and yield (index, start, end, frames) for each window.

    Keeps only the frames still needed by upcoming windows in memory.
    """
    buffer: list[SampledFrame] = []
    frames = sample_frames(path, sample_fps)
    exhausted = False
    last_time = 0.0
    seg_idx = 0
    start = 0.0
    while True:
        end = start + length
        # fill buffer until we have passed the end of this window
        while not exhausted and (not buffer or buffer[-1].time < end):
            try:
                f = next(frames)
                buffer.append(f)
                last_time = f.time
            except StopIteration:
                exhausted = True
        in_window = [f for f in buffer if start <= f.time < end]
        if not in_window:
            break
        real_end = min(end, last_time + 1.0 / sample_fps) if exhausted else end
        if duration:
            real_end = min(real_end, duration)
        yield seg_idx, round(start, 3), round(real_end, 3), in_window
        seg_idx += 1
        if exhausted and end >= last_time:
            break
        start += stride
        buffer = [f for f in buffer if f.time >= start]


def pick_evenly(items: list, n: int) -> list:
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx]


def save_thumbnail(rgb: np.ndarray, out_path: Path, max_side: int = 320) -> str:
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    img = cv2.resize(rgb, (int(w * scale), int(h * scale))) if scale < 1 else rgb
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return str(out_path)


def export_clip(video_path: Path, start: float, end: float, out_path: Path, pad: float = 1.0) -> Path:
    """Cut [start-pad, end+pad] into a browser-playable H.264 mp4.

    Uses FFmpeg when available (recommended); otherwise falls back to OpenCV
    (mp4v codec, which some browsers will not play inline).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s = max(0.0, start - pad)
    dur = (end + pad) - s
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{s:.3f}", "-i", str(video_path),
               "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
               "-an", "-movflags", "+faststart", str(out_path)]
        subprocess.run(cmd, check=True)
        return out_path

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(int(dur * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()
    cap.release()
    return out_path
