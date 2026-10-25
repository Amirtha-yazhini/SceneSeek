"""Write a short synthetic video so the pipeline and dashboard can be tried without VIRAT.

    python scripts/make_demo_video.py            # -> data/videos/demo_synthetic.mp4

It is only a smoke test: shapes move around, so search scores will be low.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "videos" / "demo_synthetic.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
fps, w, h, seconds = 30, 640, 360, 12
vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for i in range(fps * seconds):
    t = i / fps
    img = np.full((h, w, 3), (90, 110, 120), np.uint8)
    cv2.rectangle(img, (0, 250), (w, h), (60, 60, 60), -1)                         # road
    cv2.rectangle(img, (480, 80), (600, 250), (200, 200, 210), -1)                 # building
    cv2.rectangle(img, (520, 160), (560, 250), (40, 40, 80), -1)                   # door
    x = int(50 + 40 * t)                                                           # walking figure
    cv2.circle(img, (x, 175), 12, (30, 30, 30), -1)
    cv2.rectangle(img, (x - 12, 188), (x + 12, 240), (20, 20, 20), -1)
    if t > 6:                                                                      # car drives by
        cx = int((t - 6) * 120)
        cv2.rectangle(img, (cx, 280), (cx + 140, 330), (40, 40, 160), -1)
    vw.write(img)
vw.release()
print(out)
