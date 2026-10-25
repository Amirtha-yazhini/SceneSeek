"""YOLO object detection (people, vehicles, bags)."""
from __future__ import annotations

import numpy as np

from ..types import Detection


class ObjectDetector:
    def __init__(self, model_name: str = "yolo11n.pt", conf: float = 0.35,
                 classes: tuple | None = None, device: str = "cpu"):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.conf = conf
        self.classes = list(classes) if classes else None
        self.device = device
        self.names = self.model.names  # {id: name}

    def detect(self, frames_rgb: list[np.ndarray]) -> list[list[Detection]]:
        """Run detection on a batch of RGB frames. Returns one list of detections per frame."""
        if not frames_rgb:
            return []
        # ultralytics expects BGR numpy arrays
        bgr = [np.ascontiguousarray(f[:, :, ::-1]) for f in frames_rgb]
        results = self.model.predict(bgr, conf=self.conf, classes=self.classes,
                                     device=self.device, verbose=False)
        out: list[list[Detection]] = []
        for r in results:
            dets = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                for box, c, p in zip(xyxy, cls, cf):
                    dets.append(Detection(label=self.names[int(c)], confidence=float(p), box=tuple(box)))
            out.append(dets)
        return out


def crop(frame_rgb: np.ndarray, box: tuple, pad: float = 0.15) -> np.ndarray:
    """Cut a detection out of a frame with a little context around it."""
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - pad * bw)); y1 = int(max(0, y1 - pad * bh))
    x2 = int(min(w, x2 + pad * bw)); y2 = int(min(h, y2 + pad * bh))
    return frame_rgb[y1:y2, x1:x2]
