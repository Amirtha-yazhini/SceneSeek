"""Pretrained action recognition over a short clip of frames.

Uses a torchvision video model trained on Kinetics-400. Its output labels
("walking the dog", "opening door", ...) are turned into a CLIP *text*
embedding per segment, weighted by probability. That lets a free-text query
be compared to the recognised action without any fine-tuning.
"""
from __future__ import annotations

import numpy as np

_MODELS = {
    "r3d_18": ("r3d_18", "R3D_18_Weights"),
    "r2plus1d_18": ("r2plus1d_18", "R2Plus1D_18_Weights"),
    "mc3_18": ("mc3_18", "MC3_18_Weights"),
    "s3d": ("s3d", "S3D_Weights"),
    "mvit_v2_s": ("mvit_v2_s", "MViT_V2_S_Weights"),
}


class ActionRecognizer:
    def __init__(self, model_name: str = "r3d_18", device: str = "cpu", num_frames: int = 16):
        import torch
        import torchvision.models.video as tvv

        if model_name not in _MODELS:
            raise ValueError(f"Unknown action model {model_name}; choose from {list(_MODELS)}")
        fn_name, weights_name = _MODELS[model_name]
        weights = getattr(tvv, weights_name).DEFAULT
        self.torch = torch
        self.device = device
        self.model = getattr(tvv, fn_name)(weights=weights).to(device).eval()
        self.transform = weights.transforms()
        self.labels: list[str] = list(weights.meta["categories"])
        self.num_frames = num_frames

    def predict(self, frames_rgb: list[np.ndarray], top_k: int = 5) -> list[tuple[str, float]]:
        """Return the top-k (label, probability) for one clip."""
        if not frames_rgb:
            return []
        frames = list(frames_rgb)
        # pad or subsample to the expected clip length
        if len(frames) < self.num_frames:
            frames = frames + [frames[-1]] * (self.num_frames - len(frames))
        elif len(frames) > self.num_frames:
            idx = np.linspace(0, len(frames) - 1, self.num_frames).round().astype(int)
            frames = [frames[i] for i in idx]
        h = min(f.shape[0] for f in frames); w = min(f.shape[1] for f in frames)
        clip = np.stack([f[:h, :w] for f in frames])                      # T,H,W,C
        x = self.torch.from_numpy(clip).permute(0, 3, 1, 2).contiguous()  # T,C,H,W uint8
        x = self.transform(x).unsqueeze(0).to(self.device)                # 1,C,T,H,W
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(x), dim=-1)[0].cpu().numpy()
        top = np.argsort(-probs)[:top_k]
        return [(self.labels[i], float(probs[i])) for i in top]


def action_text_embedding(actions: list[tuple[str, float]], label_embeddings: dict[str, np.ndarray],
                          dim: int) -> np.ndarray:
    """Probability-weighted mix of CLIP text embeddings of the predicted action labels."""
    from .clip_encoder import normalize

    if not actions:
        return np.zeros(dim, dtype=np.float32)
    vec = np.zeros(dim, dtype=np.float32)
    for label, p in actions:
        vec += p * label_embeddings[label]
    return normalize(vec)
