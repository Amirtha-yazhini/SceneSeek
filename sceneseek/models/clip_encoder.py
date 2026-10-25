"""OpenCLIP: puts images and text in the same embedding space."""
from __future__ import annotations

import numpy as np

# Wrapping a query in a few phrasings and averaging makes CLIP text embeddings steadier.
QUERY_TEMPLATES = (
    "{}",
    "a photo of {}",
    "a surveillance camera photo of {}",
    "a CCTV video frame showing {}",
)


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-8, None)


class ClipEncoder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k",
                 device: str = "cpu"):
        import open_clip
        import torch

        self.torch = torch
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        with torch.no_grad():
            self.dim = int(self.model.encode_text(self.tokenizer(["x"]).to(device)).shape[-1])

    def encode_images(self, images_rgb: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        from PIL import Image

        if not images_rgb:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images_rgb), batch_size):
                batch = [self.preprocess(Image.fromarray(im)) for im in images_rgb[i:i + batch_size]]
                x = self.torch.stack(batch).to(self.device)
                out.append(self.model.encode_image(x).float().cpu().numpy())
        return normalize(np.concatenate(out))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self.torch.no_grad():
            tok = self.tokenizer(texts).to(self.device)
            emb = self.model.encode_text(tok).float().cpu().numpy()
        return normalize(emb)

    def encode_query(self, query: str) -> np.ndarray:
        """One normalized vector for a user query, averaged over prompt templates."""
        q = query.strip().rstrip(".")
        embs = self.encode_texts([t.format(q) for t in QUERY_TEMPLATES])
        return normalize(embs.mean(axis=0))
