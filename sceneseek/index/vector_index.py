"""FAISS similarity index with a row -> segment mapping.

Vectors are L2-normalised and stored in an inner-product index, so the
search score is cosine similarity. A copy of the vectors is kept on disk
(numpy) so candidates can be re-scored exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # keeps light-weight tests and tooling working without faiss
    faiss = None


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.row_segment: list[str] = []   # row -> segment_id
        self.row_kind: list[str] = []      # row -> "frame" | "crop:<label>" | "action"
        self._faiss = faiss.IndexFlatIP(dim) if faiss else None

    def __len__(self) -> int:
        return len(self.row_segment)

    def add(self, vectors: np.ndarray, segment_ids: list[str], kinds: list[str]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32).reshape(-1, self.dim)
        if len(vectors) != len(segment_ids) or len(vectors) != len(kinds):
            raise ValueError("vectors, segment_ids and kinds must be the same length")
        if not len(vectors):
            return
        self.vectors = np.concatenate([self.vectors, vectors])
        self.row_segment.extend(segment_ids)
        self.row_kind.extend(kinds)
        if self._faiss is not None:
            self._faiss.add(vectors)

    def remove_segments(self, prefix: str) -> int:
        """Drop all rows whose segment id starts with `prefix` (used when re-ingesting a video)."""
        keep = [i for i, s in enumerate(self.row_segment) if not s.startswith(prefix)]
        removed = len(self.row_segment) - len(keep)
        if removed:
            self.vectors = self.vectors[keep]
            self.row_segment = [self.row_segment[i] for i in keep]
            self.row_kind = [self.row_kind[i] for i in keep]
            self._rebuild()
        return removed

    def _rebuild(self) -> None:
        if faiss:
            self._faiss = faiss.IndexFlatIP(self.dim)
            if len(self.vectors):
                self._faiss.add(np.ascontiguousarray(self.vectors))

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to k (row, cosine score), best first."""
        if not len(self):
            return []
        k = min(k, len(self))
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        if self._faiss is not None:
            scores, rows = self._faiss.search(q, k)
            return [(int(r), float(s)) for r, s in zip(rows[0], scores[0]) if r >= 0]
        sims = self.vectors @ q[0]
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(r), float(sims[r])) for r in top]

    def scores_for_rows(self, query: np.ndarray, rows: list[int]) -> np.ndarray:
        if not rows:
            return np.zeros(0, dtype=np.float32)
        return self.vectors[rows] @ np.asarray(query, dtype=np.float32).reshape(-1)

    def rows_by_segment(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, s in enumerate(self.row_segment):
            out.setdefault(s, []).append(i)
        return out

    # --- persistence ------------------------------------------------------
    def save(self, folder: Path, name: str) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / f"{name}.vectors.npy", self.vectors)
        (folder / f"{name}.rows.json").write_text(json.dumps(
            {"dim": self.dim, "segment": self.row_segment, "kind": self.row_kind}))
        if faiss and self._faiss is not None:
            faiss.write_index(self._faiss, str(folder / f"{name}.faiss"))

    @classmethod
    def load(cls, folder: Path, name: str) -> "VectorIndex":
        folder = Path(folder)
        meta = json.loads((folder / f"{name}.rows.json").read_text())
        idx = cls(meta["dim"])
        idx.vectors = np.load(folder / f"{name}.vectors.npy").astype(np.float32)
        idx.row_segment = meta["segment"]
        idx.row_kind = meta["kind"]
        fpath = folder / f"{name}.faiss"
        if faiss and fpath.exists():
            idx._faiss = faiss.read_index(str(fpath))
            if idx._faiss.ntotal != len(idx.row_segment):
                idx._rebuild()
        else:
            idx._rebuild()
        return idx

    @staticmethod
    def exists(folder: Path, name: str) -> bool:
        return (Path(folder) / f"{name}.rows.json").exists()
