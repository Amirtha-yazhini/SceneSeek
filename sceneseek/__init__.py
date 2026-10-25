"""SceneSeek: natural-language semantic search over video.

Dashboard entry point:

    from sceneseek import SceneSeek, Config
    engine = SceneSeek.load()
    resp = engine.search("a person entering a building")
"""
from .config import Config
from .types import SearchResponse, SearchResult, format_ts

__all__ = ["Config", "SceneSeek", "SearchResponse", "SearchResult", "format_ts"]


def __getattr__(name):  # lazy so `import sceneseek` does not load torch
    if name == "SceneSeek":
        from .pipeline.search import SceneSeek
        return SceneSeek
    raise AttributeError(name)
