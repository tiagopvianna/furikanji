from pathlib import Path

from src.furikanji.cache import cache


class ModelCacheAdapter:
    @property
    def comic_text_detector_model_path(self) -> Path:
        return cache.comic_text_detector
