import json
from typing import Callable

import numpy as np

from src.furikanji.application.furigana_renderer import FuriganaRenderer
from src.furikanji.application.page_text_extractor import (
    PageTextExtractionResultDict,
    PageTextExtractor,
)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return json.JSONEncoder.default(self, obj)


def _dump_json(obj: object, path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, cls=_NumpyEncoder)


class ProcessImageUseCase:
    def __init__(
        self,
        page_text_extractor: PageTextExtractor,
        furigana_renderer: FuriganaRenderer,
        json_dumper: Callable[[object, str], None] = _dump_json,
    ):
        self.page_text_extractor = page_text_extractor
        self.furigana_renderer = furigana_renderer
        self.json_dumper = json_dumper

    def __call__(
        self,
        image_path: str,
        json_output_path: str = "result.json",
        overlay_output_path: str = "result_overlay.jpg",
    ) -> PageTextExtractionResultDict:
        result = self.page_text_extractor(image_path)
        self.json_dumper(result, json_output_path)
        self.furigana_renderer.render_overlay(image_path, result, overlay_output_path)
        return result
