from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from loguru import logger

from src.furikanji.application.interfaces import (
    LocalizedTextLine,
    LocalizedTextRegion,
    TextLocalizationResult,
)


class _ComicTextDetectorModelCache:
    def __init__(self, root: Path | None = None):
        self.root = root or (Path.home() / ".cache" / "manga-ocr")
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def comic_text_detector_model_path(self) -> Path:
        model_path = self.root / "comictextdetector.pt"
        model_url = (
            "https://github.com/zyddnys/manga-image-translator/releases/download/"
            "beta-0.2.1/comictextdetector.pt"
        )
        self._download_if_needed(model_path, model_url)
        return model_path

    @staticmethod
    def _download_if_needed(path: Path, url: str) -> None:
        if path.is_file():
            return

        logger.info(f"Downloading {url}")
        response = requests.get(url, stream=True, verify=True)
        if response.status_code != 200:
            raise RuntimeError(f"Failed downloading {url}")
        with path.open("wb") as file:
            for chunk in response.iter_content(1024):
                if chunk:
                    file.write(chunk)
        logger.info(f"Finished downloading {url}")


class ComicTextDetectorLocalizer:
    def __init__(
        self, input_size=1024, text_height=64, force_cpu=False, model_cache=None
    ):
        from comic_text_detector.inference import TextDetector

        if model_cache is None:
            model_cache = _ComicTextDetectorModelCache()

        self.text_height = text_height
        cuda = torch.cuda.is_available()
        device = "cuda" if cuda and not force_cpu else "cpu"
        self._detector = TextDetector(
            model_path=model_cache.comic_text_detector_model_path,
            input_size=input_size,
            device=device,
            act="leaky",
        )

    def localize_text(self, image):
        _, refined_mask, text_blocks = self._detector(
            image, refine_mode=1, keep_undetected_mask=True
        )
        localized_text_regions = []
        for blk in text_blocks:
            lines = []
            for line_idx, line_outline in enumerate(blk.lines_array()):
                line_image = blk.get_transformed_region(
                    image, line_idx, self.text_height
                )
                line_text_mask = blk.get_transformed_region(
                    refined_mask, line_idx, self.text_height
                )
                line_outline_list = line_outline.tolist()
                line_target_width, line_target_height = (
                    self._compute_line_mask_inner_dimensions(
                        refined_mask=refined_mask, line_outline=line_outline_list
                    )
                )
                lines.append(
                    LocalizedTextLine(
                        line_outline=line_outline_list,
                        line_image=line_image,
                        line_text_mask=line_text_mask,
                        line_target_width=line_target_width,
                        line_target_height=line_target_height,
                    )
                )

            localized_text_regions.append(
                LocalizedTextRegion(
                    bounding_box=list(blk.xyxy),
                    is_vertical=blk.vertical,
                    estimated_font_size=blk.font_size,
                    lines=lines,
                )
            )

        return TextLocalizationResult(
            text_mask=refined_mask,
            localized_text_regions=localized_text_regions,
        )

    def _compute_line_mask_inner_dimensions(
        self, refined_mask: np.ndarray, line_outline: list[list[float]]
    ) -> tuple[float | None, float | None]:
        if not line_outline:
            return None, None
        polygon = np.array(line_outline, dtype=np.int32)
        xs = polygon[:, 0]
        ys = polygon[:, 1]
        x_min = max(0, int(xs.min()))
        y_min = max(0, int(ys.min()))
        x_max = min(refined_mask.shape[1] - 1, int(xs.max()))
        y_max = min(refined_mask.shape[0] - 1, int(ys.max()))
        if x_max < x_min or y_max < y_min:
            return None, None

        mask_crop = refined_mask[y_min : y_max + 1, x_min : x_max + 1]
        local_polygon = polygon.copy()
        local_polygon[:, 0] -= x_min
        local_polygon[:, 1] -= y_min
        polygon_mask = np.zeros_like(mask_crop, dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [local_polygon], 255)
        active = (mask_crop > 0) & (polygon_mask > 0)
        if not np.any(active):
            return None, None

        active_ys, active_xs = np.where(active)
        width = float(active_xs.max() - active_xs.min() + 1)
        height = float(active_ys.max() - active_ys.min() + 1)
        return width, height
