from dataclasses import dataclass
from typing import List, TypedDict

import cv2
import numpy as np
from loguru import logger
from PIL import Image
from scipy.signal.windows import gaussian

from src.furikanji.application.interfaces import (
    LocalizedTextRegion,
    TextLocalizerAdapter,
    TextTranscriberAdapter,
)


class InvalidImage(Exception):
    def __init__(self, message="Animation file, Corrupted file or Unsupported type"):
        super().__init__(message)


def imread(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread, but works with unicode paths."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


LineImage = np.ndarray
LineTextMask = np.ndarray
LineCrops = List[np.ndarray]
SplitPoints = List[int]


class ExtractedTextRegionDict(TypedDict):
    bounding_box: List[int]
    is_vertical: bool
    estimated_font_size: int
    line_outline_points: List[list]
    line_target_widths: List[float | None]
    line_target_heights: List[float | None]
    line_texts: List[str]


class PageTextExtractionResultDict(TypedDict):
    image_width: int
    image_height: int
    text_regions: List[ExtractedTextRegionDict]


@dataclass(frozen=True)
class LineSplitResult:
    line_crops: LineCrops
    split_points: SplitPoints


class PageTextExtractor:
    """Extracts localized regions and transcribed text from a page image."""

    def __init__(
        self,
        text_localizer: TextLocalizerAdapter,
        text_transcriber: TextTranscriberAdapter,
        text_height=64,
        max_ratio_vert=16,
        max_ratio_hor=8,
        anchor_window=2,
        disable_ocr=False,
    ):
        """
        Args:
            text_height: Target transformed line height used for localization crops.
            max_ratio_vert: Max width/height ratio for vertical lines before splitting.
            max_ratio_hor: Max width/height ratio for horizontal lines before splitting.
            anchor_window: Split search window multiplier around each split anchor.
            disable_ocr: If True, skip localization/transcription and return only page metadata.
            text_localizer: Adapter that localizes text regions/lines on the page image.
            text_transcriber: Adapter that transcribes line image crops into text.
        """
        self.text_height = text_height
        self.max_ratio_vert = max_ratio_vert
        self.max_ratio_hor = max_ratio_hor
        self.anchor_window = anchor_window
        self.disable_ocr = disable_ocr
        self.text_localizer = text_localizer
        self.text_transcriber = text_transcriber

    def __call__(self, img_path: str) -> PageTextExtractionResultDict:
        """Run localization/transcription and return the current result schema."""
        image = imread(img_path)
        if image is None:
            raise InvalidImage()
        height, width, *_ = image.shape
        result: PageTextExtractionResultDict = {
            "image_width": width,
            "image_height": height,
            "text_regions": [],
        }
        if self.disable_ocr:
            return result

        logger.info("Running text detection")
        localization = self.text_localizer.localize_text(image)
        result["text_regions"] = self._transcribe_localized_regions(
            localization.localized_text_regions
        )

        return result

    def _transcribe_localized_regions(
        self, localized_text_regions: List[LocalizedTextRegion]
    ) -> List[ExtractedTextRegionDict]:
        """Serialize and transcribe all localized regions."""
        return [self._build_region_result(region) for region in localized_text_regions]

    def _build_region_result(self, region: LocalizedTextRegion) -> ExtractedTextRegionDict:
        """Serialize one localized region and transcribe its lines."""
        result_block = {
            "bounding_box": list(region.bounding_box),
            "is_vertical": region.is_vertical,
            "estimated_font_size": region.estimated_font_size,
            "line_outline_points": [],
            "line_target_widths": [],
            "line_target_heights": [],
            "line_texts": [],
        }

        for line in region.lines:
            line_text = self._transcribe_line_image(
                line_image=line.line_image,
                line_text_mask=line.line_text_mask,
                is_vertical=region.is_vertical,
            )
            result_block["line_outline_points"].append(line.line_outline)
            result_block["line_target_widths"].append(line.line_target_width)
            result_block["line_target_heights"].append(line.line_target_height)
            result_block["line_texts"].append(line_text)

        return result_block

    def _transcribe_line_image(
        self,
        line_image: LineImage,
        line_text_mask: LineTextMask,
        is_vertical: bool,
    ) -> str:
        """Split line image when needed, then OCR and join all split pieces."""
        split_result = self._split_line_image_for_transcription(
            line_image=line_image,
            line_text_mask=line_text_mask,
            max_ratio=self._max_ratio_for_orientation(is_vertical),
            anchor_window=self.anchor_window,
        )
        transcribed_text = ""
        for line_crop in split_result.line_crops:
            if is_vertical:
                line_crop = cv2.rotate(line_crop, cv2.ROTATE_90_CLOCKWISE)
            transcribed_text += self.text_transcriber.transcribe_text(
                Image.fromarray(line_crop)
            )
        return transcribed_text

    def _max_ratio_for_orientation(self, is_vertical: bool) -> int:
        """Choose split ratio threshold based on line orientation."""
        return self.max_ratio_vert if is_vertical else self.max_ratio_hor

    def _split_line_image_for_transcription(
        self,
        line_image: LineImage,
        line_text_mask: LineTextMask,
        max_ratio: int = 16,
        anchor_window: int = 2,
    ) -> LineSplitResult:
        """Split wide lines at low-density valleys before OCR."""
        line_height, line_width, *_ = line_image.shape
        ratio = line_width / line_height
        if ratio <= max_ratio:
            return LineSplitResult(line_crops=[line_image], split_points=[])

        textheight = line_height
        smoothing_kernel = gaussian(textheight * 2, textheight / 8)
        num_splits = int(np.ceil(ratio / max_ratio))
        split_anchors = self._compute_split_anchors(
            width=line_width, num_splits=num_splits
        )
        line_density = self._compute_line_density(
            line_text_mask=line_text_mask, smoothing_kernel=smoothing_kernel
        )
        split_points = self._find_split_points(
            split_anchors=split_anchors,
            line_density=line_density,
            window_size=anchor_window * textheight,
            width=line_width,
        )
        return LineSplitResult(
            line_crops=list(np.split(line_image, split_points, axis=1)),
            split_points=split_points,
        )

    @staticmethod
    def _compute_split_anchors(width: int, num_splits: int) -> np.ndarray:
        """Compute equally spaced x anchors where splits may occur."""
        return np.linspace(0, width, num_splits + 1)[1:-1]

    @staticmethod
    def _compute_line_density(
        line_text_mask: LineTextMask, smoothing_kernel: np.ndarray
    ) -> np.ndarray:
        """Build a smoothed 1D density signal from line text mask."""
        line_density = line_text_mask.sum(axis=0)
        line_density = np.convolve(line_density, smoothing_kernel, "same")
        line_density /= line_density.max()
        return line_density

    @staticmethod
    def _find_split_points(
        split_anchors: np.ndarray,
        line_density: np.ndarray,
        window_size: int,
        width: int,
    ) -> SplitPoints:
        """Select split x positions near anchors at local density minima."""
        split_points = []
        for anchor in split_anchors:
            anchor = int(anchor)
            n0 = np.clip(anchor - window_size // 2, 0, width)
            n1 = np.clip(anchor + window_size // 2, 0, width)
            p = line_density[n0:n1].argmin()
            p += n0
            split_points.append(p)
        return split_points
