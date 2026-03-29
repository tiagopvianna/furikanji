import cv2
import numpy as np
from PIL import Image
from loguru import logger
from scipy.signal.windows import gaussian

from src.furikanji.application.interfaces import (
    LocalizedTextLine,
    LocalizedTextRegion,
    TextLocalizerAdapter,
    TextTranscriberAdapter,
)
from src.furikanji.utils import imread


class InvalidImage(Exception):
    def __init__(self, message="Animation file, Corrupted file or Unsupported type"):
        super().__init__(message)


class PageTextExtractor:
    """Extracts localized regions and transcribed text from a page image."""

    def __init__(
        self,
        text_height=64,
        max_ratio_vert=16,
        max_ratio_hor=8,
        anchor_window=2,
        disable_ocr=False,
        text_localizer: TextLocalizerAdapter = None,
        text_transcriber: TextTranscriberAdapter = None,
    ):
        self.text_height = text_height
        self.max_ratio_vert = max_ratio_vert
        self.max_ratio_hor = max_ratio_hor
        self.anchor_window = anchor_window
        self.disable_ocr = disable_ocr

        if not self.disable_ocr:
            if text_localizer is None or text_transcriber is None:
                raise ValueError("text_localizer and text_transcriber are required when OCR is enabled")
            self.text_localizer = text_localizer
            self.text_transcriber = text_transcriber

    def __call__(self, img_path):
        """Run localization/transcription and return the current result schema."""
        image = self._read_image(img_path)
        result = self._initialize_result(image)
        if self.disable_ocr:
            return result

        localization = self._localize_text(image)
        result["blocks"] = self._transcribe_localized_regions(localization.localized_text_regions)

        return result

    @staticmethod
    def _read_image(img_path):
        """Read input image and validate it is decodable."""
        image = imread(img_path)
        if image is None:
            raise InvalidImage()
        return image

    @staticmethod
    def _initialize_result(image):
        """Initialize output container with page-level metadata."""
        height, width, *_ = image.shape
        return {"img_width": width, "img_height": height, "blocks": []}

    def _localize_text(self, image):
        """Run text localization adapter and return localization result."""
        logger.info("Running text detection")
        return self.text_localizer.localize_text(image)

    def _transcribe_localized_regions(self, localized_text_regions):
        """Serialize and transcribe all localized regions."""
        return [self._build_region_result(region) for region in localized_text_regions]

    def _build_region_result(self, region: LocalizedTextRegion):
        """Serialize one localized region and transcribe its lines."""
        result_blk = {
            "box": list(region.bounding_box),
            "vertical": region.is_vertical,
            "font_size": region.estimated_font_size,
            "lines_coords": [],
            "lines": [],
        }

        for line in region.lines:
            line_text = self._transcribe_localized_line(line, region.is_vertical)
            result_blk["lines_coords"].append(self._serialize_line_outline(line.line_outline))
            result_blk["lines"].append(line_text)

        return result_blk

    def _transcribe_localized_line(self, line: LocalizedTextLine, is_vertical: bool):
        """Split line image when needed, then OCR each split."""
        line_crops, _ = self._split_line_image_for_transcription(
            line_image=line.line_image,
            line_text_mask=line.line_text_mask,
            max_ratio=self._max_ratio_for_orientation(is_vertical),
            anchor_window=self.anchor_window,
        )
        return self._transcribe_line_crops(line_crops, is_vertical)

    def _transcribe_line_crops(self, line_crops, is_vertical):
        """Apply OCR to split crops, rotating vertical crops first."""
        line_text = ""
        for line_crop in line_crops:
            if is_vertical:
                line_crop = cv2.rotate(line_crop, cv2.ROTATE_90_CLOCKWISE)
            line_text += self.text_transcriber.transcribe_text(Image.fromarray(line_crop))
        return line_text

    def _max_ratio_for_orientation(self, is_vertical):
        """Choose split ratio threshold based on line orientation."""
        return self.max_ratio_vert if is_vertical else self.max_ratio_hor

    @staticmethod
    def _serialize_line_outline(line_outline):
        """Convert outline to plain list for JSON-friendly serialization."""
        return line_outline.tolist() if hasattr(line_outline, "tolist") else line_outline

    def _split_line_image_for_transcription(self, line_image, line_text_mask, max_ratio=16, anchor_window=2):
        """Split wide lines at low-density valleys before OCR."""
        ratio = self._line_aspect_ratio(line_image)
        if ratio <= max_ratio:
            return [line_image], []

        textheight = line_image.shape[0]
        smoothing_kernel = gaussian(textheight * 2, textheight / 8)
        num_splits = int(np.ceil(ratio / max_ratio))
        split_anchors = self._compute_split_anchors(width=line_image.shape[1], num_splits=num_splits)
        line_density = self._compute_line_density(line_text_mask=line_text_mask, smoothing_kernel=smoothing_kernel)
        split_points = self._find_split_points(
            split_anchors=split_anchors,
            line_density=line_density,
            window_size=anchor_window * textheight,
            width=line_image.shape[1],
        )
        return np.split(line_image, split_points, axis=1), split_points

    @staticmethod
    def _line_aspect_ratio(line_image):
        """Return width/height ratio for a transformed line image."""
        h, w, *_ = line_image.shape
        return w / h

    @staticmethod
    def _compute_split_anchors(width, num_splits):
        """Compute equally spaced x anchors where splits may occur."""
        return np.linspace(0, width, num_splits + 1)[1:-1]

    @staticmethod
    def _compute_line_density(line_text_mask, smoothing_kernel):
        """Build a smoothed 1D density signal from line text mask."""
        line_density = line_text_mask.sum(axis=0)
        line_density = np.convolve(line_density, smoothing_kernel, "same")
        line_density /= line_density.max()
        return line_density

    @staticmethod
    def _find_split_points(split_anchors, line_density, window_size, width):
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
