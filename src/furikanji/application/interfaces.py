from dataclasses import dataclass
from typing import List, Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class LocalizedTextLine:
    line_outline: List[List[float]]
    line_image: np.ndarray
    line_text_mask: np.ndarray
    line_target_width: float | None = None
    line_target_height: float | None = None


@dataclass(frozen=True)
class LocalizedTextRegion:
    bounding_box: List[int]
    is_vertical: bool
    estimated_font_size: int
    lines: List[LocalizedTextLine]


@dataclass(frozen=True)
class TextLocalizationResult:
    text_mask: np.ndarray
    localized_text_regions: List[LocalizedTextRegion]


# Text localization identifies where text is in a full page image.
# It returns geometry-oriented outputs (masks and line/block regions), not strings.
class TextLocalizerAdapter(Protocol):
    def localize_text(self, image: np.ndarray) -> TextLocalizationResult:
        """Return localization metadata for a full page image."""


# Text transcription converts a cropped text image region into a Unicode string.
# It returns semantic text content, not geometry.
class TextTranscriberAdapter(Protocol):
    def transcribe_text(self, image_crop: Image.Image) -> str:
        """Return transcribed text for a cropped line image."""


@dataclass(frozen=True)
class FuriganaSegment:
    base_text: str
    reading: str
    needs_furigana: bool


class FuriganaReadingGenerator(Protocol):
    def resolve_line_segments(self, line_text: str) -> List[FuriganaSegment]:
        """Return ordered furigana-ready segments for one OCR line."""
