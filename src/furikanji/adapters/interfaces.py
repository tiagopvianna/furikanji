from dataclasses import dataclass
from typing import Any, List, Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TextLocalizationResult:
    mask: np.ndarray
    refined_mask: np.ndarray
    text_blocks: List[Any]


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
