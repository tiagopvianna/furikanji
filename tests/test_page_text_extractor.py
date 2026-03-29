import unittest
from unittest.mock import patch

import numpy as np

from src.furikanji.application.interfaces import (
    LocalizedTextLine,
    LocalizedTextRegion,
    TextLocalizationResult,
)
from src.furikanji.application.page_text_extractor import PageTextExtractor


class FakeTextDetector:
    def __init__(self, localized_text_regions):
        self._localized_text_regions = localized_text_regions

    def localize_text(self, image):
        h, w = image.shape[:2]
        text_mask = np.zeros((h, w), dtype=np.uint8)
        return TextLocalizationResult(
            text_mask=text_mask,
            localized_text_regions=self._localized_text_regions,
        )


class FakeTextRecognizer:
    def __init__(self, text="テスト"):
        self.text = text
        self.calls = 0

    def transcribe_text(self, image_crop):
        self.calls += 1
        return self.text


class TestPageTextExtractor(unittest.TestCase):
    @patch("src.furikanji.application.page_text_extractor.imread")
    def test_disable_ocr_returns_empty_blocks_with_image_size(self, mock_imread):
        mock_imread.return_value = np.zeros((10, 20, 3), dtype=np.uint8)
        ocr = PageTextExtractor(disable_ocr=True)

        result = ocr("dummy.png")

        self.assertEqual(result["img_width"], 20)
        self.assertEqual(result["img_height"], 10)
        self.assertEqual(result["blocks"], [])

    @patch("src.furikanji.application.page_text_extractor.imread")
    def test_uses_injected_adapters_to_build_output(self, mock_imread):
        mock_imread.return_value = np.zeros((30, 40, 3), dtype=np.uint8)
        fake_line = LocalizedTextLine(
            line_outline=np.array([[1.0, 2.0], [20.0, 2.0], [20.0, 22.0], [1.0, 22.0]], dtype=np.float32),
            line_image=np.ones((16, 8, 3), dtype=np.uint8) * 255,
            line_text_mask=np.ones((16, 8), dtype=np.uint8) * 255,
        )
        fake_region = LocalizedTextRegion(
            bounding_box=[1, 2, 20, 22],
            is_vertical=False,
            estimated_font_size=24,
            lines=[fake_line],
        )
        detector = FakeTextDetector([fake_region])
        recognizer = FakeTextRecognizer("こんにちは")
        ocr = PageTextExtractor(
            disable_ocr=False,
            text_localizer=detector,
            text_transcriber=recognizer,
        )

        result = ocr("dummy.png")

        self.assertEqual(result["img_width"], 40)
        self.assertEqual(result["img_height"], 30)
        self.assertEqual(len(result["blocks"]), 1)
        self.assertEqual(result["blocks"][0]["lines"], ["こんにちは"])
        self.assertEqual(recognizer.calls, 1)


if __name__ == "__main__":
    unittest.main()
