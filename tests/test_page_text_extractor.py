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
        self.received_image_sizes = []

    def transcribe_text(self, image_crop):
        self.calls += 1
        self.received_image_sizes.append(image_crop.size)
        return self.text


class TestPageTextExtractor(unittest.TestCase):
    def test_requires_adapters_in_constructor(self):
        with self.assertRaises(TypeError):
            PageTextExtractor()

    @patch("src.furikanji.application.page_text_extractor.imread")
    def test_disable_ocr_returns_empty_blocks_with_image_size(self, mock_imread):
        mock_imread.return_value = np.zeros((10, 20, 3), dtype=np.uint8)
        ocr = PageTextExtractor(
            text_localizer=FakeTextDetector([]),
            text_transcriber=FakeTextRecognizer(),
            disable_ocr=True,
        )

        result = ocr("dummy.png")

        self.assertEqual(result["img_width"], 20)
        self.assertEqual(result["img_height"], 10)
        self.assertEqual(result["blocks"], [])

    @patch("src.furikanji.application.page_text_extractor.imread")
    def test_uses_injected_adapters_to_build_output(self, mock_imread):
        mock_imread.return_value = np.zeros((30, 40, 3), dtype=np.uint8)
        fake_line = LocalizedTextLine(
            line_outline=[[1.0, 2.0], [20.0, 2.0], [20.0, 22.0], [1.0, 22.0]],
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

    def test_split_line_image_for_transcription_does_not_split_small_ratio(self):
        extractor = PageTextExtractor(
            text_localizer=FakeTextDetector([]),
            text_transcriber=FakeTextRecognizer(),
            disable_ocr=True,
        )
        line_image = np.ones((16, 10, 3), dtype=np.uint8)
        line_text_mask = np.ones((16, 10), dtype=np.uint8)

        split_result = extractor._split_line_image_for_transcription(
            line_image=line_image,
            line_text_mask=line_text_mask,
            max_ratio=16,
            anchor_window=2,
        )

        self.assertEqual(len(split_result.line_crops), 1)
        self.assertEqual(split_result.split_points, [])

    def test_split_line_image_for_transcription_splits_wide_line(self):
        extractor = PageTextExtractor(
            text_localizer=FakeTextDetector([]),
            text_transcriber=FakeTextRecognizer(),
            disable_ocr=True,
        )
        line_image = np.ones((16, 120, 3), dtype=np.uint8)
        line_text_mask = np.ones((16, 120), dtype=np.uint8)
        line_text_mask[:, 58:62] = 0

        split_result = extractor._split_line_image_for_transcription(
            line_image=line_image,
            line_text_mask=line_text_mask,
            max_ratio=4,
            anchor_window=2,
        )

        self.assertGreater(len(split_result.line_crops), 1)
        self.assertGreater(len(split_result.split_points), 0)

    @patch("src.furikanji.application.page_text_extractor.imread")
    def test_vertical_line_is_rotated_before_transcription(self, mock_imread):
        mock_imread.return_value = np.zeros((30, 40, 3), dtype=np.uint8)
        fake_line = LocalizedTextLine(
            line_outline=[[1.0, 2.0], [20.0, 2.0], [20.0, 22.0], [1.0, 22.0]],
            line_image=np.ones((16, 8, 3), dtype=np.uint8) * 255,
            line_text_mask=np.ones((16, 8), dtype=np.uint8) * 255,
        )
        fake_region = LocalizedTextRegion(
            bounding_box=[1, 2, 20, 22],
            is_vertical=True,
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

        ocr("dummy.png")

        self.assertEqual(recognizer.calls, 1)
        self.assertEqual(recognizer.received_image_sizes[0], (16, 8))


if __name__ == "__main__":
    unittest.main()
