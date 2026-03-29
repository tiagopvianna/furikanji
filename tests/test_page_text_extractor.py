import unittest
from unittest.mock import patch

import numpy as np

from src.furikanji.adapters.interfaces import TextLocalizationResult
from src.furikanji.page_text_extractor import PageTextExtractor


class FakeTextDetector:
    def __init__(self, blocks):
        self._blocks = blocks

    def localize_text(self, image):
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask_refined = np.zeros((h, w), dtype=np.uint8)
        return TextLocalizationResult(mask=mask, refined_mask=mask_refined, text_blocks=self._blocks)


class FakeTextRecognizer:
    def __init__(self, text="テスト"):
        self.text = text
        self.calls = 0

    def transcribe_text(self, image_crop):
        self.calls += 1
        return self.text


class FakeBlock:
    def __init__(self, vertical=False, font_size=24):
        self.xyxy = np.array([1, 2, 20, 22], dtype=np.int32)
        self.vertical = vertical
        self.font_size = font_size
        self._line = np.array([[1.0, 2.0], [20.0, 2.0], [20.0, 22.0], [1.0, 22.0]], dtype=np.float32)

    def lines_array(self):
        return [self._line]

    def get_transformed_region(self, img, line_idx, textheight):
        return np.ones((textheight, textheight // 2, 3), dtype=np.uint8) * 255


class TestPageTextExtractor(unittest.TestCase):
    @patch("src.furikanji.page_text_extractor.imread")
    def test_disable_ocr_returns_empty_blocks_with_image_size(self, mock_imread):
        mock_imread.return_value = np.zeros((10, 20, 3), dtype=np.uint8)
        ocr = PageTextExtractor(disable_ocr=True)

        result = ocr("dummy.png")

        self.assertEqual(result["img_width"], 20)
        self.assertEqual(result["img_height"], 10)
        self.assertEqual(result["blocks"], [])

    @patch("src.furikanji.page_text_extractor.imread")
    def test_uses_injected_adapters_to_build_output(self, mock_imread):
        mock_imread.return_value = np.zeros((30, 40, 3), dtype=np.uint8)
        fake_block = FakeBlock(vertical=False, font_size=24)
        detector = FakeTextDetector([fake_block])
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
