import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from src.furikanji.application.furigana_renderer import (
    EraseConfig,
    FuriganaRenderConfig,
    FuriganaRenderer,
)


class FakeFuriganaReadingGenerator:
    def resolve_line_segments(self, line_text: str):
        return []


class TestFuriganaRendererErase(unittest.TestCase):
    def setUp(self):
        self.draw = ImageDraw.Draw(Image.new("RGB", (200, 200), (255, 255, 255)))

    def test_erase_background_for_region_calls_both_strategies(self):
        renderer = FuriganaRenderer(
            furigana_reading_generator=FakeFuriganaReadingGenerator(),
            config=FuriganaRenderConfig(erase=EraseConfig(strategy="both")),
        )
        line_outline_points = [
            [[10, 10], [20, 10], [20, 20], [10, 20]],
        ]
        planned_bounds = (12.0, 12.0, 24.0, 24.0)

        with patch.object(renderer, "_erase_detected_region_background") as detected_mock, patch.object(
            renderer, "_erase_planned_text_background"
        ) as planned_mock:
            renderer._erase_background_for_region(
                draw=self.draw,
                line_outline_points=line_outline_points,
                planned_bounds=planned_bounds,
            )

        detected_mock.assert_called_once_with(
            draw=self.draw, line_outline_points=line_outline_points
        )
        planned_mock.assert_called_once_with(
            draw=self.draw, planned_bounds=planned_bounds
        )


if __name__ == "__main__":
    unittest.main()
