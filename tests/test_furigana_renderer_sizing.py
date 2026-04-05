import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from src.furikanji.application.furigana_renderer import (
    FuriganaRenderConfig,
    FuriganaRenderer,
    SizingConfig,
)


class FakeFuriganaReadingGenerator:
    def resolve_line_segments(self, line_text: str):
        return []


class TestFuriganaRendererSizing(unittest.TestCase):
    def setUp(self):
        self.draw = ImageDraw.Draw(Image.new("RGB", (200, 200), (255, 255, 255)))

    def test_estimate_main_size_applies_one_step_correction_and_clamp(self):
        renderer = FuriganaRenderer(
            furigana_reading_generator=FakeFuriganaReadingGenerator(),
            config=FuriganaRenderConfig(
                sizing=SizingConfig(
                    min_main_size=8,
                    max_main_size=60,
                    main_scale_bias=1.0,
                    enable_one_step_correction=True,
                )
            ),
        )

        with patch.object(
            renderer, "_measure_probe_dimension", side_effect=[2.0, 10.0]
        ):
            size = renderer._estimate_main_size_from_target_dimension(
                draw=self.draw,
                target_dimension=40.0,
                vertical=True,
            )

        self.assertEqual(size, 60)

    def test_estimate_main_size_without_correction(self):
        renderer = FuriganaRenderer(
            furigana_reading_generator=FakeFuriganaReadingGenerator(),
            config=FuriganaRenderConfig(
                sizing=SizingConfig(
                    min_main_size=8,
                    max_main_size=120,
                    main_scale_bias=1.2,
                    enable_one_step_correction=False,
                )
            ),
        )

        with patch.object(renderer, "_measure_probe_dimension", return_value=2.0):
            size = renderer._estimate_main_size_from_target_dimension(
                draw=self.draw,
                target_dimension=40.0,
                vertical=True,
            )

        self.assertEqual(size, 24)

    def test_resolve_region_sizing_uses_median_and_proportional_derivatives(self):
        renderer = FuriganaRenderer(
            furigana_reading_generator=FakeFuriganaReadingGenerator(),
            config=FuriganaRenderConfig(
                sizing=SizingConfig(
                    furigana_ratio=0.5,
                    char_spacing_ratio=0.25,
                    furigana_spacing_ratio=0.2,
                    min_furigana_size=6,
                    max_furigana_size=72,
                )
            ),
        )
        text_region = {
            "bounding_box": [0, 0, 100, 100],
            "is_vertical": True,
            "estimated_font_size": 24,
            "line_outline_points": [
                [[0, 0], [20, 0], [20, 40], [0, 40]],
                [[30, 0], [50, 0], [50, 40], [30, 40]],
                [[60, 0], [80, 0], [80, 40], [60, 40]],
            ],
            "line_texts": ["a", "b", "c"],
        }

        with patch.object(
            renderer,
            "_estimate_main_size_from_target_dimension",
            side_effect=[10, 20, 30],
        ):
            sizing = renderer._resolve_region_sizing(text_region, self.draw)

        self.assertEqual(sizing.main_size, 20)
        self.assertEqual(sizing.furigana_size, 10)
        self.assertEqual(sizing.char_spacing, 5)
        self.assertEqual(sizing.furigana_spacing, 2)

    def test_resolve_region_sizing_uses_ocr_fallback_when_geometry_missing(self):
        renderer = FuriganaRenderer(
            furigana_reading_generator=FakeFuriganaReadingGenerator(),
            config=FuriganaRenderConfig(
                sizing=SizingConfig(
                    use_ocr_estimate_fallback=True,
                    ocr_fallback_main_scale=0.5,
                    furigana_ratio=0.5,
                )
            ),
        )
        text_region = {
            "bounding_box": [0, 0, 100, 100],
            "is_vertical": True,
            "estimated_font_size": 40,
            "line_outline_points": [],
            "line_texts": [],
        }

        sizing = renderer._resolve_region_sizing(text_region, self.draw)

        self.assertEqual(sizing.main_size, 20)
        self.assertEqual(sizing.furigana_size, 10)

    def test_resolve_region_sizing_uses_height_target_for_horizontal(self):
        renderer = FuriganaRenderer(
            furigana_reading_generator=FakeFuriganaReadingGenerator(),
            config=FuriganaRenderConfig(),
        )
        text_region = {
            "bounding_box": [0, 0, 120, 40],
            "is_vertical": False,
            "estimated_font_size": 24,
            "line_outline_points": [
                [[0, 0], [100, 0], [100, 20], [0, 20]],
            ],
            "line_texts": ["text"],
        }

        with patch.object(
            renderer, "_estimate_main_size_from_target_dimension", return_value=18
        ) as estimate_mock:
            sizing = renderer._resolve_region_sizing(text_region, self.draw)

        self.assertEqual(sizing.main_size, 18)
        _, kwargs = estimate_mock.call_args
        self.assertEqual(kwargs["target_dimension"], 20)
        self.assertFalse(kwargs["vertical"])


if __name__ == "__main__":
    unittest.main()
