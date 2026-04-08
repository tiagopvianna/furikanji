import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from src.furikanji.application.furigana_renderer import (
    DrawCommand,
    EraseConfig,
    FuriganaRenderConfig,
    FuriganaRenderer,
    PageRenderPlan,
    RegionRenderPlan,
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

    def test_paint_page_render_plan_erases_all_regions_before_drawing_any_region(self):
        renderer = FuriganaRenderer(
            furigana_reading_generator=FakeFuriganaReadingGenerator(),
            config=FuriganaRenderConfig(erase=EraseConfig(strategy="both")),
        )
        page_render_plan = PageRenderPlan(
            region_plans=[
                RegionRenderPlan(
                    draw_commands=[],
                    planned_bounds=(0.0, 0.0, 10.0, 10.0),
                    line_outline_points=[[[0, 0], [10, 0], [10, 10], [0, 10]]],
                ),
                RegionRenderPlan(
                    draw_commands=[DrawCommand(text="a", x=5.0, y=5.0, font=None)],
                    planned_bounds=(20.0, 20.0, 30.0, 30.0),
                    line_outline_points=[[[20, 20], [30, 20], [30, 30], [20, 30]]],
                ),
            ]
        )
        call_order: list[str] = []

        def erase_side_effect(*args, **kwargs):
            call_order.append("erase")

        def paint_side_effect(*args, **kwargs):
            call_order.append("paint")

        with patch.object(
            renderer, "_erase_background_for_region", side_effect=erase_side_effect
        ) as erase_mock, patch.object(
            renderer, "_paint_planned_region_text", side_effect=paint_side_effect
        ) as paint_mock:
            renderer.paint_page_render_plan(
                draw=self.draw,
                page_render_plan=page_render_plan,
            )

        self.assertEqual(erase_mock.call_count, 2)
        self.assertEqual(paint_mock.call_count, 2)
        self.assertEqual(call_order, ["erase", "erase", "paint", "paint"])


if __name__ == "__main__":
    unittest.main()
