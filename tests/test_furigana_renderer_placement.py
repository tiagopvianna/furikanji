import unittest

from PIL import ImageFont

from src.furikanji.application.furigana_renderer import (
    DrawCommand,
    FuriganaRenderConfig,
    FuriganaRenderer,
    PlacementConfig,
)


class FakeFuriganaReadingGenerator:
    def resolve_line_segments(self, line_text: str):
        return []


class TestFuriganaRendererPlacement(unittest.TestCase):
    def test_center_policy_places_intrinsic_layout_at_region_center(self):
        renderer = FuriganaRenderer(
            furigana_reading_generator=FakeFuriganaReadingGenerator(),
            config=FuriganaRenderConfig(
                placement=PlacementConfig(policy="center", overflow_aware_anchor="center")
            ),
        )
        font = ImageFont.load_default()
        commands = [DrawCommand(text="x", x=0.0, y=0.0, font=font)]

        translated, final_bounds, dx, dy, overflow_flags = renderer._place_region_intrinsic_layout(
            draw_commands=commands,
            intrinsic_bounds=(0.0, 0.0, 20.0, 10.0),
            region_origin=(0.0, 0.0),
            region_detected_bounds=(40.0, 30.0, 80.0, 70.0),
            image_size=(200, 200),
        )

        self.assertEqual((dx, dy), (50.0, 45.0))
        self.assertEqual(final_bounds, (50.0, 45.0, 70.0, 55.0))
        self.assertEqual((translated[0].x, translated[0].y), (50.0, 45.0))
        self.assertEqual(
            overflow_flags,
            {
                "overflow_left": False,
                "overflow_top": False,
                "overflow_right": False,
                "overflow_bottom": False,
            },
        )

    def test_overflow_aware_policy_corrects_right_overflow_after_centering(self):
        renderer = FuriganaRenderer(
            furigana_reading_generator=FakeFuriganaReadingGenerator(),
            config=FuriganaRenderConfig(
                placement=PlacementConfig(policy="overflow_aware", overflow_aware_anchor="center")
            ),
        )
        font = ImageFont.load_default()
        commands = [DrawCommand(text="x", x=0.0, y=0.0, font=font)]

        translated, final_bounds, dx, dy, overflow_flags = renderer._place_region_intrinsic_layout(
            draw_commands=commands,
            intrinsic_bounds=(0.0, 0.0, 30.0, 10.0),
            region_origin=(0.0, 0.0),
            region_detected_bounds=(90.0, 10.0, 110.0, 30.0),
            image_size=(100, 100),
        )

        self.assertEqual((dx, dy), (70.0, 15.0))
        self.assertEqual(final_bounds, (70.0, 15.0, 100.0, 25.0))
        self.assertEqual((translated[0].x, translated[0].y), (70.0, 15.0))
        self.assertEqual(
            overflow_flags,
            {
                "overflow_left": False,
                "overflow_top": False,
                "overflow_right": False,
                "overflow_bottom": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
