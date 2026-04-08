import unittest

from src.furikanji.application.furigana_renderer import (
    FuriganaRenderConfig,
    FuriganaRenderer,
    LineBounds,
    VerticalLayoutConfig,
)
from src.furikanji.application.interfaces import FuriganaSegment


class FakeFuriganaReadingGenerator:
    def resolve_line_segments(self, line_text: str):
        return []


class DummyFont:
    def __init__(self, size: int):
        self.size = size


class FakeDraw:
    def textbbox(self, xy, text, font):
        x, y = xy
        height = getattr(font, "size", 10)
        width = max(1, len(text)) * max(1, height // 2)
        return (x, y, x + width, y + height)


class TestFuriganaRendererVerticalRubyFit(unittest.TestCase):
    def setUp(self):
        self.draw = FakeDraw()

    def test_vertical_ruby_is_bounded_per_token_without_overlap(self):
        renderer = FuriganaRenderer(
            furigana_reading_generator=FakeFuriganaReadingGenerator(),
            config=FuriganaRenderConfig(
                vertical=VerticalLayoutConfig(
                    furigana_y_offset=0,
                    ruby_min_size=6,
                    ruby_min_spacing=1,
                )
            ),
        )
        renderer._load_japanese_font = lambda size: DummyFont(size)

        commands, _ = renderer._plan_vertical_line_layout(
            draw=self.draw,
            bounds=LineBounds(x_min=0, x_max=20, y_min=0, y_max=40),
            segments=[
                FuriganaSegment(base_text="漢", reading="あああ", needs_furigana=True),
                FuriganaSegment(base_text="字", reading="いいい", needs_furigana=True),
            ],
            font=DummyFont(20),
            furigana_font=DummyFont(10),
            furigana_size=10,
            line_shift=0,
            char_spacing=0,
            furigana_spacing=2,
        )

        ruby_first = commands[1:4]
        ruby_second = commands[5:8]
        first_bottom = max(command.y + command.font.size for command in ruby_first)
        second_top = min(command.y for command in ruby_second)
        self.assertLessEqual(first_bottom, second_top)

    def test_vertical_ruby_uses_shrink_path_when_default_size_does_not_fit(self):
        renderer = FuriganaRenderer(
            furigana_reading_generator=FakeFuriganaReadingGenerator(),
            config=FuriganaRenderConfig(
                vertical=VerticalLayoutConfig(
                    furigana_y_offset=0,
                    ruby_min_size=8,
                    ruby_min_spacing=1,
                )
            ),
        )
        renderer._load_japanese_font = lambda size: DummyFont(size)

        commands, _ = renderer._plan_vertical_line_layout(
            draw=self.draw,
            bounds=LineBounds(x_min=0, x_max=20, y_min=0, y_max=20),
            segments=[
                FuriganaSegment(base_text="漢", reading="ああ", needs_furigana=True),
            ],
            font=DummyFont(20),
            furigana_font=DummyFont(10),
            furigana_size=10,
            line_shift=0,
            char_spacing=0,
            furigana_spacing=4,
        )

        ruby_commands = commands[1:]
        self.assertTrue(ruby_commands)
        self.assertTrue(all(command.font.size == 8 for command in ruby_commands))

    def test_vertical_ruby_renders_at_minimum_when_budget_is_still_too_small(self):
        renderer = FuriganaRenderer(
            furigana_reading_generator=FakeFuriganaReadingGenerator(),
            config=FuriganaRenderConfig(
                vertical=VerticalLayoutConfig(
                    furigana_y_offset=0,
                    ruby_min_size=6,
                    ruby_min_spacing=1,
                )
            ),
        )
        renderer._load_japanese_font = lambda size: DummyFont(size)

        commands, _ = renderer._plan_vertical_line_layout(
            draw=self.draw,
            bounds=LineBounds(x_min=0, x_max=20, y_min=0, y_max=10),
            segments=[
                FuriganaSegment(base_text="漢", reading="あああああ", needs_furigana=True),
            ],
            font=DummyFont(10),
            furigana_font=DummyFont(8),
            furigana_size=8,
            line_shift=0,
            char_spacing=0,
            furigana_spacing=2,
        )

        self.assertEqual(len(commands), 6)
        self.assertEqual(commands[0].text, "漢")
        ruby_commands = commands[1:]
        self.assertTrue(all(command.font.size == 6 for command in ruby_commands))


if __name__ == "__main__":
    unittest.main()
