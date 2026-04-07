import importlib.util
import unittest
from unittest.mock import patch


SUDACHI_AVAILABLE = importlib.util.find_spec("sudachipy") is not None


class _FakeMorpheme:
    def __init__(self, surface: str, reading_form: str):
        self._surface = surface
        self._reading_form = reading_form

    def surface(self):
        return self._surface

    def reading_form(self):
        return self._reading_form


@unittest.skipUnless(SUDACHI_AVAILABLE, "sudachipy is required")
class TestSudachiFuriganaReadingGenerator(unittest.TestCase):
    @patch(
        "src.furikanji.adapters.sudachi_furigana_reading_generator.dictionary.Dictionary"
    )
    def test_resolve_line_segments_converts_to_hiragana_and_flags_kanji(self, dictionary_mock):
        tokenizer_mock = dictionary_mock.return_value.create.return_value
        tokenizer_mock.tokenize.return_value = [
            _FakeMorpheme("漢字", "カンジ"),
            _FakeMorpheme("かな", "カナ"),
            _FakeMorpheme("ABC", "*"),
        ]

        from src.furikanji.adapters.sudachi_furigana_reading_generator import (
            SudachiFuriganaReadingGenerator,
        )

        generator = SudachiFuriganaReadingGenerator()
        segments = generator.resolve_line_segments("漢字かなABC")

        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0].base_text, "漢字")
        self.assertEqual(segments[0].reading, "かんじ")
        self.assertTrue(segments[0].needs_furigana)

        self.assertEqual(segments[1].base_text, "かな")
        self.assertEqual(segments[1].reading, "かな")
        self.assertFalse(segments[1].needs_furigana)

        self.assertEqual(segments[2].base_text, "ABC")
        self.assertEqual(segments[2].reading, "ABC")
        self.assertFalse(segments[2].needs_furigana)

    @patch(
        "src.furikanji.adapters.sudachi_furigana_reading_generator.dictionary.Dictionary"
    )
    def test_invalid_split_mode_raises_value_error(self, dictionary_mock):
        from src.furikanji.adapters.sudachi_furigana_reading_generator import (
            SudachiFuriganaReadingGenerator,
        )

        with self.assertRaises(ValueError):
            SudachiFuriganaReadingGenerator(split_mode="D")


if __name__ == "__main__":
    unittest.main()
