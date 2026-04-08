import importlib.util
import json
import os
import tempfile
import unittest
from unittest.mock import patch


SUDACHI_AVAILABLE = importlib.util.find_spec("sudachipy") is not None


class _FakeMorpheme:
    def __init__(
        self,
        surface: str,
        reading_form: str,
        pos: tuple[str, ...] = ("名詞", "一般"),
    ):
        self._surface = surface
        self._reading_form = reading_form
        self._pos = pos

    def surface(self):
        return self._surface

    def reading_form(self):
        return self._reading_form

    def part_of_speech(self):
        return self._pos


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

    @patch(
        "src.furikanji.adapters.sudachi_furigana_reading_generator.dictionary.Dictionary"
    )
    def test_watashi_override_applies_before_common_particle(self, dictionary_mock):
        tokenizer_mock = dictionary_mock.return_value.create.return_value
        tokenizer_mock.tokenize.return_value = [
            _FakeMorpheme("私", "ワタクシ", ("代名詞", "*", "*")),
            _FakeMorpheme("が", "ガ", ("助詞", "格助詞", "一般")),
        ]

        from src.furikanji.adapters.sudachi_furigana_reading_generator import (
            SudachiFuriganaReadingGenerator,
        )

        generator = SudachiFuriganaReadingGenerator()
        segments = generator.resolve_line_segments("私が")

        self.assertEqual(segments[0].base_text, "私")
        self.assertEqual(segments[0].reading, "わたし")

    @patch(
        "src.furikanji.adapters.sudachi_furigana_reading_generator.dictionary.Dictionary"
    )
    def test_watashi_override_does_not_apply_when_not_pronoun(self, dictionary_mock):
        tokenizer_mock = dictionary_mock.return_value.create.return_value
        tokenizer_mock.tokenize.return_value = [
            _FakeMorpheme("私", "ワタクシ", ("名詞", "一般")),
            _FakeMorpheme("が", "ガ", ("助詞", "格助詞", "一般")),
        ]

        from src.furikanji.adapters.sudachi_furigana_reading_generator import (
            SudachiFuriganaReadingGenerator,
        )

        generator = SudachiFuriganaReadingGenerator()
        segments = generator.resolve_line_segments("私が")

        self.assertEqual(segments[0].base_text, "私")
        self.assertEqual(segments[0].reading, "わたくし")

    @patch(
        "src.furikanji.adapters.sudachi_furigana_reading_generator.dictionary.Dictionary"
    )
    def test_override_exception_suffix_skips_override(self, dictionary_mock):
        tokenizer_mock = dictionary_mock.return_value.create.return_value
        tokenizer_mock.tokenize.return_value = [
            _FakeMorpheme("私", "ワタクシ", ("名詞", "代名詞", "一般")),
            _FakeMorpheme("達", "タチ", ("名詞", "接尾", "一般")),
            _FakeMorpheme("が", "ガ", ("助詞", "格助詞", "一般")),
        ]

        rules_payload = {
            "rules": [
                {
                    "kanji": "私",
                    "reading": "わたし",
                    "pos_contains": ["代名詞"],
                    "next_surfaces": [],
                    "prev_surfaces": [],
                    "exception_prefixes": [],
                    "exception_suffixes": ["達"],
                }
            ]
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            json.dump(rules_payload, handle, ensure_ascii=False)
            rules_path = handle.name

        from src.furikanji.adapters.sudachi_furigana_reading_generator import (
            SudachiFuriganaReadingGenerator,
        )

        try:
            generator = SudachiFuriganaReadingGenerator(
                reading_overrides_path=rules_path
            )
            segments = generator.resolve_line_segments("私達が")
        finally:
            os.remove(rules_path)

        self.assertEqual(segments[0].base_text, "私")
        self.assertEqual(segments[0].reading, "わたくし")

    @patch(
        "src.furikanji.adapters.sudachi_furigana_reading_generator.dictionary.Dictionary"
    )
    def test_trims_kana_suffix_for_mixed_token(self, dictionary_mock):
        tokenizer_mock = dictionary_mock.return_value.create.return_value
        tokenizer_mock.tokenize.return_value = [
            _FakeMorpheme("私たち", "ワタシタチ"),
        ]

        from src.furikanji.adapters.sudachi_furigana_reading_generator import (
            SudachiFuriganaReadingGenerator,
        )

        generator = SudachiFuriganaReadingGenerator()
        segments = generator.resolve_line_segments("私たち")

        self.assertEqual(segments[0].reading, "わたし")
        self.assertTrue(segments[0].needs_furigana)

    @patch(
        "src.furikanji.adapters.sudachi_furigana_reading_generator.dictionary.Dictionary"
    )
    def test_trims_okurigana_suffix(self, dictionary_mock):
        tokenizer_mock = dictionary_mock.return_value.create.return_value
        tokenizer_mock.tokenize.return_value = [
            _FakeMorpheme("食べる", "タベル"),
        ]

        from src.furikanji.adapters.sudachi_furigana_reading_generator import (
            SudachiFuriganaReadingGenerator,
        )

        generator = SudachiFuriganaReadingGenerator()
        segments = generator.resolve_line_segments("食べる")

        self.assertEqual(segments[0].reading, "た")
        self.assertTrue(segments[0].needs_furigana)

    @patch(
        "src.furikanji.adapters.sudachi_furigana_reading_generator.dictionary.Dictionary"
    )
    def test_pure_kanji_keeps_full_reading_when_no_overlap(self, dictionary_mock):
        tokenizer_mock = dictionary_mock.return_value.create.return_value
        tokenizer_mock.tokenize.return_value = [
            _FakeMorpheme("学校", "ガッコウ"),
        ]

        from src.furikanji.adapters.sudachi_furigana_reading_generator import (
            SudachiFuriganaReadingGenerator,
        )

        generator = SudachiFuriganaReadingGenerator()
        segments = generator.resolve_line_segments("学校")

        self.assertEqual(segments[0].reading, "がっこう")
        self.assertTrue(segments[0].needs_furigana)

    @patch(
        "src.furikanji.adapters.sudachi_furigana_reading_generator.dictionary.Dictionary"
    )
    def test_full_overlap_disables_furigana(self, dictionary_mock):
        tokenizer_mock = dictionary_mock.return_value.create.return_value
        tokenizer_mock.tokenize.return_value = [
            _FakeMorpheme("かな", "カナ"),
        ]

        from src.furikanji.adapters.sudachi_furigana_reading_generator import (
            SudachiFuriganaReadingGenerator,
        )

        generator = SudachiFuriganaReadingGenerator()
        segments = generator.resolve_line_segments("かな")

        self.assertEqual(segments[0].reading, "かな")
        self.assertFalse(segments[0].needs_furigana)

    @patch(
        "src.furikanji.adapters.sudachi_furigana_reading_generator.dictionary.Dictionary"
    )
    def test_override_applies_before_suffix_trim(self, dictionary_mock):
        tokenizer_mock = dictionary_mock.return_value.create.return_value
        tokenizer_mock.tokenize.return_value = [
            _FakeMorpheme("私たち", "ワタクシタチ", ("代名詞", "*", "*")),
            _FakeMorpheme("が", "ガ", ("助詞", "格助詞", "一般")),
        ]

        rules_payload = {
            "rules": [
                {
                    "kanji": "私たち",
                    "reading": "わたしたち",
                    "pos_contains": ["代名詞"],
                    "next_surfaces": ["が"],
                    "prev_surfaces": [],
                    "exception_prefixes": [],
                    "exception_suffixes": [],
                }
            ]
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            json.dump(rules_payload, handle, ensure_ascii=False)
            rules_path = handle.name

        from src.furikanji.adapters.sudachi_furigana_reading_generator import (
            SudachiFuriganaReadingGenerator,
        )

        try:
            generator = SudachiFuriganaReadingGenerator(
                reading_overrides_path=rules_path
            )
            segments = generator.resolve_line_segments("私たちが")
        finally:
            os.remove(rules_path)

        self.assertEqual(segments[0].reading, "わたし")
        self.assertTrue(segments[0].needs_furigana)


if __name__ == "__main__":
    unittest.main()
