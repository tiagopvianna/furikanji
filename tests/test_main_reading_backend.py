import unittest

from src.furikanji.main import _build_furigana_reading_generator


class TestMainReadingBackend(unittest.TestCase):
    def test_build_sudachi_backend(self):
        from src.furikanji.adapters.sudachi_furigana_reading_generator import (
            SudachiFuriganaReadingGenerator,
        )

        generator = _build_furigana_reading_generator()
        self.assertIsInstance(generator, SudachiFuriganaReadingGenerator)


if __name__ == "__main__":
    unittest.main()
