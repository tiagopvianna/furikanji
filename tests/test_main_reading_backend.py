import importlib.util
import unittest

from src.furikanji.adapters.fugashi_furigana_reading_generator import (
    FugashiFuriganaReadingGenerator,
)
from src.furikanji.main import _build_furigana_reading_generator


SUDACHI_AVAILABLE = importlib.util.find_spec("sudachipy") is not None


class TestMainReadingBackend(unittest.TestCase):
    def test_build_fugashi_backend(self):
        generator = _build_furigana_reading_generator("fugashi")
        self.assertIsInstance(generator, FugashiFuriganaReadingGenerator)

    @unittest.skipUnless(SUDACHI_AVAILABLE, "sudachipy is required")
    def test_build_sudachi_backend(self):
        from src.furikanji.adapters.sudachi_furigana_reading_generator import (
            SudachiFuriganaReadingGenerator,
        )

        generator = _build_furigana_reading_generator("sudachi")
        self.assertIsInstance(generator, SudachiFuriganaReadingGenerator)

    def test_invalid_reading_backend_raises_value_error(self):
        with self.assertRaises(ValueError):
            _build_furigana_reading_generator("invalid")


if __name__ == "__main__":
    unittest.main()
