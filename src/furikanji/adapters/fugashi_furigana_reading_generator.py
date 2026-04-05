import re

import jaconv
from fugashi import Tagger

from src.furikanji.application.interfaces import FuriganaReadingGenerator, FuriganaSegment


class FugashiFuriganaReadingGenerator(FuriganaReadingGenerator):
    def __init__(self):
        self._tagger = Tagger()
        self._kanji_regex = re.compile(r"[\u4E00-\u9FFF]")

    def resolve_line_segments(self, line_text: str) -> list[FuriganaSegment]:
        segments: list[FuriganaSegment] = []
        for word in self._tagger(line_text):
            base_text = word.surface
            kana = getattr(word.feature, "kana", None)
            reading = jaconv.kata2hira(kana) if kana else base_text
            segments.append(
                FuriganaSegment(
                    base_text=base_text,
                    reading=reading,
                    needs_furigana=self._requires_furigana(
                        base_text=base_text, reading=reading
                    ),
                )
            )
        return segments

    def _requires_furigana(self, base_text: str, reading: str) -> bool:
        return bool(self._kanji_regex.search(base_text) and base_text != reading)
