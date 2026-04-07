import re
import shlex
from pathlib import Path

import jaconv
from fugashi import Tagger
from loguru import logger

from src.furikanji.application.interfaces import (
    FuriganaReadingGenerator,
    FuriganaSegment,
)


class FugashiFuriganaReadingGenerator(FuriganaReadingGenerator):
    def __init__(self):
        self._tagger = self._build_tagger()
        self._kanji_regex = re.compile(r"[\u4E00-\u9FFF]")

    def _build_tagger(self) -> Tagger:
        dicdir = self._try_import_dicdir("unidic")
        if dicdir is not None:
            logger.info("Loading fugashi with dictionary=unidic")
            return Tagger(f"-d {shlex.quote(dicdir)}")

        dicdir = self._try_import_dicdir("unidic_lite")
        if dicdir is not None:
            logger.info("Loading fugashi with dictionary=unidic_lite")
            return Tagger(f"-d {shlex.quote(dicdir)}")

        logger.info("Loading fugashi with dictionary=default")
        return Tagger()

    def _try_import_dicdir(self, module_name: str) -> str | None:
        try:
            module = __import__(module_name)
        except Exception:
            return None
        dicdir = getattr(module, "DICDIR", None)
        if not dicdir:
            return None
        return dicdir if Path(dicdir).exists() else None

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
