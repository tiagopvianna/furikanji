import re

import jaconv
from loguru import logger
from sudachipy import dictionary, tokenizer

from src.furikanji.application.interfaces import (
    FuriganaReadingGenerator,
    FuriganaSegment,
)


class SudachiFuriganaReadingGenerator(FuriganaReadingGenerator):
    def __init__(self, split_mode: str = "C", dictionary_type: str = "core"):
        self._kanji_regex = re.compile(r"[\u4E00-\u9FFF]")
        self._split_mode_name = split_mode.upper()
        self._split_mode = self._resolve_split_mode(self._split_mode_name)
        self._dictionary_type = dictionary_type
        self._tokenizer = dictionary.Dictionary(dict=dictionary_type).create()
        logger.info(
            "Initializing SudachiPy: dictionary={}, split_mode={}",
            self._dictionary_type,
            self._split_mode_name,
        )

    def _resolve_split_mode(self, split_mode: str):
        split_modes = tokenizer.Tokenizer.SplitMode
        if split_mode == "A":
            return split_modes.A
        if split_mode == "B":
            return split_modes.B
        if split_mode == "C":
            return split_modes.C
        raise ValueError("split_mode must be one of: A, B, C")

    def resolve_line_segments(self, line_text: str) -> list[FuriganaSegment]:
        segments: list[FuriganaSegment] = []
        for morpheme in self._tokenizer.tokenize(line_text, self._split_mode):
            base_text = morpheme.surface()
            reading_kata = morpheme.reading_form()
            reading = (
                jaconv.kata2hira(reading_kata)
                if reading_kata and reading_kata != "*"
                else base_text
            )
            segments.append(
                FuriganaSegment(
                    base_text=base_text,
                    reading=reading,
                    needs_furigana=self._requires_furigana(
                        base_text=base_text,
                        reading=reading,
                    ),
                )
            )
        return segments

    def _requires_furigana(self, base_text: str, reading: str) -> bool:
        return bool(self._kanji_regex.search(base_text) and base_text != reading)
