import json
import re
from pathlib import Path

import jaconv
from loguru import logger
from sudachipy import dictionary, tokenizer

from src.furikanji.application.interfaces import (
    FuriganaReadingGenerator,
    FuriganaSegment,
)


class SudachiFuriganaReadingGenerator(FuriganaReadingGenerator):
    def __init__(
        self,
        split_mode: str = "C",
        dictionary_type: str = "core",
        reading_overrides_path: str | None = None,
    ):
        self._kanji_regex = re.compile(r"[\u4E00-\u9FFF]")
        self._split_mode_name = split_mode.upper()
        self._split_mode = self._resolve_split_mode(self._split_mode_name)
        self._dictionary_type = dictionary_type
        self._tokenizer = dictionary.Dictionary(dict=dictionary_type).create()
        self._reading_override_rules = self._load_reading_override_rules(
            reading_overrides_path
        )
        logger.info(
            "Initializing SudachiPy: dictionary={}, split_mode={}, override_rules={}",
            self._dictionary_type,
            self._split_mode_name,
            len(self._reading_override_rules),
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

    def _load_reading_override_rules(
        self, reading_overrides_path: str | None
    ) -> list[dict]:
        default_path = Path(__file__).with_name("sudachi_reading_overrides.json")
        path = Path(reading_overrides_path) if reading_overrides_path else default_path
        if not path.exists():
            logger.warning("Sudachi reading override file not found: {}", path)
            return []

        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        rules = payload.get("rules", []) if isinstance(payload, dict) else []
        normalized_rules: list[dict] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            if not rule.get("kanji") or not rule.get("reading"):
                continue
            normalized_rules.append(
                {
                    "kanji": str(rule["kanji"]),
                    "reading": str(rule["reading"]),
                    "pos_contains": [str(v) for v in rule.get("pos_contains", [])],
                    "next_surfaces": [str(v) for v in rule.get("next_surfaces", [])],
                    "prev_surfaces": [str(v) for v in rule.get("prev_surfaces", [])],
                    "exception_prefixes": [
                        str(v) for v in rule.get("exception_prefixes", [])
                    ],
                    "exception_suffixes": [
                        str(v) for v in rule.get("exception_suffixes", [])
                    ],
                }
            )
        return normalized_rules

    def resolve_line_segments(self, line_text: str) -> list[FuriganaSegment]:
        morphemes = list(self._tokenizer.tokenize(line_text, self._split_mode))
        surfaces = [morpheme.surface() for morpheme in morphemes]
        segments: list[FuriganaSegment] = []
        for index, morpheme in enumerate(morphemes):
            base_text = surfaces[index]
            reading_kata = morpheme.reading_form()
            reading = (
                jaconv.kata2hira(reading_kata)
                if reading_kata and reading_kata != "*"
                else base_text
            )
            reading = self._apply_reading_overrides(
                morphemes=morphemes,
                surfaces=surfaces,
                index=index,
                base_text=base_text,
                reading=reading,
            )
            reading = self._trim_redundant_kana_suffix(
                base_text=base_text,
                reading=reading,
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

    def _apply_reading_overrides(
        self,
        morphemes: list,
        surfaces: list[str],
        index: int,
        base_text: str,
        reading: str,
    ) -> str:
        prev_surface = surfaces[index - 1] if index > 0 else ""
        next_surface = surfaces[index + 1] if index + 1 < len(surfaces) else ""
        prefix_text = "".join(surfaces[:index])
        suffix_text = "".join(surfaces[index + 1 :])

        for rule in self._reading_override_rules:
            if base_text != rule["kanji"]:
                continue
            if not self._rule_matches_context(
                rule=rule,
                morpheme=morphemes[index],
                prev_surface=prev_surface,
                next_surface=next_surface,
                prefix_text=prefix_text,
                suffix_text=suffix_text,
            ):
                continue
            return rule["reading"]
        return reading

    def _rule_matches_context(
        self,
        rule: dict,
        morpheme,
        prev_surface: str,
        next_surface: str,
        prefix_text: str,
        suffix_text: str,
    ) -> bool:
        pos_contains = rule.get("pos_contains", [])
        if pos_contains and not self._pos_contains_any(morpheme, pos_contains):
            return False

        prev_surfaces = rule.get("prev_surfaces", [])
        if prev_surfaces and prev_surface not in prev_surfaces:
            return False

        next_surfaces = rule.get("next_surfaces", [])
        if next_surfaces and next_surface not in next_surfaces:
            return False

        for prefix in rule.get("exception_prefixes", []):
            if prefix and prefix_text.endswith(prefix):
                return False

        for suffix in rule.get("exception_suffixes", []):
            if suffix and suffix_text.startswith(suffix):
                return False

        return True

    @staticmethod
    def _pos_contains_any(morpheme, keywords: list[str]) -> bool:
        part_of_speech = getattr(morpheme, "part_of_speech", None)
        if callable(part_of_speech):
            try:
                pos_tuple = part_of_speech()
            except Exception:
                pos_tuple = ()
        else:
            pos_tuple = ()
        if not pos_tuple:
            return False
        pos_values = [str(pos) for pos in pos_tuple]
        return any(keyword in pos for keyword in keywords for pos in pos_values)

    def _requires_furigana(self, base_text: str, reading: str) -> bool:
        return bool(self._kanji_regex.search(base_text) and base_text != reading)

    def _trim_redundant_kana_suffix(self, base_text: str, reading: str) -> str:
        base_hira = self._to_hiragana(base_text)
        reading_hira = self._to_hiragana(reading)
        trim_count = 0

        while trim_count < len(base_text) and trim_count < len(reading):
            base_char = base_hira[len(base_hira) - 1 - trim_count]
            reading_char = reading_hira[len(reading_hira) - 1 - trim_count]
            if base_char != reading_char:
                break
            if not (self._is_kana(base_char) and self._is_kana(reading_char)):
                break
            trim_count += 1

        if trim_count == 0:
            return reading
        if trim_count >= len(reading):
            return base_text
        return reading[:-trim_count]

    @staticmethod
    def _to_hiragana(text: str) -> str:
        return jaconv.kata2hira(text)

    @staticmethod
    def _is_kana(ch: str) -> bool:
        codepoint = ord(ch)
        return (
            0x3040 <= codepoint <= 0x309F
            or 0x30A0 <= codepoint <= 0x30FF
            or 0xFF66 <= codepoint <= 0xFF9D
        )
