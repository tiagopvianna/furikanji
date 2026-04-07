from src.furikanji.adapters.fugashi_furigana_reading_generator import (
    FugashiFuriganaReadingGenerator,
)
from src.furikanji.application.interfaces import (
    FuriganaReadingGenerator,
    FuriganaSegment,
    LocalizedTextLine,
    LocalizedTextRegion,
    TextLocalizationResult,
    TextLocalizerAdapter,
    TextTranscriberAdapter,
)

__all__ = [
    "LocalizedTextLine",
    "LocalizedTextRegion",
    "FuriganaReadingGenerator",
    "FuriganaSegment",
    "FugashiFuriganaReadingGenerator",
    "TextLocalizationResult",
    "TextLocalizerAdapter",
    "TextTranscriberAdapter",
]

try:
    from src.furikanji.adapters.sudachi_furigana_reading_generator import (
        SudachiFuriganaReadingGenerator,
    )

    __all__.append("SudachiFuriganaReadingGenerator")
except Exception:
    pass
