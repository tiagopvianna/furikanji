import fire
import numpy as np

from src.furikanji.adapters.comic_text_detector_localizer import (
    ComicTextDetectorLocalizer,
)
from src.furikanji.adapters.fugashi_furigana_reading_generator import (
    FugashiFuriganaReadingGenerator,
)
from src.furikanji.adapters.manga_ocr_text_transcriber import MangaOcrTextTranscriber
from src.furikanji.application.furigana_renderer import (
    FuriganaRenderConfig,
    FuriganaRenderer,
)
from src.furikanji.application.interfaces import TextLocalizationResult
from src.furikanji.application.page_text_extractor import PageTextExtractor
from src.furikanji.application.process_image_use_case import ProcessImageUseCase


class _NoopTextLocalizer:
    def localize_text(self, image: np.ndarray) -> TextLocalizationResult:
        height, width = image.shape[:2]
        return TextLocalizationResult(
            text_mask=np.zeros((height, width), dtype=np.uint8),
            localized_text_regions=[],
        )


class _NoopTextTranscriber:
    def transcribe_text(self, image_crop) -> str:
        return ""


def _build_furigana_reading_generator(reading_backend: str):
    normalized_backend = reading_backend.lower()
    if normalized_backend == "fugashi":
        return FugashiFuriganaReadingGenerator()
    if normalized_backend == "sudachi":
        from src.furikanji.adapters.sudachi_furigana_reading_generator import (
            SudachiFuriganaReadingGenerator,
        )

        return SudachiFuriganaReadingGenerator()
    raise ValueError("reading_backend must be one of: fugashi, sudachi")


def process_single_image(
    image_path: str,
    output_path: str = "output.png",
    json_output_path: str = "result.json",
    device: str = "cpu",
    force_cpu: bool = False,
    pretrained_model_name_or_path: str = "kha-white/manga-ocr-base",
    reading_backend: str = "fugashi",
    disable_ocr: bool = False,
    draw_target_boxes: bool = False,
    draw_overlay_text: bool = True,
    **extractor_kwargs,
):
    normalized_device = device.lower()
    if normalized_device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: cpu, cuda")
    effective_force_cpu = force_cpu or normalized_device == "cpu"

    detector_input_size = extractor_kwargs.pop("detector_input_size", 1024)
    text_height = extractor_kwargs.get("text_height", 64)

    if disable_ocr:
        text_localizer = _NoopTextLocalizer()
        text_transcriber = _NoopTextTranscriber()
    else:
        text_localizer = ComicTextDetectorLocalizer(
            input_size=detector_input_size,
            text_height=text_height,
            force_cpu=effective_force_cpu,
        )
        text_transcriber = MangaOcrTextTranscriber(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            force_cpu=effective_force_cpu,
        )

    page_text_extractor = PageTextExtractor(
        text_localizer=text_localizer,
        text_transcriber=text_transcriber,
        disable_ocr=disable_ocr,
        **extractor_kwargs,
    )
    furigana_reading_generator = _build_furigana_reading_generator(reading_backend)
    furigana_renderer = FuriganaRenderer(
        furigana_reading_generator=furigana_reading_generator,
        config=FuriganaRenderConfig(
            draw_target_boxes=draw_target_boxes,
            draw_overlay_text=draw_overlay_text,
        ),
    )
    process_image_use_case = ProcessImageUseCase(page_text_extractor, furigana_renderer)
    process_image_use_case(
        image_path=image_path,
        json_output_path=json_output_path,
        overlay_output_path=output_path,
    )


if __name__ == "__main__":
    fire.Fire(process_single_image)
