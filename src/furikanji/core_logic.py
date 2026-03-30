from src.furikanji.adapters.comic_text_detector_localizer import ComicTextDetectorLocalizer
from src.furikanji.adapters.manga_ocr_text_transcriber import MangaOcrTextTranscriber
from src.furikanji.application.furigana_renderer import FuriganaRenderer
from src.furikanji.application.page_text_extractor import PageTextExtractor
from src.furikanji.utils import dump_json

class CoreLogic:
    def __init__(
        self, pretrained_model_name_or_path="kha-white/manga-ocr-base", force_cpu=False, disable_ocr=False, **kwargs
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.force_cpu = force_cpu
        self.disable_ocr = disable_ocr
        self.kwargs = kwargs
        self.mpocr = None
        self.renderer = FuriganaRenderer()

    def init_models(self):
        if self.mpocr is None:
            extractor_kwargs = dict(self.kwargs)
            detector_input_size = extractor_kwargs.pop("detector_input_size", 1024)
            text_height = extractor_kwargs.get("text_height", 64)
            text_localizer = None
            text_transcriber = None
            if not self.disable_ocr:
                text_localizer = ComicTextDetectorLocalizer(
                    input_size=detector_input_size,
                    text_height=text_height,
                    force_cpu=self.force_cpu,
                )
                text_transcriber = MangaOcrTextTranscriber(
                    pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                    force_cpu=self.force_cpu,
                )
            self.mpocr = PageTextExtractor(
                text_localizer=text_localizer,
                text_transcriber=text_transcriber,
                disable_ocr=self.disable_ocr,
                **extractor_kwargs,
            )

    def process_volume(self, path: str):
        self.init_models()
        result = self.mpocr(path)
        dump_json(result, "result.json")
        self.renderer.render_overlay(
            path,
            result,
            "result_overlay.jpg",
        )
