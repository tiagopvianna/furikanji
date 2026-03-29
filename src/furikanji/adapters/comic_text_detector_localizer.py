import torch

from src.furikanji.adapters.interfaces import TextLocalizationResult
from src.furikanji.adapters.model_cache_adapter import ModelCacheAdapter


class ComicTextDetectorLocalizer:
    def __init__(self, input_size=1024, force_cpu=False, cache_adapter=None):
        from comic_text_detector.inference import TextDetector

        if cache_adapter is None:
            cache_adapter = ModelCacheAdapter()

        cuda = torch.cuda.is_available()
        device = "cuda" if cuda and not force_cpu else "cpu"
        self._detector = TextDetector(
            model_path=cache_adapter.comic_text_detector_model_path,
            input_size=input_size,
            device=device,
            act="leaky",
        )

    def localize_text(self, image):
        mask, refined_mask, text_blocks = self._detector(image, refine_mode=1, keep_undetected_mask=True)
        return TextLocalizationResult(mask=mask, refined_mask=refined_mask, text_blocks=text_blocks)
