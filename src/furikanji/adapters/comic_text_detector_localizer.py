import torch

from src.furikanji.adapters.model_cache_adapter import ModelCacheAdapter
from src.furikanji.application.interfaces import (
    LocalizedTextLine,
    LocalizedTextRegion,
    TextLocalizationResult,
)


class ComicTextDetectorLocalizer:
    def __init__(
        self, input_size=1024, text_height=64, force_cpu=False, cache_adapter=None
    ):
        from comic_text_detector.inference import TextDetector

        if cache_adapter is None:
            cache_adapter = ModelCacheAdapter()

        self.text_height = text_height
        cuda = torch.cuda.is_available()
        device = "cuda" if cuda and not force_cpu else "cpu"
        self._detector = TextDetector(
            model_path=cache_adapter.comic_text_detector_model_path,
            input_size=input_size,
            device=device,
            act="leaky",
        )

    def localize_text(self, image):
        _, refined_mask, text_blocks = self._detector(
            image, refine_mode=1, keep_undetected_mask=True
        )
        localized_text_regions = []
        for blk in text_blocks:
            lines = []
            for line_idx, line_outline in enumerate(blk.lines_array()):
                line_image = blk.get_transformed_region(
                    image, line_idx, self.text_height
                )
                line_text_mask = blk.get_transformed_region(
                    refined_mask, line_idx, self.text_height
                )
                line_outline_list = line_outline.tolist()
                lines.append(
                    LocalizedTextLine(
                        line_outline=line_outline_list,
                        line_image=line_image,
                        line_text_mask=line_text_mask,
                    )
                )

            localized_text_regions.append(
                LocalizedTextRegion(
                    bounding_box=list(blk.xyxy),
                    is_vertical=blk.vertical,
                    estimated_font_size=blk.font_size,
                    lines=lines,
                )
            )

        return TextLocalizationResult(
            text_mask=refined_mask,
            localized_text_regions=localized_text_regions,
        )
