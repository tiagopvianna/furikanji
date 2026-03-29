class MangaOcrTextTranscriber:
    def __init__(self, pretrained_model_name_or_path="kha-white/manga-ocr-base", force_cpu=False):
        from manga_ocr import MangaOcr

        self._ocr = MangaOcr(pretrained_model_name_or_path, force_cpu)

    def transcribe_text(self, image_crop):
        return self._ocr(image_crop)
