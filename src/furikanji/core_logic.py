from json import JSONDecodeError

from loguru import logger
from tqdm import tqdm

from src.furikanji.manga_page_ocr import MangaPageOcr
from src.furikanji.utils import dump_json, load_json
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Additional imports for furigana overlay
from fugashi import Tagger
import jaconv
import re

class CoreLogic:
    def __init__(
        self, pretrained_model_name_or_path="kha-white/manga-ocr-base", force_cpu=False, disable_ocr=False, **kwargs
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.force_cpu = force_cpu
        self.disable_ocr = disable_ocr
        self.kwargs = kwargs
        self.mpocr = None

    def _load_japanese_font(self, size):
        font_path = Path(__file__).parent / "fonts" / "NotoSansCJKjp-Regular.otf"
        try:
            return ImageFont.truetype(str(font_path), size)
        except (IOError, OSError):
            return ImageFont.load_default()

    def _overlay_result_on_image(self, image_path, result, overlay_output_path):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        tagger = Tagger()
        kanji_regex = re.compile(r'[\u4E00-\u9FFF]')
        # Reserve a small font size for furigana and compute offsets for all vertical lines
        furigana_size = 6
        offsets = self._compute_vertical_offsets(result, tagger, kanji_regex, furigana_size)
        vertical_counter = 0
        logger.debug(result)

        for block_index, block in enumerate(result.get("blocks", [])):
            vertical = bool(block.get("vertical", False))
            lines = block.get("lines", [])
            lines_coords = block.get("lines_coords", [])
            detected_size = block.get("font_size", 24)
            font = self._load_japanese_font(detected_size - 14)
            furigana_size = 8
            furigana_font = self._load_japanese_font(furigana_size)

            # Compute bounding box for the entire block to erase background once
            all_xs = [pt[0] for line in lines_coords for pt in line]
            all_ys = [pt[1] for line in lines_coords for pt in line]
            bx_min, bx_max = min(all_xs), max(all_xs)
            by_min, by_max = min(all_ys), max(all_ys)
            # Erase underneath
            draw.rectangle([(bx_min - 10, by_min - 10), (bx_max + 10, by_max + 10)], fill=(255, 255, 255))

            for line_text, line_coords in zip(lines, lines_coords):
                # Determine this line's index and its horizontal offset
                line_index = vertical_counter
                dx = offsets.get(line_index, 0)
                vertical_counter += 1
                # Compute bounding box
                logger.debug(f"line_text: {line_text}")
                logger.debug(f"line_coords: {line_coords}")
                xs = [pt[0] for pt in line_coords]
                ys = [pt[1] for pt in line_coords]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                if vertical:
                    # Apply precomputed horizontal shift
                    x_min_shifted = x_min + dx
                    x_max_shifted = x_max + dx
                    column_width = x_max - x_min
                    # Check if any word in the line has Kanji to adjust horizontal spacing
                    has_kanji_line = any(kanji_regex.search(word.surface) and (jaconv.kata2hira(getattr(word.feature, "kana", "")) != word.surface) for word in tagger(line_text))
                    # extra_x = furigana_size
                    extra_x = 0
                    logger.debug(f"has_kanji_line: {has_kanji_line}")
                    y_cursor = y_min
                    for word in tagger(line_text):
                        surface = word.surface
                        kana = getattr(word.feature, "kana", None)
                        reading = jaconv.kata2hira(kana) if kana else surface
                        is_kanji = bool(kanji_regex.search(surface) and surface != reading)
                        # Record start for furigana alignment
                        y_word_start = y_cursor
                        # Draw each character in surface vertically
                        for char in surface:
                            bbox = draw.textbbox((0, 0), char, font=font)
                            w_char, h_char = bbox[2] - bbox[0], bbox[3] - bbox[1]
                            x_pos = x_min_shifted + (column_width - w_char) / 2
                            draw.text((x_pos - 10, y_cursor), char, fill=(0, 0, 0), font=font)
                            y_cursor += h_char + 6
                        # Draw furigana reading to the right of the column if it's kanji
                        if is_kanji:
                            y_furi = y_word_start
                            for kana_char in reading:
                                bbox_f = draw.textbbox((0, 0), kana_char, font=furigana_font)
                                w_f, h_f = bbox_f[2] - bbox_f[0], bbox_f[3] - bbox_f[1]
                                x_f = x_max_shifted
                                draw.text((x_f - 10, y_furi), kana_char, fill=(0, 0, 0), font=furigana_font)
                                y_furi += h_f + 2
                else:
                    # Horizontal: draw words with furigana
                    x_cursor = x_min
                    for word in tagger(line_text):
                        surface = word.surface
                        kana = getattr(word.feature, 'kana', None)
                        reading = jaconv.kata2hira(kana) if kana else surface
                        is_kanji = bool(kanji_regex.search(surface) and surface != reading)
                        # Measure surface width/height
                        bbox_word = draw.textbbox((0, 0), surface, font=font)
                        w_word = bbox_word[2] - bbox_word[0]
                        # If kanji, draw reading above
                        if is_kanji:
                            # Measure reading dimensions
                            bbox_ruby = draw.textbbox((0, 0), reading, font=furigana_font)
                            w_ruby = bbox_ruby[2] - bbox_ruby[0]
                            h_ruby = bbox_ruby[3] - bbox_ruby[1]
                            # Center ruby over the surface
                            ruby_x = x_cursor + (w_word - w_ruby) / 2
                            ruby_y = y_min - h_ruby - 2
                            draw.text((ruby_x, ruby_y), reading, fill=(0, 0, 0), font=furigana_font)
                        # Draw the main surface
                        draw.text((x_cursor, y_min), surface, fill=(0, 0, 0), font=font)
                        x_cursor += w_word
        image.save(overlay_output_path)

    def init_models(self):
        if self.mpocr is None:
            self.mpocr = MangaPageOcr(
                self.pretrained_model_name_or_path,
                force_cpu=self.force_cpu,
                disable_ocr=self.disable_ocr,
                **self.kwargs,
            )

    def process_volume(self, path: str, ignore_errors=False, no_cache=False):
        self.init_models()
        result = self.mpocr(path)
        dump_json(result, "result.json")
        self._overlay_result_on_image(
            path,
            result,
            "result_overlay.jpg",
        )

    def _collect_vertical_line_info(self, result, tagger, kanji_regex, furigana_size):
        lines_info = []
        counter = 0
        for block in result.get("blocks", []):
            if not block.get("vertical", False):
                continue
            for line_text, line_coords in zip(block.get("lines", []), block.get("lines_coords", [])):
                xs = [pt[0] for pt in line_coords]
                ys = [pt[1] for pt in line_coords]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                width = x_max - x_min
                has_kanji = any(
                    kanji_regex.search(word.surface)
                    and (jaconv.kata2hira(getattr(word.feature, "kana", "")) != word.surface)
                    for word in tagger(line_text)
                )
                required_space = furigana_size + 2 if has_kanji else 0
                lines_info.append({
                    "index": counter,
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                    "width": width,
                    "required_space": required_space
                })
                counter += 1
        return lines_info

    def _group_vertical_lines(self, lines_info, x_thresh=12, y_overlap_min=1):
        groups = []
        def overlaps(a, b):
            y_overlap = min(a["y_max"], b["y_max"]) - max(a["y_min"], b["y_min"])
            if y_overlap < y_overlap_min:
                return False
            if abs(a["x_min"] - b["x_max"]) < x_thresh or abs(a["x_max"] - b["x_min"]) < x_thresh:
                return True
            return False
        for line in sorted(lines_info, key=lambda L: L["x_min"]):
            placed = False
            for group in groups:
                if any(overlaps(line, member) for member in group):
                    group.append(line)
                    placed = True
                    break
            if not placed:
                groups.append([line])
        return groups

    def _compute_offsets_from_groups(self, groups):
        offsets = {}
        for group in groups:
            group_sorted = sorted(group, key=lambda L: L["x_min"])
            # First line in group: no shift
            first = group_sorted[0]
            offsets[first["index"]] = 0
            for prev_line, line in zip(group_sorted, group_sorted[1:]):
                prev_index = prev_line["index"]
                prev_x_min = prev_line["x_min"]
                prev_width = prev_line["width"]
                prev_space = prev_line["required_space"]
                prev_dx = offsets[prev_index]
                prev_shifted_right = prev_x_min + prev_dx + prev_width + prev_space
                dx = prev_shifted_right - line["x_min"]
                offsets[line["index"]] = dx
        return offsets

    def _compute_vertical_offsets(self, result, tagger, kanji_regex, furigana_size):
        lines_info = self._collect_vertical_line_info(result, tagger, kanji_regex, furigana_size)
        groups = self._group_vertical_lines(lines_info)
        offsets = self._compute_offsets_from_groups(groups)
        return offsets
