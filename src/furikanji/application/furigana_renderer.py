from pathlib import Path
import re

import jaconv
from fugashi import Tagger
from loguru import logger
from PIL import Image, ImageDraw, ImageFont


class FuriganaRenderer:
    def _load_japanese_font(self, size):
        font_path = Path(__file__).parent.parent / "fonts" / "NotoSansCJKjp-Regular.otf"
        try:
            return ImageFont.truetype(str(font_path), size)
        except (IOError, OSError):
            return ImageFont.load_default()

    def render_overlay(self, image_path, result, overlay_output_path):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        tagger = Tagger()
        kanji_regex = re.compile(r"[\u4E00-\u9FFF]")
        # Reserve a small font size for furigana and compute offsets for all vertical lines
        furigana_size = 6
        offsets = self._compute_vertical_offsets(result, tagger, kanji_regex, furigana_size)
        vertical_counter = 0
        logger.debug(result)

        for text_region in result.get("text_regions", []):
            vertical = bool(text_region.get("is_vertical", False))
            line_texts = text_region.get("line_texts", [])
            line_outline_points = text_region.get("line_outline_points", [])
            detected_size = text_region.get("estimated_font_size", 24)
            font = self._load_japanese_font(detected_size - 14)
            furigana_size = 8
            furigana_font = self._load_japanese_font(furigana_size)

            # Compute bounding box for the entire block to erase background once
            all_xs = [pt[0] for line in line_outline_points for pt in line]
            all_ys = [pt[1] for line in line_outline_points for pt in line]
            bx_min, bx_max = min(all_xs), max(all_xs)
            by_min, by_max = min(all_ys), max(all_ys)
            # Erase underneath
            draw.rectangle(
                [(bx_min - 10, by_min - 10), (bx_max + 10, by_max + 10)],
                fill=(255, 255, 255),
            )

            for line_text, line_coords in zip(line_texts, line_outline_points):
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
                                draw.text(
                                    (x_f - 10, y_furi),
                                    kana_char,
                                    fill=(0, 0, 0),
                                    font=furigana_font,
                                )
                                y_furi += h_f + 2
                else:
                    # Horizontal: draw words with furigana
                    x_cursor = x_min
                    for word in tagger(line_text):
                        surface = word.surface
                        kana = getattr(word.feature, "kana", None)
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

    def _collect_vertical_line_info(self, result, tagger, kanji_regex, furigana_size):
        lines_info = []
        counter = 0
        for text_region in result.get("text_regions", []):
            if not text_region.get("is_vertical", False):
                continue
            for line_text, line_coords in zip(
                text_region.get("line_texts", []),
                text_region.get("line_outline_points", []),
            ):
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
                lines_info.append(
                    {
                        "index": counter,
                        "x_min": x_min,
                        "x_max": x_max,
                        "y_min": y_min,
                        "y_max": y_max,
                        "width": width,
                        "required_space": required_space,
                    }
                )
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

        for line in sorted(lines_info, key=lambda l: l["x_min"]):
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
            group_sorted = sorted(group, key=lambda l: l["x_min"])
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
