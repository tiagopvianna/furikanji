from pathlib import Path
import re
from dataclasses import dataclass

import jaconv
from fugashi import Tagger
from loguru import logger
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class RenderingContext:
    image: Image.Image
    draw: ImageDraw.ImageDraw
    tagger: Tagger
    kanji_regex: re.Pattern


@dataclass(frozen=True)
class TokenReading:
    surface: str
    reading: str
    needs_furigana: bool


@dataclass(frozen=True)
class LineBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width(self):
        return self.x_max - self.x_min


class FuriganaRenderer:
    REGION_ERASE_PADDING = 10
    VERTICAL_PLANNING_FURIGANA_SIZE = 6
    REGION_FURIGANA_FONT_SIZE = 8
    MAIN_FONT_SIZE_OFFSET = 14
    VERTICAL_MAIN_TEXT_X_OFFSET = 10
    VERTICAL_FURIGANA_X_OFFSET = 10
    VERTICAL_CHAR_SPACING = 6
    VERTICAL_FURIGANA_SPACING = 2
    HORIZONTAL_FURIGANA_GAP = 2

    def _load_japanese_font(self, size):
        font_path = Path(__file__).parent.parent / "fonts" / "NotoSansCJKjp-Regular.otf"
        try:
            return ImageFont.truetype(str(font_path), size)
        except (IOError, OSError):
            return ImageFont.load_default()

    def render_overlay(self, image_path, result, overlay_output_path):
        context = self._initialize_rendering_context(image_path=image_path)
        offsets = self._plan_vertical_layout_shifts(
            result=result,
            tagger=context.tagger,
            kanji_regex=context.kanji_regex,
            furigana_size=self.VERTICAL_PLANNING_FURIGANA_SIZE,
        )
        vertical_line_index = 0
        logger.debug(result)

        for text_region in result.get("text_regions", []):
            vertical_line_index = self._render_text_region_overlay(
                context=context,
                text_region=text_region,
                offsets=offsets,
                vertical_line_index=vertical_line_index,
            )
        context.image.save(overlay_output_path)

    def _initialize_rendering_context(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return RenderingContext(
            image=image,
            draw=ImageDraw.Draw(image),
            tagger=Tagger(),
            kanji_regex=re.compile(r"[\u4E00-\u9FFF]"),
        )

    def _render_text_region_overlay(self, context, text_region, offsets, vertical_line_index):
        vertical = bool(text_region.get("is_vertical", False))
        line_texts = text_region.get("line_texts", [])
        line_outline_points = text_region.get("line_outline_points", [])
        font, furigana_font = self._build_region_fonts(
            text_region.get("estimated_font_size", 24)
        )
        self._erase_region_background(
            draw=context.draw, line_outline_points=line_outline_points
        )
        return self._render_region_lines(
            draw=context.draw,
            vertical=vertical,
            line_texts=line_texts,
            line_outline_points=line_outline_points,
            font=font,
            furigana_font=furigana_font,
            tagger=context.tagger,
            kanji_regex=context.kanji_regex,
            offsets=offsets,
            vertical_line_index=vertical_line_index,
        )

    def _build_region_fonts(self, estimated_font_size):
        main_font_size = max(1, estimated_font_size - self.MAIN_FONT_SIZE_OFFSET)
        main_font = self._load_japanese_font(main_font_size)
        furigana_font = self._load_japanese_font(self.REGION_FURIGANA_FONT_SIZE)
        return main_font, furigana_font

    def _erase_region_background(self, draw, line_outline_points):
        all_points = [pt for line in line_outline_points for pt in line]
        if not all_points:
            return
        all_xs = [pt[0] for pt in all_points]
        all_ys = [pt[1] for pt in all_points]
        draw.rectangle(
            [
                (
                    min(all_xs) - self.REGION_ERASE_PADDING,
                    min(all_ys) - self.REGION_ERASE_PADDING,
                ),
                (
                    max(all_xs) + self.REGION_ERASE_PADDING,
                    max(all_ys) + self.REGION_ERASE_PADDING,
                ),
            ],
            fill=(255, 255, 255),
        )

    def _render_region_lines(
        self,
        draw,
        vertical,
        line_texts,
        line_outline_points,
        font,
        furigana_font,
        tagger,
        kanji_regex,
        offsets,
        vertical_line_index,
    ):
        for line_text, line_coords in self._safe_zip_region_lines(
            line_texts=line_texts, line_outline_points=line_outline_points
        ):
            logger.debug(f"line_text: {line_text}")
            logger.debug(f"line_coords: {line_coords}")
            bounds = self._measure_line_bounds(line_coords)
            if bounds is None:
                continue
            tokens = self._tokenize_with_readings(
                line_text=line_text, tagger=tagger, kanji_regex=kanji_regex
            )
            if vertical:
                line_shift = offsets.get(vertical_line_index, 0)
                self._render_vertical_line_with_furigana(
                    draw=draw,
                    bounds=bounds,
                    tokens=tokens,
                    font=font,
                    furigana_font=furigana_font,
                    line_shift=line_shift,
                )
                vertical_line_index += 1
            else:
                self._render_horizontal_line_with_furigana(
                    draw=draw,
                    bounds=bounds,
                    tokens=tokens,
                    font=font,
                    furigana_font=furigana_font,
                )
        return vertical_line_index

    def _safe_zip_region_lines(self, line_texts, line_outline_points):
        if len(line_texts) != len(line_outline_points):
            logger.warning(
                "Region has mismatched line payload sizes: "
                f"{len(line_texts)} texts vs {len(line_outline_points)} outlines"
            )
        return zip(line_texts, line_outline_points)

    def _measure_line_bounds(self, line_coords):
        if not line_coords:
            logger.warning("Skipping line with empty outline points")
            return None
        xs = [pt[0] for pt in line_coords]
        ys = [pt[1] for pt in line_coords]
        return LineBounds(x_min=min(xs), x_max=max(xs), y_min=min(ys), y_max=max(ys))

    def _tokenize_with_readings(self, line_text, tagger, kanji_regex):
        tokens = []
        for word in tagger(line_text):
            surface = word.surface
            kana = getattr(word.feature, "kana", None)
            reading = jaconv.kata2hira(kana) if kana else surface
            tokens.append(
                TokenReading(
                    surface=surface,
                    reading=reading,
                    needs_furigana=self._classify_token_requires_furigana(
                        surface=surface, reading=reading, kanji_regex=kanji_regex
                    ),
                )
            )
        return tokens

    def _classify_token_requires_furigana(self, surface, reading, kanji_regex):
        return bool(kanji_regex.search(surface) and surface != reading)

    def _render_vertical_line_with_furigana(
        self, draw, bounds, tokens, font, furigana_font, line_shift
    ):
        x_min_shifted = bounds.x_min + line_shift
        x_max_shifted = bounds.x_max + line_shift
        y_cursor = bounds.y_min
        for token in tokens:
            y_token_start = y_cursor
            for char in token.surface:
                bbox = draw.textbbox((0, 0), char, font=font)
                w_char = bbox[2] - bbox[0]
                h_char = bbox[3] - bbox[1]
                x_pos = x_min_shifted + (bounds.width - w_char) / 2
                draw.text(
                    (x_pos - self.VERTICAL_MAIN_TEXT_X_OFFSET, y_cursor),
                    char,
                    fill=(0, 0, 0),
                    font=font,
                )
                y_cursor += h_char + self.VERTICAL_CHAR_SPACING

            if token.needs_furigana:
                y_furi = y_token_start
                for kana_char in token.reading:
                    bbox_f = draw.textbbox((0, 0), kana_char, font=furigana_font)
                    h_f = bbox_f[3] - bbox_f[1]
                    draw.text(
                        (x_max_shifted - self.VERTICAL_FURIGANA_X_OFFSET, y_furi),
                        kana_char,
                        fill=(0, 0, 0),
                        font=furigana_font,
                    )
                    y_furi += h_f + self.VERTICAL_FURIGANA_SPACING

    def _render_horizontal_line_with_furigana(
        self, draw, bounds, tokens, font, furigana_font
    ):
        x_cursor = bounds.x_min
        for token in tokens:
            bbox_word = draw.textbbox((0, 0), token.surface, font=font)
            w_word = bbox_word[2] - bbox_word[0]
            if token.needs_furigana:
                bbox_ruby = draw.textbbox((0, 0), token.reading, font=furigana_font)
                w_ruby = bbox_ruby[2] - bbox_ruby[0]
                h_ruby = bbox_ruby[3] - bbox_ruby[1]
                ruby_x = x_cursor + (w_word - w_ruby) / 2
                ruby_y = bounds.y_min - h_ruby - self.HORIZONTAL_FURIGANA_GAP
                draw.text((ruby_x, ruby_y), token.reading, fill=(0, 0, 0), font=furigana_font)
            draw.text((x_cursor, bounds.y_min), token.surface, fill=(0, 0, 0), font=font)
            x_cursor += w_word

    def _describe_vertical_line_layout_needs(self, result, tagger, kanji_regex, furigana_size):
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

    def _cluster_colliding_vertical_columns(self, lines_info, x_thresh=12, y_overlap_min=1):
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

    def _resolve_column_shift_by_cluster(self, groups):
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

    def _plan_vertical_layout_shifts(self, result, tagger, kanji_regex, furigana_size):
        lines_info = self._describe_vertical_line_layout_needs(
            result, tagger, kanji_regex, furigana_size
        )
        groups = self._cluster_colliding_vertical_columns(lines_info)
        return self._resolve_column_shift_by_cluster(groups)
