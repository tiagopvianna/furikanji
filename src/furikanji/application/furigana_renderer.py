from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterable

from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from src.furikanji.application.interfaces import FuriganaReadingGenerator, FuriganaSegment
from src.furikanji.application.page_text_extractor import (
    ExtractedTextRegionDict,
    PageTextExtractionResultDict,
)


@dataclass(frozen=True)
class RenderingContext:
    image: Image.Image
    draw: ImageDraw.ImageDraw


@dataclass(frozen=True)
class LineBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width(self):
        return self.x_max - self.x_min


@dataclass(frozen=True)
class DrawCommand:
    text: str
    x: float
    y: float
    font: ImageFont.ImageFont


@dataclass(frozen=True)
class VerticalLineLayoutNeed:
    index: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    width: float
    required_space: float


Bounds = tuple[float, float, float, float]
LineOutline = list[list[float]]
LineOutlineList = list[LineOutline]


@dataclass(frozen=True)
class EraseConfig:
    strategy: str = "planned_text"
    region_padding: int = 10
    planned_text_padding: int = 4


@dataclass(frozen=True)
class TypographyConfig:
    main_font_size_offset: int = 14
    furigana_font_size: int = 8


@dataclass(frozen=True)
class VerticalLayoutConfig:
    planning_furigana_size: int = 6
    main_text_x_offset: int = 10
    furigana_x_offset: int = 10
    char_spacing: int = 6
    furigana_spacing: int = 2
    required_space_padding: int = 2
    collision_x_threshold: int = 12
    collision_y_overlap_min: int = 1


@dataclass(frozen=True)
class HorizontalLayoutConfig:
    furigana_gap: int = 2


@dataclass(frozen=True)
class FuriganaRenderConfig:
    erase: EraseConfig = field(default_factory=EraseConfig)
    typography: TypographyConfig = field(default_factory=TypographyConfig)
    vertical: VerticalLayoutConfig = field(default_factory=VerticalLayoutConfig)
    horizontal: HorizontalLayoutConfig = field(default_factory=HorizontalLayoutConfig)


class FuriganaRenderer:
    def __init__(
        self,
        furigana_reading_generator: FuriganaReadingGenerator,
        config: FuriganaRenderConfig | None = None,
    ) -> None:
        self.furigana_reading_generator = furigana_reading_generator
        self.config = config or FuriganaRenderConfig()

    def _load_japanese_font(self, size: int) -> ImageFont.ImageFont:
        font_path = Path(__file__).parent.parent / "fonts" / "NotoSansCJKjp-Regular.otf"
        try:
            return ImageFont.truetype(str(font_path), size)
        except (IOError, OSError):
            return ImageFont.load_default()

    def __call__(
        self,
        image_path: str,
        result: PageTextExtractionResultDict,
        overlay_output_path: str,
    ) -> None:
        context = self._initialize_rendering_context(image_path=image_path)
        offsets = self._plan_vertical_layout_shifts(
            result=result,
            furigana_size=self.config.vertical.planning_furigana_size,
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

    def _initialize_rendering_context(self, image_path: str) -> RenderingContext:
        image = Image.open(image_path).convert("RGB")
        return RenderingContext(
            image=image,
            draw=ImageDraw.Draw(image),
        )

    def _render_text_region_overlay(
        self,
        context: RenderingContext,
        text_region: ExtractedTextRegionDict,
        offsets: dict[int, float],
        vertical_line_index: int,
    ) -> int:
        vertical = bool(text_region.get("is_vertical", False))
        line_texts = text_region.get("line_texts", [])
        line_outline_points = text_region.get("line_outline_points", [])
        font, furigana_font = self._build_region_fonts(
            text_region.get("estimated_font_size", 24)
        )
        draw_commands, planned_bounds, vertical_line_index = self._plan_region_text_layout(
            draw=context.draw,
            vertical=vertical,
            line_texts=line_texts,
            line_outline_points=line_outline_points,
            font=font,
            furigana_font=furigana_font,
            offsets=offsets,
            vertical_line_index=vertical_line_index,
        )
        self._erase_background_for_region(
            draw=context.draw,
            line_outline_points=line_outline_points,
            planned_bounds=planned_bounds,
        )
        self._paint_planned_region_text(draw=context.draw, draw_commands=draw_commands)
        return vertical_line_index

    def _build_region_fonts(
        self, estimated_font_size: int
    ) -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
        main_font_size = max(1, estimated_font_size - self.config.typography.main_font_size_offset)
        main_font = self._load_japanese_font(main_font_size)
        furigana_font = self._load_japanese_font(self.config.typography.furigana_font_size)
        return main_font, furigana_font

    def _erase_background_for_region(
        self,
        draw: ImageDraw.ImageDraw,
        line_outline_points: LineOutlineList,
        planned_bounds: Bounds | None,
    ) -> None:
        if self.config.erase.strategy == "planned_text":
            self._erase_planned_text_background(draw=draw, planned_bounds=planned_bounds)
            return
        if self.config.erase.strategy == "detected_region":
            self._erase_detected_region_background(
                draw=draw, line_outline_points=line_outline_points
            )
            return
        logger.warning(
            f"Unknown erase strategy '{self.config.erase.strategy}', using planned_text"
        )
        self._erase_planned_text_background(draw=draw, planned_bounds=planned_bounds)

    def _erase_detected_region_background(
        self,
        draw: ImageDraw.ImageDraw,
        line_outline_points: LineOutlineList,
    ) -> None:
        all_points = [pt for line in line_outline_points for pt in line]
        if not all_points:
            return
        all_xs = [pt[0] for pt in all_points]
        all_ys = [pt[1] for pt in all_points]
        draw.rectangle(
            [
                (
                    min(all_xs) - self.config.erase.region_padding,
                    min(all_ys) - self.config.erase.region_padding,
                ),
                (
                    max(all_xs) + self.config.erase.region_padding,
                    max(all_ys) + self.config.erase.region_padding,
                ),
            ],
            fill=(255, 255, 255),
        )

    def _erase_planned_text_background(
        self,
        draw: ImageDraw.ImageDraw,
        planned_bounds: Bounds | None,
    ) -> None:
        if planned_bounds is None:
            return
        x0, y0, x1, y1 = planned_bounds
        draw.rectangle(
            [
                (
                    x0 - self.config.erase.planned_text_padding,
                    y0 - self.config.erase.planned_text_padding,
                ),
                (
                    x1 + self.config.erase.planned_text_padding,
                    y1 + self.config.erase.planned_text_padding,
                ),
            ],
            fill=(255, 255, 255),
        )

    def _plan_region_text_layout(
        self,
        draw: ImageDraw.ImageDraw,
        vertical: bool,
        line_texts: list[str],
        line_outline_points: LineOutlineList,
        font: ImageFont.ImageFont,
        furigana_font: ImageFont.ImageFont,
        offsets: dict[int, float],
        vertical_line_index: int,
    ) -> tuple[list[DrawCommand], Bounds | None, int]:
        draw_commands = []
        planned_bounds: Bounds | None = None
        for line_text, line_coords in self._safe_zip_region_lines(
            line_texts=line_texts, line_outline_points=line_outline_points
        ):
            logger.debug(f"line_text: {line_text}")
            logger.debug(f"line_coords: {line_coords}")
            bounds = self._measure_line_bounds(line_coords)
            if bounds is None:
                continue
            segments = self.furigana_reading_generator.resolve_line_segments(
                line_text=line_text
            )
            if vertical:
                line_shift = offsets.get(vertical_line_index, 0)
                line_commands, line_bounds = self._plan_vertical_line_layout(
                    draw=draw,
                    bounds=bounds,
                    segments=segments,
                    font=font,
                    furigana_font=furigana_font,
                    line_shift=line_shift,
                )
                draw_commands.extend(line_commands)
                planned_bounds = self._merge_bounds(planned_bounds, line_bounds)
                vertical_line_index += 1
            else:
                line_commands, line_bounds = self._plan_horizontal_line_layout(
                    draw=draw,
                    bounds=bounds,
                    segments=segments,
                    font=font,
                    furigana_font=furigana_font,
                )
                draw_commands.extend(line_commands)
                planned_bounds = self._merge_bounds(planned_bounds, line_bounds)
        return draw_commands, planned_bounds, vertical_line_index

    def _paint_planned_region_text(
        self, draw: ImageDraw.ImageDraw, draw_commands: list[DrawCommand]
    ) -> None:
        for command in draw_commands:
            draw.text((command.x, command.y), command.text, fill=(0, 0, 0), font=command.font)

    def _safe_zip_region_lines(
        self, line_texts: list[str], line_outline_points: LineOutlineList
    ) -> Iterable[tuple[str, LineOutline]]:
        if len(line_texts) != len(line_outline_points):
            logger.warning(
                "Region has mismatched line payload sizes: "
                f"{len(line_texts)} texts vs {len(line_outline_points)} outlines"
            )
        return zip(line_texts, line_outline_points)

    def _measure_line_bounds(self, line_coords: LineOutline) -> LineBounds | None:
        if not line_coords:
            logger.warning("Skipping line with empty outline points")
            return None
        xs = [pt[0] for pt in line_coords]
        ys = [pt[1] for pt in line_coords]
        return LineBounds(x_min=min(xs), x_max=max(xs), y_min=min(ys), y_max=max(ys))

    def _plan_vertical_line_layout(
        self,
        draw: ImageDraw.ImageDraw,
        bounds: LineBounds,
        segments: list[FuriganaSegment],
        font: ImageFont.ImageFont,
        furigana_font: ImageFont.ImageFont,
        line_shift: float,
    ) -> tuple[list[DrawCommand], Bounds | None]:
        draw_commands = []
        planned_bounds: Bounds | None = None
        x_min_shifted = bounds.x_min + line_shift
        x_max_shifted = bounds.x_max + line_shift
        y_cursor = bounds.y_min
        for segment in segments:
            y_segment_start = y_cursor
            for char in segment.base_text:
                bbox = draw.textbbox((0, 0), char, font=font)
                w_char = bbox[2] - bbox[0]
                h_char = bbox[3] - bbox[1]
                x_pos = x_min_shifted + (bounds.width - w_char) / 2
                command, command_bounds = self._build_draw_command(
                    draw=draw,
                    text=char,
                    x=x_pos - self.config.vertical.main_text_x_offset,
                    y=y_cursor,
                    font=font,
                )
                draw_commands.append(command)
                planned_bounds = self._merge_bounds(planned_bounds, command_bounds)
                y_cursor += h_char + self.config.vertical.char_spacing

            if segment.needs_furigana:
                y_furi = y_segment_start
                for kana_char in segment.reading:
                    bbox_f = draw.textbbox((0, 0), kana_char, font=furigana_font)
                    h_f = bbox_f[3] - bbox_f[1]
                    command, command_bounds = self._build_draw_command(
                        draw=draw,
                        text=kana_char,
                        x=x_max_shifted - self.config.vertical.furigana_x_offset,
                        y=y_furi,
                        font=furigana_font,
                    )
                    draw_commands.append(command)
                    planned_bounds = self._merge_bounds(planned_bounds, command_bounds)
                    y_furi += h_f + self.config.vertical.furigana_spacing
        return draw_commands, planned_bounds

    def _plan_horizontal_line_layout(
        self,
        draw: ImageDraw.ImageDraw,
        bounds: LineBounds,
        segments: list[FuriganaSegment],
        font: ImageFont.ImageFont,
        furigana_font: ImageFont.ImageFont,
    ) -> tuple[list[DrawCommand], Bounds | None]:
        draw_commands = []
        planned_bounds: Bounds | None = None
        x_cursor = bounds.x_min
        for segment in segments:
            bbox_word = draw.textbbox((0, 0), segment.base_text, font=font)
            w_word = bbox_word[2] - bbox_word[0]
            if segment.needs_furigana:
                bbox_ruby = draw.textbbox((0, 0), segment.reading, font=furigana_font)
                w_ruby = bbox_ruby[2] - bbox_ruby[0]
                h_ruby = bbox_ruby[3] - bbox_ruby[1]
                ruby_x = x_cursor + (w_word - w_ruby) / 2
                ruby_y = bounds.y_min - h_ruby - self.config.horizontal.furigana_gap
                command, command_bounds = self._build_draw_command(
                    draw=draw,
                    text=segment.reading,
                    x=ruby_x,
                    y=ruby_y,
                    font=furigana_font,
                )
                draw_commands.append(command)
                planned_bounds = self._merge_bounds(planned_bounds, command_bounds)
            command, command_bounds = self._build_draw_command(
                draw=draw,
                text=segment.base_text,
                x=x_cursor,
                y=bounds.y_min,
                font=font,
            )
            draw_commands.append(command)
            planned_bounds = self._merge_bounds(planned_bounds, command_bounds)
            x_cursor += w_word
        return draw_commands, planned_bounds

    def _build_draw_command(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: float,
        y: float,
        font: ImageFont.ImageFont,
    ) -> tuple[DrawCommand, Bounds]:
        command = DrawCommand(text=text, x=x, y=y, font=font)
        return command, draw.textbbox((x, y), text, font=font)

    def _merge_bounds(
        self, bounds_a: Bounds | None, bounds_b: Bounds | None
    ) -> Bounds | None:
        if bounds_a is None:
            return bounds_b
        if bounds_b is None:
            return bounds_a
        return (
            min(bounds_a[0], bounds_b[0]),
            min(bounds_a[1], bounds_b[1]),
            max(bounds_a[2], bounds_b[2]),
            max(bounds_a[3], bounds_b[3]),
        )

    def _describe_vertical_line_layout_needs(
        self, result: PageTextExtractionResultDict, furigana_size: int
    ) -> list[VerticalLineLayoutNeed]:
        lines_info: list[VerticalLineLayoutNeed] = []
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
                segments = self.furigana_reading_generator.resolve_line_segments(line_text)
                has_furigana_segments = any(
                    segment.needs_furigana for segment in segments
                )
                required_space = (
                    furigana_size + self.config.vertical.required_space_padding
                    if has_furigana_segments
                    else 0
                )
                lines_info.append(
                    VerticalLineLayoutNeed(
                        index=counter,
                        x_min=x_min,
                        x_max=x_max,
                        y_min=y_min,
                        y_max=y_max,
                        width=width,
                        required_space=required_space,
                    )
                )
                counter += 1
        return lines_info

    def _cluster_colliding_vertical_columns(
        self,
        lines_info: list[VerticalLineLayoutNeed],
        x_thresh: int | None = None,
        y_overlap_min: int | None = None,
    ) -> list[list[VerticalLineLayoutNeed]]:
        x_thresh = (
            self.config.vertical.collision_x_threshold if x_thresh is None else x_thresh
        )
        y_overlap_min = (
            self.config.vertical.collision_y_overlap_min
            if y_overlap_min is None
            else y_overlap_min
        )
        groups: list[list[VerticalLineLayoutNeed]] = []

        def overlaps(a: VerticalLineLayoutNeed, b: VerticalLineLayoutNeed) -> bool:
            y_overlap = min(a.y_max, b.y_max) - max(a.y_min, b.y_min)
            if y_overlap < y_overlap_min:
                return False
            if abs(a.x_min - b.x_max) < x_thresh or abs(a.x_max - b.x_min) < x_thresh:
                return True
            return False

        for line in sorted(lines_info, key=lambda l: l.x_min):
            placed = False
            for group in groups:
                if any(overlaps(line, member) for member in group):
                    group.append(line)
                    placed = True
                    break
            if not placed:
                groups.append([line])
        return groups

    def _resolve_column_shift_by_cluster(
        self, groups: list[list[VerticalLineLayoutNeed]]
    ) -> dict[int, float]:
        offsets: dict[int, float] = {}
        for group in groups:
            group_sorted = sorted(group, key=lambda l: l.x_min)
            # First line in group: no shift
            first = group_sorted[0]
            offsets[first.index] = 0
            for prev_line, line in zip(group_sorted, group_sorted[1:]):
                prev_index = prev_line.index
                prev_x_min = prev_line.x_min
                prev_width = prev_line.width
                prev_space = prev_line.required_space
                prev_dx = offsets[prev_index]
                prev_shifted_right = prev_x_min + prev_dx + prev_width + prev_space
                dx = prev_shifted_right - line.x_min
                offsets[line.index] = dx
        return offsets

    def _plan_vertical_layout_shifts(
        self, result: PageTextExtractionResultDict, furigana_size: int
    ) -> dict[int, float]:
        lines_info = self._describe_vertical_line_layout_needs(result, furigana_size)
        groups = self._cluster_colliding_vertical_columns(lines_info)
        return self._resolve_column_shift_by_cluster(groups)
