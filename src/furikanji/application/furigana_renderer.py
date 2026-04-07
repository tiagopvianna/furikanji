from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Iterable

from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from src.furikanji.application.interfaces import (
    FuriganaReadingGenerator,
    FuriganaSegment,
)
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
class RegionRenderPlan:
    draw_commands: list[DrawCommand]
    planned_bounds: "Bounds | None"
    line_outline_points: "LineOutlineList"


@dataclass(frozen=True)
class PageRenderPlan:
    region_plans: list[RegionRenderPlan]


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
    strategy: str = "both"
    region_padding: int = 10
    planned_text_padding: int = 4


@dataclass(frozen=True)
class VerticalLayoutConfig:
    main_text_x_offset: int = 10
    furigana_x_offset: int = 10
    furigana_y_offset: int = 3
    required_space_padding: int = 2
    collision_x_threshold: int = 12
    collision_y_overlap_min: int = 1


@dataclass(frozen=True)
class HorizontalLayoutConfig:
    furigana_gap: int = 2


@dataclass(frozen=True)
class SizingConfig:
    probe_glyph: str = "田"
    min_main_size: int = 8
    max_main_size: int = 120
    main_scale_bias: float = 1.0
    furigana_ratio: float = 0.5
    min_furigana_size: int = 6
    max_furigana_size: int = 72
    char_spacing_ratio: float = 0.25
    furigana_spacing_ratio: float = 0.2
    enable_one_step_correction: bool = True
    use_ocr_estimate_fallback: bool = True
    min_target_dimension: float = 1.0
    default_main_size: int = 24
    ocr_fallback_main_scale: float = 0.45
    min_char_spacing: int = 1
    min_furigana_spacing: int = 1
    target_inner_ratio_vertical: float = 1.0
    target_inner_ratio_horizontal: float = 1.0


@dataclass(frozen=True)
class ResolvedRegionSizing:
    main_size: int
    furigana_size: int
    char_spacing: int
    furigana_spacing: int


@dataclass(frozen=True)
class PlacementConfig:
    policy: str = "overflow_aware"
    overflow_aware_anchor: str = "center"
    min_margin: int = 0


@dataclass(frozen=True)
class FuriganaRenderConfig:
    erase: EraseConfig = field(default_factory=EraseConfig)
    vertical: VerticalLayoutConfig = field(default_factory=VerticalLayoutConfig)
    horizontal: HorizontalLayoutConfig = field(default_factory=HorizontalLayoutConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    placement: PlacementConfig = field(default_factory=PlacementConfig)
    draw_target_boxes: bool = False
    target_box_color: tuple[int, int, int] = (0, 255, 0)
    target_box_width: int = 2
    target_box_source: str = "selected"
    draw_overlay_text: bool = True


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
        if self.config.draw_target_boxes:
            self._draw_line_target_boxes(draw=context.draw, result=result)
        image_width, image_height = context.image.size
        logger.info(
            "Starting furigana render pass: image={}x{}, text_regions={}",
            image_width,
            image_height,
            len(result.get("text_regions", [])),
        )
        page_render_plan = self.build_page_render_plan(
            result=result,
            measure_draw=context.draw,
            image_size=(image_width, image_height),
        )
        total_commands = sum(
            len(region_plan.draw_commands)
            for region_plan in page_render_plan.region_plans
        )
        if self.config.draw_overlay_text:
            self.paint_page_render_plan(
                draw=context.draw, page_render_plan=page_render_plan
            )
        context.image.save(overlay_output_path)
        logger.info(
            "Finished furigana render pass: regions={}, draw_commands={}, output={}",
            len(page_render_plan.region_plans),
            total_commands,
            overlay_output_path,
        )

    def _draw_line_target_boxes(
        self, draw: ImageDraw.ImageDraw, result: PageTextExtractionResultDict
    ) -> None:
        for text_region in result.get("text_regions", []):
            vertical = bool(text_region.get("is_vertical", False))
            line_target_widths = text_region.get("line_target_widths", [])
            line_target_heights = text_region.get("line_target_heights", [])
            for index, line_coords in enumerate(
                text_region.get("line_outline_points", [])
            ):
                bounds = self._compute_outline_bounds(line_coords)
                if bounds is None:
                    continue
                if self.config.target_box_source == "bbox":
                    box_bounds = bounds
                else:
                    bbox_width = bounds[2] - bounds[0]
                    bbox_height = bounds[3] - bounds[1]
                    mask_target_dimension = (
                        (
                            line_target_widths[index]
                            if vertical
                            else line_target_heights[index]
                        )
                        if index
                        < len(line_target_widths if vertical else line_target_heights)
                        else None
                    )
                    selected_target_dimension = (
                        float(mask_target_dimension)
                        if mask_target_dimension is not None
                        and mask_target_dimension > 0
                        else (bbox_width if vertical else bbox_height)
                    )
                    x_center = (bounds[0] + bounds[2]) / 2
                    y_center = (bounds[1] + bounds[3]) / 2
                    if vertical:
                        half_w = selected_target_dimension / 2
                        box_bounds = (
                            x_center - half_w,
                            bounds[1],
                            x_center + half_w,
                            bounds[3],
                        )
                    else:
                        half_h = selected_target_dimension / 2
                        box_bounds = (
                            bounds[0],
                            y_center - half_h,
                            bounds[2],
                            y_center + half_h,
                        )
                draw.rectangle(
                    [box_bounds[0:2], box_bounds[2:4]],
                    outline=self.config.target_box_color,
                    width=self.config.target_box_width,
                )

    def _initialize_rendering_context(self, image_path: str) -> RenderingContext:
        image = Image.open(image_path).convert("RGB")
        return RenderingContext(
            image=image,
            draw=ImageDraw.Draw(image),
        )

    def build_page_render_plan(
        self,
        result: PageTextExtractionResultDict,
        measure_draw: ImageDraw.ImageDraw,
        image_size: tuple[int, int],
    ) -> PageRenderPlan:
        resolved_region_sizings = self._resolve_region_sizings(
            result=result, measure_draw=measure_draw
        )
        offsets = self._plan_vertical_column_shifts_for_furigana(
            result=result,
            region_sizings=resolved_region_sizings,
        )
        logger.trace("Vertical column shifts for furigana: {}", offsets)
        vertical_line_index = 0
        region_plans: list[RegionRenderPlan] = []
        for region_index, text_region in enumerate(result.get("text_regions", [])):
            region_render_plan, vertical_line_index = self._build_region_render_plan(
                measure_draw=measure_draw,
                text_region=text_region,
                offsets=offsets,
                vertical_line_index=vertical_line_index,
                region_index=region_index,
                image_size=image_size,
                region_sizing=resolved_region_sizings[region_index],
            )
            region_plans.append(region_render_plan)
        return PageRenderPlan(region_plans=region_plans)

    def paint_page_render_plan(
        self,
        draw: ImageDraw.ImageDraw,
        page_render_plan: PageRenderPlan,
    ) -> None:
        for region_index, region_plan in enumerate(page_render_plan.region_plans):
            logger.trace(
                "Painting region {}: draw_commands={}, planned_bounds={}",
                region_index,
                len(region_plan.draw_commands),
                region_plan.planned_bounds,
            )
            self._erase_background_for_region(
                draw=draw,
                line_outline_points=region_plan.line_outline_points,
                planned_bounds=region_plan.planned_bounds,
            )
            self._paint_planned_region_text(
                draw=draw, draw_commands=region_plan.draw_commands
            )

    def _build_region_render_plan(
        self,
        measure_draw: ImageDraw.ImageDraw,
        text_region: ExtractedTextRegionDict,
        offsets: dict[int, float],
        vertical_line_index: int,
        region_index: int,
        image_size: tuple[int, int],
        region_sizing: ResolvedRegionSizing,
    ) -> tuple[RegionRenderPlan, int]:
        vertical = bool(text_region.get("is_vertical", False))
        line_texts = text_region.get("line_texts", [])
        line_outline_points = text_region.get("line_outline_points", [])
        region_detected_bounds = self._compute_outline_list_bounds(line_outline_points)
        if region_detected_bounds is not None:
            region_ratio_bounds = self._to_relative_bounds(
                region_detected_bounds, image_size
            )
            logger.trace(
                "Region {} input geometry: vertical={}, lines={}, detected_bounds_px={}, detected_bounds_ratio={}",
                region_index,
                vertical,
                len(line_texts),
                region_detected_bounds,
                region_ratio_bounds,
            )
        else:
            logger.trace(
                "Region {} input geometry: vertical={}, lines={}, detected_bounds_px=None",
                region_index,
                vertical,
                len(line_texts),
            )
        if region_detected_bounds is not None:
            region_origin = (region_detected_bounds[0], region_detected_bounds[1])
            intrinsic_line_outline_points = self._translate_line_outline_points(
                line_outline_points=line_outline_points,
                dx=-region_origin[0],
                dy=-region_origin[1],
            )
        else:
            region_origin = (0.0, 0.0)
            intrinsic_line_outline_points = line_outline_points
        font, furigana_font = self._build_region_fonts(region_sizing=region_sizing)
        logger.trace(
            "Region {} font plan: estimated_font_size={}, main_font_size={}, furigana_font_size={}, char_spacing={}, furigana_spacing={}",
            region_index,
            text_region.get("estimated_font_size", 24),
            self._debug_font_size(font),
            self._debug_font_size(furigana_font),
            region_sizing.char_spacing,
            region_sizing.furigana_spacing,
        )
        intrinsic_draw_commands, intrinsic_bounds, vertical_line_index = (
            self._plan_region_text_layout(
                draw=measure_draw,
                vertical=vertical,
                line_texts=line_texts,
                line_outline_points=intrinsic_line_outline_points,
                font=font,
                furigana_font=furigana_font,
                offsets=offsets,
                vertical_line_index=vertical_line_index,
                region_sizing=region_sizing,
            )
        )
        draw_commands, planned_bounds, placement_dx, placement_dy, overflow_flags = (
            self._place_region_intrinsic_layout(
                draw_commands=intrinsic_draw_commands,
                intrinsic_bounds=intrinsic_bounds,
                region_origin=region_origin,
                region_detected_bounds=region_detected_bounds,
                image_size=image_size,
            )
        )
        planned_ratio_bounds = (
            self._to_relative_bounds(planned_bounds, image_size)
            if planned_bounds is not None
            else None
        )
        logger.debug(
            "Region {} plan output: commands={}, planned_bounds_px={}, planned_bounds_ratio={}, placement_dx={}, placement_dy={}",
            region_index,
            len(draw_commands),
            planned_bounds,
            planned_ratio_bounds,
            placement_dx,
            placement_dy,
        )
        return (
            RegionRenderPlan(
                draw_commands=draw_commands,
                planned_bounds=planned_bounds,
                line_outline_points=line_outline_points,
            ),
            vertical_line_index,
        )

    def _place_region_intrinsic_layout(
        self,
        draw_commands: list[DrawCommand],
        intrinsic_bounds: Bounds | None,
        region_origin: tuple[float, float],
        region_detected_bounds: Bounds | None,
        image_size: tuple[int, int],
    ) -> tuple[list[DrawCommand], Bounds | None, float, float, dict[str, bool]]:
        if intrinsic_bounds is None:
            return (
                draw_commands,
                None,
                0.0,
                0.0,
                {
                    "overflow_left": False,
                    "overflow_top": False,
                    "overflow_right": False,
                    "overflow_bottom": False,
                },
            )
        target_bounds = (
            region_detected_bounds
            if region_detected_bounds is not None
            else self._translate_bounds(
                bounds=intrinsic_bounds,
                dx=region_origin[0],
                dy=region_origin[1],
            )
        )
        if target_bounds is None:
            return (
                draw_commands,
                intrinsic_bounds,
                0.0,
                0.0,
                {
                    "overflow_left": False,
                    "overflow_top": False,
                    "overflow_right": False,
                    "overflow_bottom": False,
                },
            )

        policy = self.config.placement.policy
        if policy == "top_left":
            base_dx, base_dy = self._compute_anchor_translation(
                intrinsic_bounds=intrinsic_bounds,
                target_bounds=target_bounds,
                anchor="top_left",
            )
        elif policy == "center":
            base_dx, base_dy = self._compute_anchor_translation(
                intrinsic_bounds=intrinsic_bounds,
                target_bounds=target_bounds,
                anchor="center",
            )
        else:
            base_dx, base_dy = self._compute_anchor_translation(
                intrinsic_bounds=intrinsic_bounds,
                target_bounds=target_bounds,
                anchor=self.config.placement.overflow_aware_anchor,
            )
        placed_bounds = self._translate_bounds(
            bounds=intrinsic_bounds, dx=base_dx, dy=base_dy
        )
        if placed_bounds is None:
            return (
                draw_commands,
                intrinsic_bounds,
                0.0,
                0.0,
                {
                    "overflow_left": False,
                    "overflow_top": False,
                    "overflow_right": False,
                    "overflow_bottom": False,
                },
            )

        if policy == "overflow_aware":
            shift_x, shift_y = self._compute_overflow_correction_shift(
                bounds=placed_bounds,
                image_size=image_size,
                margin=self.config.placement.min_margin,
            )
        else:
            shift_x, shift_y = (0.0, 0.0)

        dx = base_dx + shift_x
        dy = base_dy + shift_y
        final_bounds = self._translate_bounds(bounds=intrinsic_bounds, dx=dx, dy=dy)
        translated_commands = self._translate_draw_commands(
            draw_commands=draw_commands, dx=dx, dy=dy
        )
        overflow_flags = self._compute_overflow_flags(
            bounds=final_bounds,
            image_size=image_size,
            margin=self.config.placement.min_margin,
        )
        logger.trace(
            "Region placement: policy={}, anchor={}, base_shift=({}, {}), overflow_correction=({}, {}), final_bounds={}, overflow_flags={}",
            policy,
            self.config.placement.overflow_aware_anchor,
            base_dx,
            base_dy,
            shift_x,
            shift_y,
            final_bounds,
            overflow_flags,
        )
        return translated_commands, final_bounds, dx, dy, overflow_flags

    def _build_region_fonts(
        self, region_sizing: ResolvedRegionSizing
    ) -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
        main_font = self._load_japanese_font(region_sizing.main_size)
        furigana_font = self._load_japanese_font(region_sizing.furigana_size)
        return main_font, furigana_font

    def _erase_background_for_region(
        self,
        draw: ImageDraw.ImageDraw,
        line_outline_points: LineOutlineList,
        planned_bounds: Bounds | None,
    ) -> None:
        if self.config.erase.strategy == "both":
            self._erase_detected_region_background(
                draw=draw, line_outline_points=line_outline_points
            )
            self._erase_planned_text_background(
                draw=draw, planned_bounds=planned_bounds
            )
            return
        if self.config.erase.strategy == "planned_text":
            self._erase_planned_text_background(
                draw=draw, planned_bounds=planned_bounds
            )
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
        logger.trace(
            "Erase detected-region background: bounds_px=({}, {}, {}, {}), padding={}",
            min(all_xs),
            min(all_ys),
            max(all_xs),
            max(all_ys),
            self.config.erase.region_padding,
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
        logger.trace(
            "Erase planned-text background: bounds_px={}, padding={}",
            planned_bounds,
            self.config.erase.planned_text_padding,
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
        region_sizing: ResolvedRegionSizing,
    ) -> tuple[list[DrawCommand], Bounds | None, int]:
        draw_commands = []
        planned_bounds: Bounds | None = None
        for line_number, (line_text, line_coords) in enumerate(
            self._safe_zip_region_lines(
                line_texts=line_texts, line_outline_points=line_outline_points
            ),
            start=1,
        ):
            line_ocr_bounds = self._compute_outline_bounds(line_coords)
            logger.trace(
                "Planning line {}: vertical={}, text_len={}, ocr_bounds_px={}",
                line_number,
                vertical,
                len(line_text),
                line_ocr_bounds,
            )
            if line_ocr_bounds is not None:
                line_ocr_width = line_ocr_bounds[2] - line_ocr_bounds[0]
                logger.trace("Line {} OCR width={} px", line_number, line_ocr_width)
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
                    char_spacing=region_sizing.char_spacing,
                    furigana_spacing=region_sizing.furigana_spacing,
                )
                draw_commands.extend(line_commands)
                planned_bounds = self._merge_bounds(planned_bounds, line_bounds)
                logger.trace(
                    "Vertical line planned: shift_dx={}, commands={}, planned_line_bounds_px={}",
                    line_shift,
                    len(line_commands),
                    line_bounds,
                )
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
                logger.trace(
                    "Horizontal line planned: commands={}, planned_line_bounds_px={}",
                    len(line_commands),
                    line_bounds,
                )
        return draw_commands, planned_bounds, vertical_line_index

    def _paint_planned_region_text(
        self, draw: ImageDraw.ImageDraw, draw_commands: list[DrawCommand]
    ) -> None:
        for command in draw_commands:
            draw.text(
                (command.x, command.y), command.text, fill=(0, 0, 0), font=command.font
            )

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
        char_spacing: int,
        furigana_spacing: int,
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
                y_cursor += h_char + char_spacing

            if segment.needs_furigana:
                y_furi = y_segment_start + self.config.vertical.furigana_y_offset
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
                    y_furi += h_f + furigana_spacing
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

    def _debug_font_size(self, font: ImageFont.ImageFont) -> int | None:
        return getattr(font, "size", None)

    def _compute_outline_bounds(self, line_coords: LineOutline) -> Bounds | None:
        if not line_coords:
            return None
        xs = [pt[0] for pt in line_coords]
        ys = [pt[1] for pt in line_coords]
        return (min(xs), min(ys), max(xs), max(ys))

    def _compute_outline_list_bounds(
        self, line_outline_points: LineOutlineList
    ) -> Bounds | None:
        all_points = [pt for line in line_outline_points for pt in line]
        if not all_points:
            return None
        xs = [pt[0] for pt in all_points]
        ys = [pt[1] for pt in all_points]
        return (min(xs), min(ys), max(xs), max(ys))

    def _to_relative_bounds(
        self, bounds: Bounds, image_size: tuple[int, int]
    ) -> tuple[float, float, float, float]:
        width, height = image_size
        if width == 0 or height == 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            bounds[0] / width,
            bounds[1] / height,
            bounds[2] / width,
            bounds[3] / height,
        )

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

    def _translate_line_outline_points(
        self, line_outline_points: LineOutlineList, dx: float, dy: float
    ) -> LineOutlineList:
        translated_lines: LineOutlineList = []
        for line in line_outline_points:
            translated_lines.append([[pt[0] + dx, pt[1] + dy] for pt in line])
        return translated_lines

    def _translate_draw_commands(
        self, draw_commands: list[DrawCommand], dx: float, dy: float
    ) -> list[DrawCommand]:
        return [
            DrawCommand(
                text=command.text,
                x=command.x + dx,
                y=command.y + dy,
                font=command.font,
            )
            for command in draw_commands
        ]

    def _translate_bounds(
        self, bounds: Bounds | None, dx: float, dy: float
    ) -> Bounds | None:
        if bounds is None:
            return None
        return (bounds[0] + dx, bounds[1] + dy, bounds[2] + dx, bounds[3] + dy)

    def _compute_anchor_translation(
        self, intrinsic_bounds: Bounds, target_bounds: Bounds, anchor: str
    ) -> tuple[float, float]:
        if anchor == "top_left":
            return (
                target_bounds[0] - intrinsic_bounds[0],
                target_bounds[1] - intrinsic_bounds[1],
            )
        intrinsic_center_x = (intrinsic_bounds[0] + intrinsic_bounds[2]) / 2
        intrinsic_center_y = (intrinsic_bounds[1] + intrinsic_bounds[3]) / 2
        target_center_x = (target_bounds[0] + target_bounds[2]) / 2
        target_center_y = (target_bounds[1] + target_bounds[3]) / 2
        return (
            target_center_x - intrinsic_center_x,
            target_center_y - intrinsic_center_y,
        )

    def _compute_overflow_correction_shift(
        self, bounds: Bounds, image_size: tuple[int, int], margin: int
    ) -> tuple[float, float]:
        x0, y0, x1, y1 = bounds
        image_width, image_height = image_size
        allowed_x0 = float(margin)
        allowed_y0 = float(margin)
        allowed_x1 = float(max(margin, image_width - margin))
        allowed_y1 = float(max(margin, image_height - margin))
        width = x1 - x0
        height = y1 - y0
        allowed_width = max(0.0, allowed_x1 - allowed_x0)
        allowed_height = max(0.0, allowed_y1 - allowed_y0)

        if width > allowed_width:
            corrected_x0 = allowed_x0
        else:
            corrected_x0 = min(max(x0, allowed_x0), allowed_x1 - width)
        if height > allowed_height:
            corrected_y0 = allowed_y0
        else:
            corrected_y0 = min(max(y0, allowed_y0), allowed_y1 - height)
        return corrected_x0 - x0, corrected_y0 - y0

    def _compute_overflow_flags(
        self, bounds: Bounds | None, image_size: tuple[int, int], margin: int
    ) -> dict[str, bool]:
        if bounds is None:
            return {
                "overflow_left": False,
                "overflow_top": False,
                "overflow_right": False,
                "overflow_bottom": False,
            }
        image_width, image_height = image_size
        allowed_x0 = float(margin)
        allowed_y0 = float(margin)
        allowed_x1 = float(max(margin, image_width - margin))
        allowed_y1 = float(max(margin, image_height - margin))
        return {
            "overflow_left": bounds[0] < allowed_x0,
            "overflow_top": bounds[1] < allowed_y0,
            "overflow_right": bounds[2] > allowed_x1,
            "overflow_bottom": bounds[3] > allowed_y1,
        }

    def _describe_vertical_columns_furigana_space_needs(
        self,
        result: PageTextExtractionResultDict,
        region_sizings: list[ResolvedRegionSizing],
    ) -> list[VerticalLineLayoutNeed]:
        lines_info: list[VerticalLineLayoutNeed] = []
        counter = 0
        for region_index, text_region in enumerate(result.get("text_regions", [])):
            if not text_region.get("is_vertical", False):
                continue
            furigana_size = region_sizings[region_index].furigana_size
            for line_text, line_coords in zip(
                text_region.get("line_texts", []),
                text_region.get("line_outline_points", []),
            ):
                xs = [pt[0] for pt in line_coords]
                ys = [pt[1] for pt in line_coords]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                width = x_max - x_min
                segments = self.furigana_reading_generator.resolve_line_segments(
                    line_text
                )
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

    def _cluster_colliding_vertical_columns_for_furigana(
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

    def _resolve_vertical_column_shift_by_cluster(
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

    def _plan_vertical_column_shifts_for_furigana(
        self,
        result: PageTextExtractionResultDict,
        region_sizings: list[ResolvedRegionSizing],
    ) -> dict[int, float]:
        lines_info = self._describe_vertical_columns_furigana_space_needs(
            result=result, region_sizings=region_sizings
        )
        groups = self._cluster_colliding_vertical_columns_for_furigana(lines_info)
        return self._resolve_vertical_column_shift_by_cluster(groups)

    def _resolve_region_sizings(
        self, result: PageTextExtractionResultDict, measure_draw: ImageDraw.ImageDraw
    ) -> list[ResolvedRegionSizing]:
        region_sizings: list[ResolvedRegionSizing] = []
        for region_index, text_region in enumerate(result.get("text_regions", [])):
            region_sizing = self._resolve_region_sizing(
                text_region=text_region, measure_draw=measure_draw
            )
            region_sizings.append(region_sizing)
            logger.debug(
                "Region {} resolved sizing: main={}, furigana={}, char_spacing={}, furigana_spacing={}",
                region_index,
                region_sizing.main_size,
                region_sizing.furigana_size,
                region_sizing.char_spacing,
                region_sizing.furigana_spacing,
            )
        return region_sizings

    def _resolve_region_sizing(
        self,
        text_region: ExtractedTextRegionDict,
        measure_draw: ImageDraw.ImageDraw,
    ) -> ResolvedRegionSizing:
        vertical = bool(text_region.get("is_vertical", False))
        line_outline_points = text_region.get("line_outline_points", [])
        line_target_widths = text_region.get("line_target_widths", [])
        line_target_heights = text_region.get("line_target_heights", [])
        line_guesses: list[int] = []

        for index, line_coords in enumerate(line_outline_points):
            line_bounds = self._compute_outline_bounds(line_coords)
            if line_bounds is None:
                continue
            bbox_target_dimension = (
                line_bounds[2] - line_bounds[0]
                if vertical
                else line_bounds[3] - line_bounds[1]
            )
            mask_target_dimension = (
                (line_target_widths[index] if vertical else line_target_heights[index])
                if index < len(line_target_widths if vertical else line_target_heights)
                else None
            )
            target_dimension = (
                float(mask_target_dimension)
                if mask_target_dimension is not None and mask_target_dimension > 0
                else bbox_target_dimension
            )
            logger.trace(
                "Line target source: vertical={}, line_index={}, mask_target={}, bbox_target={}, selected_target={}",
                vertical,
                index,
                mask_target_dimension,
                bbox_target_dimension,
                target_dimension,
            )
            effective_target_dimension = self._compute_effective_target_dimension(
                target_dimension=target_dimension,
                vertical=vertical,
            )
            guess = self._estimate_main_size_from_target_dimension(
                draw=measure_draw,
                target_dimension=effective_target_dimension,
                vertical=vertical,
            )
            if guess is not None:
                line_guesses.append(guess)

        if line_guesses:
            raw_main_size = round(median(line_guesses))
            main_size = self._clamp_main_size(raw_main_size)
            logger.debug(
                "Region dynamic sizing: line_count={}, median={}, clamped={}",
                len(line_guesses),
                raw_main_size,
                main_size,
            )
        else:
            main_size = self._resolve_main_size_fallback(
                estimated_font_size=text_region.get(
                    "estimated_font_size", self.config.sizing.default_main_size
                )
            )
            logger.debug(
                "Region sizing fallback used: estimated_font_size={}, resolved_main_size={}",
                text_region.get(
                    "estimated_font_size", self.config.sizing.default_main_size
                ),
                main_size,
            )

        furigana_size = max(
            self.config.sizing.min_furigana_size,
            min(
                self.config.sizing.max_furigana_size,
                round(main_size * self.config.sizing.furigana_ratio),
            ),
        )
        char_spacing = max(
            self.config.sizing.min_char_spacing,
            round(main_size * self.config.sizing.char_spacing_ratio),
        )
        furigana_spacing = max(
            self.config.sizing.min_furigana_spacing,
            round(furigana_size * self.config.sizing.furigana_spacing_ratio),
        )
        return ResolvedRegionSizing(
            main_size=main_size,
            furigana_size=furigana_size,
            char_spacing=char_spacing,
            furigana_spacing=furigana_spacing,
        )

    def _estimate_main_size_from_target_dimension(
        self, draw: ImageDraw.ImageDraw, target_dimension: float, vertical: bool
    ) -> int | None:
        if target_dimension < self.config.sizing.min_target_dimension:
            logger.trace(
                "Sizing micro-check skipped: target_dimension={} below min_target_dimension={}",
                target_dimension,
                self.config.sizing.min_target_dimension,
            )
            return None
        probe_dimension_at_1 = self._measure_probe_dimension(
            draw=draw,
            font_size=1,
            vertical=vertical,
        )
        if probe_dimension_at_1 <= 0:
            logger.trace(
                "Sizing micro-check failed: probe_dimension_at_1={} for vertical={}",
                probe_dimension_at_1,
                vertical,
            )
            return None

        guessed_raw = round(
            (target_dimension / probe_dimension_at_1)
            * self.config.sizing.main_scale_bias
        )
        guessed = self._clamp_main_size(guessed_raw)
        if not self.config.sizing.enable_one_step_correction:
            logger.trace(
                "Sizing micro-check: vertical={}, target={}, probe_at_1={}, s0_raw={}, s0_clamped={}, correction=disabled, final={}",
                vertical,
                target_dimension,
                probe_dimension_at_1,
                guessed_raw,
                guessed,
                guessed,
            )
            return guessed

        measured = self._measure_probe_dimension(
            draw=draw, font_size=guessed, vertical=vertical
        )
        if measured <= 0:
            logger.trace(
                "Sizing micro-check: vertical={}, target={}, probe_at_1={}, s0_raw={}, s0_clamped={}, measured_at_s0={}, correction=skipped, final={}",
                vertical,
                target_dimension,
                probe_dimension_at_1,
                guessed_raw,
                guessed,
                measured,
                guessed,
            )
            return guessed
        corrected_raw = round(guessed * (target_dimension / measured))
        corrected = self._clamp_main_size(corrected_raw)
        logger.trace(
            "Sizing micro-check: vertical={}, target={}, probe_at_1={}, s0_raw={}, s0_clamped={}, measured_at_s0={}, s1_raw={}, final={}",
            vertical,
            target_dimension,
            probe_dimension_at_1,
            guessed_raw,
            guessed,
            measured,
            corrected_raw,
            corrected,
        )
        return corrected

    def _measure_probe_dimension(
        self, draw: ImageDraw.ImageDraw, font_size: int, vertical: bool
    ) -> float:
        font = self._load_japanese_font(max(1, font_size))
        bbox = draw.textbbox((0, 0), self.config.sizing.probe_glyph, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return float(width if vertical else height)

    def _resolve_main_size_fallback(self, estimated_font_size: int) -> int:
        if self.config.sizing.use_ocr_estimate_fallback:
            scaled = round(
                estimated_font_size * self.config.sizing.ocr_fallback_main_scale
            )
            return self._clamp_main_size(scaled)
        return self._clamp_main_size(self.config.sizing.default_main_size)

    def _clamp_main_size(self, size: int) -> int:
        return max(
            self.config.sizing.min_main_size,
            min(self.config.sizing.max_main_size, size),
        )

    def _compute_effective_target_dimension(
        self, target_dimension: float, vertical: bool
    ) -> float:
        ratio = (
            self.config.sizing.target_inner_ratio_vertical
            if vertical
            else self.config.sizing.target_inner_ratio_horizontal
        )
        ratio = max(0.1, min(1.0, ratio))
        effective = target_dimension * ratio
        logger.trace(
            "Sizing target adjustment: vertical={}, target_raw={}, target_inner_ratio={}, target_effective={}",
            vertical,
            target_dimension,
            ratio,
            effective,
        )
        return effective
