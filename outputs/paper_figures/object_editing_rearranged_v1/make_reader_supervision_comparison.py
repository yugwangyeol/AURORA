"""Stack the full-model and no-reader-supervision editing figures."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
FULL_MODEL = ROOT / "object_editing_language_selected.png"
WITHOUT_READER_SUPERVISION = ROOT / "object_editing_remove_replace_3x2_woreader.png"
OUTPUT = ROOT / "object_editing_reader_supervision_comparison"


def font(size, bold=False):
    suffix = "-Bold.ttf" if bold else ".ttf"
    return ImageFont.truetype(
        f"/usr/share/fonts/truetype/dejavu/DejaVuSerif{suffix}", size
    )


def centered(draw, text, box, text_font):
    left, top, width, height = box
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=text_font)
    draw.text(
        (left + (width - (x1 - x0)) / 2 - x0,
         top + (height - (y1 - y0)) / 2 - y0),
        text,
        font=text_font,
        fill="black",
    )


def main():
    Image.init()
    with Image.open(FULL_MODEL) as image:
        full = image.convert("RGB")
    with Image.open(WITHOUT_READER_SUPERVISION) as image:
        no_reader = image.convert("RGB")
    if full.size != no_reader.size:
        raise ValueError(f"Figure sizes differ: {full.size} vs. {no_reader.size}")

    width, figure_height = full.size
    outer = 12
    heading_height = 82
    section_gap = 34
    height = outer * 2 + heading_height * 2 + figure_height * 2 + section_gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    heading_font = font(50, bold=True)

    first_heading_y = outer
    first_figure_y = first_heading_y + heading_height
    second_heading_y = first_figure_y + figure_height + section_gap
    second_figure_y = second_heading_y + heading_height

    centered(draw, "Full Model", (0, first_heading_y, width, heading_height), heading_font)
    canvas.paste(full, (0, first_figure_y))
    centered(
        draw,
        "w/o Reader Supervision",
        (0, second_heading_y, width, heading_height),
        heading_font,
    )
    canvas.paste(no_reader, (0, second_figure_y))

    divider_y = second_heading_y - section_gap // 2
    draw.line((outer, divider_y, width - outer, divider_y), fill=(175, 175, 175), width=2)

    canvas.save(OUTPUT.with_suffix(".png"), dpi=(300, 300))
    canvas.save(OUTPUT.with_suffix(".pdf"), resolution=300)
    preview_width = 1400
    canvas.resize(
        (preview_width, round(height * preview_width / width)), Image.Resampling.LANCZOS
    ).save(ROOT / "object_editing_reader_supervision_comparison_preview.png")
    print(OUTPUT.with_suffix(".png"))
    print(OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
