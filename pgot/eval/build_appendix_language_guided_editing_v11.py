"""Render E4 as one six-row table with a single, large header row."""

from pgot.eval.build_appendix_language_guided_editing_v9 import main as render_combined


def main() -> None:
    render_combined(
        show_group_titles=False,
        unified_six_rows=True,
        header_font_size=36,
        revision="v11",
        output_stem="language_guided_object_editing_v11_single_header",
    )


if __name__ == "__main__":
    main()
