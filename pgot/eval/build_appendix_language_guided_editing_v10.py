"""Render the combined E4 figure with column headers only."""

from pgot.eval.build_appendix_language_guided_editing_v9 import main as render_combined


def main() -> None:
    render_combined(
        show_group_titles=False,
        revision="v10",
        output_stem="language_guided_object_editing_v10_combined_no_group_titles",
    )


if __name__ == "__main__":
    main()
