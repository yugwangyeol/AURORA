"""Render E4 v8 with three cross-category language-guided replacements."""

from pgot.eval.build_appendix_qualitative_figures import APPENDIX_ROOT
from pgot.eval.build_appendix_language_guided_editing_v5 import render


def main() -> None:
    directory = APPENDIX_ROOT / "e4_language_guided_editing"
    root_v8 = directory / "_inference_v8_cross_category_candidates"
    right = (
        (root_v8, "appendix_v8_replace_bicycle_motorcycle", "Replace",
         "A's bicycle with B's motorcycle."),
        (root_v8, "appendix_v8_replace_cup_vase", "Replace",
         "A's cup with B's vase."),
        (root_v8, "appendix_v8_replace_horse_cow", "Replace",
         "A's horse with B's cow."),
    )
    render(
        right_override=right,
        revision="v8",
        output_stem="language_guided_object_editing_v8_cross_category",
        change_note=(
            "Kept v7 unchanged and added a cross-category alternative. Each right-side edit "
            "uses language-selected replace with full donor semantic and visual bundles. "
            "Selected bicycle-to-motorcycle, cup-to-vase, and horse-to-cow results after "
            "enlarged review of eight newly inferred cross-category requests."
        ),
    )


if __name__ == "__main__":
    main()
