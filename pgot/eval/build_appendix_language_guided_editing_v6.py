"""Render E4 v6 with two large, visually audited language-guided edits."""

from pgot.eval.build_appendix_qualitative_figures import APPENDIX_ROOT
from pgot.eval.build_appendix_language_guided_editing_v5 import render


def main() -> None:
    directory = APPENDIX_ROOT / "e4_language_guided_editing"
    root_v2 = directory / "_inference_v2"
    root_v6 = directory / "_inference_v6_candidates"
    right = (
        (root_v6, "appendix_v6_transfer_phone_modern_vintage", "Transfer",
         "B's vintage appearance to A's phone."),
        (root_v6, "appendix_v6_transfer_fridge_white_steel", "Transfer",
         "B's stainless-steel appearance to A's refrigerator."),
        (root_v2, "appendix_v2_transfer_suitcase", "Transfer",
         "B's pattern to A's suitcase."),
    )
    render(
        right_override=right,
        revision="v6",
        output_stem="language_guided_object_editing_v6",
        change_note=(
            "Replaced v5 right rows 1 and 2: the umbrella edit obscured the person and the "
            "jumping snowboard was too small at paper scale. Selected large phone and "
            "refrigerator edits after an eight-candidate enlarged visual audit."
        ),
    )


if __name__ == "__main__":
    main()
