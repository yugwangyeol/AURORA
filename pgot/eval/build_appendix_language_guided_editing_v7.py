"""Render E4 v7 with two previously unused COCO source/donor pairs."""

from pgot.eval.build_appendix_qualitative_figures import APPENDIX_ROOT
from pgot.eval.build_appendix_language_guided_editing_v5 import render


def main() -> None:
    directory = APPENDIX_ROOT / "e4_language_guided_editing"
    root_v2 = directory / "_inference_v2"
    root_v7b = directory / "_inference_v7b_candidates"
    right = (
        (root_v7b, "appendix_v7b_plane_white_to_blue", "Transfer",
         "B's blue paint to A's airplane."),
        (root_v7b, "appendix_v7b_sheep_white_to_black", "Transfer",
         "B's black wool to A's sheep."),
        (root_v2, "appendix_v2_transfer_suitcase", "Transfer",
         "B's pattern to A's suitcase."),
    )
    render(
        right_override=right,
        revision="v7",
        output_stem="language_guided_object_editing_v7",
        change_note=(
            "Replaced v6 right rows 1 and 2 with newly inferred examples whose source and "
            "donor COCO IDs do not occur in any earlier appendix editing candidate or figure "
            "manifest. Selected airplane paint and sheep wool transfers after enlarged visual "
            "review of fourteen new requests."
        ),
    )


if __name__ == "__main__":
    main()
