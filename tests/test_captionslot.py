import torch

from scale_rae.model.object_centric import build_captionslot_attention_mask


def test_captionslot_attention_mask_routes_information():
    positions = {
        "sys_s": 0,
        "sys_e": 4,
        "user_prefix_s": 4,
        "user_prefix_e": 5,
        "img_s": 5,
        "img_e": 9,
        "user_suffix_s": 9,
        "user_suffix_e": 10,
        "assistant_prefix_s": 10,
        "assistant_prefix_e": 11,
        "cap_s": 11,
        "cap_e": 15,
        "slot_s": 15,
        "slot_e": 18,
        "reg_s": 18,
        "reg_e": 20,
        "im_start_idx": 20,
        "rae_s": 21,
        "rae_e": 24,
        "im_end_idx": 24,
        "assistant_suffix_s": 25,
        "assistant_suffix_e": 26,
        "total_len": 26,
    }
    ref_spans = torch.tensor([[[0, 2], [2, 4], [-1, -1]]], dtype=torch.long)
    active_slot_mask = torch.tensor([[True, True, False]])
    caption_padding_mask = torch.tensor([[True, True, True, True]])

    bias = build_captionslot_attention_mask(
        positions=positions,
        ref_spans=ref_spans,
        active_slot_mask=active_slot_mask,
        caption_padding_mask=caption_padding_mask,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).squeeze(1)

    # slot 0 sees only the first referring-expression span and all image patches.
    slot0 = positions["slot_s"]
    assert torch.isfinite(bias[0, slot0, positions["img_s"]:positions["img_e"]]).all()
    assert torch.isfinite(bias[0, slot0, positions["cap_s"]:positions["cap_s"] + 2]).all()
    assert torch.isneginf(bias[0, slot0, positions["cap_s"] + 2:positions["cap_e"]]).all()
    assert torch.isfinite(bias[0, slot0, positions["slot_s"]]).all()
    assert torch.isneginf(bias[0, slot0, positions["slot_s"] + 1]).all()
    assert torch.isneginf(bias[0, slot0, positions["reg_s"]:positions["reg_e"]]).all()

    # registers see captions, image patches, slots, and other registers.
    reg0 = positions["reg_s"]
    assert torch.isfinite(bias[0, reg0, positions["img_s"]:positions["img_e"]]).all()
    assert torch.isfinite(bias[0, reg0, positions["cap_s"]:positions["cap_e"]]).all()
    assert torch.isfinite(bias[0, reg0, positions["slot_s"]:positions["slot_s"] + 2]).all()

    # RAE queries cannot see image patches directly but can see caption/slot/reg and im_start.
    rae0 = positions["rae_s"]
    assert torch.isneginf(bias[0, rae0, positions["img_s"]:positions["img_e"]]).all()
    assert torch.isfinite(bias[0, rae0, positions["cap_s"]:positions["cap_e"]]).all()
    assert torch.isfinite(bias[0, rae0, positions["slot_s"]:positions["slot_s"] + 2]).all()
    assert torch.isfinite(bias[0, rae0, positions["reg_s"]:positions["reg_e"]]).all()
    assert torch.isfinite(bias[0, rae0, positions["im_start_idx"]]).all()
