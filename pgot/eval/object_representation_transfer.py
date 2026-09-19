"""Explicit owner transfers: qualitative, native Reader, no instruction model."""
import re

from pgot.eval.ar_binding_router import parse_ar_slots

CONDITIONS = ("reconstruction", "text_only", "semantic_only", "visual_only", "bundle_swap")


def replace_description(text, slot_index, description, category=None):
    """Replace one AR owner's description, preserving all markers and indices."""
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Empty donor description")
    if re.search(r"<(ovt|scene_end|thing|stuff)>", description):
        raise ValueError("Description contains schema markers")
    if category is not None and (not isinstance(category,str) or not category.strip() or re.search(r"[<>:\n]",category)):
        raise ValueError("Invalid donor category")
    chunks = text.split("<ovt>")
    if not 0 <= slot_index < len(chunks)-1:
        raise ValueError("Target slot outside bound AR owners")
    slot = parse_ar_slots(text, len(chunks)-1)[slot_index]
    if not slot["parsed"] or not slot["is_thing"]:
        raise ValueError("Target must be a parsed thing")
    chunk = chunks[slot_index]
    marker = list(re.finditer(r"<thing>", chunk))[-1]
    header = re.match(r"<thing>\s*[^:<]+\s*:\s*", chunk[marker.start():])
    if header is None:
        raise ValueError("Cannot isolate target description")
    prefix=chunk[:marker.start()+header.end()]
    if category is not None:
        old_name=re.match(r"<thing>\s*([^:<]+)\s*:",chunk[marker.start():])[1].strip()
        ordinal=re.search(r"\s+(\d+)$",old_name)
        suffix=f" {ordinal[1]}" if ordinal else ""
        prefix=chunk[:marker.start()]+f"<thing> {category.strip()}{suffix}: "
    chunks[slot_index] = prefix + description.strip() + " "
    result = "<ovt>".join(chunks)
    assert result.count("<ovt>") == text.count("<ovt>")
    return result


def transfer_objects(objects, target_slot, donor_objects, donor_slot, condition, text_semantic=None):
    """Only one tuple changes; all other object/register inputs remain untouched."""
    if condition not in CONDITIONS:
        raise ValueError("Unknown transfer condition")
    if not 0 <= target_slot < len(objects) or not 0 <= donor_slot < len(donor_objects):
        raise ValueError("Transfer slot outside valid owners")
    units = list(objects)
    sa, ma = units[target_slot]
    sb, mb = donor_objects[donor_slot]
    if condition == "text_only":
        if text_semantic is None:
            raise ValueError("Text-only requires semantic state re-encoded on image A")
        units[target_slot] = (text_semantic, ma)
    elif condition == "semantic_only":
        units[target_slot] = (sb, ma)
    elif condition == "visual_only":
        units[target_slot] = (sa, mb)
    elif condition == "bundle_swap":
        units[target_slot] = (sb, mb)
    return units
