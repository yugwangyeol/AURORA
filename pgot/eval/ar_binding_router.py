"""GT-free selection of AR regions from class/point query inputs only.

No masks, annotation identities, IoUs, or expected GT indices are accepted by
the router. This is a lexical class router, not general referring-expression
understanding. One-to-one assignment uses predicted hard-mask centroids.
"""

import re

import numpy as np
from scipy.optimize import linear_sum_assignment


ALIASES = {
    "man": "person", "woman": "person", "boy": "person", "girl": "person",
    "child": "person", "kid": "person", "men": "person", "women": "person",
    "people": "person", "persons": "person", "human": "person",
    "sofa": "couch", "plane": "airplane", "aeroplane": "airplane",
    "bike": "bicycle", "motorbike": "motorcycle", "table": "dining table",
    "plant": "potted plant", "television": "tv", "tv monitor": "tv",
    "monitor": "tv", "cellphone": "cell phone", "mobile phone": "cell phone",
    "phone": "cell phone", "fridge": "refrigerator", "hair dryer": "hair drier",
}


def normalize_category(name):
    name = re.sub(r"\s+", " ", str(name).lower().replace("_", " ").replace("-", " ")).strip()
    name = re.sub(r"^(a|an|the)\s+", "", name)
    return ALIASES.get(name, name)


def parse_ar_slots(text, ovt_count):
    """Keep a record for EVERY OVT, including unparseable/stuff records.

    Readout labels are 1..ovt_count; never compact parsed entries and thereby
    attach a caption to another slot's mask.
    """
    chunks = text.split("<ovt>")[:-1]
    if len(chunks) != ovt_count:
        raise ValueError("Decoded caption OVT count differs from packed AR positions")
    slots = []
    for index, chunk in enumerate(chunks):
        markers = list(re.finditer(r"<(thing|stuff)>", chunk))
        marker = markers[-1] if markers else None
        match = re.match(r"\s*([^:<]+?)\s*:\s*(.*)", chunk[marker.end():], re.DOTALL) if marker else None
        name = re.sub(r"\s+\d+$", "", match[1].strip()) if match else None
        slots.append({"slot_index": index, "prediction_label": index + 1,
                      "is_thing": bool(marker and marker[1] == "thing"),
                      "category": name, "normalized_category": normalize_category(name) if name else None,
                      "description": match[2].strip() if match else None, "parsed": bool(match)})
    return slots


def route_ar_queries(prediction, slots, query_inputs, *, center_source="mask", use_point=True):
    """Return selected AR labels, using ONLY whitelisted class/point inputs.

    query_inputs entries must contain ONLY category and center_normalized_xy.
    Class is a hard compatibility constraint. Dummy assignments allow queries
    with no distinct compatible nonempty region to remain unmatched.
    """
    pred = np.asarray(prediction, dtype=np.int64)
    if pred.ndim != 2 or pred.min() < 0 or pred.max() > len(slots):
        raise ValueError("Prediction must have background 0 and slot labels 1..N")
    if center_source not in {"mask", "bbox"}:
        raise ValueError("Unknown predicted center source")
    for query in query_inputs:
        if set(query) != {"category", "center_normalized_xy"}:
            raise ValueError("Router accepts class/point input only, not GT scoring metadata")
        point = np.asarray(query["center_normalized_xy"], dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all() or np.any((point < 0) | (point > 1)):
            raise ValueError("Invalid query point")
    h, w = pred.shape
    candidates = []
    for index, slot in enumerate(slots):
        if slot["prediction_label"] != index + 1:
            raise ValueError("AR slot labels must retain original OVT order")
        yy, xx = np.nonzero(pred == index + 1)
        if not len(xx) or not slot["parsed"] or not slot["is_thing"]:
            continue
        if center_source == "mask":
            x, y = (xx.mean() + .5) / w, (yy.mean() + .5) / h
        else:
            x, y = (xx.min() + xx.max() + 1) / (2 * w), (yy.min() + yy.max() + 1) / (2 * h)
        candidates.append(dict(slot, predicted_center_normalized_xy=[float(x), float(y)], pixels=int(len(xx))))
    q, n = len(query_inputs), len(candidates)
    # Squared normalized distance <= 2; unmatched cost 3; incompatible 1e6.
    cost = np.full((q, n + q), 3., dtype=float)
    cost[:, :n] = 1e6
    compatible_counts = []
    for i, query in enumerate(query_inputs):
        compatible = 0
        for j, candidate in enumerate(candidates):
            if normalize_category(query["category"]) != candidate["normalized_category"]:
                continue
            compatible += 1
            delta = np.asarray(query["center_normalized_xy"]) - candidate["predicted_center_normalized_xy"]
            cost[i, j] = float(delta @ delta) if use_point else 0.
        compatible_counts.append(compatible)
    assigned = dict(zip(*linear_sum_assignment(cost))) if q else {}
    labels, records = [], []
    for i in range(q):
        j = assigned[i]
        candidate = candidates[j] if j < n and cost[i, j] < 3 else None
        label = candidate["prediction_label"] if candidate else None
        labels.append(label)
        records.append({"query_index": i, "selected_ar_prediction_label": label,
                        "selected_ar_slot_index": candidate["slot_index"] if candidate else None,
                        "selected_ar_category": candidate["category"] if candidate else None,
                        "predicted_center_normalized_xy": candidate["predicted_center_normalized_xy"] if candidate else None,
                        "point_distance_squared": float(cost[i, j]) if candidate and use_point else None,
                        "compatible_nonempty_candidates": compatible_counts[i],
                        "unmatched_reason": None if candidate else
                            "no_compatible_nonempty_slot" if not compatible_counts[i] else "one_to_one_capacity"})
    return labels, records, candidates


def routed_prediction(prediction, selected_labels):
    """Return Q query regions plus residual background; leave missing queries empty."""
    pred = np.asarray(prediction, dtype=np.int64)
    labels = [label for label in selected_labels if label is not None]
    if len(labels) != len(set(labels)) or any(label < 1 for label in labels):
        raise ValueError("Selected AR labels must be unique positive labels")
    result = np.zeros_like(pred)
    for i, label in enumerate(selected_labels):
        if label is not None:
            result[pred == label] = i + 1
    return result
