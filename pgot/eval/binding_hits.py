"""CTRL-O-style binding accuracy, with explicit query-to-instance identities.

Hungarian assignment maximizes total mask IoU. It is used ONLY for scoring:
expected target indices must already be fixed by the query manifest.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def score_binding_hits(prediction, targets, expected_indices, ignore=None, *, query_labels=None, num_predictions=None):
    """prediction: HxW, 0=register background, query i has label i+1.

    targets: NxHxW independent binary foreground instance masks (can overlap).
    All visible GT instances participate, including unqueried distractors.
    Background is a predicted competitor, not a foreground GT target.
    Unmatched query predictions count as misses, not dropped observations.

    Optional query_labels maps queries to arbitrary original prediction labels;
    None means no routed prediction and ALWAYS counts as a miss. num_predictions
    retains unselected/empty AR slots in the secondary all-slot assignment pool.
    """
    pred = np.asarray(prediction, dtype=np.int64)
    gt = np.asarray(targets, dtype=bool)
    expected = np.asarray(expected_indices, dtype=np.int64)
    q = len(expected)
    n = q if num_predictions is None else int(num_predictions)
    labels = list(range(1, q + 1)) if query_labels is None else list(query_labels)
    if n < 0 or len(labels) != q:
        raise ValueError("Invalid prediction count or query label count")
    present = [label for label in labels if label is not None]
    if len(set(present)) != len(present) or any(label < 1 or label > n for label in present):
        raise ValueError("Query prediction labels must be unique valid labels or None")
    if pred.ndim != 2 or gt.ndim != 3 or gt.shape[1:] != pred.shape:
        raise ValueError("Expected prediction HxW and targets NxHxW")
    if pred.min() < 0 or pred.max() > n:
        raise ValueError("Prediction labels must be 0..number of queries")
    if len(set(expected.tolist())) != q or np.any(expected < 0) or np.any(expected >= len(gt)):
        raise ValueError("Query targets must be unique valid GT indices")
    valid = np.ones(pred.shape, dtype=bool) if ignore is None else ~np.asarray(ignore, dtype=bool)
    if valid.shape != pred.shape:
        raise ValueError("Ignore mask shape differs from prediction")
    if len(gt) and not np.all(gt[:, valid].any(axis=1)):
        raise ValueError("Remove GT instances empty after ignore filtering before fixing query indices")
    pred_area = np.bincount(pred[valid], minlength=n + 1).astype(np.float64)
    inter = np.zeros((n + 1, len(gt)), dtype=np.float64)
    areas = np.zeros(len(gt), dtype=np.float64)
    for j, mask in enumerate(gt):
        mask = mask & valid
        areas[j] = mask.sum()
        inter[:, j] = np.bincount(pred[mask], minlength=n + 1)
    union = pred_area[:, None] + areas[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    # Place objects before aggregate background, mirroring PGOT readout ordering.
    label_order = list(range(1, n + 1)) + [0]
    rows, cols = linear_sum_assignment(iou[label_order], maximize=True)
    matched = {label_order[int(r)]: int(c) for r, c in zip(rows, cols)}
    records = []
    for i, target_idx in enumerate(expected):
        label = labels[i]
        value = float(iou[label, target_idx]) if label is not None else 0.
        hit = label is not None and matched.get(label) == int(target_idx)
        records.append({"query_index": i, "target_gt_index": int(target_idx),
                        "matched_gt_index": matched.get(label), "hit": hit,
                        "target_iou": value, "hit_iou50": hit and value >= 0.5,
                        "positive_iou_hit": hit and value > 0.0})
    return records
