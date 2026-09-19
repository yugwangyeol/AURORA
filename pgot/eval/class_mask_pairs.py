"""Class-compatible, one-to-one instance pair scoring (not Binding Hits).

Every generated thing pair is a prediction, including unknown classes, empty
masks and duplicates. All visible valid GT thing instances form recall's
denominator. Descriptive attributes/relations are NOT scored by this metric.
"""
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from pgot.eval.ar_binding_router import normalize_category, parse_ar_slots


def ar_pairs(text, ovt_count, text_mode='description'):
    """Retain original OVT identities, and compact THING readout identities.

    Remove only the schema markers and ordinal from the shared natural-language
    text. Unknown/malformed owners and trailing THING text without an OVT are
    predictions too; explicit STUFF owners are outside thing-instance scoring.
    """
    if text_mode not in {'description', 'category'}:
        raise ValueError('Unknown text mode')
    slots = parse_ar_slots(text, ovt_count)
    chunks = text.split('<ovt>')
    result = []
    rank = 0
    for slot, chunk in zip(slots, chunks):
        markers = list(re.finditer(r'<(thing|stuff)>', chunk))
        marker = markers[-1] if markers else None
        if marker and marker[1] == 'stuff':
            continue
        rank += 1
        if slot['parsed']:
            shared = (f"{slot['category']}: {slot['description']}" if text_mode == 'description'
                      else slot['normalized_category'])
        else:
            shared = re.sub(r'<[^>]+>', '', chunk).strip(' .\n')
        result.append(dict(slot, prediction_label=rank, shared_text=shared,
                           pair_index=len(result), unbound_text=False))
    tail = chunks[-1].split('<scene_end>')[0]
    markers = list(re.finditer(r'<(thing|stuff)>', tail))
    if markers and markers[-1][1] == 'thing' and tail[markers[-1].end():].strip(' .\n'):
        fragment = tail[markers[-1].start():]
        slot = parse_ar_slots(fragment + '<ovt>', 1)[0]
        shared = (f"{slot['category']}: {slot['description']}" if slot['parsed'] and text_mode == 'description'
                  else slot['normalized_category'] if slot['parsed'] else re.sub(r'<[^>]+>', '', fragment).strip())
        result.append(dict(slot, slot_index=None, prediction_label=None, shared_text=shared,
                           pair_index=len(result), unbound_text=True))
    return result


def masks_for_pairs(prediction, pairs):
    pred = np.asarray(prediction)
    if pred.ndim != 2:
        raise ValueError('Expected a hard 2-D prediction')
    return np.stack([pred == p['prediction_label'] if p['prediction_label'] is not None
                     else np.zeros_like(pred, bool) for p in pairs]) if pairs else np.zeros((0, *pred.shape), bool)


def ctrlo_pair_masks(maps, count, queries_per_pass):
    maps = np.asarray(maps)
    if maps.ndim != 3 or count < 0 or not 1 <= queries_per_pass <= 6:
        raise ValueError('Invalid CTRL-O chunk maps/count/capacity')
    if len(maps) != (count + queries_per_pass - 1) // queries_per_pass:
        raise ValueError('Missing/extra CTRL-O chunk; never truncate overflow')
    return (np.stack([maps[i//queries_per_pass] == i%queries_per_pass for i in range(count)])
            if count else np.zeros((0, *maps.shape[1:]), bool))


def iou_matrix(predictions, targets, valid, prediction_classes=None, target_classes=None):
    valid = np.asarray(valid, bool)
    p, g = np.asarray(predictions, bool), np.asarray(targets, bool)
    if p.ndim != 3 or g.ndim != 3 or p.shape[1:] != valid.shape or g.shape[1:] != valid.shape:
        raise ValueError('Mask/valid shape mismatch')
    p, g = p & valid, g & valid
    pa, ga = p.sum((1, 2)), g.sum((1, 2))
    packed_p = np.packbits(p.reshape(len(p), valid.size), axis=1)
    packed_g = np.packbits(g.reshape(len(g), valid.size), axis=1)
    lookup = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    matrix = np.zeros((len(p), len(g)), float)
    for i in range(len(p)):
        for j in range(len(g)):
            if prediction_classes is not None and (prediction_classes[i] is None or prediction_classes[i] != target_classes[j]):
                continue
            inter = int(lookup[np.bitwise_and(packed_p[i], packed_g[j])].sum())
            union = int(pa[i] + ga[j]) - inter
            matrix[i, j] = inter / union if union else 0.
    return matrix


def score_pairs(predictions, classes, targets, gt_classes, valid, thresholds=(.25, .5, .75)):
    if len(classes) != len(predictions) or len(gt_classes) != len(targets):
        raise ValueError('Class/mask count mismatch')
    classes = [normalize_category(c) if c is not None else None for c in classes]
    gt_classes = [normalize_category(c) for c in gt_classes]
    matrix = iou_matrix(predictions, targets, valid, classes, gt_classes)
    result = {}
    for threshold in thresholds:
        if not 0 < threshold <= 1:
            raise ValueError('IoU thresholds must be in (0, 1]')
        eligible = matrix >= threshold
        # Maximize number of valid matches first, summed IoU second. Background
        # and ineligible edges cannot displace a valid match with a better sum.
        reward = eligible * (min(matrix.shape, default=0) + 1 + matrix)
        rows, cols = linear_sum_assignment(reward, maximize=True)
        matches = [{'pair_index': int(i), 'gt_index': int(j), 'iou': float(matrix[i, j])}
                   for i, j in zip(rows, cols) if eligible[i, j]]
        tp = len(matches)
        result[f'{threshold:.2f}'] = dict(tp=tp, fp=len(classes)-tp, fn=len(gt_classes)-tp,
                    matches=matches, matched_iou_sum=sum(m['iou'] for m in matches))
    return result


def rates(tp, fp, fn):
    return dict(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
