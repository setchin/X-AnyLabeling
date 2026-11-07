"""Instance-level admission: strong DEIM evidence plus sustained observations."""
from __future__ import annotations

import json
import copy
from collections import defaultdict

import numpy as np

from .store import Store, atomic, box, iou

POLICY = {'version': 'instance_confidence_continuity_v1', 'candidate_threshold': .1,
          'strong_score_strictly_above': .5, 'minimum_consecutive_frames': 10,
          'matching_iou': .3, 'boundary_link_iou': .5, 'maximum_boundary_transitions': 2}


def longest_run(frames):
    longest = run = 0; previous = -2
    for n in sorted(set(map(int, frames))):
        run = run+1 if n == previous+1 else 1
        longest = max(longest, run); previous = n
    return longest


def frame_score(raw, bbox, width, height):
    rows = np.asarray(raw)
    rows = rows[np.isfinite(rows).all(axis=1) & (rows[:, 4] >= .1)].astype(float, copy=True)
    if not len(rows):
        return 0.
    rows[:, [0, 2]] = rows[:, [0, 2]].clip(0, width)
    rows[:, [1, 3]] = rows[:, [1, 3]].clip(0, height)
    return max((float(r[4]) for r in rows if r[2] > r[0] and r[3] > r[1]
                and iou(bbox, r[:4]) >= POLICY['matching_iou']), default=0.)


def evidence(observations, raw, width, height):
    scores = {int(n): frame_score(raw[int(n)-1], b, width, height) for n, b in observations.items()}
    high = [n for n, score in scores.items() if score > .5]
    run = longest_run(observations)
    reasons = []
    if not high:
        reasons.append('no_deim_score_above_0.5')
    if run < 10:
        reasons.append('fewer_than_10_consecutive_observed_frames')
    return {'passes': not reasons, 'max_deim_score': max(scores.values(), default=0.),
            'high_confidence_frames': sorted(high), 'longest_consecutive_frames': run,
            'observed_frames': len(observations), 'reasons': reasons}


def protected(instance, shapes):
    return (instance['status'] == 'accepted' or instance['origin'] == 'manual_box'
            or any(s.get('sar', {}).get('manual_geometry') or s.get('sar', {}).get('origin') == 'manual_box'
                   or s.get('sar', {}).get('review_state') == 'accepted' for s in shapes))


def apply_instance_policy(store: Store, key, link_boundaries=True):
    """One reversible transaction; never convert automatic pruning into human rejection."""
    raw = np.load(store.raw_path(key), mmap_mode='r')
    with store.locked(key):
        index = store.index(key); docs = {n: store.doc(key, n) for n in range(1, index['frames']+1)}
        original = copy.deepcopy(docs); merges = []

        def collect():
            records = defaultdict(dict)
            for n, doc in docs.items():
                for shape in doc['shapes']:
                    ident = shape.get('sar', {}).get('instance_id')
                    if ident:
                        records[ident][n] = shape
            return records

        records = collect(); before_count = len(records)
        if link_boundaries:
            while True:
                stats = {ident: evidence({n: box(s) for n, s in shapes.items()}, raw, index['width'], index['height'])
                         for ident, shapes in records.items()}
                pairs = []
                for a, left in records.items():
                    last = max(left)
                    for b, right in records.items():
                        first = min(right)
                        if a == b or not 1 <= first-last <= 2:
                            continue
                        if not (stats[a]['passes'] or stats[b]['passes']):
                            continue
                        overlap = iou(box(left[last]), box(right[first]))
                        if overlap >= .5:
                            pairs.append((overlap, a, b))
                # A join must be each end's unique best match; ambiguity is left for review.
                eligible = []
                for overlap, a, b in pairs:
                    alternatives = [ov for ov, aa, bb in pairs if (aa == a or bb == b) and (aa, bb) != (a, b)]
                    if not alternatives or overlap-max(alternatives) >= .1:
                        eligible.append((overlap, a, b))
                if not eligible:
                    break
                overlap, a, b = max(eligible)
                keep, drop = a, b
                if protected(index['instances'][b], records[b].values()) and not protected(index['instances'][a], records[a].values()):
                    keep, drop = b, a
                parent = index['instances'][keep]; child = index['instances'][drop]
                for shape in records[drop].values():
                    shape['sar'].setdefault('original_instance_id', drop)
                    shape['sar']['instance_id'] = keep; shape['group_id'] = parent['group_id']
                parent['segments'] = list(dict.fromkeys(parent['segments']+child['segments']))
                parent['person_confirmed_seed_frames'] = sorted(set(parent.get('person_confirmed_seed_frames', [])+child.get('person_confirmed_seed_frames', [])))
                parent.setdefault('linked_instances', []).append(drop)
                # Preserve per-shape acceptance; a join does not approve its new frames.
                if any(s['sar'].get('review_state') != 'accepted' for s in [*records[keep].values(), *records[drop].values()]):
                    parent['status'] = 'pending'
                child.update(status='linked', alias_of=keep)
                merges.append({'retained_instance': keep, 'linked_instance': drop, 'boundary_iou': overlap})
                records = collect()

        retained = []; removed = []; removed_shapes = 0
        for ident, shapes in records.items():
            instance = index['instances'][ident]
            stats = evidence({n: box(s) for n, s in shapes.items()}, raw, index['width'], index['height'])
            review_protected = protected(instance, shapes.values())
            instance['admission'] = {**stats, 'policy': POLICY, 'protected_by_human_review': review_protected}
            row = {'instance_id': ident, 'group_id': instance['group_id'], **stats, 'protected_by_human_review': review_protected}
            if stats['passes'] or review_protected:
                retained.append(row)
            else:
                for n, shape in shapes.items():
                    docs[n]['shapes'].remove(shape); removed_shapes += 1
                instance['status'] = 'auto_filtered'; removed.append(row)
        changed = {n: doc for n, doc in docs.items() if doc != original[n]}
        index['quality_policy'] = POLICY
        op = store.commit(key, changed, index, 'filter automatic instances by confidence and continuity')
        report = {'sequence': key, 'policy': POLICY, 'instances_before': before_count,
                  'instances_after': len(retained), 'removed_instances': removed, 'retained_instances': retained,
                  'linked_boundaries': merges, 'removed_shapes': removed_shapes, 'operation': op,
                  'max_instances_in_one_frame': max((len(d['shapes']) for d in docs.values()), default=0)}
        atomic(store.metadata(key)/'instance_filter_report.json', report)
    store.write_review_queue(key)
    return report
