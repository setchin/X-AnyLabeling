"""Frozen DEIM + one-second SAM3 forward proposals; backward is evidence only."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from .store import WORKSPACE, Store, atomic, box, iou
from .quality import POLICY, apply_instance_policy, evidence as instance_evidence, protected
from .settings import SETTINGS


def candidate_rows(raw, width, height, threshold=.1):
    rows = np.array(raw, dtype=float, copy=True)
    rows[:, [0, 2]] = rows[:, [0, 2]].clip(0, width)
    rows[:, [1, 3]] = rows[:, [1, 3]].clip(0, height)
    rows = rows[np.isfinite(rows).all(axis=1) & (rows[:, 4] >= threshold)
                & (rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1])]
    kept = []
    for r in sorted(rows.tolist(), key=lambda r: -r[4]):
        if not any(iou(r[:4], q[:4]) > .6 for q in kept):
            kept.append(r)
    result = []
    for r in sorted(kept, key=lambda r: -(r[2]-r[0])*(r[3]-r[1])):
        area = (r[2]-r[0])*(r[3]-r[1])
        covered = any(max(0, min(r[2], q[2])-max(r[0], q[0]))*
                      max(0, min(r[3], q[3])-max(r[1], q[1]))/max(area, 1e-9) >= .85 for q in result)
        if not covered:
            result.append(r)
    return result


class Engine:
    def __init__(self, store: Store, gpu=1, progress=None):
        self.store = store; self.gpu = int(gpu)
        self.progress = progress or (lambda message: print(message, flush=True))
        self._sam = None; self._deim = None; self._common = None
        self.raw = {}; self.anchor_raw = {}; self.person_checks = {}

    def raw_predictions(self, key):
        if key not in self.raw:
            path = self.store.raw_path(key)
            if not path.is_file():
                raise ValueError(f'Dense DEIM prediction cache is missing: {path}')
            self.raw[key] = np.load(path, mmap_mode='r')
        return self.raw[key]

    def candidates(self, key, frame, threshold=.1):
        raw = self.anchor_raw.get(key, {}).get(frame)
        if raw is None:
            raw = self.raw_predictions(key)[frame-1]
        index = self.store.index(key)
        return candidate_rows(raw, index['width'], index['height'], threshold)

    def refresh_anchors(self, key):
        """Actually rerun frozen DEIM on every 1-fps image, including empty references."""
        import torch
        torch.cuda.set_device(self.gpu); torch.set_num_threads(4)
        if self._common is None:
            from . import deim
            self._common = deim
        if self._deim is None:
            self.progress('Loading frozen DEIMv2-X checkpoint')
            model, post, info = self._common.load_model(torch.device(f'cuda:{self.gpu}'))
            self._deim = (model, post, info)
        index = self.store.index(key)
        frames = sorted({1, *(int(n) for n in index['anchor_references'])})
        if len(frames) == 1:
            frames = list(range(1, index['frames']+1, 30))
        result = {}
        model, post, info = self._deim
        for offset in range(0, len(frames), 8):
            chunk = frames[offset:offset+8]; samples = []
            for n in chunk:
                path = self.store.data/'FRAMES'/key/f'{Path(key).name}_{n:06d}.jpg'
                with Image.open(path) as im:
                    samples.append(self._common.TRANSFORM(im.convert('RGB')))
            predictions = self._common.infer(model, post, torch.stack(samples),
                [[index['width'], index['height']]]*len(chunk), torch.device(f'cuda:{self.gpu}'))
            result.update(zip(chunk, predictions))
            self.progress(f'DEIM refreshed {min(offset+8, len(frames))}/{len(frames)} seed/anchor frames')
        self.anchor_raw[key] = result
        directory = self.store.metadata(key)
        np.savez_compressed(directory/'deim_anchor_predictions.npz', **{str(n): a for n, a in result.items()})
        comparisons = self.compare_anchor_predictions(key, result)
        report = {'frames': frames, 'anchor_frames': sorted(map(int, index['anchor_references'])),
            'threshold': .1, 'inference_performed': True, 'model': 'DEIMv2 DINOv3-X',
            'checkpoint': str(self._common.CHECKPOINT), 'image_size': 640,
            'comparison_with_dense_cache': comparisons, 'model_info': info,
            'created_at': time.time()}
        atomic(directory/'deim_anchor_refresh.json', report)
        return report

    def compare_anchor_predictions(self, key, result):
        # Query order can change under tiny FP32 rounding differences. Match boxes
        # geometrically instead of subtracting unrelated rows of the 300-query arrays.
        matched_ious, score_differences, coordinate_differences = [], [], []
        for n, new in result.items():
            old = self.raw_predictions(key)[n-1]
            for r in new[new[:, 4] >= .1]:
                best = max(old, key=lambda b: iou(r[:4], b[:4]))
                matched_ious.append(iou(r[:4], best[:4]))
                score_differences.append(abs(float(r[4])-float(best[4])))
                coordinate_differences.append(float(np.max(np.abs(r[:4]-best[:4]))))
        return {'threshold': .1, 'compared_predictions': len(matched_ious),
            'minimum_best_match_iou': min(matched_ious, default=None),
            'max_matched_score_difference': max(score_differences, default=None),
            'max_matched_coordinate_difference_pixels': max(coordinate_differences, default=None),
            'matching': 'Each fresh prediction scoring >=0.1 matched by maximum box IoU to saved dense predictions'}

    def sam(self):
        if self._sam is None:
            import torch
            torch.cuda.set_device(self.gpu); torch.set_num_threads(4)
            for name in ('sam3_repo', 'sam3_checkpoint'):
                path = getattr(SETTINGS, name)
                if path is None or not path.exists():
                    raise ValueError(f'Set a valid {name} in FOREST_SAR_CONFIG.')
            sys.path.insert(0, str(SETTINGS.sam3_repo))
            from sam3.model.sam3_video_predictor import Sam3VideoPredictor
            self.progress('Loading local SAM3 checkpoint')
            self._sam = Sam3VideoPredictor(checkpoint_path=str(SETTINGS.sam3_checkpoint),
                strict_state_dict_loading=True, apply_temporal_disambiguation=True, compile=False)
        return self._sam

    def person_check(self, key, frame):
        """Unprompted-by-box SAM3 'person' detection is independent verification evidence."""
        path = self.store.metadata(key)/'sam3_person_checks.json'
        if key not in self.person_checks:
            self.person_checks[key] = json.loads(path.read_text()) if path.exists() else {}
        if str(frame) in self.person_checks[key]:
            return self.person_checks[key][str(frame)]
        predictor = self.sam(); session = None; index = self.store.index(key)
        self.progress(f'SAM3 person verification on frame {frame}')
        with tempfile.TemporaryDirectory(prefix='forest_sar_person_') as tmp:
            (Path(tmp)/'000000.jpg').symlink_to(self.store.data/'FRAMES'/key/f'{Path(key).name}_{frame:06d}.jpg')
            try:
                session = predictor.handle_request({'type': 'start_session', 'resource_path': tmp,
                    'offload_video_to_cpu': True, 'offload_state_to_cpu': True})['session_id']
                out = predictor.handle_request({'type': 'add_prompt', 'session_id': session,
                    'frame_index': 0, 'text': 'person', 'output_prob_thresh': .5})['outputs']
                records = []
                for b, score in zip(out.get('out_boxes_xywh', []), out.get('out_probs', [])):
                    x, y, w, h = map(float, b)
                    records.append({'bbox': [x*index['width'], y*index['height'],
                        (x+w)*index['width'], (y+h)*index['height']], 'score': float(score)})
                result = {'frame': frame, 'prompt': 'person', 'box_prompt_used': False,
                          'detections': records, 'status': 'complete', 'created_at': time.time()}
            except Exception as exc:
                result = {'frame': frame, 'prompt': 'person', 'box_prompt_used': False,
                          'detections': [], 'status': 'error', 'error': str(exc)}
            finally:
                if session:
                    predictor.handle_request({'type': 'close_session', 'session_id': session})
        self.person_checks[key][str(frame)] = result
        atomic(path, self.person_checks[key])
        return result

    def propagate(self, key, start, end, seed_box, reverse=False):
        """Direct instance box prompt (SAM corner labels 2/3); no text-detection gate."""
        predictor = self.sam(); index = self.store.index(key)
        width, height = index['width'], index['height']; session = None
        with tempfile.TemporaryDirectory(prefix='forest_sar_review_') as tmp:
            for local, n in enumerate(range(start, end+1)):
                (Path(tmp)/f'{local:06d}.jpg').symlink_to(
                    self.store.data/'FRAMES'/key/f'{Path(key).name}_{n:06d}.jpg')
            try:
                session = predictor.handle_request({'type': 'start_session', 'resource_path': tmp,
                    'offload_video_to_cpu': True, 'offload_state_to_cpu': True})['session_id']
                seed_idx = end-start if reverse else 0
                x0, y0, x1, y1 = seed_box
                prompt = predictor.handle_request({'type': 'add_prompt', 'session_id': session,
                    'frame_index': seed_idx, 'obj_id': 0,
                    'points': [[x0/width, y0/height], [x1/width, y1/height]],
                    'point_labels': [2, 3], 'rel_coordinates': True})
                records = {}
                for response in predictor.handle_stream_request({'type': 'propagate_in_video',
                    'session_id': session, 'propagation_direction': 'backward' if reverse else 'forward',
                    'start_frame_index': seed_idx, 'max_frame_num_to_track': end-start+1,
                    'force_tracker_propagation': True, 'evict_cached_frame_outputs': True}):
                    out = response['outputs']; n = start+int(response['frame_index'])
                    for ident, mask in zip(out.get('out_obj_ids', []), out.get('out_binary_masks', [])):
                        if int(ident) != 0:
                            continue
                        ys, xs = np.nonzero(mask)
                        if len(xs):
                            records[n] = {'bbox': [float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1)],
                                          'mask_area': int(len(xs))}
                return records
            finally:
                if session:
                    predictor.handle_request({'type': 'close_session', 'session_id': session})

    def track(self, key, frame, bbox, instance_id=None, origin='manual_box', seed_token=None, verify=True):
        index = self.store.initialize(key); frame = int(frame)
        bbox = list(map(float, bbox))
        if not (1 <= frame <= index['frames'] and len(bbox) == 4 and np.isfinite(bbox).all()
                and 0 <= bbox[0] < bbox[2] <= index['width'] and 0 <= bbox[1] < bbox[3] <= index['height']):
            raise ValueError('Invalid seed frame or bounding box.')
        end = min(frame+30, index['frames'])
        person_check = self.person_check(key, frame)
        person_confirmed = (any(iou(bbox, r['bbox']) >= .3 for r in person_check['detections'])
                            if person_check['status'] == 'complete' else None)
        self.progress(f'Forward tracking frame {frame} → {end}')
        forward = self.propagate(key, frame, end, bbox)
        backward = {}; backward_status = 'no_independent_endpoint_seed'; backward_seed = None
        if verify and end in forward and end > frame:
            refs = [{'bbox': r['bbox'], 'origin': 'human_reference', 'score': None}
                    for r in index['anchor_references'].get(str(end), [])]
            refs += [{'bbox': r[:4], 'origin': 'deim', 'score': float(r[4])} for r in self.candidates(key, end)]
            eligible = [(iou(forward[end]['bbox'], r['bbox']), r) for r in refs]
            eligible.sort(key=lambda pair: pair[0], reverse=True)
            if eligible and eligible[0][0] >= .15:
                self.progress(f'Backward verification frame {end} → {frame}')
                backward_seed = eligible[0][1]
                try:
                    backward = self.propagate(key, frame, end, backward_seed['bbox'], reverse=True)
                    backward_status = 'independent_endpoint_box'
                except Exception as exc:
                    backward_status = f'verification_error:{type(exc).__name__}:{exc}'
        segment = uuid.uuid4().hex
        evidence = {'seed_frame': frame, 'end_frame': end, 'seed_box': bbox, 'seed_origin': origin,
            'seed_token': seed_token, 'forward': forward, 'backward': backward,
            'seed_person_confirmed': person_confirmed, 'person_check': person_check,
            'backward_status': backward_status, 'backward_seed': backward_seed, 'forward_is_never_gated_by_backward': True,
            'seed_mask_present': frame in forward, 'seed_mask_iou': iou(bbox, forward[frame]['bbox']) if frame in forward else None,
            'checkpoint': str(SETTINGS.sam3_checkpoint)}
        with self.store.locked(key):
            index = self.store.index(key)
            if seed_token and seed_token in index['completed_auto_seeds']:
                return {'skipped': True, 'reason': 'completed seed'}
            current = self.store.doc(key, frame)
            if instance_id:
                instance_id = self.store.resolve_instance(index, instance_id)
                if instance_id not in index['instances'] or index['instances'][instance_id]['status'] == 'deleted':
                    raise ValueError('Cannot extend an unknown or deleted instance.')
            else:
                matches = sorted([(iou(box(s), bbox), s.get('sar', {}).get('instance_id')) for s in current['shapes']
                                  if s.get('sar', {}).get('instance_id') and index['instances'][s['sar']['instance_id']]['status'] != 'deleted'],
                                 key=lambda item: item[0], reverse=True)
                instance_id = matches[0][1] if matches and matches[0][0] >= .3 else None
                if instance_id is None:
                    # A newly drawn earlier seed can overlap a known future segment.
                    # Require agreement in three frames before adopting that identity.
                    votes = {}
                    for n, candidate in forward.items():
                        for s in self.store.doc(key, n)['shapes']:
                            ident = s.get('sar', {}).get('instance_id')
                            if ident and index['instances'][ident]['status'] != 'deleted' and iou(candidate['bbox'], box(s)) >= .5:
                                votes.setdefault(ident, set()).add(n)
                    eligible = sorted([(len(fs), ident) for ident, fs in votes.items() if len(fs) >= 3], reverse=True)
                    if eligible and (len(eligible) == 1 or eligible[0][0] > eligible[1][0]):
                        instance_id = eligible[0][1]
            # Probe the full one-second track before admitting an automatic instance.
            # Once its family has strong evidence, fading per-frame scores cannot end it.
            observations = {n: r['bbox'] for n, r in forward.items()}
            if instance_id:
                existing_shapes = []
                for n in range(1, index['frames']+1):
                    matches = [s for s in self.store.doc(key, n)['shapes']
                               if s.get('sar', {}).get('instance_id') == instance_id]
                    existing_shapes.extend(matches)
                    if matches:
                        observations.setdefault(n, box(matches[0]))
                human_protected = protected(index['instances'][instance_id], existing_shapes)
            else:
                human_protected = False
            admission = instance_evidence(observations, self.raw_predictions(key), index['width'], index['height'])
            if origin != 'manual_box' and not human_protected and not admission['passes']:
                evidence.update(admission=admission, policy=POLICY)
                atomic(self.store.metadata(key)/'filtered_probes'/f'{segment}.json', evidence)
                if seed_token:
                    index['completed_auto_seeds'].append(seed_token)
                    self.store.commit(key, {}, index, 'hold automatic seed failing instance admission')
                return {'skipped': True, 'reason': admission['reasons'], 'start_frame': frame,
                        'end_frame': end, 'admission': admission}
            if instance_id is None:
                instance_id = self.store.new_instance(index, frame, bbox, origin)
            instance = index['instances'][instance_id]
            instance['segments'].append(segment)
            if person_confirmed:
                instance.setdefault('person_confirmed_seed_frames', []).append(frame)
            changed = {}; added_unreviewed = False
            for n in sorted({frame, *forward}):
                doc = self.store.doc(key, n)
                raw_box = forward.get(n, {}).get('bbox', bbox)
                if origin != 'manual_box' and self.store.rejected(index, n, raw_box):
                    continue
                candidates = self.candidates(key, n)
                best = max(candidates, key=lambda r: iou(raw_box, r[:4]), default=None)
                overlap = iou(raw_box, best[:4]) if best is not None else 0.
                deim_verified = overlap >= .3
                back_overlap = iou(raw_box, backward[n]['bbox']) if n in backward else None
                # Preserve reviewed seed geometry; other frames use every nonempty forward mask.
                proposed = bbox if n == frame else raw_box
                shapes = doc['shapes']
                same = next((s for s in shapes if s.get('sar', {}).get('instance_id') == instance_id), None)
                if (origin != 'manual_box' and same and
                        (same.get('sar', {}).get('review_state') == 'accepted' or same.get('sar', {}).get('manual_geometry'))):
                    continue
                seed_original = same and same.get('sar', {}).get('origin') in ('human_reference', 'manual_box')
                if n == frame or not seed_original:
                    item = self.store.shape(index, instance_id, proposed, origin if n == frame else 'sam3_forward',
                        review_state='pending', segment_id=segment, seed_frame=frame, seed_box=bbox, forward_present=n in forward,
                        mask_area=forward.get(n, {}).get('mask_area'),
                        deim_verified=deim_verified, deim_score=float(best[4]) if best is not None else None,
                        deim_iou=overlap, backward_verified=back_overlap is not None and back_overlap >= .3,
                        backward_iou=back_overlap, backward_status=backward_status,
                        missing_human_reference=(str(n) in index['anchor_references'] and
                            not any(iou(proposed, r['bbox']) >= .3 for r in index['anchor_references'][str(n)])),
                        seed_mask_iou=evidence['seed_mask_iou'], seed_token=seed_token)
                    item['sar']['seed_person_confirmed'] = person_confirmed
                    # Adopt an unregistered user-drawn seed instead of duplicating it.
                    if n == frame:
                        shapes[:] = [s for s in shapes if s.get('sar', {}).get('instance_id') or iou(box(s), bbox) < .99]
                    if same in shapes:
                        shapes.remove(same)
                    shapes.append(item)
                    added_unreviewed = True
                changed[n] = doc
            if added_unreviewed:
                instance['status'] = 'pending'
            if seed_token:
                index['completed_auto_seeds'].append(seed_token)
            op = self.store.commit(key, changed, index, f'track instance {instance_id} from frame {frame}')
            evidence.update(instance_id=instance_id, segment_id=segment, operation=op)
            atomic(self.store.metadata(key)/'segments'/f'{segment}.json', evidence)
        return {'instance_id': instance_id, 'segment_id': segment, 'start_frame': frame, 'end_frame': end,
                'forward_frames': len(forward), 'backward_frames': len(backward), 'backward_status': backward_status,
                'seed_mask_iou': evidence['seed_mask_iou'], 'operation': op}

    def generate(self, key, refresh=True, verify=True):
        index = self.store.initialize(key)
        if refresh:
            self.refresh_anchors(key)
        elif key not in self.anchor_raw:
            p = self.store.metadata(key)/'deim_anchor_predictions.npz'
            if p.exists():
                with np.load(p) as cache:
                    self.anchor_raw[key] = {int(n): cache[n] for n in cache.files}
        starts = sorted({1, *(int(n) for n in index['anchor_references'])})
        if len(starts) == 1:
            starts = list(range(1, index['frames']+1, 30))
        for start in starts:
            self.person_check(key, start)
        results = []
        for start in starts:
            references = index['anchor_references'].get(str(start), [])
            seeds = [(r['bbox'], 'human_reference') for r in references]
            for r in self.candidates(key, start):
                if not any(iou(r[:4], b) >= .3 for b, _ in seeds):
                    seeds.append((r[:4], 'deim_seed'))
            for bbox, origin in seeds:
                index = self.store.index(key)
                token = f'{start}:{origin}:'+','.join(f'{x:.1f}' for x in bbox)
                if token in index['completed_auto_seeds'] or self.store.rejected(index, start, bbox):
                    continue
                results.append(self.track(key, start, bbox, origin=origin, seed_token=token, verify=verify))
        # Keep unmatched dense DEIM proposals at 0.1 for review, including sparse anchor frames.
        # Consecutive overlapping detector-only proposals share an instance; SAM tracks take priority.
        with self.store.locked(key):
            index = self.store.index(key); changed = {}; previous = []
            for n in range(1, index['frames']+1):
                doc = self.store.doc(key, n); now = []; used_previous = set()
                for r in self.candidates(key, n):
                    if self.store.rejected(index, n, r[:4]) or any(iou(box(s), r[:4]) >= .3 for s in doc['shapes']):
                        continue
                    matches = sorted([(iou(b, r[:4]), ident) for b, ident in previous if ident not in used_previous], reverse=True)
                    ident = matches[0][1] if matches and matches[0][0] >= .3 else self.store.new_instance(index, n, r[:4], 'deim_only')
                    used_previous.add(ident)
                    doc['shapes'].append(self.store.shape(index, ident, r[:4], 'deim_only',
                        deim_score=float(r[4]), deim_verified=False, backward_verified=False,
                        missing_human_reference=str(n) in index['anchor_references'] and
                            not any(iou(r[:4], ref['bbox']) >= .3 for ref in index['anchor_references'][str(n)])))
                    now.append((r[:4], ident)); changed[n] = doc
                previous = now
            self.store.commit(key, changed, index, 'retain unmatched dense DEIM proposals at 0.1')
        filtering = apply_instance_policy(self.store, key)
        report = {'sequence': key, 'threshold': .1, 'seconds_per_track': 1,
                  'segments_generated': [r for r in results if not r.get('skipped')],
                  'filtered_probes': [r for r in results if r.get('skipped')],
                  'forward_policy': 'all nonempty masks retained for admitted or human-controlled instances', 'backward_policy': 'verification only',
                  'instance_admission_policy': POLICY, 'instance_filtering': filtering,
                  'summary': self.store.summary(key), 'created_at': time.time()}
        atomic(self.store.metadata(key)/'generation_report.json', report)
        self.annotate_person_checks(key)
        self.store.write_review_queue(key)
        self.progress(f'Complete: {len(results)} new SAM3 segments')
        return report

    def annotate_person_checks(self, key):
        """Attach independent semantic evidence also to already generated/resumed segments."""
        checks = self.person_checks.get(key, {})
        with self.store.locked(key):
            index = self.store.index(key); changed = {}
            for instance in index['instances'].values():
                confirmed = set(instance.get('person_confirmed_seed_frames', []))
                for segment in instance['segments']:
                    p = self.store.metadata(key)/'segments'/f'{segment}.json'
                    if not p.exists():
                        continue
                    evidence = json.loads(p.read_text()); check = checks.get(str(evidence['seed_frame']))
                    if check and check['status'] == 'complete' and any(iou(evidence['seed_box'], r['bbox']) >= .3 for r in check['detections']):
                        confirmed.add(evidence['seed_frame'])
                instance['person_confirmed_seed_frames'] = sorted(confirmed)
            for n in range(1, index['frames']+1):
                doc = self.store.doc(key, n); dirty = False
                for shape in doc['shapes']:
                    meta = shape.get('sar', {}); check = checks.get(str(meta.get('seed_frame')))
                    if check and check['status'] == 'complete':
                        confirmed = any(iou(meta['seed_box'], r['bbox']) >= .3 for r in check['detections'])
                        if meta.get('seed_person_confirmed') != confirmed:
                            meta['seed_person_confirmed'] = confirmed; dirty = True
                if dirty:
                    changed[n] = doc
            self.store.commit(key, changed, index, 'attach independent SAM3 person verification')
