"""Sequence-local Labelme documents, instance lineage, and reversible operations."""
from __future__ import annotations

import copy
import csv
import fcntl
import hashlib
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .settings import ROOT, SETTINGS

WORKSPACE = ROOT
DATA = SETTINGS.data_root
DEFAULT_OUTPUT = SETTINGS.output_root


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    os.replace(tmp, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def box(shape):
    points = shape.get('points', [])
    if len(points) < 2:
        raise ValueError('Select a rectangle with at least two points.')
    xs, ys = zip(*points)
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return float(area/union) if union > 0 else 0.


class Store:
    def __init__(self, output=DEFAULT_OUTPUT, data=DATA):
        self.output = Path(output).absolute(); self.data = Path(data).absolute()

    def key(self, key):
        parts = Path(key).parts
        if len(parts) != 2 or any(p in ('..', '.', '') for p in parts) or Path(key).is_absolute():
            raise ValueError('Sequence key must be GROUP/SEQUENCE.')
        if not (self.data/'FRAMES'/key).is_dir():
            raise ValueError('Unknown source sequence.')
        return str(Path(*parts))

    def directory(self, key):
        return self.output/'annotations'/self.key(key)

    def metadata(self, key):
        return self.output/'metadata'/self.key(key)

    @contextmanager
    def locked(self, key):
        p = self.metadata(key)/'.lock'; p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield

    def index(self, key):
        return json.loads((self.metadata(key)/'instances.json').read_text())

    def path(self, key, frame):
        key = self.key(key); seq = Path(key).name
        return self.directory(key)/f'{seq}_{int(frame):06d}.json'

    def doc(self, key, frame):
        return json.loads(self.path(key, frame).read_text())

    def raw_path(self, key: str) -> Path:
        """Return the sequence-local dense DEIM cache path.

        Args:
            key: Source sequence in GROUP/SEQUENCE form.

        Returns:
            Path to the unthresholded XYXY/score NumPy array.
        """
        root = SETTINGS.raw_predictions or self.data/'raw_predictions'
        return root/f'{self.key(key)}.npy'

    def references(self, key: str) -> tuple[dict, str]:
        """Read optional sparse references without assuming complete human labels.

        Args:
            key: Source sequence in GROUP/SEQUENCE form.

        Returns:
            Frame-indexed XYXY references and the sequence split name.
        """
        manifest = SETTINGS.anchor_references or self.data/'anchor_references.json'
        if manifest.exists():
            row = json.loads(manifest.read_text()).get(key, {})
            return row.get('frames', {}), row.get('split', 'unknown')
        if SETTINGS.anchor_references:
            raise ValueError(f'Configured anchor reference file is missing: {manifest}')
        if not SETTINGS.anchor_alignment:
            return {}, 'unknown'
        if not SETTINGS.coco_annotations:
            raise ValueError('anchor_alignment requires coco_annotations.')
        audit = json.loads(SETTINGS.anchor_alignment.read_text())
        anchors = [a for a in audit['anchors']
                   if f"{a['group']}/{a['base_sequence']}" == key]
        if any(a.get('status') != 'exact_pixel_match' for a in anchors):
            raise ValueError('Sparse anchors must be verified against exact dense frames.')
        references = {}
        for split in sorted({a['split'] for a in anchors}):
            coco = json.loads((SETTINGS.coco_annotations/f'{split}.json').read_text())
            by_id = {}
            for annotation in coco['annotations']:
                by_id.setdefault(annotation['image_id'], []).append(annotation)
            for anchor in anchors:
                if anchor['split'] != split:
                    continue
                references[str(anchor['dense_frame_1based'])] = [
                    {'bbox': [b['bbox'][0], b['bbox'][1],
                              b['bbox'][0]+b['bbox'][2], b['bbox'][1]+b['bbox'][3]],
                     'annotation_id': b['id'], 'coco_image_id': anchor['coco_image_id'],
                     'split': split}
                    for b in by_id.get(anchor['coco_image_id'], [])]
        return references, anchors[0]['split'] if anchors else 'unknown'

    def initialize(self, key):
        key = self.key(key)
        with self.locked(key):
            if (self.metadata(key)/'instances.json').exists():
                # Finish rollback of any process interrupted during a multi-file write.
                for p in sorted((self.metadata(key)/'transactions').glob('*/operation.json')):
                    operation = json.loads(p.read_text())
                    if operation['status'] != 'applying':
                        continue
                    before = json.loads((p.parent/'before.json').read_text())
                    for n, doc in before.items():
                        atomic(self.path(key, n), doc)
                    atomic(self.metadata(key)/'instances.json', json.loads((p.parent/'index_before.json').read_text()))
                    operation['status'] = 'rolled_back_after_interruption'; atomic(p, operation)
                return self.index(key)
            from PIL import Image
            files = sorted((self.data/'FRAMES'/key).glob('*.jpg'))
            if not files:
                raise ValueError('No source JPEGs.')
            frames = [int(re.search(r'_(\d+)$', f.stem)[1]) for f in files]
            if frames != list(range(1, len(files)+1)):
                raise ValueError('Expected contiguous one-based source frames.')
            with Image.open(files[0]) as im:
                width, height = im.size
            references, split = self.references(key)
            directory = self.directory(key); directory.mkdir(parents=True, exist_ok=True)
            for n, image in zip(frames, files):
                link = directory/image.name
                if not os.path.lexists(link):
                    link.symlink_to(os.path.relpath(image, directory))
                p = self.path(key, n)
                if p.exists():
                    raise RuntimeError(f'Unregistered review annotations already exist: {p}')
                atomic(p, {'version': '5.5.0', 'flags': {'human_verified': False}, 'shapes': [],
                    'imagePath': image.name, 'imageData': None, 'imageWidth': width, 'imageHeight': height,
                    'description': 'Editable review proposals; empty shapes are not verified negatives.',
                    'human_reference': references.get(str(n), []),
                    'sar_review': {'sequence': key, 'frame': n, 'state': 'pending'}})
            index = {'sequence': key, 'frames': len(files), 'width': width, 'height': height,
                'source_fps': 30, 'anchor_references': references, 'instances': {}, 'next_group_id': 1,
                'rejected_observations': [], 'revision': 0, 'created_at': time.time(),
                'split': split, 'completed_auto_seeds': []}
            atomic(self.metadata(key)/'instances.json', index)
            return index

    def new_instance(self, index, frame, bbox, origin):
        ident = uuid.uuid4().hex
        index['instances'][ident] = {'instance_id': ident, 'group_id': index['next_group_id'],
            'seed_frame': int(frame), 'seed_box': bbox, 'origin': origin, 'status': 'pending',
            'segments': [], 'created_at': time.time()}
        index['next_group_id'] += 1
        return ident

    @staticmethod
    def resolve_instance(index, ident):
        seen = set()
        while ident in index['instances'] and index['instances'][ident].get('alias_of'):
            if ident in seen:
                raise RuntimeError('Cyclic instance lineage.')
            seen.add(ident); ident = index['instances'][ident]['alias_of']
        return ident

    def shape(self, index, ident, bbox, origin, **metadata):
        x0, y0, x1, y1 = bbox
        return {'label': 'person', 'shape_type': 'rectangle',
            'points': [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            'group_id': index['instances'][ident]['group_id'], 'description': f'{origin}; review pending',
            'flags': {'needs_review': True, 'human_verified': False},
            'sar': {'instance_id': ident, 'shape_id': uuid.uuid4().hex, 'origin': origin,
                    'review_state': index['instances'][ident]['status'], **metadata}}

    def commit(self, key, docs, index, title):
        """Called under the sequence lock. Before-images permit checked undo."""
        op = f'{time.time_ns()}_{uuid.uuid4().hex[:8]}'
        directory = self.metadata(key)/'transactions'/op; directory.mkdir(parents=True)
        old = self.index(key)
        atomic(directory/'index_before.json', old)
        before, after = {}, {}
        for n, doc in docs.items():
            path = self.path(key, n)
            before[str(n)] = json.loads(path.read_text())
        atomic(directory/'before.json', before)
        atomic(directory/'operation.json', {'title': title, 'status': 'applying', 'frames': list(docs)})
        for n, doc in docs.items():
            atomic(self.path(key, n), doc); after[str(n)] = digest(self.path(key, n))
        index['revision'] = old['revision']+1
        atomic(self.metadata(key)/'instances.json', index)
        atomic(directory/'operation.json', {'title': title, 'status': 'complete', 'frames': list(docs),
            'after_hashes': after, 'index_after_hash': digest(self.metadata(key)/'instances.json')})
        return op

    def instance_action(self, key, ident, action):
        if action not in ('delete', 'accept'):
            raise ValueError('Unknown instance action.')
        with self.locked(key):
            index = self.index(key)
            ident = self.resolve_instance(index, ident)
            if ident not in index['instances']:
                raise ValueError('Unknown instance ID.')
            changed = {}; removed = 0
            for n in range(1, index['frames']+1):
                doc = self.doc(key, n)
                matches = [s for s in doc['shapes'] if s.get('sar', {}).get('instance_id') == ident]
                if not matches:
                    continue
                removed += len(matches)
                if action == 'delete':
                    for s in matches:
                        index['rejected_observations'].append({'frame': n, 'bbox': box(s), 'instance_id': ident})
                    doc['shapes'] = [s for s in doc['shapes'] if s not in matches]
                else:
                    for s in matches:
                        s['sar']['review_state'] = 'accepted'
                        s['flags'].update(needs_review=False, human_verified=True)
                        s['description'] = 'Instance accepted by reviewer'
                changed[n] = doc
            index['instances'][ident]['status'] = 'deleted' if action == 'delete' else 'accepted'
            op = self.commit(key, changed, index, f'{action} instance {ident}')
            return {'instance_id': ident, 'affected_shapes': removed, 'affected_frames': len(changed), 'operation': op}

    def undo(self, key):
        with self.locked(key):
            candidates = sorted((self.metadata(key)/'transactions').glob('*/operation.json'), reverse=True)
            for p in candidates:
                record = json.loads(p.read_text())
                if record['status'] != 'complete':
                    continue
                if digest(self.metadata(key)/'instances.json') != record['index_after_hash']:
                    raise RuntimeError('Instance state changed after this operation; undo would overwrite newer work.')
                for n, expected in record['after_hashes'].items():
                    if digest(self.path(key, n)) != expected:
                        raise RuntimeError(f'Frame {n} was edited after this operation; refusing to overwrite it.')
                before = json.loads((p.parent/'before.json').read_text())
                for n, doc in before.items():
                    atomic(self.path(key, n), doc)
                atomic(self.metadata(key)/'instances.json', json.loads((p.parent/'index_before.json').read_text()))
                record['status'] = 'undone'; atomic(p, record)
                return {'undone': record['title'], 'frames': len(before)}
            raise ValueError('No operation available to undo.')

    def summary(self, key):
        index = self.index(key); counts = {}; support = {}; verification = {}; mismatches = {}; detector_only = {}; forward_only = {}; pending = set()
        for n in range(1, index['frames']+1):
            for s in self.doc(key, n)['shapes']:
                meta = s.get('sar', {}); ident = meta.get('instance_id')
                if ident:
                    if meta.get('review_state') != 'accepted':
                        pending.add(ident)
                    counts.setdefault(ident, set()).add(n)
                    support[ident] = support.get(ident, 0)+int(meta.get('deim_verified', False))
                    verification[ident] = verification.get(ident, 0)+int(meta.get('backward_verified', False))
                    if meta.get('missing_human_reference'):
                        mismatches.setdefault(ident, []).append(n)
                    detector_only[ident] = detector_only.get(ident, 0)+int(meta.get('origin') == 'deim_only')
                    forward_only[ident] = forward_only.get(ident, 0)+int(meta.get('forward_present', False) and not meta.get('backward_verified', False))
        rows = []
        for ident, instance in index['instances'].items():
            fs = sorted(counts.get(ident, []))
            rows.append({**instance, 'status': 'pending' if instance['status'] == 'accepted' and ident in pending else instance['status'],
                         'frame_count': len(fs), 'first_frame': fs[0] if fs else instance['seed_frame'],
                         'last_frame': fs[-1] if fs else instance['seed_frame'],
                         'deim_supported_frames': support.get(ident, 0), 'backward_verified_frames': verification.get(ident, 0),
                         'anchor_disagreement_frames': mismatches.get(ident, []),
                         'detector_only_frames': detector_only.get(ident, 0), 'forward_only_frames': forward_only.get(ident, 0)})
        return {'sequence': key, 'frames': index['frames'], 'directory': str(self.directory(key)),
                'revision': index['revision'], 'split': index['split'], 'instances': rows}

    def write_review_queue(self, key):
        rows = []
        for item in self.summary(key)['instances']:
            if item['status'] == 'deleted' or not item['frame_count']:
                continue
            reasons = []
            if item['anchor_disagreement_frames']:
                reasons.append('missing_from_human_anchor')
            if item['origin'] in ('deim_only', 'deim_seed'):
                reasons.append('detector_proposal')
            if item['forward_only_frames']:
                reasons.append('forward_without_backward_confirmation')
            if not item['deim_supported_frames']:
                reasons.append('no_detector_confirmation')
            if not item.get('person_confirmed_seed_frames'):
                reasons.append('no_independent_sam3_person_confirmation')
            rows.append({'instance_id': item['instance_id'], 'group_id': item['group_id'], 'state': item['status'],
                'first_frame': item['first_frame'], 'last_frame': item['last_frame'], 'frame_count': item['frame_count'],
                'origin': item['origin'], 'anchor_disagreement_frames': ','.join(map(str, item['anchor_disagreement_frames'])),
                'deim_supported_frames': item['deim_supported_frames'], 'backward_verified_frames': item['backward_verified_frames'],
                'review_reasons': ';'.join(reasons)})
        rows.sort(key=lambda row: (row['state'] == 'accepted', not bool(row['anchor_disagreement_frames']), row['group_id']))
        path = self.metadata(key)/'review_queue.csv'
        if rows:
            with path.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        else:
            path.write_text('instance_id,group_id,state\n')
        return str(path)

    @staticmethod
    def rejected(index, frame, bbox):
        return any(r['frame'] == frame and iou(r['bbox'], bbox) >= .5 for r in index['rejected_observations'])
