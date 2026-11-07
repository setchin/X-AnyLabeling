"""Behavioral checks for cascaded deletion, undo, sparse references and forward retention."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from .engine import Engine
from .store import Store, atomic, box


class FakeEngine(Engine):
    def person_check(self, key, frame):
        return {'frame': frame, 'detections': [], 'status': 'complete'}

    def propagate(self, key, start, end, seed_box, reverse=False):
        if reverse:
            return {}
        return {n: {'bbox': list(seed_box), 'mask_area': 100} for n in range(start, end+1)}


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); root = Path(self.tmp.name)
        self.key = 'test_group/test_sequence'
        directory = root/'data/FRAMES'/self.key; directory.mkdir(parents=True)
        for n in range(1, 7):
            Image.new('RGB', (100, 100)).save(directory/f'test_sequence_{n:06d}.jpg')
        self.store = Store(root/'output', root/'data'); self.store.initialize(self.key)
        self.engine = FakeEngine(self.store)
        self.engine.raw[self.key] = np.zeros((6, 1, 5))

    def tearDown(self):
        self.tmp.cleanup()

    def test_forward_survives_missing_backward_and_delete_undo(self):
        result = self.engine.track(self.key, 2, [10, 10, 30, 40])
        ident = result['instance_id']
        self.assertEqual(result['forward_frames'], 5)
        self.assertEqual(result['backward_frames'], 0)
        for n in range(2, 7):
            self.assertEqual(self.store.doc(self.key, n)['shapes'][0]['sar']['instance_id'], ident)
        other = self.engine.track(self.key, 3, [60, 60, 80, 90])['instance_id']
        removed = self.store.instance_action(self.key, ident, 'delete')
        self.assertEqual(removed['affected_frames'], 5)
        self.assertTrue(self.store.rejected(self.store.index(self.key), 2, [10, 10, 30, 40]))
        for n in range(3, 7):
            self.assertEqual([s['sar']['instance_id'] for s in self.store.doc(self.key, n)['shapes']], [other])
        self.store.undo(self.key)
        self.assertEqual(self.store.index(self.key)['instances'][ident]['status'], 'pending')
        self.assertFalse(self.store.index(self.key)['rejected_observations'])

    def test_arbitrary_frame_extension_keeps_identity(self):
        first = self.engine.track(self.key, 1, [10, 10, 30, 40])
        second = self.engine.track(self.key, 3, [10, 10, 30, 40], instance_id=first['instance_id'])
        self.assertEqual(first['instance_id'], second['instance_id'])
        self.assertEqual(len(self.store.index(self.key)['instances'][first['instance_id']]['segments']), 2)
        for n in range(1, 7):
            self.assertEqual(len(self.store.doc(self.key, n)['shapes']), 1)

    def test_undo_detects_later_manual_edit(self):
        self.engine.track(self.key, 2, [10, 10, 30, 40])
        doc = self.store.doc(self.key, 3); doc['description'] = 'manual correction'
        atomic(self.store.path(self.key, 3), doc)
        with self.assertRaisesRegex(RuntimeError, 'edited'):
            self.store.undo(self.key)

    def test_missing_human_instance_is_retained_at_anchor(self):
        index = self.store.index(self.key); index['anchor_references']['3'] = []
        atomic(self.store.metadata(self.key)/'instances.json', index)
        self.engine.track(self.key, 1, [10, 10, 30, 40])
        shape = self.store.doc(self.key, 3)['shapes'][0]
        self.assertTrue(shape['sar']['missing_human_reference'])
        self.assertEqual(box(shape), [10, 10, 30, 40])

    def test_earlier_seed_can_adopt_future_instance(self):
        future = self.engine.track(self.key, 3, [10, 10, 30, 40])
        earlier = self.engine.track(self.key, 1, [10, 10, 30, 40])
        self.assertEqual(future['instance_id'], earlier['instance_id'])

    def test_automatic_tracking_respects_rejected_observations(self):
        ident = self.engine.track(self.key, 3, [10, 10, 30, 40])['instance_id']
        self.store.instance_action(self.key, ident, 'delete')
        self.engine.track(self.key, 1, [10, 10, 30, 40], origin='human_reference')
        for n in range(3, 7):
            self.assertFalse(self.store.doc(self.key, n)['shapes'])

    def test_automatic_regeneration_preserves_accepted_geometry(self):
        ident = self.engine.track(self.key, 1, [10, 10, 30, 40])['instance_id']
        self.store.instance_action(self.key, ident, 'accept')
        before = self.store.doc(self.key, 3)['shapes'][0]
        self.engine.track(self.key, 1, [11, 10, 31, 40], instance_id=ident, origin='human_reference')
        self.assertEqual(self.store.doc(self.key, 3)['shapes'][0], before)
        self.assertEqual(self.store.index(self.key)['instances'][ident]['status'], 'accepted')

    def test_interrupted_operation_recovers_from_before_images(self):
        ident = self.engine.track(self.key, 2, [10, 10, 30, 40])['instance_id']
        deleted = self.store.instance_action(self.key, ident, 'delete')
        path = self.store.metadata(self.key)/'transactions'/deleted['operation']/'operation.json'
        record = json.loads(path.read_text()); record['status'] = 'applying'; atomic(path, record)
        self.store.initialize(self.key)
        self.assertEqual(self.store.doc(self.key, 3)['shapes'][0]['sar']['instance_id'], ident)
        self.assertEqual(json.loads(path.read_text())['status'], 'rolled_back_after_interruption')

    def test_manual_extension_reopens_new_proposals(self):
        ident = self.engine.track(self.key, 1, [10, 10, 30, 40])['instance_id']
        self.store.instance_action(self.key, ident, 'accept')
        self.engine.track(self.key, 3, [11, 10, 31, 40], instance_id=ident)
        self.assertEqual(self.store.doc(self.key, 1)['shapes'][0]['sar']['review_state'], 'accepted')
        self.assertEqual(self.store.doc(self.key, 3)['shapes'][0]['sar']['review_state'], 'pending')
        self.assertEqual(self.store.summary(self.key)['instances'][0]['status'], 'pending')


if __name__ == '__main__':
    unittest.main()
