"""Instance confidence is historical; continuity gates only new automatic objects."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from .engine import Engine
from .quality import apply_instance_policy, evidence
from .store import Store, atomic
from .test_workbench import FakeEngine


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); root = Path(self.tmp.name)
        self.key = 'quality_test/clip'
        directory = root/'data/FRAMES'/self.key; directory.mkdir(parents=True)
        for n in range(1, 41):
            Image.new('RGB', (100, 100)).save(directory/f'clip_{n:06d}.jpg')
        self.store = Store(root/'output', root/'data'); self.store.initialize(self.key)
        self.raw = np.tile(np.array([[10, 10, 30, 40, .2], [60, 60, 80, 90, .2]], dtype=float), (40, 1, 1))
        self.path = self.store.raw_path(self.key)
        self.path.parent.mkdir(parents=True); self.save_raw()
        self.engine = FakeEngine(self.store); self.engine.raw[self.key] = self.raw

    def save_raw(self):
        np.save(self.path, self.raw)

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, start, end, bbox, accepted=False):
        with self.store.locked(self.key):
            index = self.store.index(self.key)
            ident = self.store.new_instance(index, start, bbox, 'deim_seed')
            docs = {}
            for n in range(start, end+1):
                doc = self.store.doc(self.key, n)
                doc['shapes'].append(self.store.shape(index, ident, bbox, 'sam3_forward'))
                docs[n] = doc
            self.store.commit(self.key, docs, index, 'test fixture')
        if accepted:
            self.store.instance_action(self.key, ident, 'accept')
        return ident

    def test_long_low_score_probe_never_creates_instance(self):
        result = self.engine.track(self.key, 1, [10, 10, 30, 40], origin='deim_seed')
        self.assertTrue(result['skipped'])
        self.assertIn('no_deim_score_above_0.5', result['reason'])
        self.assertFalse(self.store.index(self.key)['instances'])

    def test_high_score_establishes_track_and_fading_scores_are_retained(self):
        self.raw[0, 0, 4] = .8
        result = self.engine.track(self.key, 1, [10, 10, 30, 40], origin='deim_seed')
        ident = result['instance_id']
        self.assertEqual(result['forward_frames'], 31)
        self.assertEqual(self.store.doc(self.key, 31)['shapes'][0]['sar']['instance_id'], ident)
        continuation = self.engine.track(self.key, 32, [10, 10, 30, 40], instance_id=ident, origin='deim_seed')
        self.assertEqual(continuation['forward_frames'], 9)
        self.assertEqual(self.store.doc(self.key, 40)['shapes'][0]['sar']['instance_id'], ident)

    def test_short_high_score_probe_is_not_admitted(self):
        self.raw[0, 0, 4] = .8
        self.engine.propagate = lambda *args, **kwargs: {n: {'bbox': [10, 10, 30, 40], 'mask_area': 100} for n in range(1, 10)}
        result = self.engine.track(self.key, 1, [10, 10, 30, 40], origin='deim_seed')
        self.assertTrue(result['skipped'])
        self.assertIn('fewer_than_10_consecutive_observed_frames', result['reason'])

    def test_score_boundary_is_strict_and_ten_frames_is_sufficient(self):
        frames = {n: [10, 10, 30, 40] for n in range(1, 11)}
        self.raw[0, 0, 4] = .5
        self.assertFalse(evidence(frames, self.raw, 100, 100)['passes'])
        self.raw[0, 0, 4] = .5001
        self.assertTrue(evidence(frames, self.raw, 100, 100)['passes'])

    def test_unrelated_high_score_cannot_confirm_background_track(self):
        self.raw[:, 1, 4] = .9
        observations = {n: [10, 10, 30, 40] for n in range(1, 21)}
        result = evidence(observations, self.raw, 100, 100)
        self.assertFalse(result['passes'])
        self.assertLess(result['max_deim_score'], .5)

    def test_filter_is_reversible_and_preserves_strong_track_tail(self):
        self.raw[0, 0, 4] = .8; self.save_raw()
        true = self.add(1, 31, [10, 10, 30, 40]); false = self.add(1, 31, [60, 60, 80, 90])
        report = apply_instance_policy(self.store, self.key)
        self.assertEqual(report['instances_after'], 1)
        self.assertEqual(self.store.doc(self.key, 31)['shapes'][0]['sar']['instance_id'], true)
        self.assertEqual(self.store.index(self.key)['instances'][false]['status'], 'auto_filtered')
        self.store.undo(self.key)
        self.assertEqual(len(self.store.doc(self.key, 31)['shapes']), 2)

    def test_short_fragment_joins_unique_continuation_before_filtering(self):
        self.raw[0, 0, 4] = .8; self.raw[8, 0, 4] = .8; self.save_raw()
        first = self.add(1, 8, [10, 10, 30, 40]); second = self.add(9, 20, [10, 10, 30, 40])
        report = apply_instance_policy(self.store, self.key)
        self.assertEqual(report['instances_after'], 1)
        self.assertEqual(self.store.doc(self.key, 20)['shapes'][0]['sar']['instance_id'], first)
        self.assertEqual(self.store.index(self.key)['instances'][second]['alias_of'], first)

    def test_explicit_human_acceptance_is_preserved(self):
        ident = self.add(1, 5, [10, 10, 30, 40], accepted=True)
        report = apply_instance_policy(self.store, self.key)
        self.assertEqual(report['instances_after'], 1)
        self.assertTrue(report['retained_instances'][0]['protected_by_human_review'])
        self.assertEqual(self.store.index(self.key)['instances'][ident]['status'], 'accepted')

    def test_delete_using_linked_uuid_removes_whole_family(self):
        self.raw[0, 0, 4] = .8; self.raw[8, 0, 4] = .8; self.save_raw()
        first = self.add(1, 8, [10, 10, 30, 40]); second = self.add(9, 20, [10, 10, 30, 40])
        apply_instance_policy(self.store, self.key)
        result = self.store.instance_action(self.key, second, 'delete')
        self.assertEqual(result['instance_id'], first)
        self.assertEqual(result['affected_shapes'], 20)
        self.assertTrue(all(not self.store.doc(self.key, n)['shapes'] for n in range(1, 21)))


if __name__ == '__main__':
    unittest.main()
