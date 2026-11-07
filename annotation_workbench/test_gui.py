"""Opt-in actual Qt widget checks using synthetic images and a fake GPU worker."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from .store import Store, box
from .test_workbench import FakeEngine


class LocalTestClient:
    """Exercise the review dock's job protocol without loading GPU models."""

    def __init__(self, store: Store, key: str) -> None:
        self.store = store
        self.engine = FakeEngine(store)
        self.engine.raw[key] = np.zeros((3, 1, 5))
        self.record = None

    def request(self, path: str, payload: dict | None = None) -> dict:
        """Run a fake tracking job or return an actual review-store summary."""
        if path.startswith('/summary?'):
            key = parse_qs(urlparse(path).query)['sequence'][0]
            return self.store.summary(key)
        if path.startswith('/jobs/'):
            return self.record
        key = payload['sequence']
        action = path.strip('/')
        if action == 'track':
            result = self.engine.track(key, payload['frame'], payload['bbox'],
                                       instance_id=payload.get('instance_id'))
        elif action in ('delete', 'accept'):
            result = self.store.instance_action(key, payload['instance_id'], action)
        elif action == 'undo':
            result = self.store.undo(key)
        else:
            raise ValueError(action)
        self.record = {'state': 'complete', 'action': action, 'result': result}
        return {'job_id': 'synthetic-worker-job'}


@unittest.skipUnless(os.environ.get('FOREST_SAR_RUN_GUI_TESTS') == '1',
                     'Set FOREST_SAR_RUN_GUI_TESTS=1 with Qt available.')
class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from .xanylabeling_addon import QtCore, QtWidgets
        QtCore.QCoreApplication.setAttribute(
            QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        from .xanylabeling_addon import create_window
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.key = 'group/clip'
        directory = self.root/'data/FRAMES'/self.key
        directory.mkdir(parents=True)
        for n in range(1, 4):
            Image.new('RGB', (100, 80), 'red').save(directory/f'clip_{n:06d}.jpg')
        self.store = Store(self.root/'review', self.root/'data')
        self.store.initialize(self.key)
        self.window = create_window(self.store.output, self.key,
                                    LocalTestClient(self.store, self.key))
        self.window.forest_sar.timer.stop()
        self.view = self.window.labeling_widget.view
        self.view.load_file(str(self.store.directory(self.key)/'clip_000001.jpg'))

    def tearDown(self):
        self.view.set_clean()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def draw(self):
        from .xanylabeling_addon import QtCore
        from anylabeling.views.labeling.shape import Shape
        shape = Shape(label='person', shape_type='rectangle')
        for x, y in ((10, 10), (30, 10), (30, 40), (10, 40)):
            shape.add_point(QtCore.QPointF(x, y))
        shape.close()
        self.view.load_shapes([shape], replace=True)
        self.view.canvas.select_shapes([shape])
        self.view.set_dirty()

    def test_track_save_delete_undo_and_accept_through_dock(self):
        self.draw()
        dock = self.window.forest_sar
        dock.track_selected()
        dock.poll()
        self.assertIsNotNone(dock.last_result, dock.status.text())
        ident = dock.last_result['instance_id']
        self.assertEqual(dock.last_result['forward_frames'], 3)
        self.assertEqual(self.view.canvas.shapes[0].other_data['sar']['instance_id'], ident)
        self.view.canvas.select_shapes([self.view.canvas.shapes[0]])
        dock.instance_action('delete')
        dock.poll()
        self.assertFalse(self.store.doc(self.key, 2)['shapes'])
        dock.undo()
        dock.poll()
        self.assertEqual(self.store.doc(self.key, 2)['shapes'][0]['sar']['instance_id'], ident)
        self.view.canvas.select_shapes([self.view.canvas.shapes[0]])
        dock.instance_action('accept')
        dock.poll()
        self.assertEqual(self.store.index(self.key)['instances'][ident]['status'], 'accepted')

    def test_rgb_thermal_toggle_preserves_shapes_and_saved_source(self):
        rgb = self.root/'images/pair_rgb.jpg'
        thermal = self.root/'images_ir/pair_ir.jpg'
        rgb.parent.mkdir()
        thermal.parent.mkdir()
        Image.new('RGB', (100, 80), 'red').save(rgb)
        Image.new('RGB', (100, 80), 'blue').save(thermal)
        self.view.load_file(str(rgb))
        self.draw()
        original = self.view.image_data
        shapes = [s.to_dict() for s in self.view.canvas.shapes]
        self.view._config['store_data'] = True
        self.view.toggle_rgbt_background(True)
        self.assertEqual(self.view._display_image_mode, 'thermal')
        self.assertEqual(self.view.image_data, original)
        self.assertEqual([s.to_dict() for s in self.view.canvas.shapes], shapes)
        output = self.root/'pair.json'
        self.assertTrue(self.view.save_labels(str(output)))
        from anylabeling.views.labeling.label_file import LabelFile
        self.assertEqual(LabelFile(str(output)).image_data, original)
        self.assertEqual(self.view.filename, str(rgb))
        self.assertFalse(self.view._swap_background_image(self._different_size_image()))

    def _different_size_image(self):
        path = self.root/'different.jpg'
        Image.new('RGB', (50, 40)).save(path)
        return str(path)


if __name__ == '__main__':
    unittest.main()
