"""Portable configuration and sparse-anchor import behavior."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from .settings import Settings, load_settings
from .store import Store, atomic


class ConfigurationTests(unittest.TestCase):
    def test_relative_paths_resolve_against_config_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp)/'config.json'
            config.write_text(json.dumps({'data_root': 'frames',
                                          'sam3_checkpoint': 'weights/sam3.pt'}))
            settings = load_settings(config)
            self.assertEqual(settings.data_root, Path(tmp)/'frames')
            self.assertEqual(settings.sam3_checkpoint, Path(tmp)/'weights/sam3.pt')

    def test_typo_in_configuration_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp)/'config.json'
            config.write_text('{"data_rooot": "frames"}')
            with self.assertRaisesRegex(ValueError, 'data_rooot'):
                load_settings(config)

    def test_explicit_missing_configuration_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            load_settings('/nonexistent/forest_sar.json')


class ReferenceImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.key = 'group/clip'
        frames = self.root/'data/FRAMES'/self.key
        frames.mkdir(parents=True)
        for n in range(1, 4):
            Image.new('RGB', (100, 80)).save(frames/f'clip_{n:06d}.jpg')
        self.settings = Settings(self.root/'data', self.root/'review')
        self.override = patch('annotation_workbench.store.SETTINGS', self.settings)
        self.override.start()
        self.store = Store(self.settings.output_root, self.settings.data_root)

    def tearDown(self):
        self.override.stop()
        self.temp.cleanup()

    def test_missing_optional_references_do_not_require_private_files(self):
        index = self.store.initialize(self.key)
        self.assertEqual(index['anchor_references'], {})
        self.assertTrue((self.store.directory(self.key)/'clip_000001.jpg').is_file())

    def test_empty_anchor_is_distinct_from_an_unlabeled_frame(self):
        atomic(self.settings.data_root/'anchor_references.json', {
            self.key: {'split': 'train', 'frames': {
                '1': [{'bbox': [10, 10, 25, 30]}], '3': []}}})
        index = self.store.initialize(self.key)
        self.assertIn('3', index['anchor_references'])
        self.assertNotIn('2', index['anchor_references'])
        self.assertEqual(self.store.doc(self.key, 1)['human_reference'][0]['bbox'],
                         [10, 10, 25, 30])
        self.assertEqual(index['split'], 'train')

    def test_unaligned_legacy_anchors_are_rejected(self):
        alignment = self.root/'alignment.json'
        atomic(alignment, {'anchors': [{'group': 'group', 'base_sequence': 'clip',
                                       'status': 'approximate'}]})
        settings = Settings(self.settings.data_root, self.settings.output_root,
                            anchor_alignment=alignment, coco_annotations=self.root)
        with patch('annotation_workbench.store.SETTINGS', settings):
            with self.assertRaisesRegex(ValueError, 'exact dense frames'):
                self.store.initialize(self.key)


if __name__ == '__main__':
    unittest.main()
