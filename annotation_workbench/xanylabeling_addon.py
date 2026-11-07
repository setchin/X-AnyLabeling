"""Launch the installed X-AnyLabeling with an instance-review dock (no app patch)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .store import DEFAULT_OUTPUT, WORKSPACE, box
from .settings import SETTINGS

# Reuse the already installed GUI distribution without installing into pytorch.
if SETTINGS.gui_site:
    sys.path.append(str(SETTINGS.gui_site))
from PyQt6 import QtCore, QtGui, QtWidgets


class Client:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def request(self, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.endpoint['url']+path, data=data,
            headers={'Authorization': 'Bearer '+self.endpoint['token'], 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(json.load(exc).get('error', str(exc))) from exc


def ensure_backend(output, gpu=1, port=8793):
    endpoint_file = Path(output)/'metadata/backend.json'
    if endpoint_file.exists():
        try:
            endpoint = json.loads(endpoint_file.read_text()); Client(endpoint).request('/health')
            return endpoint
        except Exception:
            pass
    log = Path(output)/'metadata/backend.log'; log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'annotation_workbench.server',
            '--output', str(output), '--gpu', str(gpu), '--port', str(port)], cwd=WORKSPACE,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError(f'Inference worker failed to start; inspect {log}')
        if endpoint_file.exists():
            try:
                endpoint = json.loads(endpoint_file.read_text())
                if endpoint['pid'] == process.pid:
                    Client(endpoint).request('/health'); return endpoint
            except Exception:
                pass
        time.sleep(.1)
    raise RuntimeError(f'Inference worker startup timed out; inspect {log}')


class ReviewDock(QtWidgets.QDockWidget):
    def __init__(self, window, client, output):
        super().__init__('ForestSAR instance review', window)
        self.setObjectName('ForestSARReviewDock')
        self.window = window; self.view = window.labeling_widget.view
        self.client = client; self.output = Path(output).absolute()
        self.job = None; self.last_result = None; self.last_error = None; self.open_after = None
        self.sequence = None; self.items = []
        native_save = self.view.save_labels
        def save_with_provenance(filename):
            path = Path(filename)
            if path.exists():
                previous = json.loads(path.read_text())
                prior = {s.get('sar', {}).get('shape_id'): s for s in previous['shapes'] if s.get('sar', {}).get('shape_id')}
                for shape in self.view.canvas.shapes:
                    meta = shape.other_data.get('sar', {})
                    old = prior.get(meta.get('shape_id'))
                    if old and any(abs(a-b) > .001 for a, b in zip(box(old), box(shape.to_dict()))):
                        meta.update(manual_geometry=True, review_state='pending')
                        shape.flags.update(needs_review=True, human_verified=False)
            return native_save(filename)
        self.view.save_labels = save_with_provenance
        panel = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(panel)
        self.status = QtWidgets.QLabel('Ready'); self.status.setWordWrap(True); layout.addWidget(self.status)
        note = QtWidgets.QLabel('Automatic instances need DEIM > 0.5 somewhere and 10 consecutive observed frames.\nDraw/select a person box to track for 1 second; accepted tracks may continue under canopy.'); note.setWordWrap(True); layout.addWidget(note)
        self.buttons = []
        for text, callback in [('Open source sequence…', self.choose_sequence),
            ('Generate labels for this sequence', self.generate),
            ('Filter automatic instances', self.filter_instances),
            ('Track selected box → 1 second', self.track_selected),
            ('Delete entire selected instance', lambda: self.instance_action('delete')),
            ('Accept entire selected instance', lambda: self.instance_action('accept')),
            ('Undo last worker operation', self.undo), ('Refresh instance list', self.refresh)]:
            b = QtWidgets.QPushButton(text); b.clicked.connect(lambda checked=False, fn=callback: self.guard(fn))
            layout.addWidget(b); self.buttons.append(b)
        row = QtWidgets.QHBoxLayout(); self.frame = QtWidgets.QSpinBox(); self.frame.setRange(1, 1000000)
        row.addWidget(self.frame); go = QtWidgets.QPushButton('Go to frame'); go.clicked.connect(lambda: self.guard(lambda: self.go(self.frame.value())))
        row.addWidget(go); self.buttons.append(go); layout.addLayout(row)
        self.filter = QtWidgets.QComboBox()
        self.filter.addItems(['All instances', 'Pending review', 'Human-anchor disagreements', 'Detector-only proposals', 'Forward without backward verification'])
        self.filter.currentIndexChanged.connect(lambda: self.guard(self.refresh))
        layout.addWidget(self.filter)
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(['ID', 'Origin', 'State', 'Frames', 'DEIM', 'Back', 'Person'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.cellDoubleClicked.connect(lambda row, col: self.guard(lambda: self.jump_instance(row)))
        self.table.cellClicked.connect(lambda row, col: self.view.canvas.deselect_shape())
        self.table.currentCellChanged.connect(lambda row, col, oldrow, oldcol: self.describe(row))
        self.view.canvas.selection_changed.connect(self.canvas_selected)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)
        self.details = QtWidgets.QLabel('DEIM and Back columns count supporting frames.\nPending proposals require review. Double-click a row to jump to its first frame.')
        self.details.setWordWrap(True); layout.addWidget(self.details)
        self.setWidget(panel); self.setMinimumWidth(410)
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.poll); self.timer.start(700)
        QtCore.QTimer.singleShot(1800, lambda: self.guard(self.refresh))

    def guard(self, callback):
        try:
            return callback()
        except Exception as exc:
            self.last_error = str(exc); self.status.setText(str(exc))
            print(f'ForestSAR: {exc}', flush=True)

    def context(self):
        filename = getattr(self.view, 'filename', None)
        if not filename:
            raise ValueError('Open a review sequence first.')
        path = Path(filename).absolute()
        try:
            relative = path.parent.relative_to(self.output/'annotations')
        except ValueError as exc:
            raise ValueError('Use Open source sequence in this panel to open the editable review dataset.') from exc
        if len(relative.parts) != 2:
            raise ValueError('Open an individual sequence directory.')
        return str(relative), int(path.stem.rsplit('_', 1)[1]), path

    def save_current(self):
        key, frame, filename = self.context()
        if self.view.dirty:
            if not self.view.save_labels(str(filename.with_suffix('.json'))):
                raise RuntimeError('Saving the current annotation failed.')
            self.view.set_clean()
        return key, frame, filename

    def submit(self, action, payload):
        if self.job:
            raise RuntimeError('A worker operation is already running.')
        self.last_error = None; self.last_result = None
        self.job = self.client.request('/'+action, payload)['job_id']
        self.view.setEnabled(False)
        self.action_states = [(a, a.isEnabled()) for a in self.window.findChildren(QtGui.QAction)]
        for action, enabled in self.action_states:
            action.setEnabled(False)
        for b in self.buttons:
            b.setEnabled(False)
        self.status.setText('Queued…')

    def generate(self):
        key, _, _ = self.save_current()
        self.submit('generate', {'sequence': key, 'refresh': True, 'verify': True})

    def filter_instances(self):
        key, _, _ = self.save_current()
        self.submit('filter', {'sequence': key})

    def selected_id(self):
        shapes = self.view.canvas.selected_shapes
        if len(shapes) > 1:
            raise ValueError('Select one linked box or one instance row.')
        if len(shapes) == 1:
            ident = shapes[0].other_data.get('sar', {}).get('instance_id')
            if ident:
                return ident
            raise ValueError('This box has no linked instance yet. Track it first.')
        row = self.table.currentRow()
        if 0 <= row < len(self.items):
            return self.items[row]['instance_id']
        raise ValueError('Select a linked box or an instance row first.')

    def canvas_selected(self, shapes):
        if shapes:
            self.table.clearSelection()
            self.table.setCurrentCell(-1, -1)

    def track_selected(self):
        selected = self.view.canvas.selected_shapes
        if len(selected) != 1 or selected[0].shape_type != 'rectangle':
            raise ValueError('Draw and select exactly one person rectangle first.')
        shape = selected[0].to_dict(); key, frame, _ = self.save_current()
        self.submit('track', {'sequence': key, 'frame': frame, 'bbox': box(shape),
                             'instance_id': shape.get('sar', {}).get('instance_id')})

    def instance_action(self, action):
        ident = self.selected_id(); key, _, _ = self.save_current()
        self.submit(action, {'sequence': key, 'instance_id': ident})

    def undo(self):
        # Avoid rewriting unchanged JSON before checked undo: X-AnyLabeling serialization differs.
        key, _, _ = self.context()
        if self.view.dirty:
            raise ValueError('Save/revert current manual edits before undoing a worker operation.')
        self.submit('undo', {'sequence': key})

    def choose_sequence(self):
        keys = self.client.request('/sequences')['sequences']
        key, ok = QtWidgets.QInputDialog.getItem(self, 'Open source sequence', 'Sequence', keys, editable=False)
        if ok:
            if self.view.filename:
                self.save_current()
            self.open_after = self.output/'annotations'/key
            self.submit('initialize', {'sequence': key})

    def go(self, frame):
        key, _, _ = self.save_current()
        path = self.output/'annotations'/key/f'{Path(key).name}_{int(frame):06d}.jpg'
        if not path.is_file():
            raise ValueError('Frame is outside this sequence.')
        self.view.load_file(str(path))

    def jump_instance(self, row):
        self.go(self.items[row]['first_frame'])
        ident = self.items[row]['instance_id']
        shapes = [s for s in self.view.canvas.shapes if s.other_data.get('sar', {}).get('instance_id') == ident]
        if shapes:
            self.view.canvas.select_shapes(shapes)

    def refresh(self):
        if self.job:
            return
        key, frame, _ = self.context(); self.sequence = key
        data = self.client.request('/summary?'+urllib.parse.urlencode({'sequence': key}))
        self.frame.setMaximum(data['frames']); self.frame.setValue(frame)
        self.items = sorted([r for r in data['instances'] if r['status'] != 'deleted' and r['frame_count']], key=lambda r: r['group_id'])
        mode = self.filter.currentIndex()
        if mode == 1:
            self.items = [r for r in self.items if r['status'] == 'pending']
        elif mode == 2:
            self.items = [r for r in self.items if r['anchor_disagreement_frames']]
        elif mode == 3:
            self.items = [r for r in self.items if r['detector_only_frames']]
        elif mode == 4:
            self.items = [r for r in self.items if r['forward_only_frames']]
        self.table.setRowCount(len(self.items))
        for row, item in enumerate(self.items):
            values = [item['group_id'], item['origin'], item['status'],
                f"{item['first_frame']}–{item['last_frame']} ({item['frame_count']})",
                item['deim_supported_frames'], item['backward_verified_frames'],
                'yes' if item.get('person_confirmed_seed_frames') else 'unconfirmed']
            for col, value in enumerate(values):
                cell = QtWidgets.QTableWidgetItem(str(value)); cell.setToolTip(item['instance_id'])
                self.table.setItem(row, col, cell)
        self.details.setText(f"{key}\n{len(self.items)} instances. Select a linked box or table row; deletion affects every related frame.\nDEIM/Back show support counts, not acceptance.")

    def describe(self, row):
        if not 0 <= row < len(self.items):
            return
        item = self.items[row]
        mismatch = ', '.join(map(str, item['anchor_disagreement_frames'])) or 'none'
        admission = item.get('admission', {})
        quality = (f"\nPeak DEIM: {admission['max_deim_score']:.3f}; longest observed run: {admission['longest_consecutive_frames']} frames"
                   if admission else '')
        self.details.setText(f"Instance {item['group_id']} · {item['instance_id'][:12]}\n"
            f"Seed: frame {item['seed_frame']} ({item['origin']})\n"
            f"Missing from human anchors: {mismatch}\n"
            f"Forward-only frames: {item['forward_only_frames']}; detector-only frames: {item['detector_only_frames']}{quality}")

    def poll(self):
        if not self.job:
            try:
                key, frame, _ = self.context()
                self.frame.setValue(frame)
                if key != self.sequence:
                    self.refresh()
            except Exception:
                pass
            return
        try:
            record = self.client.request('/jobs/'+self.job)
            self.status.setText(record.get('message', record['state']))
            if record['state'] not in ('complete', 'failed'):
                return
            self.job = None; self.view.setEnabled(True)
            for action, enabled in self.action_states:
                action.setEnabled(enabled)
            for b in self.buttons:
                b.setEnabled(True)
            if self.open_after and record['state'] == 'complete':
                self.view.import_image_folder(str(self.open_after)); self.open_after = None
            elif self.view.filename:
                self.view.load_file(self.view.filename)
            if record['state'] == 'failed':
                self.last_error = record['error']; self.status.setText('Failed: '+record['error'])
            else:
                self.last_result = record['result']; self.status.setText('Complete — '+record['action'])
            self.refresh()
        except Exception as exc:
            self.status.setText('Worker connection: '+str(exc))


def create_window(output, sequence, client, profile=None):
    from anylabeling import config as app_config
    from anylabeling.resources import resources  # noqa: F401
    from anylabeling.views.mainwindow import MainWindow
    profile = Path(profile or Path(output)/'metadata/gui_profile'); profile.mkdir(parents=True, exist_ok=True)
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.Format.IniFormat, QtCore.QSettings.Scope.UserScope, str(profile))
    app_config.set_work_directory(str(profile)); config_file = profile/'.xanylabelingrc'
    if not config_file.exists():
        config_file.write_text('auto_save: false\nstore_data: false\nlabels: [person]\n')
    app_config.current_config_file = str(config_file)
    config = app_config.get_config(str(config_file))
    config.update(auto_save=False, store_data=False, labels=['person'])
    app = QtWidgets.QApplication.instance()
    directory = Path(output)/'annotations'/sequence
    window = MainWindow(app, config=config, filename=str(directory) if directory.exists() else None)
    window.setWindowTitle('X-AnyLabeling — ForestSAR instance review')
    dock = ReviewDock(window, client, output)
    window.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
    window.forest_sar = dock; window.resize(1700, 1050)
    return window


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', default=str(DEFAULT_OUTPUT))
    p.add_argument('--sequence', required=True, help='Source GROUP/SEQUENCE key')
    p.add_argument('--gpu', type=int, default=1); p.add_argument('--port', type=int, default=8793)
    p.add_argument('--frame', type=int)
    args = p.parse_args(); endpoint = ensure_backend(args.output, args.gpu, args.port)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication([])
    window = create_window(args.output, args.sequence, Client(endpoint)); window.show()
    if args.frame is not None:
        QtCore.QTimer.singleShot(2000, lambda: window.forest_sar.guard(lambda: window.forest_sar.go(args.frame)))
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
