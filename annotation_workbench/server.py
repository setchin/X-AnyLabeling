"""Single-worker local GPU service. GUI dependencies stay out of inference."""
from __future__ import annotations

import argparse
import hmac
import json
import os
import secrets
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .engine import Engine
from .store import DEFAULT_OUTPUT, Store, atomic


def serve(output=DEFAULT_OUTPUT, gpu=1, port=8793):
    store = Store(output); jobs = {}; mutex = threading.Lock(); pool = ThreadPoolExecutor(max_workers=1)
    for p in (store.output/'metadata').glob('*/*/instances.json'):
        store.initialize(str(p.parent.relative_to(store.output/'metadata')))
    token = secrets.token_urlsafe(32)
    engine = Engine(store, gpu=gpu)

    def run(ident, action, args):
        record = jobs[ident]
        def progress(message):
            print(f'[{ident[:8]}] {message}', flush=True)
            with mutex:
                record['message'] = message; record['messages'].append(message)
        engine.progress = progress
        with mutex:
            record.update(state='running', started_at=time.time())
        try:
            key = args['sequence']
            if action == 'initialize':
                store.initialize(key); result = store.summary(key)
            elif action == 'generate':
                result = engine.generate(key, refresh=args.get('refresh', True), verify=args.get('verify', True))
                result = {'sequence': key, 'new_segments': len(result['segments_generated']), 'summary': result['summary']}
            elif action == 'track':
                result = engine.track(key, args['frame'], args['bbox'], instance_id=args.get('instance_id'))
            elif action in ('delete', 'accept'):
                result = store.instance_action(key, args['instance_id'], action)
            elif action == 'undo':
                result = store.undo(key)
            elif action == 'filter':
                from .quality import apply_instance_policy
                result = apply_instance_policy(store, key)
            else:
                raise ValueError('Unknown job action.')
            store.write_review_queue(key)
            with mutex:
                record.update(state='complete', result=result, ended_at=time.time())
        except Exception as exc:
            trace = traceback.format_exc(); print(trace, flush=True)
            with mutex:
                record.update(state='failed', error=str(exc), traceback=trace, ended_at=time.time())
        atomic(store.output/'metadata/jobs'/f'{ident}.json', record)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if '/jobs/' not in self.path:
                super().log_message(fmt, *args)

        def send(self, value, code=200):
            payload = json.dumps(value).encode()
            self.send_response(code); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)

        def authorized(self):
            return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer '+token)

        def do_GET(self):
            if not self.authorized():
                return self.send({'error': 'Unauthorized'}, 401)
            url = urlparse(self.path)
            try:
                if url.path == '/health':
                    return self.send({'status': 'ok', 'output': str(store.output), 'gpu': gpu, 'pid': os.getpid()})
                if url.path == '/sequences':
                    keys = [f'{g.name}/{s.name}' for g in sorted((store.data/'FRAMES').iterdir()) if g.is_dir()
                            for s in sorted(g.iterdir()) if s.is_dir()]
                    return self.send({'sequences': keys})
                if url.path == '/summary':
                    key = parse_qs(url.query)['sequence'][0]
                    with store.locked(key):
                        return self.send(store.summary(key))
                if url.path.startswith('/jobs/'):
                    ident = url.path.rsplit('/', 1)[1]
                    with mutex:
                        record = dict(jobs[ident]); record['messages'] = list(record['messages'])
                    return self.send(record)
                self.send({'error': 'Not found'}, 404)
            except Exception as exc:
                self.send({'error': str(exc)}, 400)

        def do_POST(self):
            if not self.authorized():
                return self.send({'error': 'Unauthorized'}, 401)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length > 1024*1024:
                    raise ValueError('Request too large.')
                args = json.loads(self.rfile.read(length)); action = self.path.strip('/')
                if action not in ('initialize', 'generate', 'track', 'delete', 'accept', 'undo', 'filter'):
                    raise ValueError('Unknown action.')
                store.key(args['sequence'])
                ident = uuid.uuid4().hex
                with mutex:
                    jobs[ident] = {'job_id': ident, 'action': action, 'sequence': args['sequence'],
                        'state': 'queued', 'message': 'Queued', 'messages': [], 'created_at': time.time()}
                pool.submit(run, ident, action, args)
                self.send({'job_id': ident}, 202)
            except Exception as exc:
                self.send({'error': str(exc)}, 400)

    server = ThreadingHTTPServer(('127.0.0.1', int(port)), Handler)
    endpoint = store.output/'metadata/backend.json'
    atomic(endpoint, {'url': f'http://127.0.0.1:{server.server_port}', 'token': token,
                      'pid': os.getpid(), 'output': str(store.output), 'gpu': gpu})
    endpoint.chmod(0o600)
    print(f'ForestSAR inference service: 127.0.0.1:{server.server_port}; GPU {gpu}', flush=True)
    try:
        server.serve_forever()
    finally:
        pool.shutdown(wait=True); server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--gpu', type=int, default=1); parser.add_argument('--port', type=int, default=8793)
    args = parser.parse_args(); serve(args.output, args.gpu, args.port)
