"""Headless entry point for generation, tracking, review and undo."""
import argparse
import json

from .engine import Engine
from .store import DEFAULT_OUTPUT, Store


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['initialize', 'generate', 'track', 'delete', 'accept', 'undo', 'status', 'filter'])
    p.add_argument('sequence'); p.add_argument('--output', default=str(DEFAULT_OUTPUT)); p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--frame', type=int); p.add_argument('--box', type=float, nargs=4); p.add_argument('--instance-id')
    p.add_argument('--cached-anchors', action='store_true'); p.add_argument('--no-backward', action='store_true')
    a = p.parse_args(); store = Store(a.output); engine = Engine(store, a.gpu)
    if a.action == 'initialize':
        result = store.initialize(a.sequence)
    elif a.action == 'generate':
        result = engine.generate(a.sequence, refresh=not a.cached_anchors, verify=not a.no_backward)
    elif a.action == 'track':
        result = engine.track(a.sequence, a.frame, a.box, instance_id=a.instance_id, verify=not a.no_backward)
    elif a.action in ('delete', 'accept'):
        result = store.instance_action(a.sequence, a.instance_id, a.action)
    elif a.action == 'undo':
        result = store.undo(a.sequence)
    elif a.action == 'filter':
        from .quality import apply_instance_policy
        result = apply_instance_policy(store, a.sequence)
    else:
        result = store.summary(a.sequence)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
