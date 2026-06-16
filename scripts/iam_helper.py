#!/usr/bin/env python3
"""
Helper for the SvelteKit review API to read the IAM dataset.

Usage:
  python3 iam_helper.py list             -> JSON [{index, text}, ...]  (excludes rejected)
  python3 iam_helper.py text <index>    -> JSON {text: "..."}
  python3 iam_helper.py reject <index>   -> marks index rejected, saves to data/rejected_review.json
  python3 iam_helper.py unreject <index> -> removes index from rejected list
"""

import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NPZ = ROOT / 'data' / 'dataset.npz'
REJECTED_FILE = ROOT / 'data' / 'rejected_review.json'


def load_npz():
    d = np.load(NPZ, allow_pickle=True)
    return d['strokes'], d['texts'], float(d['std'])


def load_rejected() -> set[str]:
    if not REJECTED_FILE.exists():
        return set()
    raw = REJECTED_FILE.read_text().strip()
    if not raw:
        return set()
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        return set()
    return set(parsed)


def save_rejected(rejected: set[str]):
    REJECTED_FILE.write_text(json.dumps(sorted(rejected)))


def cmd_list():
    _, texts, _ = load_npz()
    rejected = load_rejected()
    result = [
        {'index': i, 'text': str(t)}
        for i, t in enumerate(texts)
        if f'iam:{i}' not in rejected
    ]
    print(json.dumps(result))


def cmd_text(index: int):
    _, texts, _ = load_npz()
    print(json.dumps({'text': str(texts[index])}))


def cmd_stroke(index: int):
    strokes_arr, _, std = load_npz()
    s = strokes_arr[index]  # shape (T, 3): (dx, dy, pen_up) in dataset.npz

    x, y = 0.0, 0.0
    strokes = []
    current: list = []
    for row in s:
        dx, dy, pen_up = float(row[0]), float(row[1]), float(row[2])
        x += dx * std
        y += dy * std
        current.append([round(x, 3), round(-y, 3)])  # flip y: IAM grows up, canvas grows down
        if pen_up > 0.5:
            strokes.append(current)
            current = []
    if current:
        strokes.append(current)

    print(json.dumps({'strokes': strokes}))


def cmd_reject(index: int):
    rejected = load_rejected()
    rejected.add(f'iam:{index}')
    save_rejected(rejected)
    print(json.dumps({'ok': True}))


def cmd_unreject(index: int):
    rejected = load_rejected()
    rejected.discard(f'iam:{index}')
    save_rejected(rejected)
    print(json.dumps({'ok': True}))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'list':
        cmd_list()
    elif cmd == 'text':
        cmd_text(int(sys.argv[2]))
    elif cmd == 'stroke':
        cmd_stroke(int(sys.argv[2]))
    elif cmd == 'reject':
        cmd_reject(int(sys.argv[2]))
    elif cmd == 'unreject':
        cmd_unreject(int(sys.argv[2]))
    else:
        print(json.dumps({'error': f'unknown command: {cmd}'}), file=sys.stderr)
        sys.exit(1)
