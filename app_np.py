"""Torch-free handwriting synthesis API (numpy backend).

Loads weights.npz + stoi.json + std.json -- no torch, no dataset. Same endpoints
and input validation as the torch version, but ~instant cold start and a tiny image.

Run:  gunicorn -b :$PORT --workers 1 --threads 8 --preload app_np:app
Local: python app_np.py
"""
import json
import re
import threading
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request

from model_np import NumpyHandwritingModel

MAX_TEXT_LEN = 100
TEMP_MIN, TEMP_MAX = 0.0, 2.0

# Polish diacritics the model never saw -> closest ASCII it can draw.
_TRANSLIT = str.maketrans({
    "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n", "ó": "o", "ś": "s", "ź": "z", "ż": "z",
    "Ą": "A", "Ć": "C", "Ę": "E", "Ł": "L", "Ń": "N", "Ó": "O", "Ś": "S", "Ź": "Z", "Ż": "Z",
})

# --- load artifacts once at startup (all torch-free) ---
stoi = json.loads(Path("stoi.json").read_text(encoding="utf-8"))
std = json.loads(Path("std.json").read_text(encoding="utf-8"))["std"]
model = NumpyHandwritingModel.from_npz("weights.npz")
VOCAB = set(stoi)
_DISALLOWED = re.compile(f"[^{re.escape(''.join(sorted(VOCAB)))}]")
_lock = threading.Lock()
print(f"ready (numpy). vocab {len(stoi)}, std {std:.3f}")

app = Flask(__name__)


def prepare(text):
    """Transliterate Polish, then reject anything still outside the vocab."""
    text = text.translate(_TRANSLIT)
    bad = sorted(set(_DISALLOWED.findall(text)))
    if bad:
        raise ValueError(f"input contains characters outside the vocab: {bad}")
    if not text.strip():
        raise ValueError("text is empty")
    if len(text) > MAX_TEXT_LEN:
        raise ValueError(f"text too long ({len(text)} > {MAX_TEXT_LEN} chars)")
    return text


def parse_params():
    data = request.get_json(silent=True) or {}
    text = data.get("text", request.args.get("text", ""))
    temp_raw = data.get("temperature", request.args.get("temperature", 0.4))
    try:
        temperature = float(temp_raw)
    except (TypeError, ValueError):
        raise ValueError("temperature must be a number")
    if not (TEMP_MIN <= temperature <= TEMP_MAX):
        raise ValueError(f"temperature must be in [{TEMP_MIN}, {TEMP_MAX}]")
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return text, temperature


def _onehot(text):
    U, V = len(text), len(stoi)
    c = np.zeros((U, V), np.float32)
    c[np.arange(U), [stoi[ch] for ch in text]] = 1.0
    return c, np.ones(U, np.float32)


def synthesize(text, temperature):
    cleaned = prepare(text)
    c, c_mask = _onehot(cleaned)
    with _lock:
        pts = list(model.sample_iter(c, c_mask, temperature=temperature))

    xy = np.cumsum(np.array([[p[0], p[1]] for p in pts]) * std, axis=0)
    pen_up = [p[2] for p in pts]
    strokes, start = [], 0
    for j in range(len(pts)):
        if pen_up[j] == 1 or j == len(pts) - 1:
            seg = xy[start:j + 1]
            if len(seg) > 1:
                strokes.append([[round(float(a), 2), round(float(b), 2)] for a, b in seg])
            start = j + 1
    return {"text": cleaned, "temperature": temperature,
            "num_points": len(pts), "strokes": strokes}


@app.post("/generate")
@app.get("/generate")
def generate():
    try:
        text, temperature = parse_params()
        return jsonify(synthesize(text, temperature))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.post("/generate/stream")
@app.get("/generate/stream")
def generate_stream():
    try:
        text, temperature = parse_params()
        cleaned = prepare(text)
        c, c_mask = _onehot(cleaned)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    def stream():
        yield json.dumps({"type": "meta", "text": cleaned, "temperature": temperature}) + "\n"
        cx = cy = 0.0
        n = 0
        with _lock:
            for dx, dy, pen_up, _phi in model.sample_iter(c, c_mask, temperature=temperature):
                cx += dx * std; cy += dy * std; n += 1
                yield json.dumps({"type": "point", "x": round(cx, 2),
                                  "y": round(cy, 2), "pen_up": int(pen_up)}) + "\n"
        yield json.dumps({"type": "end", "num_points": n}) + "\n"

    return Response(stream(), mimetype="application/x-ndjson")


@app.get("/vocab")
def vocab():
    chars = sorted(stoi, key=lambda c: stoi[c])
    return jsonify({"size": len(chars), "chars": "".join(chars)})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "backend": "numpy", "model": "best-4"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
