"""Basic HTTP API around the handwriting synthesis model.

Run:   uv run app.py            (serves on http://127.0.0.1:5000)

Endpoints:
  POST /generate        JSON body  {"text": "...", "temperature": 0.4}
  GET  /generate?text=hello&temperature=0.4
       -> JSON: cleaned text, dropped chars, and the handwriting as a list of
          strokes (each stroke = polyline of [x, y] points, ready to draw).
  GET  /generate.png?text=hello&temperature=0.4
       -> PNG image of the handwriting (handy for eyeballing in a browser).
  GET  /vocab           -> the characters the model can draw.
  GET  /health          -> liveness check.
"""
import io
import json
import re
import threading

import numpy as np
import torch
from flask import Flask, Response, jsonify, request, send_file

from generate import load_model, sample, sample_iter

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")

CKPT = "best-4.pth"        # epoch 78 (epoch 80 rejected for higher loss)
MAX_TEXT_LEN = 100          # guard: long lines drift / blow past max_steps
TEMP_MIN, TEMP_MAX = 0.0, 2.0

# Polish diacritics the model never saw -> closest ASCII it can draw.
_TRANSLIT = str.maketrans({
    "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n", "ó": "o", "ś": "s", "ź": "z", "ż": "z",
    "Ą": "A", "Ć": "C", "Ę": "E", "Ł": "L", "Ń": "N", "Ó": "O", "Ś": "S", "Ź": "Z", "Ż": "Z",
})

# --- load the model once at startup ---
print(f"loading {CKPT} on {DEVICE} ...")
model, stoi = load_model(DEVICE, ckpt=CKPT)
std = json.load(open("std.json"))["std"]  # baked normalisation factor (no dataset needed)
VOCAB = set(stoi)
_lock = threading.Lock()                 # model inference is not thread-safe; serialise it

# Regex matching any single character the model CANNOT draw. Built straight from the
# vocab (re.escape handles the regex-special chars like . ( ) + ? in there), so it
# stays in sync if the vocab ever changes. Used to REJECT (not silently drop) input.
_DISALLOWED = re.compile(f"[^{re.escape(''.join(sorted(VOCAB)))}]")
print(f"ready. vocab size {len(stoi)}, std {std:.3f}")

app = Flask(__name__)


def prepare(text):
    """Transliterate Polish, then REJECT if anything is still outside the vocab.

    Raises ValueError (-> 400) listing the offending characters instead of
    silently dropping them.
    """
    text = text.translate(_TRANSLIT)
    bad = sorted(set(_DISALLOWED.findall(text)))
    if bad:
        raise ValueError(f"input contains characters outside the vocab: {bad}")
    if not text.strip():
        raise ValueError("text is empty")
    return text


def parse_params():
    """Pull text + temperature from JSON body or query string. Raises ValueError."""
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


def synthesize(text, temperature):
    """Validate -> sample -> denormalise into absolute polylines. Returns dict."""
    cleaned = prepare(text)
    if len(cleaned) > MAX_TEXT_LEN:
        raise ValueError(f"text too long ({len(cleaned)} > {MAX_TEXT_LEN} chars)")

    with _lock:
        seq, _phis = sample(model, cleaned, stoi, DEVICE, temperature=temperature)

    # offsets -> absolute coordinates, split into separate strokes on pen lifts
    xy = np.cumsum(seq[:, :2].numpy() * std, axis=0)
    pen_up = seq[:, 2].numpy()
    strokes, start = [], 0
    for j in range(len(seq)):
        if pen_up[j] == 1 or j == len(seq) - 1:
            pts = xy[start:j + 1]
            if len(pts) > 1:
                strokes.append([[round(float(x), 2), round(float(y), 2)] for x, y in pts])
            start = j + 1
    return {"text": cleaned, "temperature": temperature,
            "num_points": int(len(seq)), "strokes": strokes}


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
    """Stream the handwriting as it is drawn, NDJSON (one JSON object per line).

      {"type":"meta","text":...,"temperature":...}
      {"type":"point","x":..,"y":..,"pen_up":0|1}   (one per pen step, live)
      {"type":"end","num_points":N}
    """
    # validate up front so errors return a real 400 (once streaming starts it's 200)
    try:
        text, temperature = parse_params()
        cleaned = prepare(text)
        if len(cleaned) > MAX_TEXT_LEN:
            raise ValueError(f"text too long ({len(cleaned)} > {MAX_TEXT_LEN} chars)")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    def stream():
        yield json.dumps({"type": "meta", "text": cleaned, "temperature": temperature}) + "\n"
        cx = cy = 0.0   # integrate offsets into absolute coords as points arrive
        n = 0
        with _lock:     # one generation at a time (single GPU, non-reentrant model)
            for dx, dy, pen_up, _phi in sample_iter(model, cleaned, stoi, DEVICE,
                                                    temperature=temperature):
                cx += dx * std
                cy += dy * std
                n += 1
                yield json.dumps({"type": "point", "x": round(cx, 2),
                                  "y": round(cy, 2), "pen_up": int(pen_up)}) + "\n"
        yield json.dumps({"type": "end", "num_points": n}) + "\n"

    return Response(stream(), mimetype="application/x-ndjson")


@app.get("/generate.png")
def generate_png():
    # matplotlib is intentionally NOT installed in the slim serving image (the web
    # frontend renders strokes from /generate). This endpoint is a local debug
    # convenience; it degrades gracefully if matplotlib is absent.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return jsonify({"error": "PNG rendering unavailable (matplotlib not installed); "
                                 "use /generate and render strokes client-side"}), 503
    try:
        text, temperature = parse_params()
        result = synthesize(text, temperature)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    fig, ax = plt.subplots(figsize=(12, 3))
    for stroke in result["strokes"]:
        arr = np.array(stroke)
        ax.plot(arr[:, 0], arr[:, 1], "k", linewidth=1.8)
    ax.set_aspect("equal"); ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.get("/vocab")
def vocab():
    chars = sorted(stoi, key=lambda c: stoi[c])
    return jsonify({"size": len(chars), "chars": "".join(chars)})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "device": DEVICE, "checkpoint": CKPT})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
