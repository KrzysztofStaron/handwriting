# Handwriting API — usage & visualization

Guide for integrating the handwriting synthesis API and rendering its output.
The deployed service is a **numpy** backend (no torch, no PNG endpoint). You get
stroke polylines as JSON and draw them yourself.

**Live URL:** https://handwriting-api-687921010800.europe-central2.run.app

---

## Quick start

```bash
# Pre-warm (optional — hides cold-start on first real request)
curl -s "$BASE/health"

# Generate handwriting for a short line
curl -s "$BASE/generate?text=hello&temperature=0.5"

# Stream points as they are drawn (lower perceived latency)
curl -sN "$BASE/generate/stream?text=hello&temperature=0.5"
```

Set `BASE` to the live URL above, or `http://127.0.0.1:5000` when running
`python app_np.py` locally.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET/POST | `/generate` | Full result as one JSON object |
| GET/POST | `/generate/stream` | NDJSON stream, one pen point per line |
| GET | `/vocab` | Characters the model can draw (77 total) |
| GET | `/health` | Liveness + pre-warm ping |

There is **no** `/generate.png` on the deployed service. Render strokes
client-side (canvas/SVG) or use the local torch app (`app.py`) for PNG debug.

### Parameters

Both `/generate` and `/generate/stream` accept the same inputs:

| Param | Type | Default | Range | Notes |
|---|---|---|---|---|
| `text` | string | — | max 100 chars | Required |
| `temperature` | float | `0.4` | `[0, 2]` | `0` = deterministic, higher = messier |

Pass via query string (`?text=...&temperature=0.5`) or JSON body on POST:

```json
{"text": "hello", "temperature": 0.5}
```

### Input rules

1. **Polish diacritics** are transliterated to ASCII (`ę→e`, `ł→l`, etc.).
2. Any character **still outside the 77-char vocab** → `400` with the offending
   chars listed. Nothing is silently dropped.
3. Empty/whitespace-only text → `400`.

Fetch the allowed charset:

```bash
curl -s "$BASE/vocab"
# {"size": 77, "chars": " !\"#'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"}
```

---

## Response formats

### `/generate` — batch JSON

```json
{
  "text": "hello",
  "temperature": 0.5,
  "num_points": 110,
  "strokes": [
    [[0.02, -0.02], [0.43, -0.86], [0.78, -1.97], ...],
    [[13.46, -8.62], [13.21, -8.89], ...]
  ]
}
```

- `text` — cleaned input (after transliteration).
- `strokes` — array of polylines. Each stroke is a list of `[x, y]` points in
  **absolute coordinates** (offsets already integrated and denormalised).
- A new stroke starts after every **pen lift** (`pen_up = 1` internally).

### `/generate/stream` — NDJSON

One JSON object per line (`application/x-ndjson`):

```
{"type": "meta", "text": "hello", "temperature": 0.5}
{"type": "point", "x": 0.02, "y": -0.02, "pen_up": 0}
{"type": "point", "x": 0.43, "y": -0.86, "pen_up": 0}
...
{"type": "point", "x": 13.46, "y": -8.62, "pen_up": 1}
{"type": "end", "num_points": 110}
```

| `type` | Fields | Meaning |
|---|---|---|
| `meta` | `text`, `temperature` | Generation started |
| `point` | `x`, `y`, `pen_up` | One pen step; `pen_up: 1` ends the current stroke |
| `end` | `num_points` | Generation finished |

Use streaming when you want ink to appear while the model is still running.
First point typically arrives in ~100 ms on Cloud Run vs waiting for the full
JSON on `/generate`.

### Errors

```json
{"error": "input contains characters outside the vocab: ['@']"}
```

HTTP `400` for validation failures. Check `error` in the JSON body.

---

## Coordinate system

Understanding coordinates is the main thing you need to render correctly.

- **Origin** `(0, 0)` is where the pen starts.
- Points are **absolute** — do not cumsum them again.
- **Y is y-up** (matplotlib convention, same as `generate.py` plots). When
  rendering on a canvas/SVG (y-down), flip: `screenY = maxY - modelY`.
- Units are arbitrary model space (~tens of units per character). Scale to fit
  your viewport; preserve aspect ratio (`equal` / `preserveAspectRatio`).

---

## Visualizing outputs

### 1. Browser canvas (recommended for apps)

Draw each stroke as a connected polyline. Start a new path whenever `pen_up`
is `1` (streaming) or when moving to the next entry in `strokes` (batch).

```html
<canvas id="c" width="800" height="200"></canvas>
<script>
const BASE = "https://handwriting-api-687921010800.europe-central2.run.app";

async function draw(text, temperature = 0.5) {
  const res = await fetch(`${BASE}/generate`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text, temperature}),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error);

  const canvas = document.getElementById("c");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#111";
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  // Fit strokes into the canvas with padding
  const all = data.strokes.flat();
  const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad = 20;
  const scale = Math.min(
    (canvas.width - 2 * pad) / (maxX - minX || 1),
    (canvas.height - 2 * pad) / (maxY - minY || 1),
  );
  const tx = (x) => pad + (x - minX) * scale;
  const ty = (y) => pad + (maxY - y) * scale;  // flip y-up model coords to canvas

  for (const stroke of data.strokes) {
    if (stroke.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(tx(stroke[0][0]), ty(stroke[0][1]));
    for (let i = 1; i < stroke.length; i++) {
      ctx.lineTo(tx(stroke[i][0]), ty(stroke[i][1]));
    }
    ctx.stroke();
  }
}

draw("hello world", 0.5);
</script>
```

### 2. Streaming canvas (live ink)

```javascript
async function drawStream(text, temperature = 0.5) {
  const res = await fetch(`${BASE}/generate/stream`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text, temperature}),
  });
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  let stroke = [];
  const strokes = [];

  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});
    const lines = buf.split("\n");
    buf = lines.pop();
    for (const line of lines) {
      if (!line) continue;
      const ev = JSON.parse(line);
      if (ev.type === "point") {
        stroke.push([ev.x, ev.y]);
        if (ev.pen_up) { strokes.push(stroke); stroke = []; }
        redraw(strokes, stroke);  // your fit-and-draw helper
      }
    }
  }
  if (stroke.length) strokes.push(stroke);
}
```

Redraw after each point (or batch with `requestAnimationFrame`) for the
"typewriter" effect.

### 3. SVG

Same data, different renderer — one `<polyline>` per stroke:

```javascript
const points = stroke.map(([x, y]) => `${tx(x)},${ty(y)}`).join(" ");
// <polyline fill="none" stroke="#111" stroke-width="2" points="..."/>
```

Flip or offset `y` only if you want a math-style y-up plot; the API data is
y-down.

### 4. Python / matplotlib (local experiments)

The repo includes a torch-based plotter for offline samples. Requires torch +
matplotlib (not in the deployed image):

```bash
uv run generate.py
# writes sample_temp_0.0.png, sample_temp_0.3.png, ... and alignment.png
```

To plot API JSON without torch:

```python
import json, requests
import matplotlib.pyplot as plt

BASE = "https://handwriting-api-687921010800.europe-central2.run.app"
data = requests.post(f"{BASE}/generate",
                     json={"text": "hello", "temperature": 0.5}).json()

fig, ax = plt.subplots(figsize=(12, 3))
for stroke in data["strokes"]:
    xs, ys = zip(*stroke)
    ax.plot(xs, ys, "k", linewidth=2)
ax.set_aspect("equal")
ax.axis("off")
plt.savefig("api_sample.png", bbox_inches="tight", dpi=150)
```

`generate.py`'s `save_plot()` shows the same stroke-splitting logic the API
uses internally (cumsum offsets → split on `pen_up`).

### 5. Local torch app with PNG (debug only)

```bash
uv run app.py
# then open http://127.0.0.1:5000/generate.png?text=hello&temperature=0.5
```

`app.py` has matplotlib and a `/generate.png` endpoint. The deployed
`app_np.py` does not.

---

## Temperature

| Value | Effect |
|---|---|
| `0.0` | Most consistent, least wobbly |
| `0.3–0.5` | Natural handwriting (good default: `0.4`) |
| `0.8+` | Increasingly messy / varied |
| `2.0` | Maximum chaos (upper bound) |

Same `text` + `temperature` will **not** produce identical output — sampling
is stochastic unless `temperature=0`.

---

## Latency tips

| Scenario | Typical time (warm) |
|---|---|
| `/health` | ~80–160 ms |
| `/generate` short text (`hi`) | ~0.6–0.8 s |
| `/generate` medium text | ~2–4 s |
| `/generate/stream` first point | ~90–360 ms |

- Call `/health` on page load to pre-warm before the user submits text.
- Prefer `/generate/stream` when you want ink to appear immediately.
- Generation time grows with text length (sequential pen-step loop).

---

## Example integration checklist

1. `GET /vocab` — build input validation / character picker in your UI.
2. `GET /health` on mount — pre-warm Cloud Run instance.
3. `POST /generate/stream` on submit — stream points into a canvas.
4. Fit coordinates to viewport; keep aspect ratio; y-down rendering.
5. Handle `400` errors and show the `error` message (bad chars, empty text, etc.).
6. Cap input at 100 characters client-side.

---

## Local development

```bash
# numpy API (same as production)
python app_np.py          # http://127.0.0.1:5000

# torch API with PNG debug endpoint
uv run app.py             # http://127.0.0.1:5000
```

Artifacts the API loads at startup: `weights.npz`, `stoi.json`, `std.json`.
See `DEPLOY.md` for building and redeploying the container.
