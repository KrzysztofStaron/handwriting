"""Stream handwriting generation and encode to MP4.

Animated video: each pen step from sample_iter becomes one video frame (live
ink, not static PNGs). Use generate_variants.py only to pick temp/seed; this
script re-runs the model and streams the chosen sample into drawing.mp4.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from model_np import NumpyHandwritingModel

LINES = [
    "Hello",
    "it's not me who's writing this",
    "but a model I built recently",
    "it's called best-4",
    "because i had 4 failed training runs",
    "and didn't care to rename it",
]

TEMPS = [0.0, 0.2, 0.3, 0.5]


# extra seed offset per line — bump to regenerate fresh samples (line 4)
SEED_OFFSETS = {3: 50000}


def variant_seed(line_i, temperature):
    j = TEMPS.index(temperature)
    base = line_i * 1000 + j * 100 + int(temperature * 10)
    return base + SEED_OFFSETS.get(line_i, 0)


# Review variants/grid.png, then set one temp per line (from TEMPS):
PICKS = [0.5, 0.3, 0.0, 0.2, 0.2, 0.5]


def build_line_config(picks):
    if picks is None or len(picks) != len(LINES):
        raise SystemExit(
            "Set PICKS in render_drawing_mp4.py after reviewing variants/grid.png.\n"
            f"Need {len(LINES)} temps from {TEMPS}, e.g. PICKS = [0.5, 0.3, 0.5, 0.2, 0.2, 0.0]"
        )
    return [{"temperature": t, "seed": variant_seed(i, t)} for i, t in enumerate(picks)]

WIDTH, HEIGHT = 1920, 1080
FPS = 60
OUTPUT = "drawing.mp4"
INK = (20, 20, 19)
PEN_DOT = (204, 120, 92)
STROKE_WIDTH = 6
HOLD_AFTER_LINE = 0.5
HOLD_END = 1.0
AUDIO = "Open House Glow.mp3"

CARD = (300, 248, 1620, 820)
DRAW_PANEL = (360, 400, 1560, 680)
PANEL_PAD = 40

# Anthropic-inspired warm editorial palette (no blue-purple gradients)
CANVAS_TOP = (250, 249, 245)
CANVAS_BOT = (237, 232, 224)
CARD_BG = (255, 255, 253)
CARD_BORDER = (230, 224, 214)
SHADOW = (210, 200, 188)
PANEL_BG = (255, 255, 255)
PANEL_BORDER = (235, 230, 222)
CLAY = (204, 120, 92)
CLAY_SOFT = (235, 210, 198)
INK_TEXT = (20, 20, 19)
BODY = (61, 61, 58)
MUTED = (142, 138, 128)
MUTED_SOFT = (176, 171, 162)
HAIRLINE = (230, 224, 214)


def load_model():
    stoi = json.loads(Path("stoi.json").read_text())
    std = json.loads(Path("std.json").read_text())["std"]
    model = NumpyHandwritingModel.from_npz("weights.npz")
    return model, stoi, std


def load_font(size, kind="sans"):
    paths = {
        "serif": [
            "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/Library/Fonts/Georgia.ttf",
        ],
        "mono": [
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Menlo.ttc",
        ],
        "sans": [
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ],
    }[kind]
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_SERIF_LG = load_font(52, "serif")
FONT_SERIF = load_font(32, "serif")
FONT_SANS = load_font(20, "sans")
FONT_PROMPT = load_font(28, "serif")
FONT_EYEBROW = load_font(13, "mono")
FONT_SMALL = load_font(16, "sans")
FONT_TINY = load_font(14, "sans")


def onehot(text, stoi):
    u, v = len(text), len(stoi)
    c = np.zeros((u, v), np.float32)
    c[np.arange(u), [stoi[ch] for ch in text]] = 1.0
    return c, np.ones(u, np.float32)


def collect_strokes(model, c, c_mask, std, temperature, rng=None):
    """Run the model and return (completed_strokes, xs, ys).

    A shared helper so generate_variants.py and render_drawing_mp4.py
    don't duplicate the pen-step accumulation loop.
    """
    cx = cy = 0.0
    xs, ys = [0.0], [0.0]
    completed = []
    current = []

    for dx, dy, pen_up, _phi in model.sample_iter(c, c_mask, temperature=temperature, rng=rng):
        cx += dx * std
        cy += dy * std
        xs.append(cx)
        ys.append(cy)
        current.append((cx, cy))
        if pen_up:
            if len(current) >= 2:
                completed.append(current[:])
            current = []

    if len(current) >= 2:
        completed.append(current[:])
    return completed, xs, ys


def _warm_gradient():
    t = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None]
    top = np.array(CANVAS_TOP, dtype=np.float32)
    bot = np.array(CANVAS_BOT, dtype=np.float32)
    arr = (top * (1 - t) + bot * t).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def build_scenery():
    img = _warm_gradient()
    draw = ImageDraw.Draw(img)

    # editorial header (static)
    draw.text((WIDTH // 2, 72), "LIVE DEMO", font=FONT_EYEBROW,
              fill=MUTED, anchor="mm")
    draw.text((WIDTH // 2, 148), "Text becomes handwriting",
              font=FONT_SERIF_LG, fill=INK_TEXT, anchor="mm")
    draw.text((WIDTH // 2, 198),
              "Each stroke is sampled and streamed as the model writes",
              font=FONT_SANS, fill=MUTED, anchor="mm")

    # showcase card + soft shadow
    draw.rounded_rectangle((CARD[0] + 5, CARD[1] + 7, CARD[2] + 5, CARD[3] + 7),
                           radius=20, fill=SHADOW)
    draw.rounded_rectangle(CARD, radius=20, fill=CARD_BG, outline=CARD_BORDER, width=1)
    draw.rectangle((CARD[0], CARD[1] + 24, CARD[0] + 4, CARD[3] - 24), fill=CLAY)

    # drawing surface
    draw.rounded_rectangle(DRAW_PANEL, radius=14, fill=PANEL_BG, outline=PANEL_BORDER, width=1)

    # footer
    draw.line((300, HEIGHT - 72, 1620, HEIGHT - 72), fill=HAIRLINE, width=1)
    draw.text((WIDTH // 2, HEIGHT - 42),
              "Handwriting synthesis  ·  best-4  ·  streamed inference",
              font=FONT_TINY, fill=MUTED_SOFT, anchor="mm")

    return img


def draw_demo_ui(draw, text, index, total, drawing=False, temperature=0.3):
    cx0, cy0, cx1, cy1 = CARD

    # card header row
    draw.text((cx0 + 40, cy0 + 36), "Live output", font=FONT_SANS, fill=BODY)

    status = "Writing" if drawing else "Waiting"
    dot = CLAY if drawing else MUTED_SOFT
    sx = cx1 - 220
    draw.ellipse((sx, cy0 + 40, sx + 8, cy0 + 48), fill=dot)
    draw.text((sx + 16, cy0 + 34), status, font=FONT_SMALL, fill=BODY if drawing else MUTED)
    draw.text((cx1 - 90, cy0 + 34), f"{index + 1}/{total}", font=FONT_SMALL, fill=MUTED)

    # prompt below drawing area
    prompt_top = DRAW_PANEL[3] + 36
    draw.line((cx0 + 40, prompt_top - 12, cx1 - 40, prompt_top - 12), fill=HAIRLINE, width=1)
    draw.text((cx0 + 40, prompt_top), "Input", font=FONT_EYEBROW, fill=MUTED)
    draw.text((cx0 + 40, prompt_top + 28), f'"{text}"', font=FONT_PROMPT, fill=INK_TEXT)

    pill = (cx0 + 40, prompt_top + 72, cx0 + 160, prompt_top + 100)
    draw.rounded_rectangle(pill, radius=14, fill=CLAY_SOFT if drawing else (245, 240, 235))
    draw.text((cx0 + 58, prompt_top + 78), f"t = {temperature}", font=FONT_EYEBROW,
              fill=CLAY if drawing else MUTED)


def fit_transform(xs, ys, panel=None, pad=None, margin=0.88):
    if panel is None:
        panel = DRAW_PANEL
    if pad is None:
        pad = PANEL_PAD
    px0, py0, px1, py1 = panel
    area_w = px1 - px0 - 2 * pad
    area_h = py1 - py0 - 2 * pad
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = min(area_w / span_x, area_h / span_y) * margin
    ox = px0 + pad + (area_w - span_x * scale) / 2
    oy = py0 + pad + (area_h - span_y * scale) / 2
    tx = lambda x: ox + (x - min_x) * scale
    ty = lambda y: oy + (max_y - y) * scale
    return tx, ty


def draw_strokes(draw, completed, current, xs, ys, show_pen=True):
    if not xs:
        return
    tx, ty = fit_transform(xs, ys)
    for stroke in completed:
        if len(stroke) < 2:
            continue
        draw.line([(tx(x), ty(y)) for x, y in stroke],
                  fill=INK, width=STROKE_WIDTH, joint="curve")
    if len(current) >= 2:
        draw.line([(tx(x), ty(y)) for x, y in current],
                  fill=INK, width=STROKE_WIDTH, joint="curve")
    if show_pen and current:
        x, y = current[-1]
        px, py = tx(x), ty(y)
        r = STROKE_WIDTH // 2 + 1
        draw.ellipse((px - r - 3, py - r - 3, px + r + 3, py + r + 3),
                     fill=PANEL_BG, outline=CLAY, width=2)
        draw.ellipse((px - r, py - r, px + r, py + r), fill=PEN_DOT)


def render_frame(scenery, text, index, total, completed=None, current=None,
                 xs=None, ys=None, drawing=False, temperature=0.3):
    img = scenery.copy()
    draw = ImageDraw.Draw(img)
    draw_demo_ui(draw, text, index, total, drawing=drawing, temperature=temperature)
    if completed is not None and xs is not None and ys is not None:
        draw_strokes(draw, completed, current or [], xs, ys)
    return np.asarray(img, dtype=np.uint8)


def emit(proc, frame_rgb, stats):
    try:
        proc.stdin.write(frame_rgb.tobytes())
    except (BrokenPipeError, OSError) as e:
        raise RuntimeError(f"ffmpeg subprocess died: {e}") from e
    stats["frames"] += 1


def hold(proc, frame_rgb, seconds, stats):
    n = max(1, int(seconds * FPS))
    buf = frame_rgb.tobytes()
    try:
        for _ in range(n):
            proc.stdin.write(buf)
    except (BrokenPipeError, OSError) as e:
        raise RuntimeError(f"ffmpeg subprocess died: {e}") from e
    stats["frames"] += n


def stream_line(model, stoi, std, text, index, total, scenery, proc, stats, line_cfg):
    temperature = line_cfg["temperature"]
    rng = np.random.default_rng(line_cfg["seed"])
    c, c_mask = onehot(text, stoi)
    cx = cy = 0.0
    xs, ys = [0.0], [0.0]
    completed = []
    current = []

    for dx, dy, pen_up, _phi in model.sample_iter(
        c, c_mask, temperature=temperature, rng=rng,
    ):
        cx += dx * std
        cy += dy * std
        xs.append(cx)
        ys.append(cy)
        current.append((cx, cy))

        frame = render_frame(scenery, text, index, total, completed, current,
                             xs, ys, drawing=True, temperature=temperature)
        emit(proc, frame, stats)

        if pen_up:
            if len(current) >= 2:
                completed.append(current[:])
            current = []

    if len(current) >= 2:
        completed.append(current[:])
    done = render_frame(scenery, text, index, total, completed, [], xs, ys,
                        drawing=True, temperature=temperature)
    hold(proc, done, HOLD_AFTER_LINE, stats)
    return done


def main():
    line_config = build_line_config(PICKS)
    model, stoi, std = load_model()
    scenery = build_scenery()

    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{WIDTH}x{HEIGHT}", "-pix_fmt", "rgb24", "-r", str(FPS),
            "-i", "-",
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            OUTPUT,
        ],
        stdin=subprocess.PIPE,
    )

    stats = {"frames": 0}
    last_frame = None

    try:
        for i, text in enumerate(LINES):
            cfg = line_config[i]
            print(f"streaming line {i + 1}/{len(LINES)}: {text!r}  t={cfg['temperature']}  seed={cfg['seed']}")
            last_frame = stream_line(model, stoi, std, text, i, len(LINES), scenery, proc, stats, cfg)

        hold(proc, last_frame, HOLD_END, stats)
    except (BrokenPipeError, OSError, RuntimeError) as e:
        proc.stdin.close()
        proc.wait()
        raise SystemExit(f"ffmpeg failed: {e}") from e

    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed with code {proc.returncode}")
    duration = stats["frames"] / FPS
    print(f"saved {OUTPUT} ({duration:.1f}s, {stats['frames']} frames @ {FPS}fps)")

    if Path(AUDIO).exists():
        muxed = OUTPUT.replace(".mp4", "_muxed.mp4")
        mux = subprocess.run(
            [
                "ffmpeg", "-y", "-i", OUTPUT, "-i", AUDIO,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", muxed,
            ],
            capture_output=True,
            text=True,
        )
        if mux.returncode != 0:
            raise SystemExit(f"audio mux failed: {mux.stderr}")
        Path(muxed).replace(OUTPUT)
        print(f"added audio from {AUDIO}")


if __name__ == "__main__":
    main()
