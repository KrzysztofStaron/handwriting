"""Generate handwriting variants for each line × temperature combo.

Produces 16 PNGs + grid for picking temp/seed. The final video is rendered
separately by render_drawing_mp4.py with live stroke-by-stroke animation.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from model_np import NumpyHandwritingModel
from render_drawing_mp4 import (
    BODY,
    CARD,
    CLAY,
    CLAY_SOFT,
    DRAW_PANEL,
    FONT_EYEBROW,
    FONT_PROMPT,
    FONT_SANS,
    FONT_SMALL,
    HAIRLINE,
    INK,
    INK_TEXT,
    LINES,
    MUTED,
    PANEL_BG,
    PANEL_BORDER,
    PANEL_PAD,
    STROKE_WIDTH,
    TEMPS,
    build_scenery,
    collect_strokes,
    fit_transform,
    load_font,
    load_model,
    onehot,
    variant_seed,
)
OUT_DIR = Path("variants")
THUMB_W, THUMB_H = 420, 240


def generate_strokes(model, stoi, std, text, temperature, seed):
    rng = np.random.default_rng(seed)
    c, c_mask = onehot(text, stoi)
    return collect_strokes(model, c, c_mask, std, temperature, rng=rng)


def draw_strokes_finished(draw, completed, xs, ys):
    if not xs:
        return
    tx, ty = fit_transform(xs, ys)
    for stroke in completed:
        if len(stroke) < 2:
            continue
        draw.line([(tx(x), ty(y)) for x, y in stroke],
                  fill=INK, width=STROKE_WIDTH, joint="curve")


def draw_variant_ui(draw, text, line_i, temperature):
    cx0, cy0, cx1, _cy1 = CARD
    draw.text((cx0 + 40, cy0 + 36), "Live output", font=FONT_SANS, fill=BODY)
    draw.text((cx1 - 120, cy0 + 34), f"temp {temperature}", font=FONT_SMALL, fill=CLAY)

    prompt_top = DRAW_PANEL[3] + 36
    draw.line((cx0 + 40, prompt_top - 12, cx1 - 40, prompt_top - 12), fill=HAIRLINE, width=1)
    draw.text((cx0 + 40, prompt_top), f"LINE {line_i + 1}", font=FONT_EYEBROW, fill=MUTED)
    draw.text((cx0 + 40, prompt_top + 28), f'"{text}"', font=FONT_PROMPT, fill=INK_TEXT)

    pill = (cx0 + 40, prompt_top + 72, cx0 + 200, prompt_top + 100)
    draw.rounded_rectangle(pill, radius=14, fill=CLAY_SOFT)
    draw.text((cx0 + 58, prompt_top + 78), f"t = {temperature}", font=FONT_EYEBROW, fill=CLAY)


def render_variant(scenery, text, line_i, temperature, completed, xs, ys):
    img = scenery.copy()
    draw = ImageDraw.Draw(img)
    draw_variant_ui(draw, text, line_i, temperature)
    draw_strokes_finished(draw, completed, xs, ys)
    return img


def render_thumb(text, line_i, temperature, completed, xs, ys):
    img = Image.new("RGB", (THUMB_W, THUMB_H), PANEL_BG)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((8, 8, THUMB_W - 8, THUMB_H - 8),
                           radius=8, fill=PANEL_BG, outline=PANEL_BORDER)

    if xs:
        # Leave room for header and footer text labels
        pad, margin = 24, 0.85
        tx, ty = fit_transform(xs, ys, panel=(pad, 20, THUMB_W - pad, THUMB_H - 36), pad=0, margin=margin)
        for stroke in completed:
            if len(stroke) < 2:
                continue
            draw.line([(tx(x), ty(y)) for x, y in stroke],
                      fill=INK, width=3, joint="curve")

    font = load_font(13, "sans")
    draw.text((16, THUMB_H - 32), f"L{line_i + 1}  t={temperature}", font=font, fill=MUTED)
    draw.text((16, THUMB_H - 52), text[:28] + ("…" if len(text) > 28 else ""),
              font=load_font(11, "mono"), fill=INK_TEXT)
    return img


def build_grid(thumbs):
    cols, rows = len(TEMPS), len(LINES)
    gap = 16
    header_h = 48
    grid_w = cols * THUMB_W + (cols + 1) * gap
    grid_h = rows * THUMB_H + (rows + 1) * gap + header_h
    grid = Image.new("RGB", (grid_w, grid_h), (250, 249, 245))
    draw = ImageDraw.Draw(grid)

    font = load_font(18, "sans")
    font_sm = load_font(14, "mono")
    for j, temp in enumerate(TEMPS):
        x = gap + j * (THUMB_W + gap) + THUMB_W // 2
        draw.text((x, 14), f"t = {temp}", font=font, fill=INK_TEXT, anchor="mm")
    for i, text in enumerate(LINES):
        y = header_h + gap + i * (THUMB_H + gap) + THUMB_H // 2
        label = f"L{i + 1}"
        draw.text((14, y), label, font=font_sm, fill=MUTED, anchor="lm")

    for i in range(rows):
        for j in range(cols):
            x = gap + j * (THUMB_W + gap)
            y = header_h + gap + i * (THUMB_H + gap)
            grid.paste(thumbs[i][j], (x, y))

    return grid


def main():
    model, stoi, std = load_model()
    scenery = build_scenery()
    OUT_DIR.mkdir(exist_ok=True)

    thumbs = [[None] * len(TEMPS) for _ in LINES]
    manifest = []

    for i, text in enumerate(LINES):
        for j, temp in enumerate(TEMPS):
            seed = variant_seed(i, temp)
            completed, xs, ys = generate_strokes(model, stoi, std, text, temp, seed)
            fname = f"line{i + 1:02d}_temp{temp:.1f}.png"
            path = OUT_DIR / fname

            img = render_variant(scenery, text, i, temp, completed, xs, ys)
            img.save(path)

            thumbs[i][j] = render_thumb(text, i, temp, completed, xs, ys)
            manifest.append({
                "file": fname,
                "line": i + 1,
                "text": text,
                "temperature": temp,
                "seed": seed,
                "points": len(xs) - 1,
            })
            print(f"saved {path} ({len(xs) - 1} points)")

    grid = build_grid(thumbs)
    grid_path = OUT_DIR / "grid.png"
    grid.save(grid_path)
    print(f"saved {grid_path}")

    index_path = OUT_DIR / "index.txt"
    lines = ["Pick your favorites — note the filename:\n"]
    for entry in manifest:
        lines.append(
            f"{entry['file']:22}  line {entry['line']}  t={entry['temperature']}  "
            f"seed={entry['seed']}  \"{entry['text']}\""
        )
    index_path.write_text("\n".join(lines) + "\n")
    print(f"saved {index_path}")


if __name__ == "__main__":
    main()
