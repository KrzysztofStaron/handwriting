"""Glyph similarity from rendered handwriting.

`overlap_costs` is what the committed table uses. Each symbol is drawn once
per font, on one shared baseline, and the cost is the fraction of ink that
does not overlap after a small shift. A period stays a dot: glyphs are not
scaled to the same box.

`render_similarity` is the older cosine of a jittered raster. The table build
does not call it.
"""

import sys
from pathlib import Path

import numpy as np

FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/System/Library/Fonts/MarkerFelt.ttc",
    "/System/Library/Fonts/Noteworthy.ttc",
)

# One size and one baseline for every symbol. The canvas leaves room for
# ascenders and descenders at this size; overlap never rescales a glyph.
GLYPH_SIZE = 72
GLYPH_CANVAS = 160
GLYPH_BASELINE = 104
GLYPH_SHIFT = 4
INK_CUT = 0.08


def available_fonts(candidates=FONT_CANDIDATES):
    found = []
    try:
        from PIL import ImageFont
    except ImportError:
        return ()
    for path in candidates:
        if not Path(path).is_file():
            continue
        try:
            ImageFont.truetype(path, 28)
        except OSError:
            continue
        found.append(path)
    return tuple(found)


def _font(path, size, cache):
    key = (path, size)
    if key not in cache:
        from PIL import ImageFont
        cache[key] = ImageFont.truetype(path, size)
    return cache[key]


def pil_embed(text, rng, fonts, cache, canvas=(96, 48), out=(64, 32)):
    from PIL import Image, ImageDraw, ImageFilter

    image = Image.new("L", canvas, 255)
    draw = ImageDraw.Draw(image)
    if fonts:
        path = fonts[int(rng.integers(0, len(fonts)))]
        font = _font(path, 28, cache)
        draw.text((12 + int(rng.integers(-3, 4)), 8 + int(rng.integers(-3, 4))),
                  text, fill=0, font=font)
    else:
        # Fixed stamp so a machine without the Mac fonts still builds a table.
        pixels = image.load()
        jitter = int(rng.integers(0, 3))
        for index, ch in enumerate(text):
            x = 8 + index * 18 + jitter
            y = 10 + (ord(ch) % 5)
            for dy in range(16):
                for dx in range(10):
                    if ((ord(ch) * 17 + dx * 3 + dy) % 7) < 3:
                        px, py = x + dx, y + dy
                        if 0 <= px < canvas[0] and 0 <= py < canvas[1]:
                            pixels[px, py] = 0
        rng.random()
    angle = float(rng.uniform(-6.0, 6.0))
    image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
    width = int(rng.integers(0, 3))
    if width == 1:
        image = image.filter(ImageFilter.MinFilter(3))
    elif width == 2:
        image = image.filter(ImageFilter.MaxFilter(3))
    image = image.resize(out, Image.BILINEAR)
    ink = 1.0 - np.asarray(image, dtype=np.float64) / 255.0
    return ink.ravel()


def _unit(vector):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return vector
    return vector / norm


def _ink_from_image(image, cut=INK_CUT):
    ink = 1.0 - np.asarray(image, dtype=np.float32) / 255.0
    ink[ink < cut] = 0.0
    return ink


def _edge_ink(mask):
    return float(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum())


def _same_ink(left, right, tol=0.02):
    diff = float(np.abs(left - right).sum())
    scale = max(float(left.sum()), float(right.sum()), 1e-6)
    return diff / scale <= tol


def render_glyph(text, font, canvas=GLYPH_CANVAS, baseline=GLYPH_BASELINE):
    """Soft ink for `text`, anchored on the middle of the baseline."""
    from PIL import Image, ImageDraw

    image = Image.new("L", (canvas, canvas), 255)
    draw = ImageDraw.Draw(image)
    draw.text((canvas // 2, baseline), text, fill=0, font=font, anchor="ms")
    return _ink_from_image(image)


def stamp_glyph(text, canvas=GLYPH_CANVAS, baseline=GLYPH_BASELINE):
    """Deterministic ink when no handwriting font loads. Space stays blank."""
    ink = np.zeros((canvas, canvas), dtype=np.float32)
    if len(text) != 1 or text == " ":
        return ink
    code = ord(text)
    height = 16
    width = 8 + (code % 5)
    x0 = canvas // 2 - width // 2
    y0 = baseline - height + 2
    for dy in range(height):
        for dx in range(width):
            if (code * 17 + dx * 3 + dy * 5) % 7 >= 3:
                continue
            y = y0 + dy
            x = x0 + dx
            if 0 <= y < canvas and 0 <= x < canvas:
                ink[y, x] = 1.0
    return ink


def glyph_masks(chars, font, canvas=GLYPH_CANVAS, baseline=GLYPH_BASELINE):
    """One mask per character. Space is blank, and a missing-glyph box is blank."""
    blank = np.zeros((canvas, canvas), dtype=np.float32)
    notdef = render_glyph("\uffff", font, canvas, baseline)
    masks = {}
    for ch in chars:
        if ch == " ":
            masks[ch] = blank
            continue
        mask = render_glyph(ch, font, canvas, baseline)
        if _same_ink(mask, notdef):
            mask = blank
        masks[ch] = mask
    return masks


def iou_costs(masks, shift=GLYPH_SHIFT):
    """`(src, dst) -> 1 - best soft IoU` over translations in `[-shift, shift]`.

    Masks share one canvas. A pair with no ink costs 1, so two blank marks are
    not a free substitution. The same key is omitted. Shifts are zero-padded.
    """
    keys = list(masks)
    if not keys:
        return {}
    stack = np.stack([np.asarray(masks[key], dtype=np.float32) for key in keys])
    if stack.ndim != 3:
        raise ValueError("each glyph mask must be a 2-d image")
    count, height, width = stack.shape
    shift = int(shift)
    if shift < 0:
        raise ValueError("shift must be >= 0")
    best = np.zeros((count, count), dtype=np.float32)
    padded = np.pad(stack, ((0, 0), (shift, shift), (shift, shift)))
    chunk = 8
    for dy in range(-shift, shift + 1):
        y0 = shift + dy
        for dx in range(-shift, shift + 1):
            x0 = shift + dx
            window = padded[:, y0:y0 + height, x0:x0 + width]
            for start in range(0, count, chunk):
                block = window[start:start + chunk, None]
                inter = np.minimum(block, stack[None]).sum(axis=(2, 3))
                union = np.maximum(block, stack[None]).sum(axis=(2, 3))
                iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 1e-6)
                stop = start + block.shape[0]
                best[start:stop] = np.maximum(best[start:stop], iou)
    cost = 1.0 - best
    out = {}
    for index, src in enumerate(keys):
        row = cost[index]
        for other, dst in enumerate(keys):
            if index == other:
                continue
            out[(src, dst)] = float(row[other])
    return out


def _symmetrize(costs, places=4):
    out = {}
    for (src, dst), cost in costs.items():
        if (src, dst) in out:
            continue
        other = costs.get((dst, src), cost)
        shared = round((float(cost) + float(other)) / 2.0, places)
        out[(src, dst)] = shared
        out[(dst, src)] = shared
    return out


def overlap_costs(chars, masks=None, fonts=None, shift=GLYPH_SHIFT, size=GLYPH_SIZE):
    """Ink-overlap cost for every ordered pair of distinct single characters.

    Spans longer than one character are ignored. With `masks`, those arrays are
    the only view. Otherwise each available font is averaged in, and a font that
    drew no ink for either side is skipped. If no font drew both, the cost is 1.
    """
    chars = [ch for ch in dict.fromkeys(chars) if isinstance(ch, str) and len(ch) == 1]
    if masks is not None:
        chosen = {ch: np.asarray(masks[ch], dtype=np.float32) for ch in chars}
        return _symmetrize(iou_costs(chosen, shift=shift))
    fonts = available_fonts() if fonts is None else tuple(fonts)
    if not fonts:
        stamped = {ch: stamp_glyph(ch) for ch in chars}
        return _symmetrize(iou_costs(stamped, shift=shift))
    totals = {}
    counts = {}
    cache = {}
    clipped = []
    for path in fonts:
        font = _font(path, size, cache)
        rendered = glyph_masks(chars, font)
        masses = {ch: float(rendered[ch].sum()) for ch in chars}
        drawn = sum(mass > 1e-6 for mass in masses.values())
        print(f"{Path(path).name}: {drawn}/{len(chars)} glyphs with ink", file=sys.stderr)
        for ch, mask in rendered.items():
            if masses[ch] > 1e-6 and _edge_ink(mask) > 0:
                clipped.append(f"{Path(path).name}:{ch}")
        table = iou_costs(rendered, shift=shift)
        for (src, dst), cost in table.items():
            if masses[src] <= 1e-6 or masses[dst] <= 1e-6:
                continue
            key = (src, dst)
            totals[key] = totals.get(key, 0.0) + cost
            counts[key] = counts.get(key, 0) + 1
    if clipped:
        preview = ", ".join(clipped[:12])
        print(f"glyph ink touches the canvas edge: {preview}", file=sys.stderr)
    averaged = {}
    for src in chars:
        for dst in chars:
            if src == dst:
                continue
            key = (src, dst)
            if counts.get(key, 0) == 0:
                averaged[key] = 1.0
            else:
                averaged[key] = totals[key] / counts[key]
    return _symmetrize(averaged)


def rescale_costs(raw, low_q=5.0, low=0.2, mid=0.9):
    """Map the `low_q` percentile to `low` and the median to `mid`."""
    if not raw:
        return {}
    values = np.asarray(list(raw.values()), dtype=np.float64)
    p_low = float(np.percentile(values, low_q))
    p_mid = float(np.percentile(values, 50))
    span = p_mid - p_low
    scaled = {}
    for key, value in raw.items():
        if span < 1e-8:
            mapped = mid
        else:
            mapped = low + (mid - low) * ((value - p_low) / span)
        scaled[key] = float(min(1.0, max(0.0, mapped)))
    return scaled


def render_similarity(spans, seed=0, n_views=4, embed=None, fonts=None):
    """Cosine costs for every ordered pair of distinct spans. Identical pairs are omitted."""
    spans = list(dict.fromkeys(spans))
    rng = np.random.default_rng(seed)
    if embed is None:
        fonts = available_fonts() if fonts is None else tuple(fonts)
        cache = {}

        def embed(text, local_rng, fonts=fonts, cache=cache):
            return pil_embed(text, local_rng, fonts, cache)

    vectors = {}
    for span in spans:
        views = [_unit(np.asarray(embed(span, rng), dtype=np.float64)) for _ in range(n_views)]
        vectors[span] = _unit(np.mean(views, axis=0))
    raw = {}
    keys = list(vectors)
    for src in keys:
        for dst in keys:
            if src == dst:
                continue
            sim = float(np.dot(vectors[src], vectors[dst]))
            sim = max(-1.0, min(1.0, sim))
            raw[(src, dst)] = 1.0 - sim
    return rescale_costs(raw)
