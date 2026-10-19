"""Glyph similarity from rendered handwriting.

Each span is drawn a few times, averaged, and compared by cosine similarity.
`embed` can be replaced; the default draws with Pillow and falls back to a
pixel stamp when no handwriting font loads.
"""

from pathlib import Path

import numpy as np

FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/System/Library/Fonts/MarkerFelt.ttc",
    "/System/Library/Fonts/Noteworthy.ttc",
)


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
