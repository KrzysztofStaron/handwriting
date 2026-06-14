"""
Convert collected/*.json (from the SvelteKit collector app) into the IAM-compatible
numpy format used by data.py / train.py.

Canvas data does NOT statistically match IAM out of the box, so each sample goes
through a normalisation pipeline:

  1. flip y      — canvas y grows down, IAM y grows up
  2. rescale     — ROBUST drawing height (p5-p95 of y) -> IAM robust body height
                   (~17.4 raw units), so pixel size / screen size stop mattering.
                   Percentile span, not max-min: a stray i-dot or descender would
                   otherwise set the scale and squash the real letters (full-extent
                   norm gave a 1.7x body-size spread). Full extent floats per sample,
                   exactly as IAM does.
  3. resample    — fixed arc-length step (~1 raw unit, IAM's median pen-down step)
                   instead of one point per pointermove event. This removes the
                   dependence on display refresh rate AND writing speed, and lands
                   points-per-character in IAM's range (median ~21) automatically.
  4. offsets     — (dx, dy, pen_up), then divide by the IAM std (std.json), same
                   as data.py does for the original dataset.

Usage:
    uv run import_collected.py              # appends to collected_strokes.npz
    uv run import_collected.py --dry-run    # just print what would be imported
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

COLLECTED_DIR = Path("collected")
OUT_FILE = Path("collected_strokes.npz")
STD_FILE = Path("std.json")

# Measured from data/strokes.npy (median over the IAM training set):
TARGET_HEIGHT = 17.4   # ROBUST line height (p5-p95 of y), raw units. We deliberately
                       # scale by a percentile height, NOT max-min: a single stray
                       # i-dot / descender flourish / pen jump sets max-min and would
                       # squash the real letter body (measured: full-extent norm gave a
                       # 1.7x body-size spread across samples). p5-p95 ignores those
                       # outliers, so letter bodies land at a consistent size that
                       # matches IAM's step-size distribution. Full extent then floats
                       # per sample, exactly as IAM does (IAM only divides by global std).
HEIGHT_PLO, HEIGHT_PHI = 5, 95
TARGET_STEP = 1.4      # MEAN pen-down step length, raw units. The mean (not the
                       # median, 0.96) is what reproduces IAM's points-per-character:
                       # resampled count = arc_length / step, IAM count = arc_length / mean.
IAM_PTS_PER_CHAR = 21.4  # reference only, printed as a sanity check

MIN_HEIGHT_PX = 5.0    # reject drawings flatter than this (probably a scribble/dot)


def load_std() -> float:
    if STD_FILE.exists():
        return json.loads(STD_FILE.read_text())["std"]
    print("Warning: std.json not found — using std=1.0 (offsets will not be normalised)")
    return 1.0


def split_strokes(pts: list[dict]) -> list[np.ndarray]:
    """points [{x, y, pen_up}] -> list of (n, 2) absolute-coordinate strokes."""
    strokes, current = [], []
    for p in pts:
        current.append((p["x"], p["y"]))
        if p["pen_up"] == 1:
            strokes.append(np.array(current, dtype=np.float64))
            current = []
    if current:  # trailing stroke without a pen_up marker
        strokes.append(np.array(current, dtype=np.float64))
    return strokes


def resample_stroke(stroke: np.ndarray, step: float) -> np.ndarray:
    """Resample a (n, 2) polyline at uniform arc-length spacing `step`.

    Keeps the exact first and last points. A stroke shorter than one step
    collapses to its endpoints (or a single point for dots, e.g. an i-dot).
    """
    if len(stroke) < 2:
        return stroke
    seg = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    if total < step:
        # Stroke is too short to resample — keep as-is to avoid duplicating
        # single-point marks (e.g. i-dots) into degenerate 2-point segments.
        return stroke
    n = int(total // step) + 1
    targets = np.linspace(0.0, total, n + 1)
    x = np.interp(targets, arc, stroke[:, 0])
    y = np.interp(targets, arc, stroke[:, 1])
    return np.stack([x, y], axis=1)


def file_to_array(path: Path, std: float) -> tuple[np.ndarray, str, float] | None:
    """
    Returns (array, text, pts_per_char) where array is (T, 3): [dx, dy, pen_up],
    rescaled + resampled to IAM statistics and normalised by std.
    Returns None if the file is malformed or degenerate.
    """
    data = json.loads(path.read_text())
    text: str | None = data.get("text")
    pts: list | None = data.get("points")

    if not text or not pts or len(pts) < 2:
        return None

    strokes = split_strokes(pts)

    # 1. flip y (canvas grows down, IAM grows up)
    for s in strokes:
        s[:, 1] = -s[:, 1]

    # 2. rescale: robust drawing height -> IAM robust line height. Percentile
    #    span (not max-min) so outlier points don't dictate the scale.
    all_pts = np.concatenate(strokes)
    ys = all_pts[:, 1]
    height = np.percentile(ys, HEIGHT_PHI) - np.percentile(ys, HEIGHT_PLO)
    if height < MIN_HEIGHT_PX:
        return None
    scale = TARGET_HEIGHT / height
    strokes = [s * scale for s in strokes]

    # 3. resample at uniform arc length
    strokes = [resample_stroke(s, TARGET_STEP) for s in strokes]

    # 4. flatten to (dx, dy, pen_up) offsets, normalised
    coords = np.concatenate(strokes)
    pen_up = np.zeros(len(coords), dtype=np.float64)
    end = -1
    for s in strokes:
        end += len(s)
        pen_up[end] = 1.0

    dx = np.diff(coords[:, 0], prepend=coords[0, 0]) / std
    dy = np.diff(coords[:, 1], prepend=coords[0, 1]) / std

    arr = np.stack([dx, dy, pen_up], axis=1).astype(np.float32)  # (T, 3)
    return arr, text, len(arr) / len(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = sorted(COLLECTED_DIR.glob("*.json")) if COLLECTED_DIR.exists() else []
    if not files:
        print(f"No files found in {COLLECTED_DIR}/")
        sys.exit(0)

    std = load_std()
    print(f"Using std={std:.4f}  target height={TARGET_HEIGHT}  target step={TARGET_STEP}")
    print(f"(IAM reference: ~{IAM_PTS_PER_CHAR} points per character)\n")

    # Load existing if present
    existing_strokes: list[np.ndarray] = []
    existing_texts: list[str] = []
    existing_filenames: list[str] = []  # preserve order — set(filenames) is used for dedup
    already_imported: set[str] = set()

    if OUT_FILE.exists():
        data = np.load(OUT_FILE, allow_pickle=True)
        existing_strokes = list(data["strokes"])
        existing_texts = list(data["texts"])
        existing_filenames = list(data["filenames"])
        already_imported = set(existing_filenames)
        print(f"Existing: {len(existing_strokes)} samples in {OUT_FILE}")

    new_strokes: list[np.ndarray] = []
    new_texts: list[str] = []
    new_filenames: list[str] = []
    skipped = 0

    for f in files:
        if f.name in already_imported:
            skipped += 1
            continue
        result = file_to_array(f, std)
        if result is None:
            print(f"  skip (degenerate): {f.name}")
            skipped += 1
            continue
        arr, text, ppc = result
        flag = "" if 10 <= ppc <= 40 else "  <-- pts/char far from IAM, check this sample"
        print(f"  + {f.name}  text={text!r}  steps={len(arr)}  pts/char={ppc:.1f}{flag}")
        new_strokes.append(arr)
        new_texts.append(text)
        new_filenames.append(f.name)

    print(f"\nNew: {len(new_strokes)}  Already imported / skipped: {skipped}")

    if args.dry_run:
        print("Dry run — nothing written.")
        return

    if not new_strokes:
        print("Nothing new to import.")
        return

    all_strokes = existing_strokes + new_strokes
    all_texts = existing_texts + new_texts
    all_filenames = existing_filenames + new_filenames

    np.savez(
        OUT_FILE,
        strokes=np.array(all_strokes, dtype=object),
        texts=np.array(all_texts),
        filenames=np.array(all_filenames),
    )
    print(f"Saved {len(all_strokes)} total samples to {OUT_FILE}")


if __name__ == "__main__":
    main()
