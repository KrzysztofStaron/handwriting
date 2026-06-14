"""Export every training source into one file: data/dataset.npz.

Merges the IAM lines (data/strokes.npy + data/sentences.txt) with the
hand-collected canvas samples (collected_strokes.npz) into a single dataset that
is ALREADY normalised by std and ALREADY in (dx, dy, pen_up) order -- so training
can load it straight from one place with no per-source preprocessing.

    uv run export_dataset.py            # -> data/dataset.npz

Layout of the .npz:
    strokes : object array of (T, 3) float32 arrays, (dx, dy, pen_up), std-normalised
    texts   : object array of str, the line each stroke sequence spells
    std     : the offset std used to normalise (needed to denormalise at draw time)
"""

import os
import numpy as np
from data import load_data, build_vocab

OUT = "data/dataset.npz"
COLLECTED = "collected_strokes.npz"


def main():
    strokes, sentences, std = load_data()
    stoi = build_vocab(sentences)

    out_strokes, out_texts = [], []

    # IAM lines: normalise offsets by std, reorder (pen, dx, dy) -> (dx, dy, pen).
    for s, sent in zip(strokes, sentences):
        s = s.astype(np.float32).copy()
        s[:, 1:] /= std
        s = s[:, [1, 2, 0]]
        out_strokes.append(s)
        out_texts.append(sent)
    print(f"+{len(out_strokes)} IAM lines")

    # Collected canvas samples are already std-normalised and (dx, dy, pen)-ordered
    # (see import_collected.py), so they go in verbatim. Drop any with chars the IAM
    # vocab can't represent, matching StrokeDataset's out-of-vocab guard.
    if os.path.exists(COLLECTED):
        data = np.load(COLLECTED, allow_pickle=True)
        added = skipped = 0
        for arr, text in zip(data["strokes"], data["texts"]):
            text = str(text)
            if any(ch not in stoi for ch in text):
                skipped += 1
                continue
            out_strokes.append(np.asarray(arr, dtype=np.float32))
            out_texts.append(text)
            added += 1
        msg = f"+{added} collected canvas samples"
        if skipped:
            msg += f", skipped {skipped} out-of-vocab"
        print(msg)
    else:
        print(f"{COLLECTED} not found -- exporting IAM only")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT,
             strokes=np.array(out_strokes, dtype=object),
             texts=np.array(out_texts, dtype=object),
             std=np.float32(std))
    print(f"\nsaved {OUT}: {len(out_strokes)} samples, std={std:.4f}")


if __name__ == "__main__":
    main()
