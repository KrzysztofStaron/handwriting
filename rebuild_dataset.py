"""Rebuild data/dataset.npz, honouring the manual QA done in the review tool.

The review tool (collector/src/routes/review) records bad samples in
data/rejected_review.json as a flat list of string ids:

    "iam:<index>"             -> index into the IAM portion of the dataset
    "collected:<filename>"    -> a collected/*.json canvas sample
    "collected2:<filename>"   -> a collected2/*.json canvas sample

This script rebuilds the merged training set from the ORIGINAL sources so the
rejections are actually applied, rather than re-filtering the (stale) merged file:

  1. IAM lines (data/strokes.npy + data/sentences.txt), dropping rejected indices.
     IAM keeps the same order as export_dataset.py, so iam:<index> lines up with
     strokes.npy[index] (and with the old dataset.npz positions 0..len(IAM)-1).
  2. Every hand-collected canvas sample in collected/ and collected2/, converted
     fresh through import_collected.file_to_array (the single source of truth for
     canvas->IAM normalisation), dropping rejected / degenerate / out-of-vocab.

Output matches export_dataset.py's layout exactly:
    strokes : object array of (T, 3) float32, (dx, dy, pen_up), std-normalised
    texts   : object array of str
    std     : the offset std used to normalise
"""

import json
import shutil
from glob import glob
from pathlib import Path

import numpy as np

from data import load_data, build_vocab
from import_collected import file_to_array

OUT = Path("data/dataset.npz")
REJECTED_FILE = Path("data/rejected_review.json")
COLLECTED_DIRS = ["collected", "collected2"]


def load_rejected() -> set[str]:
    if not REJECTED_FILE.exists():
        return set()
    raw = REJECTED_FILE.read_text().strip()
    if not raw:
        return set()
    parsed = json.loads(raw)
    return set(parsed) if isinstance(parsed, list) else set()


def main():
    rejected = load_rejected()
    rejected_iam = {int(r.split(":", 1)[1]) for r in rejected if r.startswith("iam:")}
    print(f"rejected: {len(rejected)} total "
          f"({len(rejected_iam)} IAM, {len(rejected) - len(rejected_iam)} collected)")

    strokes, sentences, std = load_data()
    stoi = build_vocab(sentences)

    out_strokes, out_texts = [], []

    # 1. IAM lines: drop rejected, normalise by std, reorder (pen, dx, dy) -> (dx, dy, pen).
    kept_iam = dropped_iam = 0
    for i, (s, sent) in enumerate(zip(strokes, sentences)):
        if i in rejected_iam:
            dropped_iam += 1
            continue
        s = s.astype(np.float32).copy()
        s[:, 1:] /= std
        s = s[:, [1, 2, 0]]
        out_strokes.append(s)
        out_texts.append(sent)
        kept_iam += 1
    print(f"IAM: +{kept_iam} kept, -{dropped_iam} rejected")

    # 2. Collected canvas samples, freshly converted from JSON. file_to_array returns
    #    arrays already std-normalised and (dx, dy, pen)-ordered, so they go in verbatim.
    added = rej = degen = oov = 0
    for d in COLLECTED_DIRS:
        for path in sorted(glob(f"{d}/*.json")):
            p = Path(path)
            if f"{d}:{p.name}" in rejected:
                rej += 1
                continue
            result = file_to_array(p, std)
            if result is None:
                degen += 1
                continue
            arr, text, _ = result
            if any(ch not in stoi for ch in text):
                oov += 1
                continue
            out_strokes.append(np.asarray(arr, dtype=np.float32))
            out_texts.append(text)
            added += 1
    print(f"collected (collected/ + collected2/): +{added} kept, "
          f"-{rej} rejected, -{degen} degenerate, -{oov} out-of-vocab")

    if OUT.exists():
        backup = OUT.with_suffix(".npz.bak")
        shutil.copy2(OUT, backup)
        print(f"backed up old dataset -> {backup}")

    np.savez(OUT,
             strokes=np.array(out_strokes, dtype=object),
             texts=np.array(out_texts, dtype=object),
             std=np.float32(std))
    print(f"\nsaved {OUT}: {len(out_strokes)} samples "
          f"({kept_iam} IAM + {added} collected), std={std:.4f}")


if __name__ == "__main__":
    main()
