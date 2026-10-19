"""Build the committed cost table. No model is loaded."""

import json
from pathlib import Path

from vcer.costs import CostTable
from vcer.table_build.from_render import (
    GLYPH_CANVAS, GLYPH_SHIFT, GLYPH_SIZE, available_fonts, overlap_costs,
)
from vcer.table_build.hand_seed import SEED_PATH, load_hand_seed

ROOT = Path(__file__).resolve().parents[2]
TABLE_PATH = Path(__file__).resolve().parents[1] / "table.json"


def combine_overlap(hand, overlap):
    """Single-character costs are ink overlap. A hand-seed pair can only lower one.

    Spans longer than one character are copied from the hand list. An overlap
    entry whose span is longer than one character is dropped, so rendering
    cannot invent a new two-letter substitution.
    """
    hand = dict(hand or {})
    merged = {}
    for (src, dst), cost in overlap.items():
        if len(src) != 1 or len(dst) != 1 or src == dst:
            continue
        cost = float(cost)
        seed = hand.get((src, dst))
        if seed is not None:
            cost = min(cost, float(seed))
        merged[(src, dst)] = cost
    for (src, dst), cost in hand.items():
        if len(src) == 1 and len(dst) == 1:
            if (src, dst) not in merged:
                merged[(src, dst)] = float(cost)
            continue
        merged[(src, dst)] = float(cost)
    return merged


def build_table(alphabet, hand=None, overlap=None):
    if hand is None:
        hand, _meta = load_hand_seed()
    fonts = available_fonts()
    if overlap is None:
        chars = [ch for ch in alphabet if len(ch) == 1]
        overlap = overlap_costs(chars, fonts=fonts)
    merged = combine_overlap(hand, overlap)
    names = ",".join(Path(path).name for path in fonts) or "pixel-stamp"
    source = (
        f"hand:{SEED_PATH.name}+ink-overlap "
        f"size={GLYPH_SIZE} canvas={GLYPH_CANVAS} shift={GLYPH_SHIFT} "
        f"fonts={names}"
    )
    # The indel cost is the floor. A stray mark is cheaper than a random
    # substitution, and a deleted character is not free.
    return CostTable(
        merged,
        ins=0.6,
        delete=0.6,
        min_sub_cost=0.15,
        min_indel_cost=0.6,
        case_cost=0.2,
        k=2,
        source=source,
    )


def write_top_confusions(table, path, limit=50):
    pairs = sorted(table.sub.items(), key=lambda item: (item[1], item[0]))[:limit]
    lines = [
        "# Cheapest non-identical substitutions",
        "",
        "One-character costs are the share of ink that does not overlap.",
        "A hand-listed pair can only make that cheaper. Two-letter pairs are only the hand list.",
        "Phase 3a (baseline confusions) was not run: the recognizer trainer already holds TrOCR.",
        "",
        "| ref | hyp | cost |",
        "|---|---|---|",
    ]
    for (src, dst), cost in pairs:
        lines.append(f"| `{src}` | `{dst}` | {cost:.3f} |")
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    alphabet = list(json.loads((ROOT / "stoi.json").read_text(encoding="utf-8")))
    table = build_table(alphabet)
    table.save(TABLE_PATH)
    write_top_confusions(table, ROOT / "docs" / "top_confusions.md")
    from vcer.eval.validate import write_reports
    write_reports(table, ROOT)
    payload = table.to_payload()
    print(f"wrote {TABLE_PATH}  pairs {len(payload['pairs'])}  hash {payload['hash']}")
    print(table.source)


if __name__ == "__main__":
    main()
