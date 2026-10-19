"""Build the committed cost table. Seed 0. No model is loaded."""

import json
from pathlib import Path

from vcer.costs import CostTable
from vcer.table_build.from_render import available_fonts, render_similarity
from vcer.table_build.hand_seed import SEED_PATH, load_hand_seed
from vcer.table_build.merge import RENDER_ONLY_FLOOR, merge_costs

ROOT = Path(__file__).resolve().parents[2]
TABLE_PATH = Path(__file__).resolve().parents[1] / "table.json"


def spans_for(alphabet, hand):
    spans = set(alphabet)
    for src, dst in hand:
        spans.add(src)
        spans.add(dst)
    return sorted(spans)


def build_table(alphabet, seed=0, hand=None, render=None):
    if hand is None:
        hand, _meta = load_hand_seed()
    if render is None:
        render = render_similarity(spans_for(alphabet, hand), seed=seed)
    fonts = available_fonts()
    merged = merge_costs(hand, render=render)
    source = (
        f"hand:{SEED_PATH.name}+render seed={seed} "
        f"render_only_floor={RENDER_ONLY_FLOOR} "
        f"fonts={','.join(Path(path).name for path in fonts) or 'pixel-stamp'}"
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
        "Hand-seed pairs are an upper bound. Render-only pairs are at least 0.55.",
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
    table = build_table(alphabet, seed=0)
    table.save(TABLE_PATH)
    write_top_confusions(table, ROOT / "docs" / "top_confusions.md")
    from vcer.eval.validate import write_reports
    write_reports(table, ROOT)
    payload = table.to_payload()
    print(f"wrote {TABLE_PATH}  pairs {len(payload['pairs'])}  hash {payload['hash']}")
    print(table.source)


if __name__ == "__main__":
    main()
