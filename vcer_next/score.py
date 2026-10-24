"""String score that charges the visual distance and nothing else.

The trainer's `both` reward keeps 75% plain CER in the string score:

    live = 0.75 * plain_edits + 0.25 * visual_distance

A stored cost of 0.40 then removes 0.85 of a full edit. `score_pair` drops
the plain-CER term, so the same mistake removes 0.40. Plain CER is still
in the diagnostics. Held-out quality stays plain CER. `train_rl.py` is
unchanged.

At K=2 a case change inside a word already costs 0.2. The two-character
span is absent from the table, so the case rule fills it, and the DP
prefers that over the stored ink cost (`W`→`w` is 0.69 in the table and
0.2 in a word). The live blend still charges 0.80 of a full edit, because
plain CER counts the case change as one edit.
"""

from pathlib import Path

from vcer.costs import CostTable
from vcer.distance import edit_distance
from vcer.reward import pair_reward, plain_cer, prepare

LIVE_CER = 0.75
LIVE_VISUAL = 0.25

# Substitutions that led the from-base play log. The digraphs are the pairs
# the hand list made cheap, and they are not the pairs that log made.
OBSERVED = (
    ("o", "a"),
    ("a", "o"),
    ("W", "w"),
    ("T", "t"),
    ("n", "r"),
    ("o", "e"),
    ("h", "n"),
    ("a", "e"),
    ("h", "t"),
)
CURATED = (
    ("d", "ol"),
    ("rn", "m"),
    ("cl", "d"),
)

_TABLE = Path(__file__).resolve().parents[1] / "vcer" / "table.json"


def live_units(plain_edits, visual_distance):
    """Edit-units the live 75/25 string score removes. One full edit is 1."""
    return LIVE_CER * float(plain_edits) + LIVE_VISUAL * float(visual_distance)


def load_table(path=None):
    return CostTable.load(_TABLE if path is None else path)


def measure(ref, hyp, table, k=2):
    """Plain edits, visual distance, live units, and the next units."""
    ref_n = prepare(ref)
    hyp_n = prepare(hyp)
    distance = edit_distance(ref_n, hyp_n, table, k=k, backtrace=False, fast=True)
    if ref_n:
        plain = plain_cer(ref_n, hyp_n) * len(ref_n)
    else:
        plain = float(len(hyp_n))
    return {
        "plain": plain,
        "visual": distance,
        "live": live_units(plain, distance),
        "next": distance,
    }


def in_word(src, dst):
    """Put a one-character pair between matching letters. A longer span stays as it is."""
    if len(src) == 1 and len(dst) == 1:
        return f"a{src}a", f"a{dst}a"
    return src, dst


def score_pair(ref, hyp, table, k=2):
    """`(reward, diagnostics)` with the visual distance as the whole score."""
    return pair_reward(ref, hyp, table, k=k, alpha=1.0)


def format_report(table=None, k=2):
    table = load_table() if table is None else table
    lines = [
        f"{'ref':>3} {'hyp':<4}  {'stored':>6}  {'paid':>7}  {'live':>5}  {'next':>5}",
    ]
    for src, dst in OBSERVED + CURATED:
        ref, hyp = in_word(src, dst)
        row = measure(ref, hyp, table, k=k)
        lines.append(
            f"{src:>3} {dst:<4}  {table.sub_cost(src, dst):6.3f}  "
            f"{row['visual']:7.3f}  {row['live']:5.3f}  {row['next']:5.3f}"
        )
    indel = measure("aa", "aba", table, k=k)
    lines.append("")
    lines.append(
        f"one insertion inside 'aa'   live {indel['live']:.2f}   next {indel['next']:.2f}"
    )
    lines.append("live is above 1 when plain CER counts a digraph as two edits.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_report())
