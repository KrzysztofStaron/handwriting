"""Proposed visual orderings, the hacking suite, and the sensitivity sweep.

The triples in `sanity_pairs.jsonl` are proposed labels, not a study. Each row
says hypothesis A looks closer to the reference than hypothesis B. A person
should spot-check them before treating the agreement rate as a human result.
"""

import json
from pathlib import Path

from vcer.costs import CostTable
from vcer.eval.hacking_cases import evaluate
from vcer.reward import pair_reward

SANITY_PATH = Path(__file__).resolve().parent / "sanity_pairs.jsonl"


def _span(ref, start, length, visual, wrong):
    better = ref[:start] + visual + ref[start + length:]
    worse = ref[:start] + wrong + ref[start + length:]
    return {
        "ref": ref,
        "hyp_a": better,
        "hyp_b": worse,
        "prefer": "a",
        "source": "proposed",
        "note": f"{ref[start:start + length]!r} -> {visual!r} should beat {wrong!r}",
    }


def proposed_pairs():
    """50+ triples. `hyp_a` is the one that should look closer."""
    rows = [
        _span("d", 0, 1, "ol", "e"),
        _span("m", 0, 1, "rn", "n"),
        _span("w", 0, 1, "vv", "x"),
        _span("world", 0, 1, "vv", "x"),
        _span("mad", 0, 1, "rn", "x"),
        _span("dog", 0, 1, "cl", "e"),
        _span("red", 2, 1, "ol", "e"),
        _span("warm", 0, 1, "vv", "x"),
        _span("moon", 0, 1, "rn", "x"),
        _span("wide", 0, 1, "vv", "x"),
        _span("made", 0, 1, "rn", "x"),
        _span("wood", 0, 1, "vv", "x"),
        _span("mind", 0, 1, "rn", "x"),
        _span("dwell", 0, 1, "cl", "x"),
        _span("and", 2, 1, "ol", "e"),
        _span("hand", 0, 1, "b", "x"),
        _span("love", 2, 1, "u", "x"),
        _span("cat", 1, 1, "o", "x"),
        _span("bet", 0, 1, "h", "x"),
        _span("cost", 1, 1, "a", "x"),
        _span("gel", 0, 1, "9", "x"),
        _span("Oat", 0, 1, "0", "x"),
        _span("Ice", 0, 1, "l", "x"),
        _span("hello", 2, 1, "1", "x"),
        _span("Juliet", 2, 1, "1", "x"),
    ]
    obvious = [
        ("hello", "hello", "xxxxx"),
        ("hello", "he11o", "xxxxx"),
        ("Juliet", "Juliet", "xxxxxxx"),
        ("Juliet", "Ju1iet", "Jxxxxx"),
        ("Romeo", "Romeo", "xxxxx"),
        ("Romeo", "R0meo", "Rxxxx"),
        ("the old man", "the old man", "xxxxxxxxxxx"),
        ("the old man", "the 0ld man", "the xxx man"),
        ("Verona", "Verona", "xxxxxx"),
        ("Verona", "Ver0na", "Vxxxxx"),
        ("wherefore", "wherefore", "xxxxxxxxx"),
        ("wherefore", "wheref0re", "wxxxxxxxx"),
        ("banished", "banished", "xxxxxxxx"),
        ("morning", "morning", "xxxxxxx"),
        ("morning", "m0rning", "xxxxxxx"),
        ("Capulet", "Capulet", "xxxxxxx"),
        ("love", "love", "xxxx"),
        ("nurse", "nurse", "xxxxx"),
        ("sword", "svvord", "sxxxx"),
        ("death", "death", "xxxxx"),
        ("death", "0eath", "xxxxx"),
        ("poison", "poison", "xxxxxx"),
        ("friar", "friar", "xxxxx"),
        ("balcony", "balcony", "xxxxxxx"),
        ("Tybalt", "Tybalt", "xxxxxx"),
        ("Mercutio", "Mercutio", "xxxxxxxx"),
        ("Mantua", "Mantua", "xxxxxx"),
        ("gold", "g0ld", "xxxx"),
        ("lady", "1ady", "xxxx"),
        ("come", "c0me", "xxxx"),
        ("night", "niqht", "nxxxx"),
        ("soft", "s0ft", "xxxx"),
        ("parting", "parting", "xxxxxxx"),
        ("sorrow", "sorrow", "xxxxxx"),
        ("exile", "exi1e", "exxxx"),
    ]
    for ref, hyp_a, hyp_b in obvious:
        rows.append({
            "ref": ref,
            "hyp_a": hyp_a,
            "hyp_b": hyp_b,
            "prefer": "a",
            "source": "proposed",
            "note": "closer transcription should beat the garbled one",
        })
    return rows


def _prefers_a(ref, hyp_a, hyp_b, table, k=2, alpha=0.8):
    reward_a, diag_a = pair_reward(ref, hyp_a, table, k=k, alpha=alpha)
    reward_b, diag_b = pair_reward(ref, hyp_b, table, k=k, alpha=alpha)
    visual = reward_a > reward_b + 1e-9
    plain = diag_a["cer"] + 1e-9 < diag_b["cer"]
    return visual, plain


def agreement(table, pairs=None, k=2, alpha=0.8):
    pairs = proposed_pairs() if pairs is None else pairs
    visual_hits = plain_hits = visual_only = 0
    for row in pairs:
        visual, plain = _prefers_a(row["ref"], row["hyp_a"], row["hyp_b"], table, k=k, alpha=alpha)
        visual_hits += visual
        plain_hits += plain
        visual_only += visual and not plain
    n = len(pairs)
    return {
        "n": n,
        "vcer": visual_hits / n,
        "cer": plain_hits / n,
        "vcer_only": visual_only,
        "vcer_hits": visual_hits,
        "cer_hits": plain_hits,
    }


def sweep(table):
    """Rewards for a few pairs across K, alpha, and the substitution floor."""
    probes = (
        ("d", "ol"),
        ("d", "e"),
        ("world", "vvorld"),
        ("world", "xorld"),
        ("hello", "he11o"),
        ("hello", "hexlo"),
        ("the old man", "the old man"),
        ("the old man", "the old man."),
        ("the old man", "the old mar"),
        ("the old man", "iiiiiiiiiii"),
    )
    lines = ["# vCER sensitivity", ""]
    lines.append("Rewards use the merged table. `K=1` cannot substitute a digraph in one edit,")
    lines.append("so `d -> ol` stops being cheaper than `d -> e` until `K >= 2`.")
    lines.append("")
    lines.append("## K and alpha")
    lines.append("")
    columns = [f"K={k} a={alpha}" for k in (1, 2, 3) for alpha in (0.6, 0.8, 1.0)]
    lines.append("| ref | hyp | " + " | ".join(columns) + " |")
    lines.append("|" + "---|" * (2 + len(columns)))
    for ref, hyp in probes:
        cells = [f"`{ref}`", f"`{hyp}`"]
        for k in (1, 2, 3):
            for alpha in (0.6, 0.8, 1.0):
                reward, _diag = pair_reward(ref, hyp, table, k=k, alpha=alpha)
                cells.append(f"{reward:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Substitution floor")
    lines.append("")
    lines.append("Stored costs are lifted when the substitution floor rises. One-character costs are the unmatched share of the ink.")
    lines.append("")
    lines.append("| floor | d→ol | d→e | trailing dot | one wrong letter | all i |")
    lines.append("|---|---|---|---|---|---|")
    for floor in (0.15, 0.30, 0.50):
        lifted = CostTable(
            table.sub, ins=table.ins, delete=table.delete,
            min_sub_cost=floor, min_indel_cost=table.min_indel_cost,
            case_cost=table.case_cost, k=table.k, source=table.source,
        )
        cells = [f"{floor:.2f}"]
        for ref, hyp in (
            ("d", "ol"),
            ("d", "e"),
            ("the old man", "the old man."),
            ("the old man", "the old mar"),
            ("the old man", "iiiiiiiiiii"),
        ):
            reward, _diag = pair_reward(ref, hyp, lifted)
            cells.append(f"{reward:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Indel floor")
    lines.append("")
    lines.append("| indel | trailing dot | dropped word `the man` | reasonable `the 0ld man` |")
    lines.append("|---|---|---|---|")
    for indel in (0.4, 0.6, 0.8, 1.0):
        lifted = CostTable(
            table.sub, ins=indel, delete=indel,
            min_sub_cost=table.min_sub_cost, min_indel_cost=min(indel, table.min_indel_cost),
            case_cost=table.case_cost, k=table.k, source=table.source,
        )
        cells = [f"{indel:.1f}"]
        for ref, hyp in (
            ("the old man", "the old man."),
            ("the old man", "the man"),
            ("the old man", "the 0ld man"),
        ):
            reward, _diag = pair_reward(ref, hyp, lifted)
            cells.append(f"{reward:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_sanity(path=None):
    path = Path(path) if path else SANITY_PATH
    rows = proposed_pairs()
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    return rows


def load_sanity(path=None):
    path = Path(path) if path else SANITY_PATH
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_reports(table, root):
    root = Path(root)
    rows = write_sanity()
    stats = agreement(table, rows)
    reasonable, hacking_rows, failures = evaluate(
        lambda ref, hyp, cost_table: pair_reward(ref, hyp, cost_table), table)
    (root / "docs" / "sweeps.md").write_text(sweep(table), encoding="utf-8")
    lines = [
        "# vCER validation",
        "",
        "The ordering set is proposed by the hand seed, not labeled by a person.",
        "Spot-check `vcer/eval/sanity_pairs.jsonl` before reading the rate as a human study.",
        "Hypothesis A is the one that should look closer. A metric agrees when it strictly prefers A.",
        "",
        f"- Pairs: {stats['n']}",
        f"- vCER agreement: {stats['vcer_hits']}/{stats['n']} ({stats['vcer']:.3f})",
        f"- Plain CER agreement: {stats['cer_hits']}/{stats['n']} ({stats['cer']:.3f})",
        f"- Pairs vCER gets and plain CER misses: {stats['vcer_only']}",
        "",
        "## Hacking suite",
        "",
        f"Bar: reward for `the old man` -> `the 0ld man` is {reasonable:.3f}.",
        "Every degenerate string has to land strictly below that.",
        "",
        "| case | reward | wCER | below bar |",
        "|---|---|---|---|",
    ]
    for row in hacking_rows:
        lines.append(
            f"| {row['name']} | {row['reward']:.3f} | {row['wcer']:.3f} | "
            f"{'yes' if row['below'] else 'NO'} |"
        )
    lines.append("")
    if failures:
        lines.append(f"Failures: {len(failures)}.")
    else:
        lines.append("The hacking suite passed.")
    lines.append("")
    lines.append("## Ablation to run later")
    lines.append("")
    lines.append("Do not start a long job for this without a go-ahead. The live trainer is a separate process.")
    lines.append("")
    lines.append("1. Plain CER (`--reward exp`) on `data/romeo_juliet.txt`.")
    lines.append("2. vCER (`--reward vcer`) on the same file.")
    lines.append("3. vCER with annealing: start from this table, then linearly raise substitution costs toward 1 so the end of training is close to exact match.")
    lines.append("")
    lines.append("Score each run with plain CER and exact match on text that was not in the prompt file. vCER is only the training signal.")
    lines.append("")
    (root / "docs" / "validation.md").write_text("\n".join(lines), encoding="utf-8")
    return stats, failures
