"""What the loaded vCER table actually charges, and why.

The training table prices every symbol by how much ink it shares with every
other symbol. A hand-listed pair can only make that cheaper. Two-letter pairs
are only the hand list. This module labels each discounted pair and explains
one alignment, so a rule can be checked without reading the JSON.
"""

from pathlib import Path

from vcer.costs import CostTable
from vcer.distance import edit_distance
from vcer.reward import pair_reward, prepare
from vcer.table_build.hand_seed import SEED_PATH, load_hand_seed
from vcer.table_build.merge import RENDER_ONLY_FLOOR

_TABLE = Path(__file__).resolve().parent / "table.json"
_ALPHA = 0.8


def _origin(src, dst, cost, hand):
    seed = hand.get((src, dst))
    if seed is not None:
        if cost + 1e-9 < float(seed):
            return "hand+render", float(seed)
        return "hand", float(seed)
    if cost >= 1.0 - 1e-9:
        return "full", None
    return "render", None


def rule_report(table_path=None, seed_path=None):
    """Discounted substitutions, plus the floors that produced them."""
    table_path = Path(table_path) if table_path else _TABLE
    table = CostTable.load(table_path)
    hand, _meta = load_hand_seed(seed_path or SEED_PATH)
    rules = []
    counts = {"pairs": 0, "full": 0, "hand": 0, "hand_render": 0, "floor": 0, "render": 0}
    for (src, dst), cost in table.sub.items():
        counts["pairs"] += 1
        origin, seed = _origin(src, dst, cost, hand)
        if origin == "full":
            counts["full"] += 1
            continue
        bucket = {"hand": "hand", "hand+render": "hand_render", "floor": "floor", "render": "render"}[origin]
        counts[bucket] += 1
        rules.append({
            "ref": src,
            "hyp": dst,
            "cost": cost,
            "origin": origin,
            "seed": seed,
            "cheaper_than_delete": cost < table.min_indel_cost - 1e-12,
        })
    rules.sort(key=lambda item: (item["cost"], item["ref"], item["hyp"]))
    missing = sorted(f"{src}->{dst}" for (src, dst) in hand if (src, dst) not in table.sub)
    return {
        "table": str(table_path),
        "hash": table.to_payload()["hash"],
        "source": table.source,
        "k": table.k,
        "alpha": _ALPHA,
        "ins": table.ins_cost(""),
        "delete": table.del_cost(""),
        "min_sub_cost": table.min_sub_cost,
        "min_indel_cost": table.min_indel_cost,
        "case_cost": table.case_cost,
        "render_floor": RENDER_ONLY_FLOOR,
        "counts": counts,
        "hand_missing": missing,
        "rules": rules,
    }


def _why(op, table, hand):
    kind = op["op"]
    if kind == "eq":
        return "same characters", "same"
    if kind == "ins":
        return f"any insertion costs {table.ins_cost(''):.2f}", "indel"
    if kind == "del":
        return f"any deletion costs {table.del_cost(''):.2f}", "indel"
    src, dst = op["ref"], op["hyp"]
    listed = table.sub.get((src, dst))
    if listed is not None and abs(listed - op["cost"]) <= 1e-6:
        origin, seed = _origin(src, dst, listed, hand)
        if origin == "hand+render":
            return f"hand list said {seed:.2f}, ink overlap pulled it to {listed:.2f}", origin
        if origin == "hand":
            return "hand list", origin
        if origin == "floor":
            return f"ink overlap, held at the {RENDER_ONLY_FLOOR:.2f} floor", origin
        return "ink overlap", origin
    if src.casefold() == dst.casefold() and abs(op["cost"] - table.case_cost) <= 1e-6:
        return "same letters, different case, no listed pair", "case"
    return "no listed pair, full substitution", "default"


def explain_pair(ref, hyp, table_path=None, seed_path=None, alpha=_ALPHA, k=None):
    """Score one pair and name the rule behind each edit."""
    if len(str(ref)) > 500 or len(str(hyp)) > 500:
        raise ValueError("ref and hyp must be at most 500 characters")
    table_path = Path(table_path) if table_path else _TABLE
    table = CostTable.load(table_path)
    hand, _meta = load_hand_seed(seed_path or SEED_PATH)
    span = table.k if k is None else int(k)
    reward, diag = pair_reward(ref, hyp, table, k=span, alpha=alpha, fast=False)
    _distance, raw_ops = edit_distance(
        prepare(ref), prepare(hyp), table, k=span, backtrace=True, fast=False)
    ops = []
    for op in raw_ops:
        why, origin = _why(op, table, hand)
        ops.append({
            "op": op["op"],
            "ref": op["ref"],
            "hyp": op["hyp"],
            "cost": op["cost"],
            "why": why,
            "origin": origin,
        })
    return {
        "ref": diag["ref"],
        "hyp": diag["hyp"],
        "reward": reward,
        "wcer": diag["wcer"],
        "cer": diag["cer"],
        "exact": diag["exact"],
        "distance": diag["distance"],
        "alpha": alpha,
        "k": span,
        "ops": ops,
    }
