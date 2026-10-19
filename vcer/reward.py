"""Visually weighted CER, as a bounded reward on decoded strings.

    wcer   = distance(ref, hyp) / max(len(ref), 1)
    reward = alpha * (1 - min(wcer, 1)) + (1 - alpha) * exact

`exact` is 1 when the normalized strings match. An empty reference scores
1 when the hypothesis is empty and 0 otherwise.
"""

import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from vcer.costs import CostTable
from vcer.distance import edit_distance

_QUOTES = str.maketrans({
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2014": "-", "\u2013": "-",
})

_DEFAULT_TABLE = Path(__file__).resolve().parent / "table.json"
_CACHE = {}


def tighten_punctuation(text):
    """Drop a space that sits immediately before punctuation.

    TrOCR often writes " :" where the line has ":". That space is not a
    handwriting error. A space after punctuation stays, so "hello,world"
    still differs from "hello, world".
    """
    out = []
    for ch in text:
        if out and out[-1] == " " and unicodedata.category(ch).startswith("P"):
            out[-1] = ch
        else:
            out.append(ch)
    return "".join(out)


def prepare(text, policy="strip_collapse"):
    """NFC, then the whitespace policy. Curly quotes and dashes are straightened.

    `strip_collapse` also drops a space before punctuation.
    """
    text = unicodedata.normalize("NFC", "" if text is None else str(text))
    text = text.translate(_QUOTES)
    if policy == "none":
        return text
    if policy == "strip_collapse":
        return tighten_punctuation(" ".join(text.strip().split()))
    raise ValueError(f"unknown normalization {policy!r}")


def _levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def plain_cer(ref, hyp):
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def _diagnostics(ref, hyp, distance, ops, wcer, cer, exact, reward):
    counts = {"eq": 0, "sub": 0, "ins": 0, "del": 0}
    for op in ops:
        counts[op["op"]] = counts.get(op["op"], 0) + 1
    top = sorted((op for op in ops if op["cost"] > 0), key=lambda op: -op["cost"])[:5]
    return {
        "wcer": wcer,
        "cer": cer,
        "exact": exact,
        "reward": reward,
        "distance": distance,
        "counts": counts,
        "top": top,
        "ref": ref,
        "hyp": hyp,
    }


def pair_reward(ref, hyp, table, k=2, alpha=0.8, normalize="strip_collapse", fast=True):
    """Return `(reward, diagnostics)` for one decoded pair."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    ref_n = prepare(ref, normalize)
    hyp_n = prepare(hyp, normalize)
    exact = ref_n == hyp_n
    if not ref_n:
        reward = 1.0 if exact else 0.0
        wcer = 0.0 if exact else 1.0
        cer = plain_cer(ref_n, hyp_n)
        return reward, _diagnostics(ref_n, hyp_n, 0.0, [], wcer, cer, exact, reward)
    distance, ops = edit_distance(ref_n, hyp_n, table, k=k, backtrace=True, fast=fast)
    wcer = distance / len(ref_n)
    cer = plain_cer(ref_n, hyp_n)
    reward = alpha * (1.0 - min(wcer, 1.0)) + (1.0 - alpha) * float(exact)
    return reward, _diagnostics(ref_n, hyp_n, distance, ops, wcer, cer, exact, reward)


def rewards(refs, hyps, table, k=2, alpha=0.8, normalize="strip_collapse", workers=8):
    """Batch rewards in sample order. One distance per pair, in a small pool."""
    refs = list(refs)
    hyps = list(hyps)
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must have the same length")

    def one(pair):
        return pair_reward(pair[0], pair[1], table, k=k, alpha=alpha, normalize=normalize)[0]

    if len(refs) <= 1 or workers <= 1:
        values = [one(pair) for pair in zip(refs, hyps)]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(refs))) as pool:
            values = list(pool.map(one, zip(refs, hyps)))
    return np.asarray(values, dtype=np.float64)


class VcerReward:
    """Loaded cost table plus the scalar reward used by the trainer."""

    def __init__(self, table, k=2, alpha=0.8, normalize="strip_collapse", table_path=None):
        self.table = table
        self.k = int(k)
        self.alpha = float(alpha)
        self.normalize = normalize
        payload = table.to_payload()
        self.config = {
            "table": str(table_path) if table_path else table.source,
            "hash": payload["hash"],
            "k": self.k,
            "alpha": self.alpha,
            "min_sub_cost": table.min_sub_cost,
            "min_indel_cost": table.min_indel_cost,
            "case_cost": table.case_cost,
            "normalize": normalize,
        }

    @classmethod
    def load(cls, path=None, k=2, alpha=0.8, normalize="strip_collapse"):
        path = Path(path) if path else _DEFAULT_TABLE
        key = str(path.resolve())
        table = _CACHE.get(key)
        if table is None:
            table = CostTable.load(path)
            _CACHE[key] = table
        return cls(table, k=k, alpha=alpha, normalize=normalize, table_path=path)

    def score_pair(self, ref, hyp):
        return pair_reward(
            ref, hyp, self.table, k=self.k, alpha=self.alpha, normalize=self.normalize)

    def score_batch(self, refs, hyps):
        values = []
        details = []
        for ref, hyp in zip(refs, hyps):
            reward, diag = self.score_pair(ref, hyp)
            values.append(reward)
            details.append(diag)
        return values, details
