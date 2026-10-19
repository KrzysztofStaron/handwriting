"""Turn decoded (reference, hypothesis) pairs into a confusion cost table.

The recognizer run already holds TrOCR, so this pass does not load a second
model. `harvest` is the pure counting step: feed it held-out strings when a
baseline pass is available. Those strings must not be the RL prompts.
"""

from collections import Counter

INF = float("inf")


def levenshtein_ops(ref, hyp):
    """Standard unit-cost alignment. Ops are `(eq|sub|ins|del, ref_char, hyp_char)`."""
    n, m = len(ref), len(hyp)
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    prev = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best = INF
            choice = None
            if i and j:
                step = 0 if ref[i - 1] == hyp[j - 1] else 1
                cand = dp[i - 1][j - 1] + step
                if cand < best:
                    best = cand
                    kind = "eq" if step == 0 else "sub"
                    choice = (i - 1, j - 1, kind, ref[i - 1], hyp[j - 1])
            if i:
                cand = dp[i - 1][j] + 1
                if cand < best:
                    best = cand
                    choice = (i - 1, j, "del", ref[i - 1], "")
            if j:
                cand = dp[i][j - 1] + 1
                if cand < best:
                    best = cand
                    choice = (i, j - 1, "ins", "", hyp[j - 1])
            dp[i][j] = best
            prev[i][j] = choice
    ops = []
    i, j = n, m
    while i or j:
        choice = prev[i][j]
        if choice is None:
            break
        pi, pj, kind, src, dst = choice
        ops.append((kind, src, dst))
        i, j = pi, pj
    ops.reverse()
    return ops


def merge_edits(ops, k=2):
    """Collapse a run of adjacent edits into spans of length at most `k`."""
    merged = []
    buf_ref = []
    buf_hyp = []

    def flush():
        src = "".join(buf_ref)
        dst = "".join(buf_hyp)
        buf_ref.clear()
        buf_hyp.clear()
        while src or dst:
            if src and dst:
                a = min(k, len(src))
                b = min(k, len(dst))
                piece_src, piece_dst = src[:a], dst[:b]
                kind = "eq" if piece_src == piece_dst else "sub"
                merged.append((kind, piece_src, piece_dst))
                src, dst = src[a:], dst[b:]
            elif src:
                a = min(k, len(src))
                merged.append(("del", src[:a], ""))
                src = src[a:]
            else:
                b = min(k, len(dst))
                merged.append(("ins", "", dst[:b]))
                dst = dst[b:]

    for kind, src, dst in ops:
        if kind == "eq":
            flush()
            merged.append((kind, src, dst))
        else:
            buf_ref.append(src)
            buf_hyp.append(dst)
    flush()
    return merged


def harvest(pairs, k=2, alpha=0.5, min_count=2, floor=0.15):
    """Map confusion counts to costs in `[floor, 1]`.

    `cost = 1 - P_norm`, and `P_norm` is scaled so the most common kept
    confusion lands on `floor`. Pairs seen fewer than `min_count` times are
    dropped. The same seed and the same pairs always return the same dict.
    """
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    counts = Counter()
    occurrences = Counter()
    for ref, hyp in pairs:
        for length in range(1, k + 1):
            for i in range(len(ref) - length + 1):
                occurrences[ref[i:i + length]] += 1
        for kind, src, dst in merge_edits(levenshtein_ops(ref, hyp), k=k):
            if kind == "sub" and src != dst and 1 <= len(src) <= k and 1 <= len(dst) <= k:
                counts[(src, dst)] += 1
    kept = []
    for pair, count in counts.items():
        if count < min_count:
            continue
        src = pair[0]
        prob = (count + alpha) / (occurrences[src] + alpha)
        kept.append((pair, prob))
    if not kept:
        return {}
    peak = max(prob for _, prob in kept)
    costs = {}
    for pair, prob in kept:
        scaled = 0.0 if peak <= 0 else prob / peak
        cost = 1.0 - (1.0 - floor) * scaled
        costs[pair] = min(1.0, max(floor, cost))
    return costs
