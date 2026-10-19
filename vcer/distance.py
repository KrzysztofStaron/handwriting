"""Generalized weighted edit distance.

A substitution may replace one to K reference characters with one to K
hypothesis characters. Identical spans cost nothing. Insertion and deletion
are one character at a time.

`edit_distance(..., fast=False)` is the direct recurrence. `fast=True` looks
up the same costs from integer-indexed arrays. Both return the same distance.
"""

INF = float("inf")


def edit_distance(ref, hyp, table, k=2, backtrace=False, fast=False):
    """Minimum cost to turn `ref` into `hyp` under `table`.

    With `backtrace`, return `(cost, ops)` where each op is
    `{op, ref, hyp, cost}`. `op` is `eq`, `sub`, `ins`, or `del`.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if fast:
        return _edit_distance_fast(ref, hyp, table, k, backtrace)
    return _edit_distance_ref(ref, hyp, table, k, backtrace)


def _edit_distance_ref(ref, hyp, table, k, backtrace):
    n, m = len(ref), len(hyp)
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    prev = [[None] * (m + 1) for _ in range(n + 1)] if backtrace else None

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best = INF
            choice = None
            if i:
                cand = dp[i - 1][j] + table.del_cost(ref[i - 1])
                if cand < best:
                    best = cand
                    choice = (i - 1, j, "del", ref[i - 1], "")
            if j:
                cand = dp[i][j - 1] + table.ins_cost(hyp[j - 1])
                if cand < best:
                    best = cand
                    choice = (i, j - 1, "ins", "", hyp[j - 1])
            for a in range(1, k + 1):
                if a > i:
                    break
                src = ref[i - a:i]
                for b in range(1, k + 1):
                    if b > j:
                        break
                    dst = hyp[j - b:j]
                    cand = dp[i - a][j - b] + table.sub_cost(src, dst)
                    if cand < best:
                        best = cand
                        choice = (i - a, j - b, "eq" if src == dst else "sub", src, dst)
            dp[i][j] = best
            if prev is not None:
                prev[i][j] = choice
    return _finish(dp, prev, n, m, backtrace)


def _edit_distance_fast(ref, hyp, table, k, backtrace):
    """Same recurrence, with span costs cached on (a, b, i, j)."""
    n, m = len(ref), len(hyp)
    sub = {}
    for a in range(1, k + 1):
        for b in range(1, k + 1):
            rows = [None] * (n + 1)
            for i in range(a, n + 1):
                src = ref[i - a:i]
                row = [0.0] * (m + 1)
                for j in range(b, m + 1):
                    row[j] = table.sub_cost(src, hyp[j - b:j])
                rows[i] = row
            sub[(a, b)] = rows
    del_c = [table.del_cost(ch) for ch in ref]
    ins_c = [table.ins_cost(ch) for ch in hyp]

    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    prev = [[None] * (m + 1) for _ in range(n + 1)] if backtrace else None
    for i in range(n + 1):
        dpi = dp[i]
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best = INF
            choice = None
            if i:
                cand = dp[i - 1][j] + del_c[i - 1]
                if cand < best:
                    best = cand
                    choice = (i - 1, j, "del", ref[i - 1], "")
            if j:
                cand = dpi[j - 1] + ins_c[j - 1]
                if cand < best:
                    best = cand
                    choice = (i, j - 1, "ins", "", hyp[j - 1])
            for a in range(1, k + 1):
                if a > i:
                    break
                for b in range(1, k + 1):
                    if b > j:
                        break
                    cand = dp[i - a][j - b] + sub[(a, b)][i][j]
                    if cand < best:
                        src = ref[i - a:i]
                        dst = hyp[j - b:j]
                        best = cand
                        choice = (i - a, j - b, "eq" if src == dst else "sub", src, dst)
            dpi[j] = best
            if prev is not None:
                prev[i][j] = choice
    return _finish(dp, prev, n, m, backtrace)


def _finish(dp, prev, n, m, backtrace):
    total = dp[n][m]
    if not backtrace:
        return total
    ops = []
    i, j = n, m
    while i or j:
        choice = prev[i][j]
        if choice is None:
            break
        pi, pj, kind, src, dst = choice
        ops.append({
            "op": kind,
            "ref": src,
            "hyp": dst,
            "cost": dp[i][j] - dp[pi][pj],
        })
        i, j = pi, pj
    ops.reverse()
    return total, ops
