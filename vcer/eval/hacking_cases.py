"""Degenerate strings that should lose to a wrong-but-plausible transcription.

The bar is `the old man` read as `the 0ld man` (a curated `o`/`0` mixup).
Spam, dropped text, and strings made only of thin glyphs have to score lower.
"""

REF = "the old man"
REASONABLE = "the 0ld man"
THIN = "il1.'"


def cases():
    return (
        ("thin_spam", (THIN + " ") * 8),
        ("only_i", "i" * len(REF)),
        ("repeat_i", REF + "i" * 12),
        ("very_short", "i"),
        ("very_long", "l" * 80),
        ("neutral_dots", "." * len(REF)),
        ("quotes", "'" * len(REF)),
        ("dropped_word", "the man"),
        ("empty", ""),
    )


def evaluate(score_pair, table):
    """Return `(reasonable_reward, failures)` where each failure is a dict."""
    reasonable, _diag = score_pair(REF, REASONABLE, table)
    failures = []
    rows = []
    for name, hyp in cases():
        reward, diag = score_pair(REF, hyp, table)
        row = {
            "name": name,
            "hyp": hyp,
            "reward": reward,
            "wcer": diag["wcer"],
            "below": reward < reasonable,
        }
        rows.append(row)
        if not row["below"]:
            failures.append(row)
    return reasonable, rows, failures
