"""Combine hand, confusion, and render costs.

A hand-seed entry is an upper bound: a data source can make that pair
cheaper, and it cannot make it more expensive. A pair that only the renderer
produced cannot go below `render_only_floor`, so a font accident cannot
undercut a curated confusion such as `d -> ol`.
"""

RENDER_ONLY_FLOOR = 0.55


def merge_costs(hand, render=None, confusion=None,
                weights=(1.0, 1.0, 1.0), floor=0.15, render_only_floor=RENDER_ONLY_FLOOR):
    """Return `(src, dst) -> cost` with every cost in `[floor, 1]`."""
    hand = dict(hand or {})
    render = dict(render or {})
    confusion = dict(confusion or {})
    sources = (hand, confusion, render)
    merged = {}
    keys = set(hand) | set(render) | set(confusion)
    for src, dst in keys:
        if src == dst:
            continue
        total_w = 0.0
        total = 0.0
        present = False
        for weight, table in zip(weights, sources):
            if (src, dst) not in table or weight <= 0:
                continue
            total += weight * float(table[(src, dst)])
            total_w += weight
            present = True
        if not present or total_w <= 0:
            continue
        cost = total / total_w
        if (src, dst) in hand:
            cost = min(cost, float(hand[(src, dst)]))
        elif (src, dst) not in confusion:
            cost = max(cost, render_only_floor)
        merged[(src, dst)] = min(1.0, max(floor, cost))
    return merged
