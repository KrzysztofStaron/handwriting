"""Blind pairwise model comparison for the RL dashboard (/blind).

Sheets and key.json come from scripts/blind_generate.py. For each prompt the
four anonymous sheets play a knockout: two semifinals, then the final and the
3rd-place match, giving places 1-4 from four picks. Picks live in picks.json,
the derived places in results.json. The key is revealed only when every
prompt is finished.
"""
import json
import random
import re
import threading
from pathlib import Path

BLIND_DIR = Path(__file__).resolve().parent / "samples" / "blind"
PAGE = Path(__file__).with_name("blind_dashboard.html")
LABELS = ("A", "B", "C", "D")
MATCHES = 4
_IMG = re.compile(r"^r\d{2}-[A-D]\.png$")
_LOCK = threading.Lock()


def _read(name, default):
    path = BLIND_DIR / name
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write(name, data):
    tmp = BLIND_DIR / (name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(BLIND_DIR / name)


def _picks():
    data = _read("picks.json", [])
    return data if isinstance(data, list) else []


def _bracket(rounds):
    """Random semifinal pairing per prompt, made once and kept."""
    data = _read("bracket.json", {})
    changed = False
    rng = random.SystemRandom()
    for r in rounds:
        key = str(r["round"])
        if key not in data:
            labels = list(LABELS)
            rng.shuffle(labels)
            data[key] = [labels[:2], labels[2:]]
            changed = True
    if changed:
        _write("bracket.json", data)
    return data


def _side(pair):
    pair = list(pair)
    random.SystemRandom().shuffle(pair)
    return pair


def _round_picks(picks, n):
    return [p for p in picks if p["round"] == n]


def _next_pair(bracket, done):
    """Pair for match len(done)+1 of a prompt, or None when finished."""
    k = len(done)
    if k < 2:
        return bracket[k]
    if k == 2:
        return [done[0]["winner"], done[1]["winner"]]
    if k == 3:
        return [done[0]["loser"], done[1]["loser"]]
    return None


def _places(done):
    final, third = done[2], done[3]
    return {final["winner"]: 1, final["loser"]: 2, third["winner"]: 3, third["loser"]: 4}


def _sync_results(rounds, picks):
    results = {}
    for r in rounds:
        done = _round_picks(picks, r["round"])
        if len(done) == MATCHES:
            results[str(r["round"])] = _places(done)
    _write("results.json", results)
    return results


def _current(rounds, picks, bracket):
    for r in rounds:
        done = _round_picks(picks, r["round"])
        if len(done) < MATCHES:
            pending = _read("pending.json", None)
            pair = _next_pair(bracket[str(r["round"])], done)
            want = {"round": r["round"], "match": len(done) + 1}
            if (not isinstance(pending, dict) or pending.get("round") != want["round"]
                    or pending.get("match") != want["match"]
                    or sorted(pending.get("pair", [])) != sorted(pair)):
                pending = dict(want, pair=_side(pair))
                _write("pending.json", pending)
            left, right = pending["pair"]
            stage = ["semifinal 1", "semifinal 2", "final", "3rd place"][len(done)]
            return {"round": r["round"], "prompt": r["prompt"], "match": len(done) + 1,
                    "stage": stage, "left": left, "right": right,
                    "images": {"left": f"/blind/img/r{r['round']:02d}-{left}.png",
                               "right": f"/blind/img/r{r['round']:02d}-{right}.png"}}
    return None


def leaderboard(key, results, picks):
    stats, reveal = {}, []
    labels = {r["round"]: r["labels"] for r in key["rounds"]}
    for r in key["rounds"]:
        places = results.get(str(r["round"]), {})
        reveal.append({"round": r["round"], "prompt": r["prompt"],
                       "labels": r["labels"], "places": places})
        for label, model in r["labels"].items():
            s = stats.setdefault(model, {"places": [], "firsts": 0, "borda": 0,
                                         "wins": 0, "games": 0})
            place = places[label]
            s["places"].append(place)
            s["firsts"] += place == 1
            s["borda"] += 4 - place
    h2h = {}
    for p in picks:
        w, l = labels[p["round"]][p["winner"]], labels[p["round"]][p["loser"]]
        stats[w]["wins"] += 1
        stats[w]["games"] += 1
        stats[l]["games"] += 1
        for a, b, won in ((w, l, 1), (l, w, 0)):
            cell = h2h.setdefault(a, {}).setdefault(b, {"wins": 0, "games": 0})
            cell["wins"] += won
            cell["games"] += 1
    board = [{"model": m, "avg_place": sum(s["places"]) / len(s["places"]),
              "firsts": s["firsts"], "borda": s["borda"], "wins": s["wins"],
              "games": s["games"], "win_rate": s["wins"] / s["games"] if s["games"] else 0}
             for m, s in stats.items()]
    board.sort(key=lambda s: (-s["borda"], s["avg_place"]))
    return {"leaderboard": board, "head_to_head": h2h, "rounds": reveal}


def state():
    prompts = _read("prompts.json", None)
    if prompts is None:
        return {"ready": False}
    rounds = prompts["rounds"]
    with _LOCK:
        picks = _picks()
        bracket = _bracket(rounds)
        results = _sync_results(rounds, picks)
        current = _current(rounds, picks, bracket)
    out = {"ready": True, "temperature": prompts.get("temperature"),
           "tries": prompts.get("tries"), "prompts": len(rounds), "matches": MATCHES,
           "picked": len(picks), "total": len(rounds) * MATCHES,
           "current": current, "done": current is None, "can_undo": bool(picks)}
    if current is None:
        out["reveal"] = leaderboard(_read("key.json", {}), results, picks)
    return out


def post(action, body):
    prompts = _read("prompts.json", None)
    if prompts is None:
        raise ValueError("no blind sheets yet")
    rounds = prompts["rounds"]
    with _LOCK:
        picks = _picks()
        if action == "pick":
            current = _current(rounds, picks, _bracket(rounds))
            if current is None:
                raise ValueError("every match is already picked")
            if body.get("round") != current["round"] or body.get("match") != current["match"]:
                raise ValueError("stale match, reload the page")
            side = body.get("winner")
            if side not in ("left", "right"):
                raise ValueError("winner must be left or right")
            winner = current[side]
            loser = current["right" if side == "left" else "left"]
            picks.append({"round": current["round"], "match": current["match"],
                          "left": current["left"], "right": current["right"],
                          "winner": winner, "loser": loser})
        elif action == "undo":
            if picks:
                last = picks.pop()
                _write("pending.json", {"round": last["round"], "match": last["match"],
                                        "pair": [last["left"], last["right"]]})
        else:
            raise ValueError("unknown action")
        _write("picks.json", picks)
    return state()


def image(name):
    if not _IMG.fullmatch(name):
        return None
    path = BLIND_DIR / name
    return path.read_bytes() if path.is_file() else None
