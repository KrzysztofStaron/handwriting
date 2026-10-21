"""Blind model comparison for the RL dashboard (/blind).

Sheets and key.json come from scripts/blind_generate.py. The key is only
read server side and is revealed once every round has a full ranking.
"""
import json
import re
import threading
from pathlib import Path

BLIND_DIR = Path(__file__).resolve().parent / "samples" / "blind"
PAGE = Path(__file__).with_name("blind_dashboard.html")
LABELS = ("A", "B", "C", "D")
_IMG = re.compile(r"^r\d{2}-[A-D]\.png$")
_LOCK = threading.Lock()


def _read(name, default):
    path = BLIND_DIR / name
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_results():
    data = _read("results.json", {})
    return data if isinstance(data, dict) else {}


def _complete(places):
    return isinstance(places, dict) and sorted(places.get(l) for l in LABELS) == [1, 2, 3, 4]


def leaderboard(key, results):
    stats = {}
    reveal = []
    for r in key["rounds"]:
        places = results.get(str(r["round"]), {})
        reveal.append({"round": r["round"], "prompt": r["prompt"],
                       "labels": r["labels"], "places": places})
        for label, model in r["labels"].items():
            s = stats.setdefault(model, {"model": model, "places": [], "firsts": 0, "borda": 0})
            place = places.get(label)
            s["places"].append(place)
            s["firsts"] += place == 1
            s["borda"] += 4 - place
    board = []
    for s in stats.values():
        board.append({"model": s["model"], "avg_place": sum(s["places"]) / len(s["places"]),
                      "firsts": s["firsts"], "borda": s["borda"]})
    board.sort(key=lambda s: (-s["borda"], s["avg_place"]))
    return {"leaderboard": board, "rounds": reveal}


def state():
    prompts = _read("prompts.json", None)
    if prompts is None:
        return {"ready": False}
    results = load_results()
    rounds = [{"round": r["round"], "prompt": r["prompt"],
               "images": {l: f"/blind/img/r{r['round']:02d}-{l}.png" for l in LABELS}}
              for r in prompts["rounds"]]
    done = all(_complete(results.get(str(r["round"]))) for r in rounds)
    out = {"ready": True, "temperature": prompts.get("temperature"),
           "tries": prompts.get("tries"), "rounds": rounds,
           "results": results, "done": done}
    if done:
        out["reveal"] = leaderboard(_read("key.json", {}), results)
    return out


def rank(body):
    """Save one round. body = {"round": n, "places": {"A": 1, ...}}."""
    prompts = _read("prompts.json", None)
    if prompts is None:
        raise ValueError("no blind sheets yet")
    n = body.get("round")
    valid = {r["round"] for r in prompts["rounds"]}
    if not isinstance(n, int) or n not in valid:
        raise ValueError("bad round")
    places = body.get("places")
    if not _complete(places) or set(places) != set(LABELS):
        raise ValueError("places must use 1,2,3,4 once each for A-D")
    with _LOCK:
        results = load_results()
        results[str(n)] = {l: int(places[l]) for l in LABELS}
        tmp = BLIND_DIR / "results.json.tmp"
        tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
        tmp.replace(BLIND_DIR / "results.json")
    return state()


def image(name):
    if not _IMG.fullmatch(name):
        return None
    path = BLIND_DIR / name
    return path.read_bytes() if path.is_file() else None
