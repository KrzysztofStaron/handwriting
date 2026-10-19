"""Append-only event log for the RL loops.

The trainers write one JSON object per line. The dashboard tails the same
file. A line is flushed before the call returns, so a reader polling the
file sees a finished record or nothing.
"""

import ast
import json
import math
import re
import time
from pathlib import Path


EVENTS_NAME = "events.jsonl"

_BLOCK = re.compile(r"^block -> ckpt-(\d+)")
_UPDATE = re.compile(
    r"^  update (\d+)  ([\d.]+) min  reward ([\d.]+)  cer ([\d.]+)  kl ([\d.]+)"
    r"(?:  loss (\S+)  grad (\S+))?  (.*)$"
)
_DRAW = re.compile(r"^draw (\d+)/(\d+)  \((\d+) steps\)$")
_CHECKPOINT = re.compile(r"^  checkpoint: (\S+)  \((\d+) updates, ([\d.]+) min\)")
_REVIEW = re.compile(r"^  review (.*) x (\d+)$")
_SCORE = re.compile(r"^ {4,6}(\d+)  cer ([\d.]+)  reward ([\d.]+)  -> (.*)$")
_BEST = re.compile(r"^  new best review reward ([\d.]+)")
_DEVICE = re.compile(r"^device: (\S+)\s+reward: (\S+)\s+policy: (\S+)")
_ANCHOR = re.compile(r"^KL anchor: (\S+)")
_PROMPTS = re.compile(r"^prompts: (\d+)\s+review sentence: (.*) x (\d+)")


def json_ready(value):
    """Replace non-finite floats with None so the file stays strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


class EventLog:
    """Append JSON lines to `path`, stamping each with the run id and time."""

    def __init__(self, path, run=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run = run or time.strftime("%Y%m%d-%H%M%S")
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)

    def write(self, record):
        row = {"ts": time.time(), "run": self.run, **json_ready(record)}
        self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._handle.flush()
        return row

    def close(self):
        self._handle.close()


def pack_draws(texts, steps, rewards, cers, hyps, advantages, traces, wcers=None,
               confidences=None, temperatures=None, images=None):
    """One record per rollout, in sample order. Missing traces stay null."""
    traces = list(traces or [])
    rewards = list(rewards or [])
    cers = list(cers or [])
    hyps = list(hyps or [])
    advantages = list(advantages or [])
    wcers = list(wcers or [])
    confidences = list(confidences or [])
    temperatures = list(temperatures or [])
    images = list(images or [])
    draws = []
    for i, text in enumerate(texts):
        trace = traces[i] if i < len(traces) else {}
        draws.append(json_ready({
            "text": text,
            "steps": steps[i] if i < len(steps) else None,
            "temperature": (
                temperatures[i] if i < len(temperatures) else trace.get("temperature")
            ),
            "reward": rewards[i] if i < len(rewards) else None,
            "cer": cers[i] if i < len(cers) else None,
            "wcer": wcers[i] if i < len(wcers) else None,
            "confidence": confidences[i] if i < len(confidences) else None,
            "hyp": hyps[i] if i < len(hyps) else None,
            "image": images[i] if i < len(images) else None,
            "advantage": advantages[i] if i < len(advantages) else trace.get("advantage"),
            "logp": trace.get("logp"),
            "kl": trace.get("kl"),
            "length": trace.get("length"),
            "skipped": trace.get("skipped"),
        }))
    return draws


def _mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _num(text):
    if text is None:
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def take_literal(text):
    """Parse a Python literal at the start of `text`. Return (value, rest)."""
    body = text.lstrip()
    if not body or body[0] not in "'\"":
        raise ValueError(f"not a quoted literal: {text!r}")
    quote = body[0]
    i = 1
    while i < len(body):
        if body[i] == "\\":
            i += 2
            continue
        if body[i] == quote:
            return ast.literal_eval(body[:i + 1]), body[i + 1:]
        i += 1
    raise ValueError(f"unclosed literal: {text!r}")


def write_events(path, events):
    """Replace `path` with `events`, one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(json_ready(event), ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_events(path, offset=0):
    """Return (events after `offset`, next offset, reset).

    `offset` is a count of complete lines. A trailing partial line is left
    for the next read. `reset` is true when `offset` is past the end of the
    file, which happens if the file was replaced.
    """
    path = Path(path)
    if not path.is_file():
        return [], 0, offset > 0
    text = path.read_text(encoding="utf-8")
    if not text:
        return [], 0, offset > 0
    lines = text.splitlines()
    if not text.endswith("\n"):
        lines = lines[:-1]
    reset = False
    if offset > len(lines):
        offset = 0
        reset = True
    events = []
    for line in lines[offset:]:
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events, len(lines), reset


def parse_stdout_log(text, run="imported"):
    """Recover events from a train_rl.py terminal log.

    The terminal log has batch means and review scores. It does not have
    loss, gradient, advantages, or per-draw rewards, so those fields are null.
    Draws collected after the last complete update are dropped.
    """
    header = {
        "device": None,
        "reward_device": None,
        "policy": None,
        "ref": None,
        "prompts": None,
        "review_text": None,
        "review_k": None,
    }
    events = []
    block = None
    pending = []
    last_update = 0
    review_text = None
    review_steps = []
    review_scores = []
    review_best = False
    review_open = False

    def flush_review():
        nonlocal review_open, review_best, review_steps, review_scores
        if not review_open or not review_scores:
            review_open = False
            review_steps = []
            review_scores = []
            review_best = False
            return
        cers = [row["cer"] for row in review_scores]
        rewards = [row["reward"] for row in review_scores]
        draws = []
        for i, row in enumerate(review_scores):
            draws.append({
                "text": review_text,
                "steps": review_steps[i] if i < len(review_steps) else None,
                "reward": row["reward"],
                "cer": row["cer"],
                "hyp": row["hyp"],
                "advantage": None,
                "logp": None,
                "kl": None,
                "length": None,
                "skipped": None,
            })
        events.append({
            "run": run,
            "type": "review",
            "block": block,
            "update": last_update,
            "text": review_text,
            "reward_mean": _mean(rewards),
            "cer_mean": _mean(cers),
            "cer_min": min(cers) if cers else None,
            "cer_max": max(cers) if cers else None,
            "best": review_best,
            "sheet": f"ckpt-{block:04d}.png" if block else None,
            "draws": draws,
            "source": "stdout",
        })
        review_open = False
        review_steps = []
        review_scores = []
        review_best = False

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.strip() == "stopped.":
            flush_review()
            continue
        device = _DEVICE.match(line)
        if device:
            header["device"] = device.group(1)
            header["reward_device"] = device.group(2)
            header["policy"] = device.group(3)
            continue
        anchor = _ANCHOR.match(line)
        if anchor:
            header["ref"] = anchor.group(1)
            continue
        prompts = _PROMPTS.match(line)
        if prompts:
            header["prompts"] = int(prompts.group(1))
            header["review_text"] = take_literal(prompts.group(2))[0]
            header["review_k"] = int(prompts.group(3))
            continue
        block_match = _BLOCK.match(line)
        if block_match:
            flush_review()
            block = int(block_match.group(1))
            pending = []
            continue
        update = _UPDATE.match(line)
        if update:
            prompt, rest = take_literal(update.group(8))
            arrow = rest.lstrip()
            if arrow.startswith("->"):
                arrow = arrow[2:]
            hyp = take_literal(arrow)[0]
            draws = []
            for item in pending:
                for steps in item["steps"]:
                    draws.append({
                        "text": item["text"],
                        "steps": steps,
                        "reward": None,
                        "cer": None,
                        "hyp": None,
                        "advantage": None,
                        "logp": None,
                        "kl": None,
                        "length": None,
                        "skipped": None,
                    })
            if draws and draws[0]["text"] == prompt:
                draws[0]["hyp"] = hyp
            last_update = int(update.group(1))
            events.append({
                "run": run,
                "type": "update",
                "update": last_update,
                "block": block,
                "elapsed_min": float(update.group(2)),
                "reward_mean": float(update.group(3)),
                "reward_min": None,
                "reward_max": None,
                "cer_mean": float(update.group(4)),
                "cer_min": None,
                "cer_max": None,
                "kl": float(update.group(5)),
                "loss": _num(update.group(6)),
                "grad": _num(update.group(7)),
                "used": None,
                "stepped": None,
                "steps_mean": _mean(item["steps"] for item in draws),
                "example": {"text": prompt, "hyp": hyp},
                "draws": draws,
                "source": "stdout",
            })
            pending = []
            continue
        draw = _DRAW.match(line.strip())
        if draw and line.startswith(" "):
            steps = int(draw.group(3))
            if review_open:
                review_steps.append(steps)
            elif pending:
                pending[-1]["steps"].append(steps)
            continue
        checkpoint = _CHECKPOINT.match(line)
        if checkpoint:
            events.append({
                "run": run,
                "type": "checkpoint",
                "block": block,
                "update": last_update,
                "block_updates": int(checkpoint.group(2)),
                "elapsed_min": float(checkpoint.group(3)),
                "checkpoint": checkpoint.group(1),
                "source": "stdout",
            })
            continue
        review = _REVIEW.match(line)
        if review:
            review_text = take_literal(review.group(1))[0]
            review_open = True
            review_steps = []
            review_scores = []
            review_best = False
            continue
        score = _SCORE.match(line)
        if score and review_open:
            review_scores.append({
                "cer": float(score.group(2)),
                "reward": float(score.group(3)),
                "hyp": take_literal(score.group(4))[0],
            })
            continue
        if _BEST.match(line):
            review_best = True
            continue
        if review_open or not line.startswith("    "):
            continue
        body = line[4:]
        if body[:1] not in "'\"":
            continue
        try:
            prompt, rest = take_literal(body)
        except ValueError:
            continue
        if rest.strip():
            continue
        pending.append({"text": prompt, "steps": []})

    flush_review()
    session = {
        "run": run,
        "type": "session",
        "loop": "recognizer",
        "source": "stdout",
        "policy": header["policy"],
        "ref": header["ref"],
        "device": header["device"],
        "reward_device": header["reward_device"],
        "prompts": header["prompts"],
        "review_text": header["review_text"],
        "review_k": header["review_k"],
        "note": "Recovered from the terminal log. Loss, gradient, and per-draw rewards were not printed.",
    }
    stop = {
        "run": run,
        "type": "stop",
        "reason": "interrupt" if "\nstopped." in text or text.rstrip().endswith("stopped.") else "done",
        "update": last_update,
        "source": "stdout",
    }
    return [session, *events, stop]
