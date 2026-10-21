"""RL from human feedback, served at /rlhf through the dashboard.

A separate process holds the policy in memory (port 8766). The dashboard
proxies /rlhf and /api/rlhf/* to it.

    uv run --no-sync rl_hf.py --init     # copy the newest rl-words checkpoint to samples/rlhf/start.pth
    uv run --no-sync rl_hf.py            # serve on 127.0.0.1:8766
    uv run --no-sync rl_hf.py --reset    # throw away feedback and go back to start.pth

Each round draws 4 samples of one prompt. The person picks the best (and
optionally the worst). Advantages: best +1, worst -1, others 0; with no worst,
best +1 and others -1/3. One REINFORCE step with KL to start.pth every
--accumulate rounds, using the trainer's reinforce_step, lr and clip.
"""
import argparse
import base64
import io
import json
import math
import random
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch

from generate import load_std
from rl import (
    draw_temperatures, load_dictionary, load_policy_and_ref, load_prompts,
    reinforce_step, rollout_batch, save_policy, strokes_to_images, word_line,
)
from train_rl import load_sidecar, move_optimizer_state, save_sidecar

ROOT = Path(__file__).resolve().parent
DIR = ROOT / "samples" / "rlhf"
SOURCE_DIR = ROOT / "samples" / "rl-words"
SOURCE_RUN = "20261021-214443"
PAGE = ROOT / "rlhf_dashboard.html"
FIXED = [
    "Krzysztof Szymon Serafin Staron",
    "The quick brown fox jumps over the lazy dog.",
    "Meet me at 7:45 on 21.10.2026, okay?",
    "hello world",
    "Wszystko bedzie dobrze, zobaczysz.",
]
K = 4


def init_start():
    """Copy the newest numbered checkpoint of the source run into samples/rlhf."""
    newest = None
    for line in (SOURCE_DIR / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if SOURCE_RUN not in line or '"checkpoint"' not in line:
            continue
        event = json.loads(line)
        if event.get("run") == SOURCE_RUN and event.get("type") == "checkpoint":
            newest = event
    if newest is None:
        raise SystemExit(f"no checkpoint events for run {SOURCE_RUN}")
    ckpt = ROOT / newest["checkpoint"]
    optim = ckpt.with_name(ckpt.stem + ".optim.pth")
    DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt, DIR / "start.pth")
    if optim.is_file():
        shutil.copy2(optim, DIR / "start.optim.pth")
    meta = {"source_run": SOURCE_RUN, "source_checkpoint": newest["checkpoint"],
            "source_optim": str(optim.relative_to(ROOT)) if optim.is_file() else None,
            "source_update": newest.get("update"), "source_block": newest.get("block"),
            "copied_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def reset():
    """Back to the fresh start copy. Keeps start.pth and meta.json."""
    for name in ("rlhf.pth", "rlhf.optim.pth", "feedback.jsonl", "state.json"):
        (DIR / name).unlink(missing_ok=True)
    for pattern in ("ckpt-*.pth", "rounds"):
        for path in DIR.glob(pattern):
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    print("rlhf state reset to start.pth")


class Session:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.device = args.device
        start = DIR / "start.pth"
        if not start.is_file():
            raise SystemExit("samples/rlhf/start.pth missing, run with --init first")
        policy = DIR / "rlhf.pth"
        policy_path = policy if policy.is_file() else start
        self.model, self.ref, self.stoi = load_policy_and_ref(str(policy_path), str(start), self.device)
        self.std = load_std()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        sidecar = load_sidecar(DIR / "rlhf.optim.pth") if policy.is_file() else None
        sidecar = sidecar or load_sidecar(DIR / "start.optim.pth")
        self.updates = 0
        if sidecar:
            self.optimizer.load_state_dict(sidecar["optimizer"])
            for group in self.optimizer.param_groups:
                group["lr"] = args.lr
            move_optimizer_state(self.optimizer, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        state_path = DIR / "state.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        self.rounds = int(state.get("rounds", 0))
        self.steps = int(state.get("steps", 0))
        self.ckpt_index = int(state.get("ckpt_index", 0))
        self.history = state.get("history", [])
        self.pending_grads = 0
        self.current = None
        self.romeo = load_prompts(str(ROOT / "data/romeo_juliet.txt"), self.stoi, 8, 48)
        try:
            self.lexicon = load_dictionary(args.dict, self.stoi, 48)
        except (OSError, SystemExit):
            self.lexicon = None

    def _save_state(self):
        (DIR / "state.json").write_text(json.dumps({
            "rounds": self.rounds, "steps": self.steps, "ckpt_index": self.ckpt_index,
            "history": self.history[-200:]}, indent=2))

    def pick_prompt(self, source):
        if source == "auto":
            source = random.choice(["words", "words", "romeo", "fixed"])
        if source == "words" and self.lexicon:
            return word_line(self.lexicon, max_chars=48), "words"
        if source == "fixed":
            return random.choice(FIXED), "fixed"
        return random.choice(self.romeo), "romeo"

    def temperatures(self):
        a = self.args
        if a.temp_mode == "fixed":
            return [a.temperature] * K
        return [float(t) for t in draw_temperatures(K, a.temp_min, a.temp_max)]

    def generate(self, body):
        text = str(body.get("text") or "").strip()
        if text:
            bad = sorted({c for c in text if c not in self.stoi})
            if bad:
                raise ValueError(f"characters not in the model alphabet: {''.join(bad)}")
            if len(text) > 80:
                raise ValueError("prompt is longer than 80 characters")
            source = "custom"
        else:
            text, source = self.pick_prompt(str(body.get("source") or "auto"))
        with self.lock:
            t0 = time.time()
            temps = self.temperatures()
            with torch.no_grad():
                seqs = rollout_batch(self.model, [text] * K, self.stoi, self.device,
                                     temps, self.args.sharpen)
            images = strokes_to_images(seqs, self.std)
            gen_s = time.time() - t0
            token = f"{self.rounds + 1}-{random.getrandbits(32):08x}"
            self.current = {"token": token, "text": text, "source": source,
                            "temps": temps, "seqs": seqs, "gen_s": gen_s}
        pngs = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            pngs.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
        return {"token": token, "text": text, "source": source, "temps": temps,
                "images": pngs, "gen_s": round(gen_s, 2)}

    def submit(self, body):
        with self.lock:
            cur = self.current
            if cur is None or body.get("token") != cur["token"]:
                raise ValueError("this round is gone, generate a new one")
            best = body.get("best")
            worst = body.get("worst")
            if best not in range(K):
                raise ValueError("best must be 0-3")
            if worst is not None and (worst not in range(K) or worst == best):
                raise ValueError("worst must be 0-3 and not the best")
            if worst is None:
                advs = [1.0 if i == best else -1.0 / (K - 1) for i in range(K)]
            else:
                advs = [1.0 if i == best else -1.0 if i == worst else 0.0 for i in range(K)]
            t0 = time.time()
            rollouts = [{"text": cur["text"], "seq": s, "temperature": t}
                        for s, t in zip(cur["seqs"], cur["temps"])]
            n = K * self.args.accumulate
            if self.pending_grads == 0:
                self.optimizer.zero_grad(set_to_none=True)
            stats = reinforce_step(self.model, self.ref, rollouts, advs, self.stoi, self.device,
                                   self.args.temperature, self.args.sharpen, self.args.kl,
                                   self.optimizer, self.args.clip, apply=False, normalizer=n)
            self.model.eval()
            self.pending_grads += 1
            grad, stepped = float("nan"), False
            if self.pending_grads >= self.args.accumulate:
                params = [p for p in self.model.parameters() if p.requires_grad]
                grad = float(torch.nn.utils.clip_grad_norm_(params, self.args.clip))
                if math.isfinite(grad) and stats["used"]:
                    self.optimizer.step()
                    stepped = True
                    self.steps += 1
                self.optimizer.zero_grad(set_to_none=True)
                self.pending_grads = 0
            self.rounds += 1
            step_s = time.time() - t0
            rdir = DIR / "rounds"
            rdir.mkdir(parents=True, exist_ok=True)
            seq_file = rdir / f"r{self.rounds:05d}.pt"
            torch.save({"text": cur["text"], "temps": cur["temps"],
                        "seqs": [s.cpu() for s in cur["seqs"]], "advs": advs}, seq_file)
            record = {"ts": time.time(), "round": self.rounds, "prompt": cur["text"],
                      "source": cur["source"], "temps": cur["temps"], "best": best,
                      "worst": worst, "advantages": advs, "loss": stats["loss"],
                      "kl": stats["kl"], "used": stats["used"], "grad": grad,
                      "stepped": stepped, "step": self.steps, "gen_s": cur["gen_s"],
                      "step_s": step_s, "sequences": str(seq_file.relative_to(DIR))}
            with open(DIR / "feedback.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            self.history.append({k: record[k] for k in
                                 ("round", "prompt", "best", "worst", "loss", "kl", "grad",
                                  "stepped", "step")})
            self.current = None
            saved = None
            if self.rounds % 5 == 0:
                self._save_latest()
                saved = "rlhf.pth"
            self._save_state()
        return {"record": record, "saved": saved, "status": self.status()}

    def _save_latest(self):
        save_policy(self.model, DIR / "rlhf.pth")
        save_sidecar(DIR / "rlhf.optim.pth", self.optimizer, self.steps, None, None, None,
                     self.ckpt_index)

    def save_checkpoint(self):
        with self.lock:
            self.ckpt_index += 1
            path = DIR / f"ckpt-{self.ckpt_index:04d}.pth"
            save_policy(self.model, path)
            save_sidecar(path.with_name(path.stem + ".optim.pth"), self.optimizer,
                         self.steps, None, None, None, self.ckpt_index)
            self._save_latest()
            self._save_state()
        return {"saved": path.name, "status": self.status()}

    def skip(self):
        with self.lock:
            self.current = None
        return self.status()

    def status(self):
        meta_path = DIR / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        recent = [h["kl"] for h in self.history[-10:] if h.get("kl") is not None
                  and math.isfinite(h["kl"])]
        a = self.args
        return {"rounds": self.rounds, "steps": self.steps, "ckpt_index": self.ckpt_index,
                "accumulate": a.accumulate, "pending": self.pending_grads,
                "recent_kl": sum(recent) / len(recent) if recent else None,
                "history": self.history[-30:], "device": self.device, "meta": meta,
                "temps": {"mode": a.temp_mode, "fixed": a.temperature,
                          "min": a.temp_min, "max": a.temp_max},
                "lr": a.lr, "kl_coef": a.kl, "clip": a.clip}


def make_handler(session):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj, ctype="application/json"):
            body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/rlhf":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/rlhf/status":
                self._send(200, session.status())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}") if length else {}
                if not isinstance(body, dict):
                    raise ValueError("body must be an object")
                action = {"/api/rlhf/generate": session.generate,
                          "/api/rlhf/submit": session.submit,
                          "/api/rlhf/skip": lambda b: session.skip(),
                          "/api/rlhf/save": lambda b: session.save_checkpoint()}.get(path)
                if action is None:
                    self._send(404, {"error": "not found"})
                    return
                self._send(200, action(body))
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": str(exc)})

        def log_message(self, *args):
            pass
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--device", default="cpu", help="cpu or mps")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--temp-mode", choices=["normal", "fixed"], default="normal",
                    help="normal = draw_temperatures(temp-min, temp-max) like training")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--temp-min", type=float, default=0.1)
    ap.add_argument("--temp-max", type=float, default=0.9)
    ap.add_argument("--sharpen", type=float, default=2.0)
    ap.add_argument("--kl", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--accumulate", type=int, default=1, help="rounds per optimizer step")
    ap.add_argument("--dict", default="/usr/share/dict/words")
    args = ap.parse_args()
    if args.init:
        init_start()
        return
    if args.reset:
        reset()
        return
    torch.set_num_threads(args.threads)
    session = Session(args)
    print(f"rlhf on http://127.0.0.1:{args.port}/rlhf  device {args.device}  "
          f"rounds {session.rounds}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session)).serve_forever()


if __name__ == "__main__":
    main()
