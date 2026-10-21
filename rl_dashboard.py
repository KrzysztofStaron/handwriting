"""Local dashboard for the RL event logs.

By default every directory under samples/ that has an events.jsonl is one
line. Notes for a line, and for each start inside it, live in that
directory's notes.json. The page polls once a second.

    uv run python rl_dashboard.py
    uv run python rl_dashboard.py --dir samples/rl-words
    uv run python rl_dashboard.py --import-log path/to/train.log --run 2026-10-10 --replace
"""

import argparse
import json
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from rl_events import (
    EVENTS_NAME, apply_run_aliases, load_run_aliases, parse_stdout_log, read_events, write_events,
)

PAGE = Path(__file__).with_name("rl_dashboard.html")
VCER_PAGE = Path(__file__).with_name("vcer_dashboard.html")
NOTES_NAME = "notes.json"
_SHEET = re.compile(r"^[A-Za-z0-9._-]+\.png$")
_SERIES_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
_EVENT_TYPE = re.compile(r'"type": "([^"]+)"')
_EVENT_RUN = re.compile(r'"run": "([^"]+)"')
_LABEL_LIMIT = 80
_COMMENT_LIMIT = 4000
_INDEX_CACHE = {}


def note_text(value, limit):
    """A stored label or comment. Empty is allowed. Other types are refused."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("label and comment must be text")
    text = "".join(ch for ch in value if ch in "\n\t" or ord(ch) >= 32)
    text = text.strip()
    if len(text) > limit:
        raise ValueError("note is too long")
    return text


def load_notes(directory):
    path = Path(directory) / NOTES_NAME
    empty = {"label": "", "comment": "", "runs": {}}
    if not path.is_file():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(data, dict):
        return empty
    runs = data.get("runs") if isinstance(data.get("runs"), dict) else {}
    clean_runs = {}
    for key, value in runs.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        clean_runs[key] = {
            "label": value.get("label") if isinstance(value.get("label"), str) else "",
            "comment": value.get("comment") if isinstance(value.get("comment"), str) else "",
        }
    return {
        "label": data.get("label") if isinstance(data.get("label"), str) else "",
        "comment": data.get("comment") if isinstance(data.get("comment"), str) else "",
        "runs": clean_runs,
    }


def save_notes(directory, notes):
    path = Path(directory) / NOTES_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(notes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _mix_summary(session):
    """Flags the trainer recorded, short enough for a list row."""
    bits = []
    policy = session.get("policy") or ""
    name = Path(str(policy)).name if policy else ""
    if session.get("resuming"):
        bits.append(("resumed " + name).strip())
    elif name:
        bits.append(name)
    novel = session.get("novel")
    words = session.get("words")
    if novel is not None or words is not None:
        try:
            novel_f = float(novel or 0)
            words_f = float(words or 0)
        except (TypeError, ValueError):
            novel_f = 0.0
            words_f = 0.0
        play = max(0.0, 1.0 - novel_f - words_f)
        parts = []
        if words_f >= 0.999:
            parts.append("words only")
        elif novel_f >= 0.999:
            parts.append("novel only")
        elif play >= 0.999 and novel_f == 0 and words_f == 0:
            parts.append("play only")
        else:
            if play > 0.001:
                parts.append(f"play {play:.0%}")
            if words_f > 0.001:
                parts.append(f"words {words_f:.0%}")
            if novel_f > 0.001:
                parts.append(f"novel {novel_f:.0%}")
        if parts:
            bits.append(" · ".join(parts))
    reward = session.get("reward")
    if reward:
        bits.append(str(reward))
    return " · ".join(bit for bit in bits if bit)


def scan_series(directory):
    """Session rows and counts. The file is read again only when it changes."""
    path = Path(directory) / EVENTS_NAME
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _INDEX_CACHE.get(key[0])
    if cached and cached[0] == key:
        return cached[1]
    aliases = {}
    try:
        aliases = load_run_aliases(directory)
    except (OSError, ValueError, json.JSONDecodeError):
        aliases = {}
    sessions = []
    seen = {}
    updates = 0
    reviews = 0
    per_updates = {}
    per_reviews = {}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        type_match = _EVENT_TYPE.search(line)
        if not type_match:
            continue
        kind = type_match.group(1)
        run_match = _EVENT_RUN.search(line)
        run = aliases.get(run_match.group(1), run_match.group(1)) if run_match else ""
        if kind == "session":
            try:
                session = json.loads(line)
            except json.JSONDecodeError:
                continue
            session["run"] = aliases.get(session.get("run"), session.get("run"))
            run = session.get("run") or ""
            if run not in seen:
                seen[run] = session
                sessions.append(session)
        elif kind == "update":
            updates += 1
            per_updates[run] = per_updates.get(run, 0) + 1
        elif kind == "review":
            reviews += 1
            per_reviews[run] = per_reviews.get(run, 0) + 1
    scanned = {
        "sessions": sessions,
        "updates": updates,
        "reviews": reviews,
        "per_updates": per_updates,
        "per_reviews": per_reviews,
    }
    _INDEX_CACHE[key[0]] = (key, scanned)
    return scanned


def series_record(directory):
    """One training directory: its notes, its starts, and how far it has got."""
    directory = Path(directory)
    scanned = scan_series(directory)
    notes = load_notes(directory)
    path = directory / EVENTS_NAME
    mtime = path.stat().st_mtime
    sessions = []
    for session in scanned["sessions"]:
        run = str(session.get("run") or "")
        note = notes["runs"].get(run) or {"label": "", "comment": ""}
        sessions.append({
            "run": run,
            "ts": session.get("ts"),
            "summary": _mix_summary(session),
            "updates": scanned["per_updates"].get(run, 0),
            "reviews": scanned["per_reviews"].get(run, 0),
            "label": note.get("label") or "",
            "comment": note.get("comment") or "",
        })
    return {
        "id": directory.name,
        "label": notes["label"],
        "comment": notes["comment"],
        "updates": scanned["updates"],
        "reviews": scanned["reviews"],
        "live": time.time() - mtime < 180,
        "mtime": mtime,
        "sessions": sessions,
    }


class Handler(BaseHTTPRequestHandler):
    directory = Path("samples/rl")
    root = None

    def series_directories(self):
        if self.root is None:
            return [self.directory]
        root = Path(self.root)
        if not root.is_dir():
            return []
        found = []
        for child in root.iterdir():
            if not child.is_dir() or not _SERIES_NAME.fullmatch(child.name):
                continue
            if (child / EVENTS_NAME).is_file():
                found.append(child)
        found.sort(key=lambda path: (path / EVENTS_NAME).stat().st_mtime, reverse=True)
        return found

    def series_directory(self, name):
        if not name or not _SERIES_NAME.fullmatch(name):
            return None
        if self.root is None:
            if Path(self.directory).name != name:
                return None
            return Path(self.directory)
        root = Path(self.root).resolve()
        path = (root / name).resolve()
        if path != root and root not in path.parents:
            return None
        if not path.is_dir() or not (path / EVENTS_NAME).is_file():
            return None
        return path

    def default_directory(self):
        found = self.series_directories()
        if not found:
            return None
        return found[0]

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/notes":
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send(400, "text/plain; charset=utf-8", b"bad length")
            return
        if length < 0 or length > 65536:
            self._send(400, "text/plain; charset=utf-8", b"note is too large")
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("note must be an object")
            directory = self.series_directory(str(body.get("series") or ""))
            if directory is None:
                self._send(404, "text/plain; charset=utf-8", b"unknown run")
                return
            label = note_text(body.get("label", ""), _LABEL_LIMIT)
            comment = note_text(body.get("comment", ""), _COMMENT_LIMIT)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._send(400, "text/plain; charset=utf-8", str(exc).encode("utf-8"))
            return
        notes = load_notes(directory)
        run = body.get("run")
        if run:
            if not isinstance(run, str) or len(run) > 80:
                self._send(400, "text/plain; charset=utf-8", b"bad run id")
                return
            notes["runs"][run] = {"label": label, "comment": comment}
        else:
            notes["label"] = label
            notes["comment"] = comment
        save_notes(directory, notes)
        self._send(200, "application/json", json.dumps(series_record(directory)).encode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            if not PAGE.is_file():
                self._send(500, "text/plain; charset=utf-8", b"rl_dashboard.html is missing")
                return
            self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
            return
        if parsed.path == "/vcer":
            if not VCER_PAGE.is_file():
                self._send(500, "text/plain; charset=utf-8", b"vcer_dashboard.html is missing")
                return
            self._send(200, "text/html; charset=utf-8", VCER_PAGE.read_bytes())
            return
        if parsed.path == "/api/vcer":
            self._send(200, "application/json", _vcer_report())
            return
        if parsed.path == "/api/vcer/score":
            query = parse_qs(parsed.query)
            try:
                body = _vcer_score(query.get("ref", [""])[0], query.get("hyp", [""])[0])
            except ValueError as exc:
                self._send(400, "text/plain; charset=utf-8", str(exc).encode("utf-8"))
                return
            self._send(200, "application/json", body)
            return
        if parsed.path == "/api/series":
            records = [series_record(path) for path in self.series_directories()]
            self._send(200, "application/json", json.dumps({"series": records}).encode("utf-8"))
            return
        if parsed.path == "/api/events":
            query = parse_qs(parsed.query)
            try:
                offset = int(query.get("offset", ["0"])[0])
            except ValueError:
                self._send(400, "text/plain; charset=utf-8", b"offset must be an integer")
                return
            name = query.get("series", [None])[0]
            if name:
                directory = self.series_directory(name)
                if directory is None:
                    self._send(404, "text/plain; charset=utf-8", b"unknown run")
                    return
            else:
                directory = self.default_directory()
            if directory is None:
                payload = {"offset": 0, "reset": False, "events": [], "series": None}
                self._send(200, "application/json", json.dumps(payload).encode("utf-8"))
                return
            events, next_offset, reset = read_events(directory / EVENTS_NAME, max(offset, 0))
            try:
                apply_run_aliases(events, load_run_aliases(directory))
            except (OSError, ValueError):
                pass
            payload = json.dumps({
                "offset": next_offset,
                "reset": reset,
                "series": directory.name,
                "events": events,
            }).encode("utf-8")
            self._send(200, "application/json", payload)
            return
        if parsed.path.startswith("/sheets/"):
            path = self._sheet(unquote(parsed.path[len("/sheets/"):]))
            if path is None:
                self._send(404, "text/plain; charset=utf-8", b"not found")
                return
            self._send(200, "image/png", path.read_bytes())
            return
        self._send(404, "text/plain; charset=utf-8", b"not found")

    def _sheet_file(self, directory, name):
        if not _SHEET.fullmatch(name):
            return None
        root = Path(directory).resolve()
        path = (root / name).resolve()
        if path != root and root not in path.parents:
            return None
        if not path.is_file():
            return None
        return path

    def _sheet(self, name):
        if "/" in name:
            series, leaf = name.split("/", 1)
            directory = self.series_directory(series)
            if directory is None:
                return None
            return self._sheet_file(directory, leaf)
        if self.root is not None:
            return None
        return self._sheet_file(self.directory, name)

    def _send(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


def _vcer_report():
    from vcer.inspect import rule_report
    return json.dumps(rule_report()).encode("utf-8")


def _vcer_score(ref, hyp):
    from vcer.inspect import explain_pair
    return json.dumps(explain_pair(ref, hyp)).encode("utf-8")


def make_server(directory, port, root=None):
    """One directory, or every events.jsonl under `root` when directory is None."""
    if directory is not None:
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
    if root is not None:
        root = Path(root).resolve()

    class Bound(Handler):
        pass

    Bound.directory = directory
    Bound.root = root
    return ThreadingHTTPServer(("127.0.0.1", port), Bound)


def import_log(log_path, directory, run, replace):
    dest = Path(directory) / EVENTS_NAME
    if dest.exists() and not replace:
        raise SystemExit(f"{dest} already exists. Pass --replace to overwrite it.")
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    events = parse_stdout_log(text, run=run or Path(log_path).stem)
    write_events(dest, events)
    return events


def main():
    parser = argparse.ArgumentParser(description="Serve the RL training dashboard.")
    parser.add_argument("--dir", default=None,
                        help="One series directory. Default: every events.jsonl under --root.")
    parser.add_argument("--root", default="samples",
                        help="Parent of the series directories when --dir is omitted.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--import-log", type=Path, help="Recover events from a train_rl.py terminal log.")
    parser.add_argument("--run", default=None, help="Run id to stamp on an imported log.")
    parser.add_argument("--replace", action="store_true", help="Overwrite events.jsonl when importing.")
    parser.add_argument("--import-only", action="store_true", help="Import the log and exit.")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in a browser.")
    args = parser.parse_args()
    if args.import_log and args.dir is None:
        raise SystemExit("--import-log needs --dir")
    if args.dir is not None:
        directory = Path(args.dir)
        directory.mkdir(parents=True, exist_ok=True)
        if args.import_log:
            events = import_log(args.import_log, directory, args.run, args.replace)
            updates = sum(1 for event in events if event["type"] == "update")
            reviews = sum(1 for event in events if event["type"] == "review")
            print(f"imported {updates} updates and {reviews} reviews -> {directory / EVENTS_NAME}",
                  flush=True)
            if args.import_only:
                return
        server = make_server(directory, args.port)
        reading = directory.resolve() / EVENTS_NAME
    else:
        root = Path(args.root)
        server = make_server(None, args.port, root=root)
        reading = root.resolve()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"RL dashboard  {url}", flush=True)
    print(f"reading       {reading}", flush=True)
    if args.open and sys.platform == "darwin":
        subprocess.run(["open", url], check=False)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
