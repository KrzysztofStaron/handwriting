"""Local dashboard for the RL event log.

The trainers append JSON lines to samples/rl/events.jsonl. This serves that
file, plus the review sheets, on localhost. The page polls once a second.

    uv run python rl_dashboard.py
    uv run python rl_dashboard.py --dir samples/rlhf
    uv run python rl_dashboard.py --import-log path/to/train.log --run 2026-10-10 --replace
"""

import argparse
import json
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from rl_events import (
    EVENTS_NAME, apply_run_aliases, load_run_aliases, parse_stdout_log, read_events, write_events,
)

PAGE = Path(__file__).with_name("rl_dashboard.html")
VCER_PAGE = Path(__file__).with_name("vcer_dashboard.html")
_SHEET = re.compile(r"^[A-Za-z0-9._-]+\.png$")


class Handler(BaseHTTPRequestHandler):
    directory = Path("samples/rl")

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
        if parsed.path == "/api/events":
            query = parse_qs(parsed.query)
            try:
                offset = int(query.get("offset", ["0"])[0])
            except ValueError:
                self._send(400, "text/plain; charset=utf-8", b"offset must be an integer")
                return
            events, next_offset, reset = read_events(self.directory / EVENTS_NAME, max(offset, 0))
            apply_run_aliases(events, load_run_aliases(self.directory))
            payload = json.dumps({
                "offset": next_offset,
                "reset": reset,
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

    def _sheet(self, name):
        if not _SHEET.fullmatch(name):
            return None
        root = self.directory.resolve()
        path = (root / name).resolve()
        if path != root and root not in path.parents:
            return None
        if not path.is_file():
            return None
        return path

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


def make_server(directory, port):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    class Bound(Handler):
        pass

    Bound.directory = directory
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
    parser.add_argument("--dir", default="samples/rl", help="Directory with events.jsonl and review sheets.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--import-log", type=Path, help="Recover events from a train_rl.py terminal log.")
    parser.add_argument("--run", default=None, help="Run id to stamp on an imported log.")
    parser.add_argument("--replace", action="store_true", help="Overwrite events.jsonl when importing.")
    parser.add_argument("--import-only", action="store_true", help="Import the log and exit.")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in a browser.")
    args = parser.parse_args()
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
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"RL dashboard  {url}", flush=True)
    print(f"reading       {directory.resolve() / EVENTS_NAME}", flush=True)
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
