"""Event log, terminal-log import, and the local dashboard server."""

import json
import math
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboards.rl_dashboard import make_server, series_record
from training.rl_events import EventLog, pack_draws, parse_stdout_log, read_events

LOG = """\
device: cpu   reward: mps   policy: models/best-4.pth   (base model)
KL anchor: models/best-4.pth
prompts: 12   review sentence: 'the old man' x 2

block -> ckpt-0001  (about 10 min of updates)
    'little lace'
      draw 1/2  (10 steps)
      draw 2/2  (12 steps)
    '"Oh, Lord"'
      draw 1/2  (8 steps)
      draw 2/2  (9 steps)
  update 1  0.5 min  reward 0.793  cer 0.249  kl 0.0000  'little lace' -> 'with an list'
  checkpoint: samples/rl/ckpt-0001.pth  (1 updates, 0.5 min)
  review 'the old man' x 2
    draw 1/2  (20 steps)
    draw 2/2  (21 steps)
     1  cer 0.18  reward 0.834  -> 'the old man .'
     2  cer 0.00  reward 1.000  -> 'the old man'
  new best review reward 0.917 -> models/rl-trocr-best.pth

block -> ckpt-0002  (about 10 min of updates)
    'power at the office'
      draw 1/2  (30 steps)
      draw 2/2  (31 steps)
  update 2  0.4 min  reward 0.800  cer 0.200  kl 0.3000  loss 0.1200  grad 0.840  'power at the office' -> 'power at the notice'
    'dropped on interrupt'
      draw 1/2  (4 steps)

stopped.
"""


class EventLogTests(unittest.TestCase):
    def test_nan_becomes_null_and_lines_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            log = EventLog(path, run="t")
            log.write({"type": "update", "loss": float("nan"), "kl": 0.25})
            log.close()
            events, offset, reset = read_events(path)
            self.assertEqual(offset, 1)
            self.assertFalse(reset)
            self.assertEqual(events[0]["run"], "t")
            self.assertEqual(events[0]["type"], "update")
            self.assertIsNone(events[0]["loss"])
            self.assertEqual(events[0]["kl"], 0.25)
            self.assertTrue(math.isfinite(events[0]["ts"]))

    def test_partial_line_is_left_for_the_next_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text('{"type": "session"}\n{"type": "upd', encoding="utf-8")
            events, offset, reset = read_events(path)
            self.assertEqual([event["type"] for event in events], ["session"])
            self.assertEqual(offset, 1)
            self.assertFalse(reset)
            path.write_text(
                '{"type": "session"}\n{"type": "update"}\n', encoding="utf-8")
            more, offset, reset = read_events(path, offset)
            self.assertEqual([event["type"] for event in more], ["update"])
            self.assertEqual(offset, 2)
            self.assertFalse(reset)

    def test_replaced_file_returns_the_new_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text('{"type": "session"}\n{"type": "stop"}\n', encoding="utf-8")
            events, offset, reset = read_events(path, offset=9)
            self.assertTrue(reset)
            self.assertEqual(len(events), 2)
            self.assertEqual(offset, 2)


class ParseTests(unittest.TestCase):
    def test_stdout_log_keeps_means_reviews_and_drops_a_partial_update(self):
        events = parse_stdout_log(LOG, run="2026-10-10")
        kinds = [event["type"] for event in events]
        self.assertEqual(kinds[0], "session")
        self.assertEqual(kinds[-1], "stop")
        self.assertEqual(events[0]["review_text"], "the old man")
        self.assertEqual(events[0]["policy"], "models/best-4.pth")
        self.assertEqual(events[0]["ref"], "models/best-4.pth")
        updates = [event for event in events if event["type"] == "update"]
        self.assertEqual([event["update"] for event in updates], [1, 2])
        self.assertEqual(len(updates[0]["draws"]), 4)
        self.assertEqual(updates[0]["draws"][0]["steps"], 10)
        self.assertEqual(updates[0]["draws"][0]["hyp"], "with an list")
        self.assertIsNone(updates[0]["draws"][1]["hyp"])
        self.assertIsNone(updates[0]["loss"])
        self.assertEqual(updates[0]["example"]["text"], "little lace")
        self.assertAlmostEqual(updates[0]["steps_mean"], (10 + 12 + 8 + 9) / 4)
        self.assertAlmostEqual(updates[1]["loss"], 0.12)
        self.assertAlmostEqual(updates[1]["grad"], 0.84)
        self.assertEqual(updates[1]["example"]["hyp"], "power at the notice")
        review = next(event for event in events if event["type"] == "review")
        self.assertEqual(review["block"], 1)
        self.assertTrue(review["best"])
        self.assertEqual(review["sheet"], "ckpt-0001.png")
        self.assertEqual([draw["steps"] for draw in review["draws"]], [20, 21])
        self.assertEqual(review["draws"][0]["hyp"], "the old man .")
        self.assertAlmostEqual(review["cer_mean"], 0.09)
        self.assertEqual(events[-1]["reason"], "interrupt")
        self.assertEqual(events[-1]["update"], 2)
        self.assertNotIn("dropped on interrupt", json.dumps(events))

    def test_quoted_prompt_with_an_apostrophe(self):
        text = """\
block -> ckpt-0001  (about 10 min of updates)
    '"I don\\'t know -."'
      draw 1/2  (3 steps)
      draw 2/2  (4 steps)
  update 7  1.0 min  reward 0.500  cer 0.500  kl 0.1000  '"I don\\'t know -."' -> 'no'
stopped.
"""
        events = parse_stdout_log(text, run="q")
        update = next(event for event in events if event["type"] == "update")
        self.assertEqual(update["example"]["text"], '"I don\'t know -."')
        self.assertEqual(update["draws"][0]["text"], '"I don\'t know -."')
        self.assertEqual(len(update["draws"]), 2)


class PackTests(unittest.TestCase):
    def test_traces_line_up_with_draws(self):
        draws = pack_draws(
            ["hi", "hi"],
            [3, 4],
            [1.0, 0.0],
            [0.0, 1.0],
            ["hi", "no"],
            [0.5, -0.5],
            [{"logp": -1.5, "kl": 0.2, "length": 3, "skipped": None},
             {"logp": None, "kl": None, "length": None, "skipped": "nonfinite_logprob"}],
        )
        self.assertEqual(draws[0]["hyp"], "hi")
        self.assertEqual(draws[0]["kl"], 0.2)
        self.assertEqual(draws[1]["skipped"], "nonfinite_logprob")
        self.assertIsNone(draws[1]["logp"])


class DashboardTests(unittest.TestCase):
    def test_server_serves_events_page_and_sheets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "events.jsonl").write_text(
                '{"type": "session", "run": "t", "review_text": "hi"}\n',
                encoding="utf-8",
            )
            (root / "ckpt-0001.png").write_bytes(b"\x89PNG\r\n")
            (root / "secret.txt").write_text("nope", encoding="utf-8")
            server = make_server(root, 0)
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{port}"
                with urllib.request.urlopen(base + "/") as response:
                    page = response.read().decode("utf-8")
                self.assertIn("Handwriting RL", page)
                with urllib.request.urlopen(base + "/api/events") as response:
                    body = json.load(response)
                self.assertEqual(body["events"][0]["review_text"], "hi")
                self.assertEqual(body["offset"], 1)
                with urllib.request.urlopen(base + "/api/events?offset=1") as response:
                    body = json.load(response)
                self.assertEqual(body["events"], [])
                self.assertFalse(body["reset"])
                with urllib.request.urlopen(base + "/sheets/ckpt-0001.png") as response:
                    self.assertEqual(response.status, 200)
                    self.assertTrue(response.read().startswith(b"\x89PNG"))
                with urllib.request.urlopen(base + "/vcer") as response:
                    self.assertIn("vCER rules", response.read().decode("utf-8"))
                with urllib.request.urlopen(base + "/api/vcer") as response:
                    rules = json.load(response)
                self.assertEqual(rules["counts"]["hand"] + rules["counts"]["hand_render"], 40)
                with urllib.request.urlopen(base + "/api/vcer/score?ref=d&hyp=ol") as response:
                    scored = json.load(response)
                self.assertAlmostEqual(scored["wcer"], 0.25)
                for path in ("/sheets/secret.txt", "/sheets/..%2Fsecret.txt", "/nope"):
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(base + path)
                    self.assertEqual(caught.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()

    def test_lines_keep_their_sheets_and_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = root / "rl-a"
            two = root / "rl-b"
            one.mkdir()
            two.mkdir()
            (one / "events.jsonl").write_text(
                '{"type": "session", "run": "old", "policy": "models/best-4.pth", '
                '"novel": 0, "words": 1, "resuming": false, "reward": "both"}\n'
                '{"type": "update", "run": "old"}\n'
                '{"type": "session", "run": "new", "policy": "models/rl-trocr.pth", "resuming": true}\n'
                '{"type": "update", "run": "new"}\n',
                encoding="utf-8",
            )
            (one / "run_aliases.json").write_text('{"new": "old"}\n', encoding="utf-8")
            (two / "events.jsonl").write_text(
                '{"type": "session", "run": "b1", "policy": "models/best-4.pth", "resuming": false}\n',
                encoding="utf-8",
            )
            (one / "ckpt-0001.png").write_bytes(b"\x89PNG\r\nA")
            (two / "ckpt-0001.png").write_bytes(b"\x89PNG\r\nB")
            folded = series_record(one)
            self.assertEqual([item["run"] for item in folded["sessions"]], ["old"])
            self.assertEqual(folded["sessions"][0]["updates"], 2)
            self.assertIn("words only", folded["sessions"][0]["summary"])

            server = make_server(None, 0, root=root)
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{port}"
                with urllib.request.urlopen(base + "/api/series") as response:
                    body = json.load(response)
                ids = sorted(item["id"] for item in body["series"])
                self.assertEqual(ids, ["rl-a", "rl-b"])
                request = urllib.request.Request(
                    base + "/api/notes",
                    data=json.dumps({
                        "series": "rl-a",
                        "run": "old",
                        "label": "words only",
                        "comment": "Novel and the play are off.",
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request) as response:
                    saved = json.load(response)
                self.assertEqual(saved["sessions"][0]["label"], "words only")
                self.assertEqual(saved["sessions"][0]["comment"], "Novel and the play are off.")
                with urllib.request.urlopen(base + "/api/series") as response:
                    again = json.load(response)
                listed = next(item for item in again["series"] if item["id"] == "rl-a")
                self.assertEqual(listed["sessions"][0]["label"], "words only")
                with urllib.request.urlopen(base + "/api/events?series=rl-b") as response:
                    events = json.load(response)
                self.assertEqual(events["series"], "rl-b")
                self.assertEqual(events["events"][0]["run"], "b1")
                with urllib.request.urlopen(base + "/sheets/rl-a/ckpt-0001.png") as response:
                    self.assertTrue(response.read().endswith(b"A"))
                with urllib.request.urlopen(base + "/sheets/rl-b/ckpt-0001.png") as response:
                    self.assertTrue(response.read().endswith(b"B"))
                for path in ("/sheets/ckpt-0001.png", "/sheets/rl-a/../rl-b/ckpt-0001.png", "/api/events?series=../rl-b"):
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(base + path)
                    self.assertEqual(caught.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
