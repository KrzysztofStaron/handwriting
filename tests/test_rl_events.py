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

from rl_dashboard import make_server
from rl_events import EventLog, pack_draws, parse_stdout_log, read_events

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
                for path in ("/sheets/secret.txt", "/sheets/..%2Fsecret.txt", "/nope"):
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(base + path)
                    self.assertEqual(caught.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
