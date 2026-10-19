"""Visually weighted CER: distance, costs, reward, and the committed table."""

import random
import unittest

from vcer.costs import CostTable, unit_table
from vcer.distance import edit_distance
from vcer.eval.hacking_cases import evaluate
from vcer.eval.validate import agreement, load_sanity, proposed_pairs
from vcer.reward import pair_reward, prepare, rewards
from vcer.table_build.from_confusions import harvest, merge_edits, levenshtein_ops
from vcer.table_build.from_render import render_similarity, rescale_costs
from vcer.table_build.hand_seed import load_hand_seed
from vcer.table_build.merge import merge_costs


def _levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class DistanceTests(unittest.TestCase):
    def test_unit_costs_match_levenshtein(self):
        table = unit_table()
        rng = random.Random(0)
        alphabet = "abcde "
        for _ in range(200):
            ref = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
            hyp = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
            got = edit_distance(ref, hyp, table, k=1)
            self.assertEqual(got, _levenshtein(ref, hyp))

    def test_empty_strings(self):
        table = unit_table()
        self.assertEqual(edit_distance("", "abc", table, k=1), 3)
        self.assertEqual(edit_distance("abc", "", table, k=1), 3)
        self.assertEqual(edit_distance("", "", table, k=1), 0)

    def test_hand_seed_prefers_lookalikes(self):
        subs, _meta = load_hand_seed()
        table = CostTable(subs, min_sub_cost=0.15, min_indel_cost=0.6, case_cost=0.2, k=2)
        self.assertLess(edit_distance("d", "ol", table), edit_distance("d", "e", table))
        self.assertLess(edit_distance("m", "rn", table), edit_distance("m", "n", table))
        self.assertLess(edit_distance("w", "vv", table), edit_distance("w", "x", table))

    def test_fast_matches_reference_on_random_pairs(self):
        subs, _meta = load_hand_seed()
        table = CostTable(subs, k=2)
        rng = random.Random(1)
        alphabet = "abcdefghijklmnopqrstuvwxyz01"
        for _ in range(10000):
            ref = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
            hyp = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
            self.assertAlmostEqual(
                edit_distance(ref, hyp, table, k=2, fast=False),
                edit_distance(ref, hyp, table, k=2, fast=True),
            )

    def test_backtrace_explains_the_distance(self):
        table = unit_table()
        total, ops = edit_distance("cat", "cut", table, k=1, backtrace=True)
        self.assertAlmostEqual(total, 1.0)
        self.assertAlmostEqual(sum(op["cost"] for op in ops), total)


class CostTests(unittest.TestCase):
    def test_roundtrip_floors_and_unknown(self):
        table = CostTable({("d", "ol"): 0.01, ("a", "b"): 0.4}, case_cost=0.2)
        self.assertAlmostEqual(table.sub_cost("d", "ol"), 0.15)
        self.assertAlmostEqual(table.sub_cost("a", "b"), 0.4)
        self.assertAlmostEqual(table.sub_cost("q", "z"), 1.0)
        self.assertAlmostEqual(table.sub_cost("A", "a"), 0.2)
        self.assertAlmostEqual(table.sub_cost("aa", "aa"), 0.0)
        self.assertGreaterEqual(table.ins_cost("i"), 0.6)
        payload = table.to_payload()
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.json"
            table.save(path)
            loaded = CostTable.load(path)
        self.assertEqual(loaded.to_payload()["hash"], payload["hash"])
        self.assertAlmostEqual(loaded.sub_cost("d", "ol"), 0.15)
        self.assertIn("hash", payload)

    def test_hash_mismatch_is_rejected(self):
        table = CostTable({("a", "o"): 0.4})
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.json"
            table.save(path)
            text = path.read_text(encoding="utf-8").replace("0.4", "0.5", 1)
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                CostTable.load(path)


class BuildTests(unittest.TestCase):
    def test_confusion_merge_counts_a_digraph_once(self):
        ops = merge_edits(levenshtein_ops("mad", "rnad"), k=2)
        self.assertIn(("sub", "m", "rn"), ops)
        costs = harvest([("d", "ol")] * 8 + [("d", "e")], k=2, min_count=2, floor=0.15)
        self.assertIn(("d", "ol"), costs)
        self.assertNotIn(("d", "e"), costs)
        self.assertAlmostEqual(costs[("d", "ol")], 0.15)
        again = harvest([("d", "ol")] * 8 + [("d", "e")], k=2, min_count=2, floor=0.15)
        self.assertEqual(costs, again)

    def test_render_is_deterministic_and_rescale_hits_the_anchors(self):
        raw = {(str(i), "x"): float(i) for i in range(100)}
        scaled = rescale_costs(raw)
        import numpy as np
        values = list(scaled.values())
        self.assertAlmostEqual(float(np.percentile(values, 5)), 0.2, places=5)
        self.assertAlmostEqual(float(np.percentile(values, 50)), 0.9, places=5)

        def embed(text, rng):
            rng.random()
            vector = np.zeros(8)
            for index, ch in enumerate(text):
                vector[index % 8] += ord(ch)
            return vector

        first = render_similarity(["a", "b", "ol"], seed=3, n_views=3, embed=embed)
        second = render_similarity(["a", "b", "ol"], seed=3, n_views=3, embed=embed)
        self.assertEqual(first, second)
        self.assertNotIn(("a", "a"), first)

    def test_merge_bounds(self):
        hand = {("d", "ol"): 0.25}
        render = {("d", "ol"): 0.9, ("d", "e"): 0.1, ("a", "a"): 0.2}
        merged = merge_costs(hand, render=render, floor=0.15, render_only_floor=0.55)
        self.assertLessEqual(merged[("d", "ol")], 0.25)
        self.assertGreaterEqual(merged[("d", "e")], 0.55)
        self.assertNotIn(("a", "a"), merged)
        for cost in merged.values():
            self.assertGreaterEqual(cost, 0.15)
            self.assertLessEqual(cost, 1.0)


class RewardTests(unittest.TestCase):
    def setUp(self):
        subs, _meta = load_hand_seed()
        self.table = CostTable(subs, k=2)

    def test_bounds_exact_and_ordering(self):
        exact, diag = pair_reward("Hello", "Hello", self.table)
        self.assertAlmostEqual(exact, 1.0)
        self.assertTrue(diag["exact"])
        look, _look_diag = pair_reward("d", "ol", self.table)
        other, _other_diag = pair_reward("d", "e", self.table)
        self.assertGreater(look, other)
        self.assertGreaterEqual(look, 0.0)
        self.assertLessEqual(other, 1.0)
        blank, _blank = pair_reward("", "", self.table)
        filled, _filled = pair_reward("", "abc", self.table)
        self.assertAlmostEqual(blank, 1.0)
        self.assertAlmostEqual(filled, 0.0)
        spaced, _spaced = pair_reward("  the   old man ", "the old man", self.table)
        self.assertAlmostEqual(spaced, 1.0)
        composed = prepare("e\u0301")
        self.assertEqual(composed, "\u00e9")

    def test_batch_matches_pairs(self):
        refs = ["d", "m", "w"]
        hyps = ["ol", "n", "vv"]
        batch = rewards(refs, hyps, self.table, workers=2)
        solo = [pair_reward(ref, hyp, self.table)[0] for ref, hyp in zip(refs, hyps)]
        self.assertEqual(list(batch), solo)


class PromptAndTrainerTests(unittest.TestCase):
    def test_romeo_lines_fit_the_alphabet(self):
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        alphabet = set(json.loads((root / "stoi.json").read_text(encoding="utf-8")))
        lines = (root / "data" / "romeo_juliet.txt").read_text(encoding="utf-8").splitlines()
        self.assertGreater(len(lines), 1000)
        self.assertNotIn("the old man", lines)
        for line in lines:
            self.assertGreaterEqual(len(line), 8)
            self.assertLessEqual(len(line), 32)
            self.assertTrue(set(line) <= alphabet)

    def test_recognizer_defaults_to_vcer_and_the_play(self):
        import train_rl
        args = train_rl.parse_args([])
        self.assertEqual(args.reward, "vcer")
        self.assertEqual(args.prompts, "data/romeo_juliet.txt")
        self.assertAlmostEqual(args.vcer_alpha, 0.8)
        self.assertEqual(args.vcer_k, 2)


class TableAndSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "vcer" / "table.json"
        cls.table = CostTable.load(path)

    def test_committed_table_keeps_the_lookalikes_cheaper(self):
        table = self.table
        self.assertEqual(table.sub_cost("d", "d"), 0.0)
        self.assertLess(edit_distance("d", "ol", table), edit_distance("d", "e", table))
        self.assertLess(edit_distance("m", "rn", table), edit_distance("m", "n", table))
        self.assertLess(edit_distance("w", "vv", table), edit_distance("w", "x", table))
        for cost in table.sub.values():
            self.assertGreaterEqual(cost, table.min_sub_cost)
            self.assertLessEqual(cost, 1.0)

    def test_proposed_orderings_beat_plain_cer(self):
        stats = agreement(self.table, load_sanity())
        self.assertGreaterEqual(stats["n"], 50)
        self.assertEqual(load_sanity(), proposed_pairs())
        self.assertGreater(stats["vcer"], stats["cer"])
        self.assertGreaterEqual(stats["vcer_only"], 8)

    def test_hacking_suite_stays_under_a_plausible_miss(self):
        _reasonable, _rows, failures = evaluate(
            lambda ref, hyp, table: pair_reward(ref, hyp, table), self.table)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
