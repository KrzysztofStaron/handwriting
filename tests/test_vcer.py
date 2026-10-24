"""Visually weighted CER: distance, costs, reward, and the committed table."""

import random
import unittest

from vcer.costs import CostTable, unit_table
from vcer.distance import edit_distance
from vcer.eval.hacking_cases import evaluate
from vcer.eval.validate import agreement, load_sanity, proposed_pairs
from vcer.inspect import explain_pair, rule_report
from vcer.reward import pair_reward, prepare, rewards
from vcer.table_build.from_confusions import harvest, merge_edits, levenshtein_ops
from vcer.table_build.build import build_table, combine_overlap
from vcer.table_build.from_render import iou_costs, overlap_costs, render_similarity, rescale_costs
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

    def test_overlap_is_the_unmatched_share_of_the_ink(self):
        import numpy as np
        block = np.zeros((8, 8), dtype=np.float32)
        block[3:5, 3:5] = 1
        shifted = np.zeros((8, 8), dtype=np.float32)
        shifted[4:6, 3:5] = 1
        far = np.zeros((8, 8), dtype=np.float32)
        far[0:1, 7:8] = 1
        blank = np.zeros((8, 8), dtype=np.float32)
        chipped = block.copy()
        chipped[3, 3] = 0
        costs = overlap_costs(
            ["a", "o", "x", " ", "ol"],
            masks={"a": block, "o": shifted, "x": far, " ": blank, "ol": block},
            shift=1,
        )
        self.assertNotIn(("a", "a"), costs)
        self.assertNotIn(("a", "ol"), costs)
        self.assertNotIn(("ol", "a"), costs)
        for src, dst in costs:
            self.assertEqual(len(src), 1)
            self.assertEqual(len(dst), 1)
        self.assertAlmostEqual(costs[("a", "o")], 0.0)
        self.assertAlmostEqual(costs[("o", "a")], costs[("a", "o")])
        self.assertAlmostEqual(costs[("a", "x")], 1.0)
        self.assertAlmostEqual(costs[("a", " ")], 1.0)
        self.assertAlmostEqual(costs[(" ", "a")], 1.0)
        partial = iou_costs({"a": block, "b": chipped}, shift=0)
        self.assertAlmostEqual(partial[("a", "b")], 0.25)

    def test_combine_keeps_hand_digraphs_and_drops_new_ones(self):
        hand = {("d", "ol"): 0.25, ("a", "o"): 0.40, ("o", "a"): 0.40, ("rn", "m"): 0.25}
        overlap = {
            ("a", "o"): 0.10,
            ("o", "a"): 0.10,
            ("a", "x"): 0.80,
            ("x", "a"): 0.80,
            ("d", "ol"): 0.05,
            ("ol", "x"): 0.05,
            ("cl", "d"): 0.05,
        }
        merged = combine_overlap(hand, overlap)
        self.assertAlmostEqual(merged[("a", "o")], 0.10)
        self.assertAlmostEqual(merged[("o", "a")], 0.10)
        self.assertAlmostEqual(merged[("a", "x")], 0.80)
        self.assertAlmostEqual(merged[("d", "ol")], 0.25)
        self.assertAlmostEqual(merged[("rn", "m")], 0.25)
        self.assertNotIn(("ol", "x"), merged)
        self.assertNotIn(("cl", "d"), merged)
        table = build_table(["a", "o", "x", "d"], hand=hand, overlap=overlap)
        self.assertAlmostEqual(table.sub_cost("a", "o"), 0.15)
        self.assertAlmostEqual(table.sub_cost("d", "ol"), 0.25)
        self.assertNotIn(("ol", "x"), table.sub)
        self.assertIn("ink-overlap", table.source)

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
        self.assertEqual(prepare("hello : world"), "hello: world")
        self.assertEqual(prepare("end ."), "end.")
        tight, diag = pair_reward("it is: so", "it is : so", self.table)
        self.assertAlmostEqual(tight, 1.0)
        self.assertTrue(diag["exact"])
        self.assertEqual(diag["cer"], 0.0)
        apart, apart_diag = pair_reward("hello, world", "hello,world", self.table)
        self.assertLess(apart, 1.0)
        self.assertGreater(apart_diag["cer"], 0.0)

    def test_rule_report_prices_symbols_by_ink(self):
        report = rule_report()
        self.assertEqual(report["counts"]["hand"] + report["counts"]["hand_render"], 40)
        self.assertEqual(report["hand_missing"], [])
        self.assertEqual(report["counts"]["floor"], 0)
        self.assertIn("ink-overlap", report["source"])
        pulled = [item for item in report["rules"] if item["ref"] == "5" and item["hyp"] == "S"]
        self.assertIn(pulled[0]["origin"], ("hand", "hand+render"))
        self.assertLessEqual(pulled[0]["cost"], 0.30)
        self.assertAlmostEqual(pulled[0]["seed"], 0.30)
        bang = [item for item in report["rules"] if item["ref"] == "!" and item["hyp"] == "("]
        if bang:
            self.assertEqual(bang[0]["origin"], "render")
            self.assertGreater(bang[0]["cost"], 0.15)

    def test_explain_names_the_rule_behind_an_edit(self):
        looked = explain_pair("d", "ol")
        subs = [op for op in looked["ops"] if op["op"] == "sub"]
        self.assertEqual(subs[0]["origin"], "hand")
        self.assertAlmostEqual(subs[0]["cost"], 0.25)
        self.assertAlmostEqual(looked["wcer"], 0.25)
        self.assertGreater(looked["cer"], looked["wcer"])
        ink = explain_pair("c", "e")
        ink_sub = [op for op in ink["ops"] if op["op"] == "sub"]
        self.assertEqual(
            ink_sub[0]["why"],
            "hand list said 0.35, ink overlap pulled it to 0.3475",
        )
        self.assertAlmostEqual(ink_sub[0]["cost"], 0.3475)
        same = explain_pair("hello: world", "hello : world")
        self.assertTrue(same["exact"])
        self.assertEqual([op for op in same["ops"] if op["op"] != "eq"], [])

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
        self.assertGreater(len(lines), 200)
        self.assertNotIn("the old man", lines)
        for line in lines:
            self.assertGreaterEqual(len(line), 8)
            self.assertLessEqual(len(line), 48)
            self.assertTrue(set(line) <= alphabet)

    def test_recognizer_defaults_to_plain_and_visual_cer(self):
        import training.train_rl as train_rl
        from training.rl import blend_string_reward
        from vcer.reward import VcerReward
        args = train_rl.parse_args([])
        self.assertEqual(args.reward, "both")
        pure, diag = VcerReward.load().score_pair("d", "ol")
        mixed = blend_string_reward(diag["cer"], diag["wcer"])
        self.assertGreater(mixed, 0.0)
        self.assertLess(mixed, pure)
        self.assertAlmostEqual(
            mixed,
            0.75 * (1.0 - min(diag["cer"], 1.0)) + 0.25 * (1.0 - min(diag["wcer"], 1.0)),
        )
        self.assertEqual(args.prompts, "data/romeo_juliet.txt")
        self.assertEqual(args.max_chars, 48)
        self.assertAlmostEqual(args.vcer_alpha, 0.8)
        self.assertEqual(args.vcer_k, 2)
        self.assertAlmostEqual(args.confidence, 0.2)
        self.assertAlmostEqual(args.straightness, 0.05)
        self.assertAlmostEqual(args.spacing, 0.0)
        self.assertAlmostEqual(args.char_penalty, 0.05)
        self.assertEqual(args.sample_every, 5)
        self.assertEqual(args.k, 8)
        self.assertEqual(args.beam, 16)
        self.assertEqual(args.keep, 4)
        self.assertIsNone(args.tree)
        self.assertAlmostEqual(args.novel, 0.05)
        self.assertAlmostEqual(args.words, 0.10)
        self.assertEqual(args.easy_leaves, 0)
        self.assertEqual(args.novel_leaves, 64)
        self.assertEqual(args.temp_min, 0.1)
        self.assertEqual(args.temp_max, 0.9)

    def test_labels_keep_their_order_and_rewards_go_high_to_low(self):
        import training.train_rl as train_rl
        refs = ["beta", "alpha", "beta", "alpha"]
        rewards = [0.2, 0.4, 0.9, 0.1]
        order = train_rl.label_reward_order(refs, rewards)
        self.assertEqual([refs[i] for i in order], ["beta", "beta", "alpha", "alpha"])
        self.assertEqual([rewards[i] for i in order], [0.9, 0.2, 0.4, 0.1])

    def test_training_sheet_saves_the_batch(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        import training.train_rl as train_rl
        images = [
            Image.new("RGB", (48, 20), "white"),
            Image.new("RGB", (36, 18), "white"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gen-000005.png"
            name = train_rl.save_training_sheet(
                images,
                [
                    "Two households, both alike in dignity",
                    "From ancient grudge break to new mutiny",
                ],
                ["Two households, both alike", "From ancient grudge"],
                [0.25, 0.4],
                [0.9, 0.4],
                path,
                5,
            )
            self.assertEqual(name, "gen-000005.png")
            self.assertTrue(path.is_file())
            with Image.open(path) as sheet:
                self.assertEqual(sheet.format, "PNG")
                self.assertGreater(sheet.size[0], 48)
                self.assertGreater(sheet.size[1], 40)

    def test_each_sample_saves_its_own_picture(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        import training.train_rl as train_rl
        from training.rl_events import pack_draws
        images = [
            Image.new("RGB", (40, 200), "white"),
            Image.new("RGB", (30, 40), "white"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            names = train_rl.save_sample_images(images, Path(tmp), 5)
            self.assertEqual(names, ["gen-000005-00.png", "gen-000005-01.png"])
            with Image.open(Path(tmp) / names[0]) as preview:
                self.assertEqual(preview.format, "PNG")
                self.assertEqual(preview.size, (32, 160))
            with Image.open(Path(tmp) / names[1]) as preview:
                self.assertEqual(preview.size, (30, 40))
        draws = pack_draws(
            ["beta", "alpha"], [10, 12], [0.2, 0.9], [0.1, 0.0],
            ["b", "alpha"], [0.0, 0.0], None, images=names,
        )
        self.assertEqual([draw["image"] for draw in draws], names)


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
        digraphs = {
            ("rn", "m"), ("m", "rn"),
            ("cl", "d"), ("d", "cl"),
            ("d", "ol"), ("ol", "d"),
            ("vv", "w"), ("w", "vv"),
        }
        multi = {pair for pair in table.sub if len(pair[0]) != 1 or len(pair[1]) != 1}
        self.assertEqual(multi, digraphs)
        import json
        from pathlib import Path
        alphabet = list(json.loads(
            (Path(__file__).resolve().parents[1] / "stoi.json").read_text(encoding="utf-8")))
        self.assertEqual(len(alphabet), 77)
        for src in alphabet:
            for dst in alphabet:
                if src == dst:
                    continue
                self.assertIn((src, dst), table.sub)
        self.assertAlmostEqual(table.sub[("a", "x")], table.sub[("x", "a")])
        self.assertGreater(table.sub[("a", "x")], table.sub[("c", "e")])

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
