"""The next visual score charges the distance. The live blend still adds plain CER."""

import unittest

from vcer.eval.hacking_cases import evaluate
from vcer.reward import pair_reward
from vcer_next.score import format_report, in_word, live_units, load_table, measure, score_pair


class LiveUnitTests(unittest.TestCase):
    def test_three_quarters_of_a_full_edit_stay_when_the_visual_cost_is_low(self):
        self.assertAlmostEqual(live_units(1, 0.40), 0.85)
        self.assertAlmostEqual(live_units(1, 0.64), 0.91)
        self.assertAlmostEqual(live_units(1, 0.60), 0.90)
        self.assertAlmostEqual(live_units(1, 1.0), 1.0)


class DistanceTests(unittest.TestCase):
    def setUp(self):
        self.table = load_table()

    def test_a_logged_lookalike_keeps_its_stored_cost_inside_a_word(self):
        ref, hyp = in_word("o", "a")
        row = measure(ref, hyp, self.table)
        self.assertAlmostEqual(row["visual"], 0.40)
        self.assertAlmostEqual(row["plain"], 1.0)
        self.assertAlmostEqual(row["live"], 0.85)
        self.assertAlmostEqual(row["next"], 0.40)

    def test_case_inside_a_word_already_uses_the_case_rule(self):
        self.assertGreater(self.table.sub_cost("W", "w"), 0.5)
        self.assertGreater(self.table.sub_cost("T", "t"), 0.5)
        row = measure(*in_word("W", "w"), self.table)
        self.assertAlmostEqual(row["visual"], self.table.case_cost)
        self.assertAlmostEqual(row["plain"], 1.0)
        self.assertAlmostEqual(row["live"], live_units(1, self.table.case_cost))
        self.assertAlmostEqual(row["next"], self.table.case_cost)

    def test_the_curated_digraph_is_cheaper_than_the_logged_pairs(self):
        logged = measure(*in_word("n", "r"), self.table)
        curated = measure("d", "ol", self.table)
        self.assertAlmostEqual(curated["visual"], 0.25)
        self.assertGreater(logged["visual"], curated["visual"])
        self.assertGreater(logged["live"], 0.90)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.table = load_table()

    def test_one_lookalike_costs_its_visual_share(self):
        reward, diag = score_pair("d", "ol", self.table)
        self.assertAlmostEqual(diag["wcer"], 0.25)
        self.assertAlmostEqual(reward, 0.75)
        self.assertAlmostEqual(diag["cer"], 2.0)

    def test_visual_reward_matches_alpha_one_on_the_same_table(self):
        nxt, next_diag = score_pair("cold", "cald", self.table)
        old, old_diag = pair_reward("cold", "cald", self.table, alpha=1.0)
        self.assertEqual(nxt, old)
        self.assertAlmostEqual(next_diag["wcer"], old_diag["wcer"])
        self.assertAlmostEqual(next_diag["wcer"], 0.40 / 4)

    def test_hacking_cases_still_lose_to_a_lookalike(self):
        _reasonable, _rows, failures = evaluate(score_pair, self.table)
        self.assertEqual(failures, [])

    def test_report_lists_the_logged_pairs_and_the_case_bypass(self):
        text = format_report(self.table)
        self.assertIn("o a", text)
        self.assertIn("W w", text)
        self.assertIn("d ol", text)
        self.assertIn("insertion", text)
