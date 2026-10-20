"""Reward, log-prob, and REINFORCE tests.

The only checkpoint this suite loads is models/best-4.pth. Other .pth files
are experiment runs and are not the base model.
"""

import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate import load_model, sample, sample_batch, stop_after_point, stop_rows
from model import M, HandwritingModel, sequence_loss
from rl import (
    BASE_CKPT,
    advantages,
    apply_temperature,
    cer,
    cer_and_reward,
    choice_sheet,
    consistency_sheet,
    group_advantages,
    beam_factors,
    choose_label,
    chunk_advantages,
    default_branches,
    tree_plan,
    history_state,
    load_dictionary,
    word_line,
    load_policy_and_ref,
    logprobs_from_states,
    novel_line,
    refuse_overwrite_base,
    reinforce_step,
    reward_from_cer,
    select_top,
    step_logprob,
    strokes_to_image,
    trajectory_logprob,
    trajectory_logprobs,
    word_prefixes,
    draw_temperatures,
    reinforce_loss,
    blend_token_advantages,
    class_budget,
    segment_budgets,
    token_reinforce_loss,
)


def craft_out(mx=0.3, my=-0.2, sx=0.4, sy=0.4):
    """One peaked mixture component on (mx, my), the rest numerically quiet."""
    out = torch.zeros(1, 1, 1 + M * 6)
    out[..., 0] = 8.0
    rest = out[..., 1:].view(1, 1, M, 6)
    rest[..., 0] = -20.0
    rest[..., 3] = 0.0
    rest[..., 4] = 0.0
    rest[0, 0, 0, 0] = 20.0
    rest[0, 0, 0, 1] = mx
    rest[0, 0, 0, 2] = my
    rest[0, 0, 0, 3] = math.log(sx)
    rest[0, 0, 0, 4] = math.log(sy)
    return out


def _words(text, word_gaps, letter_gap=3.0, word_width=16.0):
    """Pen strokes for `text`. `word_gaps` are the blanks between the words."""
    import numpy as np
    words = text.split()
    gaps = list(word_gaps)
    if len(gaps) != max(len(words) - 1, 0):
        raise ValueError("one gap per space")
    pieces = []
    x = 0.0
    for index, word in enumerate(words):
        # Two strokes per word, so a gap inside the word exists and must not count.
        pieces.append((x, x + word_width * 0.45))
        x = pieces[-1][1] + letter_gap
        pieces.append((x, x + word_width * 0.45))
        x = pieces[-1][1]
        if index < len(gaps):
            x += gaps[index]
    pts = []
    pens = []
    for x0, x1 in pieces:
        xs = np.linspace(x0, x1, 5)
        for step, value in enumerate(xs):
            pts.append((float(value), 0.0))
            pens.append(1.0 if step == len(xs) - 1 else 0.0)
    xy = np.asarray(pts, dtype=np.float64)
    delta = np.diff(xy, axis=0, prepend=[[0.0, 0.0]])
    pen = np.asarray(pens, dtype=np.float64)[:, None]
    return np.concatenate([delta, pen], axis=1)


def _band_seq(xs, y0, y1):
    """Stroke offsets whose ink lies between two y curves. std=1 reconstructs them."""
    import numpy as np
    xs = np.asarray(xs, dtype=np.float64)
    y0 = np.broadcast_to(np.asarray(y0, dtype=np.float64), xs.shape)
    y1 = np.broadcast_to(np.asarray(y1, dtype=np.float64), xs.shape)
    pts = [(float(x), float(y)) for x, a, b in zip(xs, y0, y1) for y in (a, (a + b) / 2, b)]
    xy = np.asarray(pts)
    delta = np.diff(xy, axis=0, prepend=xy[:1])
    pen = np.zeros((len(xy), 1))
    return np.concatenate([delta, pen], axis=1)


class StraightnessTests(unittest.TestCase):
    def test_a_flat_band_scores_zero_and_an_arch_does_not(self):
        import numpy as np
        from rl import line_straightness
        xs = np.linspace(0, 400, 48)
        flat = line_straightness(_band_seq(xs, 0.0, 10.0), std=1.0)
        mild = line_straightness(_band_seq(xs, 0.002 * xs, 10 + 0.002 * xs), std=1.0)
        arch = 10 * np.sin(np.pi * xs / xs[-1])
        arched = line_straightness(_band_seq(xs, arch, arch + 10), std=1.0)
        tilted = line_straightness(_band_seq(xs, 0.025 * xs, 10 + 0.025 * xs), std=1.0)
        self.assertAlmostEqual(flat, 0.0, places=6)
        self.assertLess(mild, 0.15)
        self.assertGreater(arched, 0.4)
        self.assertGreater(tilted, 0.5)
        self.assertEqual(line_straightness(np.zeros((3, 3)), std=1.0), 0.0)
        sunk = line_straightness(_band_seq(xs, np.where(xs < 300, 0.0, -10.0), np.where(xs < 300, 10.0, 0.0)), std=1.0)
        self.assertGreater(sunk, 0.5)

    def test_word_gap_penalty_ignores_a_small_wobble_and_gaps_inside_words(self):
        from rl import spacing_inconsistency
        even = spacing_inconsistency(
            _words("the old man", (20, 20), letter_gap=3), std=1.0, text="the old man")
        mild = spacing_inconsistency(
            _words("the old man", (18, 22), letter_gap=3), std=1.0, text="the old man")
        wild = spacing_inconsistency(
            _words("the old man", (2, 80), letter_gap=3), std=1.0, text="the old man")
        inside = spacing_inconsistency(
            _words("the old man", (30, 30), letter_gap=14), std=1.0, text="the old man")
        one = spacing_inconsistency(
            _words("hello", (), letter_gap=2), std=1.0, text="hello")
        self.assertAlmostEqual(even, 0.0, places=6)
        self.assertLess(mild, 0.02)
        self.assertGreater(wild, 0.5)
        self.assertLess(inside, 0.02)
        self.assertEqual(one, 0.0)


class RewardTests(unittest.TestCase):
    def test_cer_known_distance(self):
        self.assertAlmostEqual(cer("kitten", "sitting"), 0.5)
        self.assertAlmostEqual(cer("cat", "cats"), 1.0 / 3.0)
        self.assertEqual(cer("same", "same"), 0.0)
        self.assertEqual(cer("abc", ""), 1.0)

    def test_cer_ignores_spacing_and_curly_quotes(self):
        self.assertEqual(cer("a  b", "a b"), 0.0)
        self.assertEqual(cer('say "hi"', "say \u201chi\u201d"), 0.0)

    def test_string_reward_charges_a_lookalike_the_full_edit(self):
        from rl import string_reward
        value, reward = string_reward("d", "ol")
        self.assertEqual(value, 2.0)
        self.assertAlmostEqual(reward, 0.0)
        same, perfect = string_reward("hello: world", "hello : world")
        self.assertEqual(same, 0.0)
        self.assertAlmostEqual(perfect, 1.0)
        slipped, partial = string_reward("cat", "cats")
        self.assertAlmostEqual(slipped, 1.0 / 3.0)
        self.assertAlmostEqual(partial, 0.8 * (1.0 - 1.0 / 3.0))

    def test_blend_reward_splits_plain_and_visual_error(self):
        from rl import blend_string_reward
        self.assertAlmostEqual(blend_string_reward(0.0, 0.0), 1.0)
        self.assertAlmostEqual(blend_string_reward(1.0, 0.25), 0.1875)
        self.assertAlmostEqual(blend_string_reward(2.0, 3.0), 0.0)

    def test_cer_ignores_a_space_before_punctuation(self):
        self.assertEqual(cer("hello: world", "hello : world"), 0.0)
        self.assertEqual(cer("end.", "end ."), 0.0)
        self.assertEqual(cer(":", " : "), 0.0)
        self.assertEqual(cer("a; b: c", "a ; b : c"), 0.0)
        self.assertGreater(cer("hello, world", "hello,world"), 0.0)
        self.assertGreater(cer("hello:", "helo:"), 0.0)

    def test_reward_shapes(self):
        self.assertAlmostEqual(reward_from_cer(0.0, "exp"), 1.0)
        self.assertAlmostEqual(reward_from_cer(0.25, "linear"), 0.75)
        self.assertAlmostEqual(reward_from_cer(1.5, "linear"), -0.5)
        self.assertAlmostEqual(reward_from_cer(0.05, "soft", tau=0.1), 1.0)
        self.assertAlmostEqual(reward_from_cer(0.55, "soft", tau=0.1), 0.5)
        self.assertEqual(reward_from_cer(1.2, "soft", tau=0.1), 0.0)

    def test_cer_cap_limits_linear_reward(self):
        _raw, reward = cer_and_reward("a", "a" + "x" * 100, kind="linear")
        self.assertAlmostEqual(reward, -1.0)

    def test_temperature_zero_has_no_density(self):
        pi = torch.tensor([1.0])
        sig = torch.tensor([1.0])
        with self.assertRaises(ValueError):
            apply_temperature(pi, sig, sig, 0.0)

    def test_temperature_reweights_like_the_sampler(self):
        pi = torch.tensor([0.7, 0.2, 0.1])
        sig = torch.ones(3)
        same, sx, sy = apply_temperature(pi, sig, sig, 1.0)
        self.assertTrue(torch.allclose(same, pi))
        self.assertTrue(torch.allclose(sx, sig))
        biased, sx, _sy = apply_temperature(pi, sig, sig, 0.5)
        expected = torch.softmax(torch.log(pi) * (1.0 - math.log(0.5)), dim=-1)
        self.assertTrue(torch.allclose(biased, expected, atol=1e-6))
        self.assertTrue(torch.allclose(sx, sig * 0.5))


class LogprobTests(unittest.TestCase):
    def test_matches_sequence_nll_at_temperature_one(self):
        out = craft_out()
        y = torch.tensor([[[0.3, -0.2, 1.0]]])
        logp = step_logprob(out, y, temperature=1.0)
        nll = sequence_loss(out, y, torch.ones(1, 1))
        self.assertTrue(torch.allclose(-logp.mean(), nll, atol=1e-4))

    def test_narrower_temperature_raises_density_at_the_mean(self):
        out = craft_out()
        y = torch.tensor([[[0.3, -0.2, 1.0]]])
        at_one = step_logprob(out, y, temperature=1.0)
        at_half = step_logprob(out, y, temperature=0.5)
        # Both stds scale by 1/2, so the peak density grows by 4.
        self.assertAlmostEqual(
            (at_half - at_one).item(), math.log(4.0), places=3)

    def test_positive_advantage_increases_logprob(self):
        logp = torch.tensor(0.0, requires_grad=True)
        ref = torch.tensor(0.0)
        loss = reinforce_loss(logp, ref, 10, advantage=1.0, kl_coef=0.0)
        loss.backward()
        self.assertAlmostEqual(logp.grad.item(), -0.1, places=6)

    def test_negative_advantage_decreases_logprob(self):
        logp = torch.tensor(0.0, requires_grad=True)
        ref = torch.tensor(0.0)
        loss = reinforce_loss(logp, ref, 10, advantage=-1.0, kl_coef=0.0)
        loss.backward()
        self.assertAlmostEqual(logp.grad.item(), 0.1, places=6)

    def test_kl_pulls_back_toward_the_anchor(self):
        logp = torch.tensor(2.0, requires_grad=True)
        ref = torch.tensor(0.0)
        loss = reinforce_loss(logp, ref, 1, advantage=0.0, kl_coef=1.0)
        loss.backward()
        self.assertAlmostEqual(logp.grad.item(), 2.0, places=5)

    def test_constant_token_advantage_matches_the_scalar_loss(self):
        per = torch.tensor([0.2, -0.4, 0.1], requires_grad=True)
        ref = torch.zeros(3)
        scalar = reinforce_loss(per.sum(), ref.sum(), 3, advantage=1.5, kl_coef=0.1)
        token = token_reinforce_loss(per, ref, 1.5, 0.1)
        self.assertTrue(torch.allclose(scalar, token))
        faded = token_reinforce_loss(per, ref, [0.0, 1.5, 3.0], 0.0)
        faded.backward()
        # The middle step is the only one whose advantage matches the scalar 1.5,
        # and the loss averages three steps, so that step's gradient is -1.5/3.
        self.assertAlmostEqual(per.grad[1].item(), -0.5, places=5)


class AdvantageTests(unittest.TestCase):
    def test_best_of_three_centers_on_the_group(self):
        adv = group_advantages([1.0, 0.0, 0.0], 3)
        self.assertAlmostEqual(sum(adv), 0.0, places=6)
        self.assertAlmostEqual(adv[0], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(adv[1], -1.0 / 3.0, places=6)
        self.assertAlmostEqual(adv[2], -1.0 / 3.0, places=6)

    def test_single_sample_uses_ema_after_the_first_batch(self):
        adv, ema = advantages([1.0, 3.0], k=1, ema=None)
        self.assertAlmostEqual(sum(adv), 0.0, places=6)
        self.assertAlmostEqual(ema, 2.0, places=6)
        adv2, ema2 = advantages([1.0, 3.0], k=1, ema=ema)
        self.assertEqual(adv2, [-1.0, 1.0])
        self.assertAlmostEqual(ema2, 2.0, places=6)


class RenderTests(unittest.TestCase):
    def test_line_image_has_ink_and_nan_is_blank(self):
        seq = np.zeros((40, 3), np.float32)
        seq[:, 0] = 0.4
        seq[:, 1] = np.sin(np.linspace(0, 6, 40)).astype(np.float32) * 0.2
        seq[-1, 2] = 1.0
        img = strokes_to_image(seq, std=20.0, height=64, supersample=1)
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(img.size[1], 64)
        self.assertLess(np.asarray(img).min(), 250)

        seq[0, 0] = np.nan
        blank = strokes_to_image(seq, std=1.0, height=32, supersample=1)
        self.assertEqual(int(np.asarray(blank).min()), 255)

    def test_choice_sheet_holds_every_sample(self):
        from PIL import Image
        images = [Image.new("RGB", (80, 40), "white") for _ in range(3)]
        sheet = choice_sheet(images, "hello")
        self.assertEqual(sheet.mode, "RGB")
        self.assertGreater(sheet.size[0], 80 * 3)

    def test_consistency_sheet_stacks_one_sentence(self):
        from PIL import Image
        images = [Image.new("RGB", (200, 60), "white") for _ in range(10)]
        captions = [f"{i}  cer 0.00" for i in range(1, 11)]
        sheet = consistency_sheet(images, "the old man", captions, step=3, summary="mean cer 0.00")
        self.assertEqual(sheet.mode, "RGB")
        self.assertGreater(sheet.size[1], 60 * 10)


class ScorerTests(unittest.TestCase):
    def test_score_maps_cer_without_loading_trocr(self):
        from rl import TrocrScorer
        scorer = TrocrScorer.__new__(TrocrScorer)
        scorer.kind = "linear"
        scorer.tau = 0.1
        scorer.read = lambda images, max_new_tokens=64: ["hi", "hx"]
        rewards, cers, hyps = scorer.score([None, None], ["hi", "hi"])
        self.assertEqual(hyps, ["hi", "hx"])
        self.assertEqual(cers, [0.0, 0.5])
        self.assertAlmostEqual(rewards[0], 1.0)
        self.assertAlmostEqual(rewards[1], 0.5)

    def test_confidence_is_the_geometric_mean_of_chosen_tokens(self):
        import torch
        from rl import mix_confidence, token_confidence
        # Vocab of 2. Token 1 is certain in the first step and a coin flip in the second.
        sure = torch.tensor([[0.0, 20.0]])
        coin = torch.tensor([[0.0, 0.0]])
        sequences = torch.tensor([[0, 1, 1]])
        both = token_confidence((sure, coin), sequences, skip_ids={0})[0]
        only_sure = token_confidence((sure,), sequences[:, :2], skip_ids={0})[0]
        self.assertGreater(only_sure, 0.99)
        self.assertAlmostEqual(both, only_sure ** 0.5 * 0.5 ** 0.5, places=5)
        self.assertEqual(token_confidence((sure,), sequences[:, :2], skip_ids={0, 1}), [0.0])
        self.assertAlmostEqual(mix_confidence(1.0, 0.5, 0.2), 0.9)
        self.assertAlmostEqual(mix_confidence(1.0, 0.5, 0.0), 1.0)
        close = mix_confidence(1.0, 1.0, 0.2)
        unsure_close = mix_confidence(1.0, 0.0, 0.2)
        wrong = mix_confidence(0.0, 1.0, 0.2)
        unsure_wrong = mix_confidence(0.0, 0.0, 0.2)
        self.assertGreater(close, unsure_close)
        self.assertGreater(wrong, unsure_wrong)
        self.assertAlmostEqual(mix_confidence(0.5, 1.0, 0.2), 0.6)

    def test_score_mixes_in_a_stashed_confidence(self):
        from rl import TrocrScorer
        scorer = TrocrScorer.__new__(TrocrScorer)
        scorer.kind = "linear"
        scorer.tau = 0.1
        scorer.confidence_weight = 0.25

        def fake_read(images, max_new_tokens=64):
            scorer.last_confidence = [1.0, 0.0]
            return ["hi", "hx"]

        scorer.read = fake_read
        rewards, cers, hyps = scorer.score([None, None], ["hi", "hi"])
        self.assertEqual(cers, [0.0, 0.5])
        self.assertAlmostEqual(rewards[0], 1.0)
        self.assertAlmostEqual(rewards[1], 0.375)

        def fake_wrong(images, max_new_tokens=64):
            scorer.last_confidence = [1.0, 0.0]
            return ["no", "no"]

        scorer.read = fake_wrong
        rewards, cers, hyps = scorer.score([None, None], ["hi", "hi"])
        self.assertEqual(cers, [1.0, 1.0])
        self.assertAlmostEqual(rewards[0], 0.25)
        self.assertAlmostEqual(rewards[1], 0.0)
        self.assertGreater(rewards[0], rewards[1])


class CheckpointTests(unittest.TestCase):
    def test_next_index_ignores_optimizer_sidecars(self):
        from train_rl import next_checkpoint_index
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ckpt-0002.pth").write_bytes(b"w")
            (root / "ckpt-0002.optim.pth").write_bytes(b"o")
            (root / "latest.optim.pth").write_bytes(b"l")
            self.assertEqual(next_checkpoint_index(root), 3)

    def test_run_alias_keeps_a_resume_on_the_earlier_series(self):
        from rl_events import apply_run_aliases, load_run_aliases
        events = [
            {"run": "20261019-235110", "type": "update", "update": 2552},
            {"run": "20261020-001005", "type": "update", "update": 2553},
        ]
        apply_run_aliases(events, {"20261020-001005": "20261019-235110"})
        self.assertEqual([row["run"] for row in events], ["20261019-235110", "20261019-235110"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_aliases.json"
            path.write_text('{"new": "old"}\n', encoding="utf-8")
            self.assertEqual(load_run_aliases(tmp), {"new": "old"})
            self.assertEqual(load_run_aliases(Path(tmp) / "missing"), {})

    def test_re_run_uses_latest_weights_when_present(self):
        from train_rl import default_policy_ckpt
        with tempfile.TemporaryDirectory() as tmp:
            latest = Path(tmp) / "rl-trocr.pth"
            self.assertEqual(default_policy_ckpt(latest), "models/best-4.pth")
            latest.write_bytes(b"w")
            self.assertEqual(default_policy_ckpt(latest), str(latest))

    def test_recognizer_uses_mps_while_the_sampler_stays_on_cpu(self):
        from train_rl import choose_devices
        self.assertEqual(choose_devices(None, None, False, True), ("cpu", "mps"))
        self.assertEqual(choose_devices(None, None, True, False), ("cuda", "cuda"))
        self.assertEqual(choose_devices(None, None, False, False), ("cpu", "cpu"))
        self.assertEqual(choose_devices("mps", None, False, True), ("mps", "mps"))
        self.assertEqual(choose_devices("cpu", "cpu", False, True), ("cpu", "cpu"))

    def test_review_sentence_sticks_for_the_run(self):
        from train_rl import resolve_review_text, write_json
        stoi = {ch: i for i, ch in enumerate("the old man")}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            self.assertEqual(resolve_review_text(None, path, stoi), "the old man")
            write_json(path, {"review_text": "old man"})
            self.assertEqual(resolve_review_text(None, path, stoi), "old man")
            self.assertEqual(resolve_review_text("the man", path, stoi), "the man")


class BatchTests(unittest.TestCase):
    def test_stop_rows_match_the_scalar_rule(self):
        torch.manual_seed(0)
        reached = torch.full((6,), -1)
        take = torch.tensor([True, True, True, False, True, True])
        pen = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        phi_end = torch.tensor([0.2, 0.9, 0.1, 5.0, 0.4, 0.8])
        phi_max = torch.tensor([0.5, 0.3, 0.4, 0.0, 0.4, 0.2])
        scalar = []
        for i in range(6):
            if not bool(take[i]):
                scalar.append((None, False))
                continue
            scalar.append(stop_after_point(
                None, float(pen[i]), float(phi_end[i]), float(phi_max[i]),
                3, min_steps=2, tail_steps=4))
        new_reached, done = stop_rows(reached, take, pen, phi_end, phi_max, 3, 2, 4)
        for i, (when, ended) in enumerate(scalar):
            if not bool(take[i]):
                self.assertEqual(int(new_reached[i]), -1)
                self.assertFalse(bool(done[i]))
                continue
            self.assertEqual(int(new_reached[i]), -1 if when is None else when)
            self.assertEqual(bool(done[i]), ended)

    def test_padded_phantom_stays_on_the_real_last_character(self):
        model = HandwritingModel(vocab_size=5)
        h1 = torch.randn(2, model.hidden_size)
        kappa = torch.zeros(2, 10)
        chars = torch.zeros(2, 4, 5)
        mask = torch.zeros(2, 4)
        mask[0, :2] = 1.0
        mask[1, :] = 1.0
        _, _, phi, phi_end = model.compute_window(h1, chars, mask, kappa)
        _, _, phi_short, end_short = model.compute_window(
            h1[:1], chars[:1, :2], mask[:1, :2], kappa[:1])
        self.assertTrue(torch.allclose(phi_end[:1], end_short))
        self.assertTrue(torch.allclose(phi[:1, :2], phi_short))

    def test_batched_logprob_matches_one_row(self):
        torch.manual_seed(1)
        model = HandwritingModel(vocab_size=5).eval()
        stoi = {ch: i for i, ch in enumerate("abcde")}
        seqs = [torch.randn(6, 3), torch.randn(3, 3)]
        texts = ["ab", "abcde"]
        singles = [
            trajectory_logprob(model, seq, text, stoi, "cpu", 0.5, 2.0)
            for seq, text in zip(seqs, texts)
        ]
        totals, lengths = trajectory_logprobs(
            model, seqs, texts, stoi, "cpu", 0.5, 2.0)
        for index, (logp, length) in enumerate(singles):
            self.assertTrue(torch.allclose(totals[index], logp, rtol=1e-4, atol=1e-4))
            self.assertEqual(int(lengths[index]), int(length))

    def test_each_drawing_gets_a_temperature_in_range(self):
        random_state = random.getstate()
        random.seed(0)
        try:
            temps = draw_temperatures(4000, 0.1, 0.9)
        finally:
            random.setstate(random_state)
        self.assertEqual(len(temps), 4000)
        self.assertTrue(all(0.1 <= value <= 0.9 for value in temps))
        mean = sum(temps) / len(temps)
        self.assertAlmostEqual(mean, 0.5, delta=0.01)
        near = sum(0.4 <= value <= 0.6 for value in temps) / len(temps)
        self.assertGreater(near, 0.8)
        tails = sum(value < 0.35 or value > 0.65 for value in temps) / len(temps)
        self.assertLess(tails, 0.05)
        self.assertEqual(draw_temperatures(3, 0.4, 0.4), [0.4, 0.4, 0.4])
        self.assertEqual(draw_temperatures(0, 0.1, 0.9), [])

    def test_mixed_temperatures_match_one_row_at_a_time(self):
        torch.manual_seed(1)
        model = HandwritingModel(vocab_size=5).eval()
        stoi = {ch: i for i, ch in enumerate("abcde")}
        seqs = [torch.randn(6, 3), torch.randn(3, 3)]
        texts = ["ab", "abcde"]
        temps = [0.3, 0.7]
        singles = [
            trajectory_logprob(model, seq, text, stoi, "cpu", temp, 2.0)
            for seq, text, temp in zip(seqs, texts, temps)
        ]
        totals, lengths = trajectory_logprobs(
            model, seqs, texts, stoi, "cpu", 0.5, 2.0, temperatures=temps)
        for index, (logp, length) in enumerate(singles):
            self.assertTrue(torch.allclose(totals[index], logp, rtol=1e-4, atol=1e-4))
            self.assertEqual(int(lengths[index]), int(length))

    def test_sample_batch_accepts_a_temperature_per_row(self):
        torch.manual_seed(2)
        model = HandwritingModel(vocab_size=4).eval()
        stoi = {ch: i for i, ch in enumerate("abcd")}
        seqs = sample_batch(
            model, ["ab", "abcd"], stoi, "cpu", temperature=[0.0, 0.7],
            max_steps=4, min_steps=4, tail_steps=0, window_sharpen=2.0,
        )
        self.assertEqual([tuple(seq.shape) for seq in seqs], [(4, 3), (4, 3)])
        self.assertTrue(all(torch.isfinite(seq).all() for seq in seqs))

    def test_zero_temperature_stays_out_of_the_gradient(self):
        torch.manual_seed(3)
        model = HandwritingModel(vocab_size=4)
        ref = HandwritingModel(vocab_size=4)
        ref.load_state_dict(model.state_dict())
        for param in ref.parameters():
            param.requires_grad_(False)
        ref.eval()
        stoi = {ch: i for i, ch in enumerate("abcd")}
        rollouts = [
            {"text": "ab", "seq": torch.randn(4, 3), "temperature": 0.0},
            {"text": "ab", "seq": torch.randn(4, 3), "temperature": 0.4},
        ]
        before = model.fc.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        stats = reinforce_step(
            model, ref, rollouts, [1.0, -1.0], stoi, "cpu",
            temperature=0.5, sharpen=1.0, kl_coef=0.0, optimizer=optimizer, clip=1.0,
        )
        self.assertEqual(stats["used"], 1)
        self.assertEqual(stats["traces"][0]["skipped"], "zero_temperature")
        self.assertIsNone(stats["traces"][1]["skipped"])
        self.assertGreater(float((model.fc.weight.detach() - before).abs().sum()), 0.0)

    def test_sample_batch_keeps_each_budget(self):
        torch.manual_seed(2)
        model = HandwritingModel(vocab_size=4).eval()
        stoi = {ch: i for i, ch in enumerate("abcd")}
        seqs = sample_batch(
            model, ["ab", "abcd"], stoi, "cpu", temperature=0.5,
            max_steps=4, min_steps=4, tail_steps=0, window_sharpen=2.0,
        )
        self.assertEqual(len(seqs), 2)
        for seq in seqs:
            self.assertEqual(tuple(seq.shape), (4, 3))
            self.assertTrue(torch.isfinite(seq).all())


class BaseModelTests(unittest.TestCase):
    def test_refuses_to_overwrite_best4(self):
        with self.assertRaises(SystemExit):
            refuse_overwrite_base(BASE_CKPT)

    def test_best4_logprob_backward(self):
        ckpt = Path(BASE_CKPT)
        if not ckpt.exists():
            self.skipTest(f"base model {BASE_CKPT} is not present")
        model, stoi = load_model("cpu", ckpt=str(ckpt))
        seq = torch.zeros(4, 3)
        seq[:, 0] = 0.1
        logp, length = trajectory_logprob(
            model, seq, "hi", stoi, "cpu", temperature=1.0, sharpen=1.0)
        self.assertTrue(torch.isfinite(logp).item())
        self.assertEqual(int(length), 4)
        logp.backward()
        self.assertIsNotNone(model.fc.weight.grad)
        self.assertGreater(float(model.fc.weight.grad.abs().sum()), 0.0)

        model.zero_grad(set_to_none=True)
        sampled, _phis = sample(
            model, "hi", stoi, "cpu", temperature=1.0,
            max_steps=3, min_steps=3, tail_steps=0, window_sharpen=1.0,
        )
        logp2, _length = trajectory_logprob(
            model, sampled, "hi", stoi, "cpu", temperature=1.0, sharpen=1.0)
        self.assertTrue(torch.isfinite(logp2).item())
        self.assertEqual(tuple(sampled.shape), (3, 3))

    def test_best4_reinforce_step_moves_weights(self):
        ckpt = Path(BASE_CKPT)
        if not ckpt.exists():
            self.skipTest(f"base model {BASE_CKPT} is not present")
        model, ref, stoi = load_policy_and_ref(str(ckpt), str(ckpt), "cpu")
        rollouts = []
        for _ in range(2):
            seq, _phis = sample(
                model, "hi", stoi, "cpu", temperature=1.0,
                max_steps=4, min_steps=4, tail_steps=0, window_sharpen=1.0,
            )
            rollouts.append({"text": "hi", "seq": seq})
        before = model.fc.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        stats = reinforce_step(
            model, ref, rollouts, [1.0, -1.0], stoi, "cpu",
            temperature=1.0, sharpen=1.0, kl_coef=0.1, optimizer=optimizer, clip=1.0,
        )
        self.assertEqual(stats["used"], 2)
        self.assertTrue(math.isfinite(stats["loss"]))
        self.assertTrue(math.isfinite(stats["kl"]))
        self.assertGreater(float((model.fc.weight.detach() - before).abs().sum()), 0.0)


def _pen_seq(rows):
    seq = torch.randn(rows, 3)
    seq[:, 2] = (seq[:, 2] > 0).float()
    return seq


class BeamTests(unittest.TestCase):
    def test_prefixes_keep_every_word_written_so_far(self):
        self.assertEqual(
            word_prefixes("john is a handsome man"),
            ["john", "john is", "john is a", "john is a handsome", "john is a handsome man"],
        )
        self.assertEqual(word_prefixes("hello, world"), ["hello,", "hello, world"])
        with self.assertRaises(ValueError):
            word_prefixes("   ")

    def test_select_top_breaks_ties_toward_the_earlier_index(self):
        self.assertEqual(select_top([0.2, 0.9, 0.9, 0.1], 2), [1, 2])

    def test_beam_factors_are_the_divisors(self):
        self.assertEqual(beam_factors(16), [1, 2, 4, 8, 16])
        self.assertEqual(beam_factors(9), [1, 3, 9])
        self.assertEqual(beam_factors(7), [1, 7])
        self.assertEqual(beam_factors(1), [1])
        with self.assertRaises(ValueError):
            beam_factors(0)

    def test_a_line_averages_every_generation(self):
        import train_rl
        stats = train_rl.summarize_line([
            {"loss": 1.0, "kl": 2.0, "used": 1, "grad": 0.5, "stepped": True},
            {"loss": 3.0, "kl": 6.0, "used": 3, "grad": 1.5, "stepped": False},
        ])
        self.assertAlmostEqual(stats["loss"], 2.5)
        self.assertAlmostEqual(stats["kl"], 5.0)
        self.assertAlmostEqual(stats["grad"], 1.0)
        self.assertEqual(stats["used"], 4)
        self.assertTrue(stats["stepped"])

    def test_novel_line_is_ten_short_words_from_the_alphabet(self):
        state = random.getstate()
        random.seed(0)
        try:
            stoi = {ch: i for i, ch in enumerate("abC12 !")}
            letters = set("abC12")
            seen = set()
            for _ in range(40):
                line = novel_line(stoi)
                words = line.split(" ")
                self.assertEqual(len(words), 10)
                self.assertEqual(line, " ".join(words))
                self.assertLessEqual(len(line), 48)
                for word in words:
                    self.assertGreaterEqual(len(word), 1)
                    self.assertLessEqual(len(word), 4)
                    self.assertTrue(set(word) <= letters)
                seen.update(line)
            self.assertTrue(seen & set("ab"))
            self.assertIn("C", seen)
            self.assertTrue(seen & set("12"))
            tight = novel_line(stoi, max_chars=19)
            self.assertEqual([len(word) for word in tight.split(" ")], [1] * 10)
            self.assertLessEqual(len(tight), 19)
        finally:
            random.setstate(state)

    def test_choose_label_mixes_the_play_with_a_fresh_line(self):
        state = random.getstate()
        random.seed(1)
        try:
            stoi = {ch: i for i, ch in enumerate("abcABC012")}
            prompts = ["the old man"]
            self.assertEqual(
                [choose_label(prompts, stoi, 0.0) for _ in range(8)],
                [("the old man", "play")] * 8,
            )
            fresh = [choose_label(prompts, stoi, 1.0) for _ in range(8)]
            self.assertTrue(all(kind == "novel" for _line, kind in fresh))
            self.assertTrue(all(len(line.split()) == 10 for line, _kind in fresh))
            self.assertTrue(all(len(word) <= 4 for line, _kind in fresh for word in line.split()))
            self.assertTrue(all(set(line) <= set("abcABC012 ") for line, _kind in fresh))
            lexicon = ["stone", "quiet", "apple", "river"]
            dictated = [choose_label(prompts, stoi, 0.0, words=1.0, lexicon=lexicon) for _ in range(8)]
            self.assertTrue(all(kind == "words" for _line, kind in dictated))
            self.assertTrue(all(set(line.split()) <= set(lexicon) for line, _kind in dictated))
        finally:
            random.setstate(state)

    def test_dictionary_words_are_lowercase_and_no_longer_than_the_play(self):
        state = random.getstate()
        random.seed(2)
        try:
            lexicon = ["aa" + chr(ord("a") + i) for i in range(20)]
            for _ in range(30):
                line = word_line(lexicon, max_chars=48)
                words = line.split(" ")
                self.assertEqual(line, " ".join(words))
                self.assertGreaterEqual(len(words), 2)
                self.assertLessEqual(len(line), 48)
                self.assertGreaterEqual(len(line), 48 - min(len(word) for word in lexicon))
                for word in words:
                    self.assertRegex(word, r"^[a-z]+$")
                    self.assertIn(word, lexicon)
            self.assertLessEqual(len(word_line(["alphabet"], max_chars=8)), 8)
            with self.assertRaises(ValueError):
                word_line(["toolong"], max_chars=4)
        finally:
            random.setstate(state)

    def test_dictionary_loader_drops_capitals_and_punctuation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "words.txt"
            path.write_text("Apple\nDon't\nwell-known\nstone\nSTONE\nCafé\n123\na\n\n", encoding="utf-8")
            stoi = {ch: i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz ")}
            self.assertEqual(load_dictionary(path, stoi, 48), ["apple", "stone"])

    def test_split_logprob_matches_one_replay_of_the_same_text(self):
        torch.manual_seed(0)
        model = HandwritingModel(vocab_size=2, hidden_size=16).eval()
        stoi = {"a": 0, "b": 1}
        text = "ab"
        full = _pen_seq(8)
        prefix, new = full[:3], full[3:]
        lp_prefix, n_prefix = trajectory_logprobs(
            model, [prefix], [text], stoi, "cpu", 1.0, 1.0)
        state = history_state(model, [(text, prefix)], stoi, "cpu", 1.0)
        lp_new, n_new = logprobs_from_states(
            model, [state], [new], [text], stoi, "cpu", 1.0, 1.0)
        lp_full, n_full = trajectory_logprobs(
            model, [full], [text], stoi, "cpu", 1.0, 1.0)
        self.assertEqual(int(n_prefix + n_new), int(n_full))
        self.assertTrue(torch.allclose(lp_prefix + lp_new, lp_full, rtol=1e-4, atol=1e-4))

        a, b, c = full[:3], full[3:5], full[5:]
        lp_a, _ = trajectory_logprobs(model, [a], [text], stoi, "cpu", 1.0, 1.0)
        state_a = history_state(model, [(text, a)], stoi, "cpu", 1.0)
        lp_b, _ = logprobs_from_states(model, [state_a], [b], [text], stoi, "cpu", 1.0, 1.0)
        state_ab = history_state(model, [(text, a), (text, b)], stoi, "cpu", 1.0)
        lp_c, _ = logprobs_from_states(model, [state_ab], [c], [text], stoi, "cpu", 1.0, 1.0)
        self.assertTrue(torch.allclose(lp_a + lp_b + lp_c, lp_full, rtol=1e-4, atol=1e-4))

        # A later word is conditioned on the longer string. Replaying the whole
        # stroke under only that longer string is a different density.
        short, longer = "a", "ab"
        head, tail = full[:4], full[4:]
        lp_head, _ = trajectory_logprobs(model, [head], [short], stoi, "cpu", 1.0, 1.0)
        carried = history_state(model, [(short, head)], stoi, "cpu", 1.0)
        lp_tail, _ = logprobs_from_states(
            model, [carried], [tail], [longer], stoi, "cpu", 1.0, 1.0)
        lp_longer, _ = trajectory_logprobs(model, [full], [longer], stoi, "cpu", 1.0, 1.0)
        self.assertFalse(torch.allclose(lp_head + lp_tail, lp_longer, rtol=1e-4, atol=1e-3))

    def test_continued_rollout_uses_the_saved_pen_state(self):
        torch.manual_seed(1)
        model = HandwritingModel(vocab_size=2, hidden_size=16)
        ref = HandwritingModel(vocab_size=2, hidden_size=16)
        ref.load_state_dict(model.state_dict())
        for param in ref.parameters():
            param.requires_grad_(False)
        stoi = {"a": 0, "b": 1}
        text = "ab"
        prefix = _pen_seq(4)
        new = _pen_seq(4)
        state = history_state(model, [(text, prefix)], stoi, "cpu", 1.0)
        expected, _ = logprobs_from_states(
            model, [state], [new], [text], stoi, "cpu", 1.0, 1.0)
        from_scratch, _ = trajectory_logprobs(
            model, [new], [text], stoi, "cpu", 1.0, 1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        stats = reinforce_step(
            model, ref,
            [{
                "text": text,
                "seq": new,
                "temperature": 1.0,
                "state": state,
                "history": [(text, prefix)],
            }],
            [1.0], stoi, "cpu", 1.0, 1.0, 0.1, optimizer, 1.0,
        )
        got = torch.tensor(stats["traces"][0]["logp"])
        self.assertTrue(torch.allclose(got, expected, rtol=1e-4, atol=1e-4))
        self.assertFalse(torch.allclose(got, from_scratch, rtol=1e-3, atol=1e-3))
        self.assertTrue(math.isfinite(stats["kl"]))

    def test_sample_batch_resumes_and_returns_only_the_new_points(self):
        torch.manual_seed(0)
        model = HandwritingModel(vocab_size=4, hidden_size=16).eval()
        stoi = {ch: i for i, ch in enumerate("abcd")}
        seqs, states = sample_batch(
            model, ["ab"], stoi, "cpu", temperature=1.0,
            max_steps=5, min_steps=2, tail_steps=1, steps_per_char=2,
            window_sharpen=1.0, return_state=True,
        )
        self.assertGreaterEqual(seqs[0].shape[0], 1)
        self.assertEqual(states[0]["x"].shape, (3,))
        self.assertIsInstance(states[0]["stopped"], bool)
        more = sample_batch(
            model, ["abcd"], stoi, "cpu", temperature=1.0,
            max_steps=4, min_steps=2, tail_steps=1, steps_per_char=2,
            window_sharpen=1.0, states=states,
        )
        self.assertGreaterEqual(more[0].shape[0], 1)
        self.assertEqual(more[0].shape[1], 3)

    def test_a_nine_word_line_forks_into_thirds(self):
        self.assertEqual(default_branches(9, 16), (4, 2, 2))
        self.assertEqual(default_branches(6, 16), (4, 2, 2))
        self.assertEqual(default_branches(10, 16), (4, 2, 2))
        plan = tree_plan(9, 16)
        self.assertEqual([stage["end"] for stage in plan], [3, 6, 9])
        self.assertEqual([stage["copies"] for stage in plan], [4, 2, 2])
        self.assertEqual(tree_plan(9, 16, (16,)), [{"end": 9, "copies": 16}])

    def test_a_short_line_forks_once_and_one_word_does_not(self):
        self.assertEqual(default_branches(4, 16), (4, 4))
        self.assertEqual([stage["end"] for stage in tree_plan(4, 16)], [2, 4])
        self.assertEqual(default_branches(2, 16), (4, 4))
        self.assertEqual([stage["copies"] for stage in tree_plan(1, 16)], [16])
        self.assertEqual(default_branches(9, 9), (3, 3))
        with self.assertRaises(ValueError):
            tree_plan(2, 16, (4, 2, 2))

    def test_chunk_advantage_is_how_the_leaf_mean_moved(self):
        def node(reward=None, parent=None):
            item = {"parent": parent, "children": [], "reward": reward}
            if parent is not None:
                parent["children"].append(item)
            return item

        roots = [node(), node()]
        mids = [node(parent=root) for root in roots for _ in range(2)]
        leaves = [node(parent=mid) for mid in mids for _ in range(2)]
        for leaf, reward in zip(leaves, [0, 0, 0, 4, 10, 10, 10, 10]):
            leaf["reward"] = reward
        adv = chunk_advantages(leaves)
        self.assertAlmostEqual(adv[id(roots[0])], 1 - 5.5)
        self.assertAlmostEqual(adv[id(roots[1])], 10 - 5.5)
        self.assertAlmostEqual(adv[id(mids[0])], 0 - 1)
        self.assertAlmostEqual(adv[id(mids[1])], 2 - 1)
        self.assertAlmostEqual(adv[id(mids[2])], 0.0)
        self.assertAlmostEqual(adv[id(leaves[3])], 4 - 2)
        self.assertAlmostEqual(adv[id(leaves[0])], 0.0)
        self.assertAlmostEqual(adv[id(leaves[4])], 0.0)
        flat = [{"parent": None, "children": [], "reward": reward} for reward in (1.0, 3.0)]
        flat_adv = chunk_advantages(flat)
        self.assertEqual([flat_adv[id(item)] for item in flat], [-1.0, 1.0])

    def test_a_boundary_fades_and_a_flat_advantage_does_not(self):
        self.assertEqual(blend_token_advantages(2.0, 4), [2.0, 2.0, 2.0, 2.0])
        faded = blend_token_advantages(2.0, 4, parent=0.0, radius=4)
        self.assertEqual(faded[0], 0.0)
        self.assertAlmostEqual(faded[-1], 1.5)
        tail = blend_token_advantages(2.0, 4, child=0.0, radius=4)
        self.assertEqual(tail[-1], 0.0)
        self.assertAlmostEqual(tail[0], 1.5)

    def test_forks_are_equal_pen_step_windows_on_the_full_line(self):
        self.assertEqual(segment_budgets("a" * 45, 1), [None])
        self.assertEqual(segment_budgets("a" * 45, 3), [450, 450, None])

    def test_each_class_gets_its_own_leaf_budget(self):
        width, branches = class_budget("words", 5, 16, pinned=(4, 2, 2), easy_leaves=6)
        self.assertEqual((width, branches), (6, (6,)))
        width, branches = class_budget("novel", 10, 16, pinned=(4, 2, 2), novel_leaves=64)
        self.assertEqual((width, branches), (64, (64,)))
        width, branches = class_budget("play", 9, 16, pinned=(4, 2, 2))
        self.assertEqual((width, branches), (16, (4, 2, 2)))
        width, branches = class_budget("play", 9, 16, pinned=(16,))
        self.assertEqual((width, branches), (16, (16,)))
        random.seed(0)
        seen = set()
        for _ in range(30):
            width, branches = class_budget("play", 9, 16)
            self.assertEqual(width, 16)
            self.assertIn(branches, ((4, 2, 2), (4, 4), (2, 8), (16,)))
            seen.add(branches)
        self.assertIn((4, 2, 2), seen)
        self.assertGreater(len(seen), 1)
        random.seed(0)
        easy, easy_branches = class_budget("words", 4, 16)
        self.assertEqual(easy_branches, (easy,))
        self.assertGreaterEqual(easy, 4)
        self.assertLessEqual(easy, 8)

    def test_a_fork_scores_only_the_finished_lines(self):
        import generate
        import train_rl
        from types import SimpleNamespace

        calls = []
        recorded = []

        issued = []

        def fake_temps(count, low, high):
            batch = [0.2 + 0.3 * i for i in range(count)]
            issued.append(list(batch))
            return list(batch)

        def fake_sample(model, texts, stoi, device, temperature=1.0, states=None,
                        return_state=False, window_sharpen=2.0, max_steps=None, **kwargs):
            self.assertEqual(window_sharpen, 1.0)
            self.assertTrue(return_state)
            call = len(calls)
            calls.append({
                "texts": list(texts),
                "states": states,
                "max_steps": max_steps,
                "temperature": [float(value) for value in temperature],
            })
            seqs = []
            saved = []
            for index in range(len(texts)):
                seq = torch.zeros(4, 3)
                seq[:, 0] = call * 100 + index
                seq[-1, 2] = 1.0
                seqs.append(seq)
                parent = None if states is None else states[index]["id"]
                saved.append({
                    "id": (call, index), "parent": parent, "x": seq[-1].clone(),
                    "stopped": False,
                })
            return seqs, saved

        def fake_reinforce(model, ref, rollouts, advs, stoi, device, temperature,
                           sharpen, kl_coef, optimizer, clip, apply=True, normalizer=None):
            recorded.append((rollouts, list(advs), apply, normalizer))
            return {
                "loss": 0.0,
                "kl": 0.0,
                "used": len(rollouts),
                "grad": 0.0,
                "stepped": False,
                "traces": [{} for _ in rollouts],
            }

        class Scorer:
            def __init__(self):
                self.refs = []
                self.last_confidence = None

            def score(self, images, refs):
                self.refs.append(list(refs))
                return [0.0, 4.0, 10.0, 10.0], [0.1, 0.2, 0.0, 0.0], list(refs)

        args = SimpleNamespace(
            beam=4, tree="2x2", temp_min=0.5, temp_max=0.5, sharpen=1.0,
            straightness=0.0, spacing=0.0, temperature=0.5, kl=0.0, clip=1.0,
            sample_every=0,
        )
        scorer = Scorer()
        real_sample = generate.sample_batch
        real_step = train_rl.reinforce_step
        real_temps = train_rl.draw_temperatures
        generate.sample_batch = fake_sample
        train_rl.reinforce_step = fake_reinforce
        train_rl.draw_temperatures = fake_temps
        try:
            stages = list(train_rl.beam_stages(
                None, None, scorer, None, ["john is a man"],
                {ch: i for i, ch in enumerate("john isaman")},
                1.0, "cpu", args, 3.0,
            ))
        finally:
            generate.sample_batch = real_sample
            train_rl.reinforce_step = real_step
            train_rl.draw_temperatures = real_temps

        line = "john is a man"
        self.assertEqual(scorer.refs, [[line] * 4])
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0][-1], {
            "tree": "2x2",
            "cuts": segment_budgets(line, 2),
            "branches": [2, 2],
            "method": "tree",
        })
        self.assertEqual(stages[0][5], 3.0)
        self.assertEqual(stages[0][0]["used"], 6)
        self.assertEqual(stages[0][1], [0.0, 4.0, 10.0, 10.0])
        self.assertEqual([item["text"] for item in stages[0][6]], [line] * 4)
        self.assertEqual(calls[0]["texts"], [line] * 2)
        self.assertIsNone(calls[0]["states"])
        self.assertEqual(calls[0]["max_steps"], segment_budgets(line, 2)[0])
        self.assertEqual(calls[1]["texts"], [line] * 4)
        self.assertIsNone(calls[1]["max_steps"])
        self.assertEqual(
            [item["id"] for item in calls[1]["states"]],
            [(0, 0), (0, 0), (0, 1), (0, 1)],
        )
        roots, root_advs, root_apply, root_norm = recorded[0]
        self.assertFalse(root_apply)
        self.assertEqual(root_norm, 4)
        self.assertEqual([item["text"] for item in roots], [line] * 2)
        self.assertTrue(all("state" not in item for item in roots))
        self.assertEqual(root_advs, [-4.0, 4.0])
        # The root has children, so its last pen steps fade toward their mean.
        self.assertNotEqual(roots[0]["adv_steps"][-1], root_advs[0])
        leaves, leaf_advs, leaf_apply, leaf_norm = recorded[1]
        self.assertFalse(leaf_apply)
        self.assertEqual(leaf_norm, 8)
        self.assertEqual(leaf_advs, [-2.0, 2.0, 0.0, 0.0])
        for index, item in enumerate(leaves):
            parent = index // 2
            self.assertEqual(item["text"], line)
            self.assertEqual(item["state"]["id"], (0, parent))
            self.assertEqual(len(item["history"]), 1)
            self.assertEqual(item["history"][0][0], line)
            self.assertEqual(float(item["history"][0][1][0, 0]), parent)
            self.assertEqual(item["seq"].shape[0], 4)
        # Roots draw a temperature. Both forks of a root keep it, including
        # the log-prob the update reinforces.
        self.assertEqual(issued, [[0.2, 0.5]])
        self.assertEqual(calls[0]["temperature"], [0.2, 0.5])
        self.assertEqual(calls[1]["temperature"], [0.2, 0.2, 0.5, 0.5])
        self.assertEqual([item["temperature"] for item in roots], [0.2, 0.5])
        self.assertEqual([item["temperature"] for item in leaves], [0.2, 0.2, 0.5, 0.5])

    def test_a_novel_exact_read_is_cloned_and_the_misses_are_left_out(self):
        import generate
        import train_rl
        from types import SimpleNamespace

        calls = []
        recorded = []

        def fake_sample(model, texts, stoi, device, temperature=1.0, states=None,
                        return_state=False, window_sharpen=2.0, max_steps=None, **kwargs):
            calls.append({"texts": list(texts), "states": states, "max_steps": max_steps})
            seqs = []
            saved = []
            for index in range(len(texts)):
                seq = torch.zeros(3, 3)
                seq[-1, 2] = 1.0
                seqs.append(seq)
                saved.append({"id": index, "x": seq[-1].clone(), "stopped": False})
            return seqs, saved

        def fake_reinforce(model, ref, rollouts, advs, stoi, device, temperature,
                           sharpen, kl_coef, optimizer, clip, apply=True, normalizer=None):
            recorded.append((rollouts, list(advs), apply, normalizer))
            return {
                "loss": 0.0, "kl": 0.0, "used": len(rollouts), "grad": 0.0,
                "stepped": False, "traces": [{} for _ in rollouts],
            }

        class Scorer:
            def __init__(self):
                self.last_confidence = None

            def score(self, images, refs):
                n = len(refs)
                cers = [0.0] + [0.5] * (n - 1)
                return [1.0] * n, cers, list(refs)

        args = SimpleNamespace(
            beam=16, tree="4x2x2", temp_min=0.5, temp_max=0.5, sharpen=1.0,
            straightness=0.0, spacing=0.0, temperature=0.5, kl=0.0, clip=1.0,
            sample_every=0, novel=1.0, words=0.0, novel_leaves=4, easy_leaves=6,
            max_chars=48, lexicon=None, text=None,
        )
        stoi = {ch: i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz0123456789")}
        real_sample = generate.sample_batch
        real_step = train_rl.reinforce_step
        generate.sample_batch = fake_sample
        train_rl.reinforce_step = fake_reinforce
        try:
            stages = list(train_rl.beam_stages(
                None, None, Scorer(), None, ["unused"], stoi, 1.0, "cpu", args, 0.0,
            ))
        finally:
            generate.sample_batch = real_sample
            train_rl.reinforce_step = real_step

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]["texts"]), 4)
        self.assertIsNone(calls[0]["states"])
        self.assertIsNone(calls[0]["max_steps"])
        self.assertEqual(len(set(calls[0]["texts"])), 1)
        self.assertEqual(stages[0][-1]["method"], "sft")
        self.assertEqual(stages[0][-1]["tree"], "4")
        self.assertEqual(stages[0][9], "novel")
        rollouts, advs, apply, normalizer = recorded[0]
        self.assertFalse(apply)
        self.assertEqual(normalizer, 1)
        self.assertEqual(advs[0], 1.0)
        self.assertFalse(rollouts[0].get("skip"))
        self.assertTrue(all(item.get("skip") == "rejected" for item in rollouts[1:]))

    def test_a_dictionary_line_is_a_few_separate_drawings(self):
        import generate
        import train_rl
        from types import SimpleNamespace

        calls = []

        def fake_sample(model, texts, stoi, device, temperature=1.0, states=None,
                        return_state=False, window_sharpen=2.0, max_steps=None, **kwargs):
            calls.append({"n": len(texts), "states": states, "max_steps": max_steps,
                          "texts": list(texts)})
            seqs = [torch.zeros(3, 3) for _ in texts]
            saved = [{"x": seq[-1].clone(), "stopped": False} for seq in seqs]
            return seqs, saved

        def fake_reinforce(*args, **kwargs):
            rollouts = args[2]
            return {
                "loss": 0.0, "kl": 0.0, "used": len(rollouts), "grad": 0.0,
                "stepped": False, "traces": [{} for _ in rollouts],
            }

        class Scorer:
            def __init__(self):
                self.calls = 0
                self.last_confidence = None

            def score(self, images, refs):
                self.calls += 1
                n = len(refs)
                return list(range(n)), [0.2] * n, list(refs)

        args = SimpleNamespace(
            beam=16, tree="4x2x2", temp_min=0.5, temp_max=0.5, sharpen=1.0,
            straightness=0.0, spacing=0.0, temperature=0.5, kl=0.0, clip=1.0,
            sample_every=0, novel=0.0, words=1.0, novel_leaves=64, easy_leaves=5,
            max_chars=48, lexicon=["man", "john", "sam"], text=None,
        )
        stoi = {ch: i for i, ch in enumerate("john isaman")}
        real_sample = generate.sample_batch
        real_step = train_rl.reinforce_step
        generate.sample_batch = fake_sample
        train_rl.reinforce_step = fake_reinforce
        try:
            stages = list(train_rl.beam_stages(
                None, None, Scorer(), None, ["the play"], stoi, 1.0, "cpu", args, 0.0,
            ))
        finally:
            generate.sample_batch = real_sample
            train_rl.reinforce_step = real_step

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["n"], 5)
        self.assertIsNone(calls[0]["states"])
        self.assertIsNone(calls[0]["max_steps"])
        self.assertEqual(len(set(calls[0]["texts"])), 1)
        self.assertEqual(stages[0][-1]["method"], "flat")
        self.assertEqual(stages[0][-1]["tree"], "5")
        self.assertEqual(stages[0][9], "words")

    def test_one_step_applies_the_accumulated_gradient(self):
        import train_rl

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.tensor(1.0))

        model = Tiny()
        model.w.grad = torch.tensor(1.0)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        grad, stepped = train_rl.commit_grads(model, opt, 1.0, used=1)
        self.assertTrue(stepped)
        self.assertAlmostEqual(grad, 1.0, places=5)
        self.assertAlmostEqual(float(model.w.detach()), 0.9, places=5)
        _, stepped = train_rl.commit_grads(model, opt, 1.0, used=0)
        self.assertFalse(stepped)
        self.assertAlmostEqual(float(model.w.detach()), 0.9, places=5)


if __name__ == "__main__":
    unittest.main()
