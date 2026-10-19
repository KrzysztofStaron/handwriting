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
    choose_label,
    history_state,
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
        arch = 10 * np.sin(np.pi * xs / xs[-1])
        arched = line_straightness(_band_seq(xs, arch, arch + 10), std=1.0)
        tilted = line_straightness(_band_seq(xs, 0.025 * xs, 10 + 0.025 * xs), std=1.0)
        self.assertAlmostEqual(flat, 0.0, places=6)
        self.assertGreater(arched, 0.2)
        self.assertGreater(tilted, 0.5)
        self.assertLess(arched, tilted)
        self.assertEqual(line_straightness(np.zeros((3, 3)), std=1.0), 0.0)
        sunk = line_straightness(_band_seq(xs, np.where(xs < 300, 0.0, -10.0), np.where(xs < 300, 10.0, 0.0)), std=1.0)
        self.assertGreater(sunk, arched)


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
                ["the old man"] * 8,
            )
            fresh = [choose_label(prompts, stoi, 1.0) for _ in range(8)]
            self.assertTrue(all(len(line.split()) == 10 for line in fresh))
            self.assertTrue(all(len(word) <= 4 for line in fresh for word in line.split()))
            self.assertTrue(all(set(line) <= set("abcABC012 ") for line in fresh))
        finally:
            random.setstate(state)

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
        more = sample_batch(
            model, ["abcd"], stoi, "cpu", temperature=1.0,
            max_steps=4, min_steps=2, tail_steps=1, steps_per_char=2,
            window_sharpen=1.0, states=states,
        )
        self.assertGreaterEqual(more[0].shape[0], 1)
        self.assertEqual(more[0].shape[1], 3)

    def test_each_stage_scores_the_prefix_and_carries_its_history(self):
        import generate
        import train_rl
        from types import SimpleNamespace

        calls = []

        def fake_sample(model, texts, stoi, device, temperature=1.0, states=None,
                        return_state=False, window_sharpen=2.0):
            self.assertEqual(window_sharpen, 1.0)
            self.assertTrue(return_state)
            call = len(calls)
            calls.append({"texts": list(texts), "states": states})
            seqs = []
            saved = []
            for index in range(len(texts)):
                seq = torch.zeros(4, 3)
                seq[:, 0] = call * 100 + index
                seq[-1, 2] = 1.0
                seqs.append(seq)
                saved.append({"id": (call, index), "x": seq[-1].clone()})
            return seqs, saved

        recorded = []

        def fake_reinforce(model, ref, rollouts, advs, stoi, device, temperature,
                           sharpen, kl_coef, optimizer, clip):
            recorded.append(rollouts)
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
                n = len(refs)
                return [float(n - i) for i in range(n)], [0.0] * n, list(refs)

        args = SimpleNamespace(
            beam=9, keep=3, temp_min=0.5, temp_max=0.5, sharpen=1.0,
            straightness=0.0, temperature=0.5, kl=0.0, clip=1.0, sample_every=0,
        )
        scorer = Scorer()
        real_sample = generate.sample_batch
        real_step = train_rl.reinforce_step
        generate.sample_batch = fake_sample
        train_rl.reinforce_step = fake_reinforce
        try:
            stages = list(train_rl.beam_stages(
                None, None, scorer, None, ["john is a"],
                {ch: i for i, ch in enumerate("john isa")},
                1.0, "cpu", args, None,
            ))
        finally:
            generate.sample_batch = real_sample
            train_rl.reinforce_step = real_step

        self.assertEqual(len(stages), 1)
        self.assertEqual(len(recorded), 3)
        self.assertEqual(stages[0][0]["used"], 27)
        self.assertEqual(len(stages[0][1]), 27)
        texts = [item["text"] for item in stages[0][6]]
        self.assertEqual(texts[:9], ["john"] * 9)
        self.assertEqual(texts[-9:], ["john is a"] * 9)
        self.assertEqual(scorer.refs, [
            ["john"] * 9,
            ["john is"] * 9,
            ["john is a"] * 9,
        ])
        self.assertEqual([call["texts"] for call in calls], scorer.refs)
        self.assertIsNone(calls[0]["states"])
        self.assertEqual(
            [item["id"] for item in calls[1]["states"]],
            [(0, parent) for parent in range(3) for _ in range(3)],
        )
        self.assertEqual([item["text"] for item in recorded[0]], ["john"] * 9)
        self.assertTrue(all("state" not in item for item in recorded[0]))

        stage1 = recorded[1]
        self.assertEqual([item["text"] for item in stage1], ["john is"] * 9)
        for parent in range(3):
            for item in stage1[parent * 3:(parent + 1) * 3]:
                self.assertEqual(item["state"]["id"], (0, parent))
                self.assertEqual(len(item["history"]), 1)
                self.assertEqual(item["history"][0][0], "john")
                self.assertEqual(item["seq"].shape[0], 4)
                self.assertEqual(float(item["history"][0][1][0, 0]), parent)

        # The top three of stage 1 are the first parent's three continuations.
        for child, mark in enumerate((100.0, 101.0, 102.0)):
            for item in recorded[2][child * 3:(child + 1) * 3]:
                self.assertEqual(item["text"], "john is a")
                self.assertEqual(item["state"]["id"], (1, child))
                self.assertEqual([past[0] for past in item["history"]], ["john", "john is"])
                self.assertEqual(float(item["history"][0][1][0, 0]), 0.0)
                self.assertEqual(float(item["history"][1][1][0, 0]), mark)
                self.assertEqual(item["seq"].shape[0], 4)


if __name__ == "__main__":
    unittest.main()
