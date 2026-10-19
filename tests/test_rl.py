"""Reward, log-prob, and REINFORCE tests.

The only checkpoint this suite loads is models/best-4.pth. Other .pth files
are experiment runs and are not the base model.
"""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate import load_model, sample
from model import M, sequence_loss
from rl import (
    BASE_CKPT,
    advantages,
    apply_temperature,
    cer,
    cer_and_reward,
    choice_sheet,
    consistency_sheet,
    group_advantages,
    load_policy_and_ref,
    refuse_overwrite_base,
    reinforce_step,
    reward_from_cer,
    step_logprob,
    strokes_to_image,
    trajectory_logprob,
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


class RewardTests(unittest.TestCase):
    def test_cer_known_distance(self):
        self.assertAlmostEqual(cer("kitten", "sitting"), 0.5)
        self.assertAlmostEqual(cer("cat", "cats"), 1.0 / 3.0)
        self.assertEqual(cer("same", "same"), 0.0)
        self.assertEqual(cer("abc", ""), 1.0)

    def test_cer_ignores_spacing_and_curly_quotes(self):
        self.assertEqual(cer("a  b", "a b"), 0.0)
        self.assertEqual(cer('say "hi"', "say \u201chi\u201d"), 0.0)

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
        scorer.read = lambda images: ["hi", "hx"]
        rewards, cers, hyps = scorer.score([None, None], ["hi", "hi"])
        self.assertEqual(hyps, ["hi", "hx"])
        self.assertEqual(cers, [0.0, 0.5])
        self.assertAlmostEqual(rewards[0], 1.0)
        self.assertAlmostEqual(rewards[1], 0.5)


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


if __name__ == "__main__":
    unittest.main()
