"""Post-training pieces shared by the recognizer loop and the human loop.

The policy is the Graves handwriting model. A trajectory is a stroke sequence
sampled the same way as generate.py (temperature bias and window sharpening
included). The reward is one number for the whole trajectory, so the update is
sequence-level REINFORCE:

    grad = advantage * grad mean_t log pi(a_t | history, text)

Length-normalized, not the raw sum, so a long line does not take a step
proportional to its length. When several samples share a prompt, the advantage
subtracts that prompt's mean reward. A KL penalty toward a frozen copy of
models/best-4.pth is folded into the advantage with a stop-gradient (the
InstructGPT form). The anchor is the base model; other checkpoints are
experiment runs and this module refuses to write best-4.pth.

Recognizer reward, from the character error rate of TrOCR-Large:

    linear    1 - CER
    exp       exp(-CER)                         default
    soft      1 if CER <= tau, else (1 - CER) / (1 - tau) until CER = 1, then 0

CER is capped at 2 before that map.
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from data import collate
from generate import load_model, sample
from model import split_params

BASE_CKPT = "models/best-4.pth"
CER_CAP = 2.0

_QUOTES = str.maketrans({
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2014": "-", "\u2013": "-",
})


def refuse_overwrite_base(path):
    """best-4.pth is the base model. Post-training writes somewhere else."""
    if Path(path).expanduser().resolve() == Path(BASE_CKPT).resolve():
        raise SystemExit(f"refusing to write {path}: {BASE_CKPT} is the base model")


def save_policy(model, path):
    refuse_overwrite_base(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_text(text):
    """Case-sensitive, with whitespace collapsed and curly quotes straightened.

    TrOCR sometimes emits typographic quotes for the straight ones in the prompt.
    Those are not handwriting errors.
    """
    if text is None:
        return ""
    return " ".join(str(text).translate(_QUOTES).strip().split())


def levenshtein(a, b):
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
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def cer(reference, hypothesis):
    """Character error rate, edit distance divided by the reference length.

    Can exceed 1 when the hypothesis is longer than the reference. An empty
    reference scores 0 against an empty hypothesis and 1 otherwise.
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def reward_from_cer(value, kind="exp", tau=0.1):
    """Map a (possibly capped) CER to a scalar reward.

    linear  1 - CER. Proportional, and negative once CER exceeds 1.
    exp     exp(-CER). Always in (0, 1]. The first errors cost more than later ones.
    soft    Thresholded. CER at or below tau scores 1, so recognizer noise and a
            slipped character on a long line are not punished. Above tau the score
            falls linearly to 0 at CER = 1, and stays 0 after that.
    """
    if kind == "linear":
        return 1.0 - value
    if kind == "exp":
        return math.exp(-value)
    if kind == "soft":
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must be between 0 and 1")
        if value <= tau:
            return 1.0
        if value >= 1.0:
            return 0.0
        return (1.0 - value) / (1.0 - tau)
    raise ValueError(f"unknown reward {kind!r} (expected linear, exp, soft)")


def cer_and_reward(reference, hypothesis, kind="exp", tau=0.1, cer_cap=CER_CAP):
    """Return (raw CER, reward). The reward sees CER capped at cer_cap."""
    raw = cer(reference, hypothesis)
    return raw, reward_from_cer(min(raw, cer_cap), kind=kind, tau=tau)


def apply_temperature(pi, sig_x, sig_y, temperature):
    """Graves probability bias (sec. 5.4). temperature = exp(-b).

    Mixture weights become pi^(1+b) / Z and each std is multiplied by temperature.
    This is the density generate.sample_iter actually draws from, so the REINFORCE
    score has to use it too. temperature <= 0 is a point mass and has no density.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0; 0 is a point mass and has no density")
    if abs(temperature - 1.0) < 1e-8:
        return pi, sig_x, sig_y
    b = -math.log(temperature)
    log_pi = torch.log(pi.clamp_min(1e-12)) * (1.0 + b)
    return torch.softmax(log_pi, dim=-1), sig_x * temperature, sig_y * temperature


def _log_gaussian_2d(dx, dy, mu_x, mu_y, sig_x, sig_y, rho):
    """Log of model.gaussian_2d. Log-domain so a tight component does not overflow."""
    one_m_rho2 = (1.0 - rho ** 2).clamp(min=1e-6)
    z1 = (dx - mu_x) ** 2 / sig_x ** 2
    z2 = (dy - mu_y) ** 2 / sig_y ** 2
    z3 = 2.0 * rho * (dx - mu_x) * (dy - mu_y) / (sig_x * sig_y)
    log_norm = (
        -math.log(2.0 * math.pi)
        - torch.log(sig_x)
        - torch.log(sig_y)
        - 0.5 * torch.log(one_m_rho2)
    )
    return log_norm - (z1 + z2 - z3) / (2.0 * one_m_rho2)


def step_logprob(out, y, temperature=1.0):
    """Per-step log pi(y_t | model output). out (B, T, 121), y (B, T, 3) -> (B, T).

    The action is the pen offset and the pen-up bit. The mixture component is a
    latent, so this is the marginal density, which is what was sampled.
    """
    pi, mu_x, mu_y, sig_x, sig_y, rho, pen = split_params(out)
    pi, sig_x, sig_y = apply_temperature(pi, sig_x, sig_y, temperature)
    dx = y[:, :, 0].unsqueeze(-1)
    dy = y[:, :, 1].unsqueeze(-1)
    log_xy = torch.logsumexp(torch.log(pi.clamp_min(1e-12)) + _log_gaussian_2d(
        dx, dy, mu_x, mu_y, sig_x, sig_y, rho), dim=-1)
    pen_up = y[:, :, 2]
    log_pen = (
        pen_up * torch.log(pen.clamp_min(1e-8))
        + (1.0 - pen_up) * torch.log((1.0 - pen).clamp_min(1e-8))
    )
    return log_xy + log_pen


def trajectory_logprob(model, seq, text, stoi, device, temperature, sharpen):
    """Teacher-forced sum of log pi(seq | text), and the number of real steps.

    Sampling feeds each point back as the next input. Replaying those points
    through forward() is the same autoregressive density, and it is the graph
    REINFORCE differentiates. `sharpen` has to match the value used while
    sampling: it scales beta inside compute_window, so it is part of the policy.
    """
    missing = [ch for ch in text if ch not in stoi]
    if missing:
        raise ValueError(f"text has characters outside the vocab: {sorted(set(missing))}")
    seq = seq.detach().float().cpu()
    if seq.ndim != 2 or seq.shape[-1] != 3 or seq.shape[0] == 0:
        raise ValueError("seq must be (T, 3) with T > 0")
    start = torch.zeros(1, 3)
    inp = torch.cat([start, seq[:-1]], dim=0)
    idx = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    x, y, mask, c, c_mask = collate([(inp, seq, idx)], vocab_size=len(stoi))
    x, y, mask, c, c_mask = [t.to(device) for t in (x, y, mask, c, c_mask)]
    model.window_sharpen = sharpen
    out, _, _, _ = model(x, c, c_mask)
    total = (step_logprob(out, y, temperature=temperature) * mask).sum()
    return total, mask.sum().detach()


def reinforce_loss(logp_sum, logp_ref_sum, length, advantage, kl_coef):
    """Scalar loss to minimize for one trajectory.

    advantage is treated as a constant (the reward already happened).
    kl_coef * per-step KL is subtracted from it, with the KL stopped so the
    backward is only the score function. The direct gradient of the KL has
    expectation zero under the policy, so leaving it out does not bias the update.
    """
    if not torch.is_tensor(length):
        length = torch.tensor(float(length), dtype=logp_sum.dtype, device=logp_sum.device)
    length = length.clamp(min=1)
    mean_logp = logp_sum / length
    step_kl = (logp_sum.detach() - logp_ref_sum.detach()) / length
    shaped = advantage - kl_coef * step_kl
    if torch.is_tensor(shaped):
        shaped = shaped.detach()
    return -(shaped * mean_logp)


def group_advantages(rewards, k):
    """Subtract each prompt's mean reward. rewards are grouped in blocks of k."""
    if k < 2:
        raise ValueError("a within-group baseline needs k >= 2")
    if len(rewards) % k != 0:
        raise ValueError("number of rewards must be a multiple of k")
    out = []
    for i in range(0, len(rewards), k):
        group = rewards[i:i + k]
        base = sum(group) / k
        out.extend(r - base for r in group)
    return out


def advantages(rewards, k, ema):
    """Return (advantages, new_ema).

    k > 1 compares the samples of one prompt with each other, so a hard word
    is not punished just for being hard. k == 1 falls back to an EMA of the
    reward. The first batch centers on its own mean.
    """
    if k > 1:
        return group_advantages(rewards, k), ema
    mean = sum(rewards) / len(rewards)
    baseline = mean if ema is None else ema
    new_ema = mean if ema is None else (0.9 * ema + 0.1 * mean)
    return [r - baseline for r in rewards], new_ema


def reinforce_step(model, ref, rollouts, advs, stoi, device, temperature,
                   sharpen, kl_coef, optimizer, clip):
    """On-policy update over a list of {text, seq} rollouts. One backward at a time.

    A full line is hundreds of sequential LSTM steps, so the trajectories are
    differentiated one by one and the gradients are averaged.
    """
    optimizer.zero_grad(set_to_none=True)
    n = len(rollouts)
    used = 0
    loss_acc = 0.0
    kl_acc = 0.0
    traces = []
    model.train()
    for item, adv in zip(rollouts, advs):
        adv_f = float(adv)
        trace = {
            "advantage": adv_f,
            "logp": None,
            "kl": None,
            "length": None,
            "skipped": None,
        }
        if not math.isfinite(adv_f):
            trace["skipped"] = "nonfinite_advantage"
            traces.append(trace)
            continue
        logp, length = trajectory_logprob(
            model, item["seq"], item["text"], stoi, device, temperature, sharpen)
        if kl_coef:
            with torch.no_grad():
                logp_ref, _ = trajectory_logprob(
                    ref, item["seq"], item["text"], stoi, device, temperature, sharpen)
        else:
            logp_ref = torch.zeros((), device=logp.device)
        if not torch.isfinite(logp) or not torch.isfinite(logp_ref):
            print(f"  non-finite logprob for {item['text']!r}, skipping", flush=True)
            trace["skipped"] = "nonfinite_logprob"
            traces.append(trace)
            continue
        loss = reinforce_loss(logp, logp_ref, length, adv, kl_coef)
        (loss / n).backward()
        used += 1
        loss_acc += float(loss.detach())
        trace["logp"] = float(logp.detach())
        trace["length"] = int(length.detach().item())
        if kl_coef:
            step_kl = float((logp.detach() - logp_ref.detach()) / length.clamp(min=1))
            kl_acc += step_kl
            trace["kl"] = step_kl
        else:
            trace["kl"] = 0.0
        traces.append(trace)
    grad_norm = 0.0
    stepped = False
    if used:
        trainable = [p for p in model.parameters() if p.requires_grad]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, clip))
        if math.isfinite(grad_norm):
            optimizer.step()
            stepped = True
        else:
            print("  non-finite gradient, skipping the update", flush=True)
            optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss_acc / used if used else float("nan"),
        "kl": kl_acc / used if used else float("nan"),
        "used": used,
        "grad": grad_norm,
        "stepped": stepped,
        "traces": traces,
    }


def load_policy_and_ref(ckpt, ref, device):
    """Load the policy and a frozen KL anchor. The anchor is not trained."""
    for label, path in (("policy", ckpt), ("KL anchor", ref)):
        if not Path(path).exists():
            raise SystemExit(f"missing {label} checkpoint: {path}")
    model, stoi = load_model(device, ckpt=ckpt)
    ref_model, _ = load_model(device, ckpt=ref)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()
    return model, ref_model, stoi


def rollout(model, text, stoi, device, temperature, sharpen):
    """One trajectory from the same sampler generate.py uses. Returns (T, 3) on CPU."""
    training = model.training
    model.eval()
    seq, _phis = sample(
        model, text, stoi, device,
        temperature=temperature, window_sharpen=sharpen,
    )
    if training:
        model.train()
    seq = seq.detach().float().cpu()
    if seq.ndim != 2 or seq.shape[0] == 0:
        raise RuntimeError(f"model produced an empty trajectory for {text!r}")
    return seq


def load_prompts(path, stoi, min_chars, max_chars):
    vocab = set(stoi)
    prompts = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not (min_chars <= len(text) <= max_chars):
            continue
        if any(ch not in vocab for ch in text):
            continue
        prompts.append(text)
    if not prompts:
        raise SystemExit(
            f"no prompts in {path} with length in [{min_chars}, {max_chars}] "
            "and characters inside the model vocab"
        )
    return prompts


def resolve_prompts(text, path, stoi, min_chars, max_chars):
    if text:
        missing = sorted({ch for ch in text if ch not in stoi})
        if missing:
            raise SystemExit(f"--text has characters outside the vocab: {missing}")
        return [text]
    return load_prompts(path, stoi, min_chars, max_chars)


def pick_prompts(prompts, n):
    if len(prompts) >= n:
        return random.sample(prompts, n)
    return [random.choice(prompts) for _ in range(n)]


def _split_strokes(xy, pen):
    """Same cut as generate.save_plot: the pen-up point belongs to the stroke that ends."""
    strokes = []
    current = []
    for i, (x, y) in enumerate(xy):
        current.append((float(x), float(y)))
        if pen[i] >= 0.5 or i == len(xy) - 1:
            strokes.append(current)
            current = []
    return [s for s in strokes if s]


def _blank(height):
    from PIL import Image
    return Image.new("RGB", (height * 4, height), "white")


def strokes_to_image(seq, std, height=384, stroke_width=5, pad=28, supersample=2):
    """Render a normalised (T, 3) trajectory to a white line image.

    Data y grows up; image y grows down. Height matches TrOCR's input so the
    recognizer's resize does not have to invent resolution. A single-point
    stroke is drawn as a dot, otherwise i and j lose their marks.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise SystemExit(
            "rendering handwriting needs pillow:\n  uv sync --extra rl\n"
            f"({exc})"
        ) from exc

    seq = np.asarray(seq, dtype=np.float64)
    if seq.ndim != 2 or seq.shape[1] < 3 or len(seq) == 0 or not np.isfinite(seq).all():
        return _blank(height)

    xy = np.cumsum(seq[:, :2] * float(std), axis=0)
    strokes = _split_strokes(xy, seq[:, 2])
    if not strokes:
        return _blank(height)

    xs = [p[0] for s in strokes for p in s]
    ys = [p[1] for s in strokes for p in s]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    span_x = max(maxx - minx, 1e-3)
    span_y = max(maxy - miny, 1e-3)

    ss = max(int(supersample), 1)
    height_s = height * ss
    pad_s = pad * ss
    width_px = max(stroke_width * ss, 1)
    inner_h = max(height_s - 2 * pad_s, 1)
    scale = inner_h / span_y
    width = int(math.ceil(span_x * scale)) + 2 * pad_s
    max_w = 2048 * ss
    if width > max_w:
        scale *= (max_w - 2 * pad_s) / max(span_x * scale, 1e-3)
        width = max_w
    width = max(width, height_s)

    img = Image.new("RGB", (width, height_s), "white")
    draw = ImageDraw.Draw(img)
    used_h = span_y * scale
    y_origin = pad_s + (inner_h - used_h) / 2.0

    def tx(x):
        return pad_s + (x - minx) * scale

    def ty(y):
        return y_origin + (maxy - y) * scale

    for stroke in strokes:
        pts = [(tx(x), ty(y)) for x, y in stroke]
        if len(pts) == 1:
            x, y = pts[0]
            r = max(width_px / 2.0, 1.0)
            draw.ellipse((x - r, y - r, x + r, y + r), fill="black")
        else:
            draw.line(pts, fill="black", width=width_px, joint="curve")

    if ss > 1:
        resample = getattr(Image, "Resampling", Image).LANCZOS
        img = img.resize((max(1, width // ss), height), resample)
    return img


def consistency_sheet(images, text, captions, step=None, summary=""):
    """Stack every sample of one sentence, with a caption above each line.

    Used by the recognizer loop so a step's draws can be compared side by side
    as the run goes. Captions are the recognizer read and the CER.
    """
    from PIL import Image, ImageDraw, ImageFont

    if len(images) != len(captions):
        raise ValueError("captions must line up with images")

    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()

    def fit(img, max_w=980, max_h=150):
        scale = min(max_w / img.width, max_h / img.height, 1.0)
        if scale >= 1.0:
            return img
        resample = getattr(Image, "Resampling", Image).LANCZOS
        return img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))), resample)

    fitted = [fit(img) for img in images]
    margin, gap, cap_h, header_h = 16, 12, 26, 48
    row_w = max(img.width for img in fitted)
    width = margin * 2 + row_w
    height = margin * 2 + header_h + sum(cap_h + img.height + gap for img in fitted)
    sheet = Image.new("RGB", (max(width, 480), height), "white")
    draw = ImageDraw.Draw(sheet)
    title = text if step is None else f"step {step}   {text}"
    if summary:
        title = f"{title}   {summary}"
    draw.text((margin, margin), title, fill="black", font=font)
    y = margin + header_h
    for caption, img in zip(captions, fitted):
        draw.text((margin, y), caption, fill=(20, 20, 20), font=font)
        y += cap_h
        sheet.paste(img, (margin, y))
        draw.rectangle(
            (margin, y, margin + img.width - 1, y + img.height - 1),
            outline=(210, 210, 210))
        y += img.height + gap
    return sheet


def choice_sheet(images, text):
    """One image with the prompt and the numbered samples, left to right."""
    from PIL import Image, ImageDraw, ImageFont

    def fit(img, max_w=640, max_h=220):
        scale = min(max_w / img.width, max_h / img.height, 1.0)
        if scale >= 1.0:
            return img
        resample = getattr(Image, "Resampling", Image).LANCZOS
        return img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), resample)

    fitted = [fit(img) for img in images]
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()

    margin, gap, header, label_h = 16, 16, 48, 36
    body_h = max(img.height for img in fitted)
    width = margin * 2 + sum(img.width for img in fitted) + gap * (len(fitted) - 1)
    height = margin * 2 + header + body_h + label_h
    sheet = Image.new("RGB", (max(width, 320), height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, margin), text, fill="black", font=font)
    x = margin
    top = margin + header
    for i, img in enumerate(fitted, start=1):
        y = top + (body_h - img.height) // 2
        sheet.paste(img, (x, y))
        draw.rectangle((x, y, x + img.width - 1, y + img.height - 1), outline=(210, 210, 210))
        draw.text((x + img.width / 2.0 - 6, top + body_h + 6), str(i), fill="black", font=font)
        x += img.width + gap
    return sheet


def reveal(path):
    """Open an image on macOS or Linux, and always print the path."""
    import subprocess
    import sys
    print(f"saved {path}", flush=True)
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", str(path)], check=False)


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class TrocrScorer:
    """TrOCR-Large handwritten as a CER reward.

    The first construction downloads the weights (about 1.5 GB) into the
    Hugging Face cache.
    """

    def __init__(self, device, model_name="microsoft/trocr-large-handwritten",
                 kind="exp", tau=0.1, beams=1, vcer_alpha=0.8, vcer_k=2, vcer_table=None):
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise SystemExit(
                "Recognizer RL needs the rl extra:\n"
                "  uv sync --extra rl\n"
                f"({exc})"
            ) from exc
        print(f"loading {model_name} on {device}", flush=True)
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.kind = kind
        self.tau = tau
        self.beams = beams
        self._vcer = None
        self.vcer_config = None
        # Phase 6 integration point: vCER is a second reward on the decoded string.
        if kind == "vcer":
            from vcer.reward import VcerReward
            self._vcer = VcerReward.load(vcer_table, k=vcer_k, alpha=vcer_alpha)
            self.vcer_config = self._vcer.config

    @torch.no_grad()
    def read(self, images):
        encoded = self.processor(images=images, return_tensors="pt", padding=True)
        pixel_values = encoded.pixel_values.to(self.device)
        ids = self.model.generate(
            pixel_values, num_beams=self.beams, max_new_tokens=64)
        return self.processor.batch_decode(ids, skip_special_tokens=True)

    def score(self, images, references):
        hyps = self.read(images)
        vcer = getattr(self, "_vcer", None)
        if vcer is not None:
            rewards, details = vcer.score_batch(references, hyps)
            self.last_details = details
            return rewards, [item["cer"] for item in details], hyps
        self.last_details = None
        cers, rewards = [], []
        for ref, hyp in zip(references, hyps):
            raw, reward = cer_and_reward(ref, hyp, kind=self.kind, tau=self.tau)
            cers.append(raw)
            rewards.append(reward)
        return rewards, cers, hyps


def add_policy_args(parser, default_save):
    parser.add_argument("--ckpt", default=None,
                        help="Policy checkpoint to start from. Defaults depend on the script. "
                             f"The base model is {BASE_CKPT}.")
    parser.add_argument("--ref", default=BASE_CKPT,
                        help=f"Frozen KL anchor. Default: {BASE_CKPT}.")
    parser.add_argument("--save", default=default_save, help="Where to write the trained policy.")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Sampling temperature. The log-prob uses the same value.")
    parser.add_argument("--sharpen", type=float, default=2.0,
                        help="Window beta scale for sampling AND the log-prob. "
                             "2 matches generate.py. The numpy server uses 1.")
    parser.add_argument("--kl", type=float, default=0.1,
                        help="How hard the per-step KL pulls back toward --ref.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient norm clip.")
    parser.add_argument("--prompts", default="data/romeo_juliet.txt",
                        help="Line pool. Sampled at random. Default is the whole of "
                             "Romeo and Juliet, cut into 8–32 character lines.")
    parser.add_argument("--text", default=None,
                        help="Use this prompt every time instead of sampling --prompts.")
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser


def short(text, n=60):
    text = str(text).replace("\n", " ")
    return text if len(text) <= n else text[:n - 1] + "…"
