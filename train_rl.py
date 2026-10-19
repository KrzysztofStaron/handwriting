"""Post-train the handwriting policy with a TrOCR-Large reward.

Training itself is unchanged in shape: several sentences, two draws each,
REINFORCE on the CER reward. The 10 samples are only a review sheet.

About every 10 minutes the loop writes a checkpoint and draws one fixed short
sentence 10 times, so the sheets are comparable. Re-running with no arguments
loads models/rl-trocr.pth when that file exists. Pass --ckpt to continue from
any numbered checkpoint. models/best-4.pth is the KL anchor and is never written.

Each update is also appended to samples/rl/events.jsonl. The local dashboard
reads that file while training runs:

    uv run python rl_dashboard.py

    uv sync --extra rl
    uv run train_rl.py
    uv run train_rl.py --ckpt samples/rl/ckpt-0003.pth
    uv run train_rl.py --ckpt models/best-4.pth   # start over from the base model
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch

from generate import load_std
from rl import (
    BASE_CKPT,
    TrocrScorer,
    add_policy_args,
    advantages,
    consistency_sheet,
    draw_temperatures,
    line_straightness,
    load_policy_and_ref,
    pick_prompts,
    refuse_overwrite_base,
    reinforce_step,
    resolve_prompts,
    reveal,
    save_policy,
    set_seed,
    short,
    strokes_to_images,
    rollout_batch,
)
from rl_events import EVENTS_NAME, EventLog, pack_draws

THIN_GLYPHS = set("il1.'")
LATEST_NAME = "rl-trocr.pth"
BEST_NAME = "rl-trocr-best.pth"
DEFAULT_REVIEW = "the old man"
_CKPT_NAME = re.compile(r"^ckpt-(\d+)\.pth$")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="REINFORCE the handwriting model against a TrOCR-Large CER reward. "
                    "A checkpoint and a 10-sample review sheet are written about every "
                    "10 minutes. Re-running resumes from the latest checkpoint. "
                    f"The base model is {BASE_CKPT} and is not modified.",
    )
    add_policy_args(parser, default_save="models/rl-trocr.pth")
    parser.add_argument("--batch", type=int, default=4, help="Prompts per update.")
    parser.add_argument("-k", type=int, default=8,
                        help="Draws per prompt. The baseline is that prompt's mean reward.")
    # Phase 6 integration point. vCER is the training default; exp/linear/soft stay available.
    parser.add_argument("--reward", choices=("vcer", "exp", "linear", "soft"), default="vcer",
                        help="vcer: visually weighted CER (default). exp: exp(-CER). "
                             "linear: 1-CER. soft: 1 at or below --tau, then linear to 0 at CER=1.")
    parser.add_argument("--vcer-alpha", type=float, default=0.8,
                        help="Weight on (1 - wCER). The rest is an exact-match bonus.")
    parser.add_argument("--vcer-k", type=int, default=2,
                        help="Longest substitution span in the visual distance. 1, 2, or 3.")
    parser.add_argument("--vcer-table", default=None,
                        help="Cost table JSON. Default: vcer/table.json.")
    parser.add_argument("--confidence", type=float, default=0.2,
                        help="Share of the reward taken from TrOCR's token confidence. "
                             "High confidence raises the reward when CER is low and lowers "
                             "it when CER is high. 0 leaves the string score unchanged. "
                             "Requires --beams 1.")
    parser.add_argument("--straightness", type=float, default=0.05,
                        help="Most a crooked line can lose. Two parallel lines sit on the "
                             "top and bottom of the ink. A flat pair (slope 0, no arch) "
                             "loses nothing. 0 turns the penalty off.")
    parser.add_argument("--tau", type=float, default=0.1,
                        help="CER forgiven outright by --reward soft.")
    parser.add_argument("--reward-model", default="microsoft/trocr-large-handwritten")
    parser.add_argument("--reward-device", default=None,
                        help="Device for TrOCR. Default: cuda with a cuda policy, "
                             "mps when Apple Silicon is available, otherwise cpu.")
    parser.add_argument("--beams", type=int, default=1, help="TrOCR beam width. 1 is greedy.")
    parser.add_argument("--block-minutes", type=float, default=10,
                        help="Train for about this long, then write a checkpoint and a review sheet.")
    parser.add_argument("--checkpoints", type=int, default=0,
                        help="Stop after this many new checkpoints. 0 runs until interrupted.")
    parser.add_argument("--review-text", default=None,
                        help=f"Sentence drawn on every review sheet. Default: the sentence "
                             f"already stored for this run, or {DEFAULT_REVIEW!r}.")
    parser.add_argument("--review-k", type=int, default=10,
                        help="Draws of --review-text on each checkpoint sheet.")
    parser.add_argument("--temp-min", type=float, default=0.1,
                        help="Lowest temperature. Draws come from a normal centered on the "
                             "midpoint of this and --temp-max; values outside are dropped. "
                             "0 draws the mode and stays out of the gradient.")
    parser.add_argument("--temp-max", type=float, default=0.9,
                        help="Highest temperature. The default range keeps most drawings near 0.5. "
                             "The log-prob uses the value that drawing was sampled at.")
    parser.add_argument("--sample-every", type=int, default=5,
                        help="Save a sheet of the training prompts every this many updates. "
                             "0 saves none. The fixed review sentence is still drawn at each checkpoint.")
    parser.add_argument("--sheet-dir", default="samples/rl")
    parser.add_argument("--no-open", action="store_true",
                        help="Save each review sheet without opening it.")
    return parser.parse_args(argv)


def choose_devices(requested_policy, requested_reward, has_cuda, has_mps):
    """Pick where the sampler and where TrOCR run.

    The sampler draws one pen step at a time. On Apple Silicon that launch
    overhead makes MPS slower than the CPU (a short line was 0.2s on CPU and
    3.4s on MPS). TrOCR-Large is the model that belongs on the GPU, so an
    automatic run keeps the policy on the CPU and the recognizer on MPS.
    Pass --device mps to put the policy on the GPU anyway.
    """
    if requested_policy:
        policy = requested_policy
    elif has_cuda:
        policy = "cuda"
    else:
        policy = "cpu"
    if requested_reward:
        reward = requested_reward
    elif policy == "cuda":
        reward = "cuda"
    elif has_mps or policy == "mps":
        reward = "mps"
    else:
        reward = "cpu"
    return policy, reward


def checkpoint_indices(sheet_dir):
    nums = []
    for path in Path(sheet_dir).glob("ckpt-*.pth"):
        match = _CKPT_NAME.match(path.name)
        if match:
            nums.append(int(match.group(1)))
    return sorted(nums)


def next_checkpoint_index(sheet_dir):
    nums = checkpoint_indices(sheet_dir)
    return (nums[-1] + 1) if nums else 1


def default_policy_ckpt(save_path, base=BASE_CKPT):
    """Re-runs load the latest weights. An explicit --ckpt overrides this."""
    latest = Path(save_path)
    if latest.is_file():
        return str(latest)
    return base


def sidecar_for(weights_path, sheet_dir):
    """Optimizer sidecar for a weights file, or None when there isn't one."""
    path = Path(weights_path)
    if path.name == LATEST_NAME:
        return Path(sheet_dir) / "latest.optim.pth"
    if _CKPT_NAME.match(path.name):
        return path.with_name(path.stem + ".optim.pth")
    return None


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_review_text(explicit, run_path, stoi):
    stored = read_json(run_path).get("review_text")
    text = explicit or stored or DEFAULT_REVIEW
    missing = sorted({ch for ch in text if ch not in stoi})
    if missing:
        raise SystemExit(f"review sentence has characters outside the vocab: {missing}")
    return text


def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_sidecar(path):
    if path is None or not Path(path).is_file():
        return None
    return torch.load(path, map_location="cpu")


def save_sidecar(path, optimizer, updates, best_reward, review_text, ema, index):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "updates": updates,
        "best_reward": best_reward,
        "review_text": review_text,
        "ema": ema,
        "checkpoint_index": index,
    }, path)


def _wcer_list(scorer, n):
    details = getattr(scorer, "last_details", None) or []
    if len(details) != n:
        return [None] * n
    return [item["wcer"] for item in details]


def _text_stats(hyps):
    chars = "".join(hyps)
    thin = (sum(ch in THIN_GLYPHS for ch in chars) / len(chars)) if chars else 0.0
    length = sum(len(hyp) for hyp in hyps) / len(hyps) if hyps else 0.0
    return thin, length


def label_reward_order(refs, rewards):
    """Labels stay in first-seen order. Draws of one label go by reward, high first."""
    first = {}
    for index, ref in enumerate(refs):
        first.setdefault(ref, index)

    def key(index):
        reward = rewards[index]
        rank = -(reward if reward is not None else float("-inf"))
        return (first[refs[index]], rank, index)

    return sorted(range(len(refs)), key=key)


def save_sample_images(images, directory, update_no, max_h=160):
    """One preview PNG per drawing, named gen-NNNNNN-II.png, in sample order."""
    from PIL import Image

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    resample = getattr(Image, "Resampling", Image).LANCZOS
    names = []
    for index, image in enumerate(images):
        preview = image
        if image.height > max_h:
            scale = max_h / image.height
            preview = image.resize(
                (max(1, int(image.width * scale)), max_h), resample)
        name = f"gen-{update_no:06d}-{index:02d}.png"
        preview.save(directory / name)
        names.append(name)
    return names


def save_training_sheet(images, refs, hyps, cers, rewards, path, update_no):
    """One contact sheet of the prompts this update actually trained on."""
    order = label_reward_order(refs, rewards)
    images = [images[i] for i in order]
    refs = [refs[i] for i in order]
    hyps = [hyps[i] for i in order]
    cers = [cers[i] for i in order]
    rewards = [rewards[i] for i in order]
    captions = [
        f"{short(ref, 40)}   reward {reward:.3f}   cer {cer:.2f}   {short(hyp, 32)}"
        for ref, hyp, cer, reward in zip(refs, hyps, cers, rewards)
    ]
    sheet = consistency_sheet(images, f"update {update_no}   training samples", captions)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path.name


def one_update(model, ref, scorer, optimizer, prompts, stoi, std, device, args, ema,
               log=None, update_no=None, block=None, sheet_path=None):
    """One on-policy update: `batch` sentences, `k` draws each."""
    chosen = pick_prompts(prompts, args.batch)
    texts = []
    for text in chosen:
        print(f"    {text!r}", flush=True)
        texts.extend([text] * args.k)
    temperatures = draw_temperatures(len(texts), args.temp_min, args.temp_max)
    print(f"    drawing {len(texts)} lines in one batch", flush=True)
    sequences = rollout_batch(
        model, texts, stoi, device, temperatures, args.sharpen)
    rollouts = []
    for index, (text, seq, temp) in enumerate(zip(texts, sequences, temperatures)):
        draw = index % args.k + 1
        rollouts.append({"text": text, "seq": seq, "temperature": temp})
        print(f"      draw {draw}/{args.k}  temp {temp:.2f}  ({seq.shape[0]} steps)", flush=True)
        if log is not None:
            log.write({
                "type": "draw",
                "update": update_no,
                "block": block,
                "text": text,
                "draw": draw,
                "k": args.k,
                "temperature": temp,
                "steps": int(seq.shape[0]),
            })
    images = strokes_to_images([item["seq"] for item in rollouts], std)
    refs = [item["text"] for item in rollouts]
    rewards, cers, hyps = scorer.score(images, refs)
    bends = [line_straightness(item["seq"], std) for item in rollouts]
    if args.straightness > 0:
        rewards = [
            float(reward) - args.straightness * bend
            for reward, bend in zip(rewards, bends)
        ]
    wcors = _wcer_list(scorer, len(hyps))
    confs = getattr(scorer, "last_confidence", None)
    advs, ema = advantages(rewards, args.k, ema)
    stats = reinforce_step(
        model, ref, rollouts, advs, stoi, device,
        args.temperature, args.sharpen, args.kl, optimizer, args.clip,
    )
    image_names = []
    if sheet_path is not None:
        image_names = save_sample_images(images, sheet_path.parent, update_no)
    draws = pack_draws(
        refs,
        [int(item["seq"].shape[0]) for item in rollouts],
        rewards, cers, hyps, advs, stats.get("traces"), wcors, confs,
        [item["temperature"] for item in rollouts],
        image_names,
        bends,
    )
    sheet_name = image_names[0] if image_names else None
    return stats, rewards, cers, hyps, refs, ema, draws, wcors, sheet_name


def write_review(model, scorer, text, stoi, std, device, args, index, out_dir):
    """Draw the fixed sentence `review_k` times from the weights just saved."""
    print(f"  review {text!r} x {args.review_k}", flush=True)
    temperatures = draw_temperatures(args.review_k, args.temp_min, args.temp_max)
    sequences = rollout_batch(
        model, [text] * args.review_k, stoi, device, temperatures, args.sharpen)
    for draw, (seq, temp) in enumerate(zip(sequences, temperatures), start=1):
        print(f"    draw {draw}/{args.review_k}  temp {temp:.2f}  ({seq.shape[0]} steps)", flush=True)
    images = strokes_to_images(sequences, std)
    rewards, cers, hyps = scorer.score(images, [text] * len(images))
    bends = [line_straightness(seq, std) for seq in sequences]
    if args.straightness > 0:
        rewards = [
            float(reward) - args.straightness * bend
            for reward, bend in zip(rewards, bends)
        ]
    wcors = _wcer_list(scorer, len(hyps))
    confs = list(getattr(scorer, "last_confidence", None) or [])
    for i, (cer, hyp, reward, wcer) in enumerate(zip(cers, hyps, rewards, wcors), start=1):
        extra = "" if wcer is None else f"  wcer {wcer:.3f}"
        conf = confs[i - 1] if i - 1 < len(confs) else None
        conf_bit = "" if conf is None else f"  conf {conf:.2f}"
        print(f"    {i:2d}  cer {cer:.2f}{extra}{conf_bit}  reward {reward:.3f}  -> {short(hyp)!r}",
              flush=True)
    mean_c = sum(cers) / len(cers)
    mean_r = sum(rewards) / len(rewards)
    present = [value for value in wcors if value is not None]
    mean_w = sum(present) / len(present) if present else None
    captions = [
        f"{i}   cer {cer:.2f}   read: {short(hyp, 70)}"
        for i, (cer, hyp) in enumerate(zip(cers, hyps), start=1)
    ]
    summary = (f"mean cer {mean_c:.2f}   min {min(cers):.2f}   max {max(cers):.2f}"
               "   saved weights")
    sheet = consistency_sheet(images, text, captions, step=index, summary=summary)
    path = out_dir / f"ckpt-{index:04d}.png"
    latest = out_dir / "latest.png"
    sheet.save(path)
    sheet.save(latest)
    if args.no_open:
        print(f"  saved {path}", flush=True)
    else:
        reveal(path)
    return {
        "reward_mean": mean_r,
        "cer_mean": mean_c,
        "cer_min": min(cers),
        "cer_max": max(cers),
        "wcer_mean": mean_w,
        "sheet": path.name,
        "draws": [
            {
                "text": text,
                "steps": int(seq.shape[0]),
                "temperature": temperatures[i],
                "cer": cer,
                "wcer": wcer,
                "confidence": confs[i] if i < len(confs) else None,
                "straightness": bends[i],
                "reward": reward,
                "hyp": hyp,
            }
            for i, (seq, cer, wcer, reward, hyp) in enumerate(
                zip(sequences, cers, wcors, rewards, hyps))
        ],
    }


def main():
    args = parse_args()
    if args.temp_min < 0 or args.temp_max < 0:
        raise SystemExit("--temp-min and --temp-max must be >= 0")
    if args.temp_max < args.temp_min:
        raise SystemExit("--temp-max must be >= --temp-min")
    if args.temp_max <= 0:
        raise SystemExit("--temp-max must be > 0 so some drawings have a density")
    if args.straightness < 0:
        raise SystemExit("--straightness must be >= 0")
    if args.k < 1 or args.batch < 1:
        raise SystemExit("--k and --batch must be >= 1")
    if args.review_k < 1:
        raise SystemExit("--review-k must be >= 1")
    if args.block_minutes <= 0:
        raise SystemExit("--block-minutes must be > 0")
    if args.reward == "soft" and not 0.0 < args.tau < 1.0:
        raise SystemExit("--tau must be between 0 and 1")
    if args.reward == "vcer" and not 0.0 <= args.vcer_alpha <= 1.0:
        raise SystemExit("--vcer-alpha must be between 0 and 1")
    if args.reward == "vcer" and args.vcer_k not in (1, 2, 3):
        raise SystemExit("--vcer-k must be 1, 2, or 3")
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1")
    if args.confidence > 0 and args.beams != 1:
        raise SystemExit("--confidence needs greedy decoding (--beams 1)")
    if args.sample_every < 0:
        raise SystemExit("--sample-every must be >= 0")
    if args.min_chars > args.max_chars:
        raise SystemExit("--min-chars cannot exceed --max-chars")

    out_dir = Path(args.sheet_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / "run.json"
    best_path = Path(args.save).with_name(BEST_NAME)
    refuse_overwrite_base(args.save)
    refuse_overwrite_base(best_path)

    ckpt = args.ckpt or default_policy_ckpt(args.save)
    device, rdev = choose_devices(
        args.device,
        args.reward_device,
        torch.cuda.is_available(),
        torch.backends.mps.is_available(),
    )
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    if device == "mps" or rdev == "mps":
        # A few ops inside TrOCR generate have no MPS kernel. Those fall
        # back to CPU; the models themselves stay on the GPU.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    set_seed(args.seed)

    resuming = Path(ckpt).is_file() and Path(ckpt).resolve() != Path(BASE_CKPT).resolve()
    print(f"device: {device}   reward: {rdev}   policy: {ckpt}"
          + ("   (resuming)" if resuming else "   (base model)"), flush=True)
    if device == "cpu" and rdev == "mps":
        print("sampler on cpu (one pen step at a time is faster there); TrOCR on mps",
              flush=True)
    print(f"KL anchor: {args.ref}", flush=True)
    print(f"updates: {args.batch} sentences x {args.k} draws, sampled in one batch   "
          f"then a checkpoint about every {args.block_minutes:g} min", flush=True)
    print(f"temperature: normal around {(args.temp_min + args.temp_max) / 2:g}, "
          f"kept inside {args.temp_min:g} to {args.temp_max:g}",
          flush=True)
    print(f"latest -> {args.save}", flush=True)
    print(f"numbered checkpoints and review sheets -> {out_dir}/ckpt-NNNN.pth and .png",
          flush=True)
    events_path = out_dir / EVENTS_NAME
    print(f"events -> {events_path}", flush=True)
    print("dashboard -> uv run python rl_dashboard.py", flush=True)

    log = None
    session_open = False
    reason = "done"
    updates = 0
    try:
        model, ref, stoi = load_policy_and_ref(ckpt, args.ref, device)
        review_text = resolve_review_text(args.review_text, run_path, stoi)
        prompts = resolve_prompts(args.text, args.prompts, stoi, args.min_chars, args.max_chars)
        std = load_std()
        print(f"prompts: {len(prompts)}   review sentence: {review_text!r} x {args.review_k}",
              flush=True)

        scorer = TrocrScorer(
            rdev,
            model_name=args.reward_model, kind=args.reward, tau=args.tau, beams=args.beams,
            vcer_alpha=args.vcer_alpha, vcer_k=args.vcer_k, vcer_table=args.vcer_table,
            confidence_weight=args.confidence,
        )
        if scorer.vcer_config:
            cfg = scorer.vcer_config
            print(f"vCER alpha {cfg['alpha']}  k {cfg['k']}  table {cfg['hash']}  "
                  f"confidence {args.confidence:g}  straightness {args.straightness:g}  "
                  f"prompts {args.prompts}", flush=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        sidecar = load_sidecar(sidecar_for(ckpt, out_dir))
        ema = None
        stored = read_json(run_path)
        best_reward = stored.get("best_reward")
        if best_reward is None:
            best_reward = float("-inf")
        if sidecar:
            optimizer.load_state_dict(sidecar["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = args.lr
            move_optimizer_state(optimizer, device)
            ema = sidecar.get("ema")
            updates = int(sidecar.get("updates") or 0)
            if sidecar.get("best_reward") is not None and not args.review_text:
                # The sidecar belongs to this weights file. The run file holds the
                # best score across the whole series, which is what we already loaded.
                pass
            print(f"restored optimizer at update {updates}", flush=True)
        elif resuming:
            print("no optimizer sidecar for this checkpoint; Adam starts fresh", flush=True)

        log = EventLog(events_path)
        log.write({
            "type": "session",
            "loop": "recognizer",
            "policy": ckpt,
            "ref": args.ref,
            "save": args.save,
            "device": device,
            "reward_device": rdev,
            "reward": args.reward,
            "tau": args.tau,
            "prompts_file": args.prompts,
            "vcer": scorer.vcer_config,
            "confidence_weight": args.confidence,
            "straightness": args.straightness,
            "batch": args.batch,
            "k": args.k,
            "temperature_min": args.temp_min,
            "temperature_max": args.temp_max,
            "temperature_distribution": "normal",
            "sharpen": args.sharpen,
            "kl_coef": args.kl,
            "lr": args.lr,
            "clip": args.clip,
            "prompts": len(prompts),
            "review_text": review_text,
            "review_k": args.review_k,
            "resuming": resuming,
            "updates": updates,
        })
        session_open = True

        new_checkpoints = 0
        try:
            while args.checkpoints <= 0 or new_checkpoints < args.checkpoints:
                index = next_checkpoint_index(out_dir)
                print(f"\nblock -> ckpt-{index:04d}  (about {args.block_minutes:g} min of updates)",
                      flush=True)
                started = time.monotonic()
                block_updates = 0
                while True:
                    upcoming = updates + 1
                    sheet_path = None
                    if args.sample_every > 0 and upcoming % args.sample_every == 0:
                        sheet_path = out_dir / f"gen-{upcoming:06d}.png"
                    stats, rewards, cers, hyps, refs, ema, draws, wcors, sheet_name = one_update(
                        model, ref, scorer, optimizer, prompts, stoi, std, device, args, ema,
                        log=log, update_no=upcoming, block=index, sheet_path=sheet_path)
                    if sheet_name:
                        print(f"  samples {sheet_name}", flush=True)
                    updates += 1
                    block_updates += 1
                    elapsed = (time.monotonic() - started) / 60.0
                    reward_mean = sum(rewards) / len(rewards)
                    cer_mean = sum(cers) / len(cers)
                    present = [value for value in wcors if value is not None]
                    wcer_mean = sum(present) / len(present) if present else None
                    thin, hyp_len = _text_stats(hyps)
                    confs = [item.get("confidence") for item in draws]
                    confs = [value for value in confs if value is not None]
                    conf_mean = sum(confs) / len(confs) if confs else None
                    wcer_bit = "" if wcer_mean is None else f"wcer {wcer_mean:.3f}  "
                    conf_bit = "" if conf_mean is None else f"conf {conf_mean:.2f}  "
                    print(f"  update {updates}  {elapsed:.1f} min  "
                          f"reward {reward_mean:.3f}  "
                          f"cer {cer_mean:.3f}  "
                          f"{wcer_bit}"
                          f"{conf_bit}"
                          f"kl {stats['kl']:.4f}  "
                          f"loss {stats['loss']:.4f}  grad {stats['grad']:.3f}  "
                          f"{refs[0]!r} -> {short(hyps[0])!r}",
                          flush=True)
                    log.write({
                        "type": "update",
                        "update": updates,
                        "block": index,
                        "elapsed_min": elapsed,
                        "reward_mean": reward_mean,
                        "reward_min": min(rewards),
                        "reward_max": max(rewards),
                        "cer_mean": cer_mean,
                        "cer_min": min(cers),
                        "cer_max": max(cers),
                        "wcer_mean": wcer_mean,
                        "confidence_mean": conf_mean,
                        "wcer_min": min(present) if present else None,
                        "wcer_max": max(present) if present else None,
                        "hyp_len_mean": hyp_len,
                        "thin_rate": thin,
                        "kl": stats["kl"],
                        "loss": stats["loss"],
                        "grad": stats["grad"],
                        "used": stats["used"],
                        "stepped": stats["stepped"],
                        "steps_mean": (
                            sum(item["steps"] for item in draws) / len(draws) if draws else None
                        ),
                        "example": {"text": refs[0], "hyp": hyps[0]},
                        "sheet": sheet_name,
                        "draws": draws,
                    })
                    if elapsed * 60.0 >= args.block_minutes * 60.0:
                        break

                # Weights first, so a failure while drawing the sheet keeps the block.
                numbered = out_dir / f"ckpt-{index:04d}.pth"
                save_policy(model, numbered)
                save_policy(model, args.save)
                payload_best = None if best_reward == float("-inf") else best_reward
                save_sidecar(
                    numbered.with_name(numbered.stem + ".optim.pth"),
                    optimizer, updates, payload_best, review_text, ema, index)
                save_sidecar(
                    out_dir / "latest.optim.pth",
                    optimizer, updates, payload_best, review_text, ema, index)
                print(f"  weights: {args.save}", flush=True)
                print(f"  checkpoint: {numbered}  ({block_updates} updates, {elapsed:.1f} min)",
                      flush=True)
                log.write({
                    "type": "checkpoint",
                    "block": index,
                    "update": updates,
                    "block_updates": block_updates,
                    "elapsed_min": elapsed,
                    "weights": args.save,
                    "checkpoint": str(numbered),
                })

                review = write_review(
                    model, scorer, review_text, stoi, std, device, args, index, out_dir)
                is_best = review["reward_mean"] > best_reward
                if is_best:
                    best_reward = review["reward_mean"]
                    save_policy(model, best_path)
                    print(f"  new best review reward {best_reward:.3f} -> {best_path}", flush=True)
                log.write({
                    "type": "review",
                    "block": index,
                    "update": updates,
                    "text": review_text,
                    "reward_mean": review["reward_mean"],
                    "cer_mean": review["cer_mean"],
                    "cer_min": review["cer_min"],
                    "cer_max": review["cer_max"],
                    "wcer_mean": review["wcer_mean"],
                    "best": is_best,
                    "sheet": review["sheet"],
                    "draws": review["draws"],
                })
                write_json(run_path, {
                    "review_text": review_text,
                    "best_reward": None if best_reward == float("-inf") else best_reward,
                    "updates": updates,
                    "last_checkpoint": index,
                })
                new_checkpoints += 1
        except KeyboardInterrupt:
            reason = "interrupt"
            print("\nstopped.", flush=True)
            if Path(args.save).is_file():
                print(f"latest weights: {args.save}", flush=True)
                print("re-run to continue, or pass --ckpt samples/rl/ckpt-NNNN.pth "
                      "to continue from an earlier sheet.", flush=True)
            else:
                print("no checkpoint written yet.", flush=True)
        except Exception:
            reason = "error"
            raise
        else:
            print(f"done. latest {args.save}   best review reward "
                  f"{best_reward:.3f} -> {best_path}", flush=True)
    finally:
        if log is not None and session_open:
            log.write({"type": "stop", "reason": reason, "update": updates})
        if log is not None:
            log.close()


if __name__ == "__main__":
    main()
