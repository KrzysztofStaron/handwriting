"""Post-train from a human best-of-k pick.

Each round draws k samples (default 3) of one prompt, opens a contact sheet,
and asks which one is best. That sample gets reward 1 and the others get 0.
The update is the same REINFORCE step as train_rl.py.

The policy starts from models/rl-human.pth if a previous session wrote one,
else models/rl-trocr.pth, else the base model models/best-4.pth. The KL
anchor stays the base model. best-4.pth is never written.

Picks are appended to data/rlhf_prefs.jsonl before the gradient step, so a
crash during the update still keeps the choice. The same rounds are appended
to samples/rlhf/events.jsonl for the dashboard:

    uv run python -m dashboards.rl_dashboard --dir samples/rlhf

    uv run python -m training.train_rlhf
    uv run python -m training.train_rlhf --text "hello world" --rounds 10
"""

import argparse
import sys
from pathlib import Path

import torch

from training.generate import default_device, load_std
from training.rl import (
    BASE_CKPT,
    add_policy_args,
    advantages,
    append_jsonl,
    choice_sheet,
    load_policy_and_ref,
    pick_prompts,
    refuse_overwrite_base,
    resolve_prompts,
    reveal,
    reinforce_step,
    rollout_batch,
    save_policy,
    set_seed,
    strokes_to_images,
)
from training.rl_events import EVENTS_NAME, EventLog, pack_draws


def default_ckpt():
    """Resume a human session, else the recognizer run, else the base model."""
    if Path("models/rl-human.pth").exists():
        return "models/rl-human.pth"
    if Path("models/rl-trocr.pth").exists():
        return "models/rl-trocr.pth"
    return BASE_CKPT


def parse_args():
    parser = argparse.ArgumentParser(
        description="REINFORCE from a human best-of-k pick. "
                    f"The KL anchor defaults to {BASE_CKPT} and that file is not modified.",
    )
    add_policy_args(parser, default_save="models/rl-human.pth")
    parser.add_argument("-k", type=int, default=3, help="Samples to draw and choose among.")
    parser.add_argument("--rounds", type=int, default=0,
                        help="Stop after this many accepted picks. 0 runs until quit.")
    parser.add_argument("--prefs", default="data/rlhf_prefs.jsonl")
    parser.add_argument("--sheet-dir", default="samples/rlhf")
    return parser.parse_args()


def ask_best(k):
    while True:
        try:
            raw = input(f"best sample [1-{k}], s=skip, q=quit: ").strip().lower()
        except EOFError:
            print()
            return "quit"
        if raw in {"q", "quit"}:
            return "quit"
        if raw in {"s", "skip"}:
            return "skip"
        if raw.isdigit() and 1 <= int(raw) <= k:
            return int(raw) - 1
        print(f"enter a number from 1 to {k}, s, or q", flush=True)


def main():
    args = parse_args()
    if not sys.stdin.isatty():
        raise SystemExit("train_rlhf.py needs an interactive terminal")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be > 0 (0 has no density to reinforce)")
    if args.k < 2:
        raise SystemExit("--k must be >= 2 so there is a comparison to make")
    if args.min_chars > args.max_chars:
        raise SystemExit("--min-chars cannot exceed --max-chars")
    refuse_overwrite_base(args.save)

    ckpt = args.ckpt or default_ckpt()
    device = args.device or default_device()
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    print(f"device: {device}   policy: {ckpt}   KL anchor: {args.ref}", flush=True)
    print(f"k: {args.k}   temperature: {args.temperature}   sharpen: {args.sharpen}   "
          f"kl: {args.kl}   lr: {args.lr}", flush=True)
    print(f"saving {args.save}    prefs {args.prefs}", flush=True)
    print(f"{BASE_CKPT} is the base model and is not modified.", flush=True)

    model, ref, stoi = load_policy_and_ref(ckpt, args.ref, device)
    prompts = resolve_prompts(args.text, args.prompts, stoi, args.min_chars, args.max_chars)
    std = load_std()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sheet_dir = Path(args.sheet_dir)
    events_path = sheet_dir / EVENTS_NAME
    print(f"events -> {events_path}", flush=True)
    print("dashboard -> uv run python -m dashboards.rl_dashboard --dir samples/rlhf", flush=True)
    accepted = 0
    shown = 0
    log = EventLog(events_path)
    reason = "done"
    try:
        log.write({
            "type": "session",
            "loop": "human",
            "policy": ckpt,
            "ref": args.ref,
            "save": args.save,
            "device": device,
            "k": args.k,
            "temperature": args.temperature,
            "sharpen": args.sharpen,
            "kl_coef": args.kl,
            "lr": args.lr,
            "clip": args.clip,
            "prompts": len(prompts),
        })
        while args.rounds <= 0 or accepted < args.rounds:
            text = pick_prompts(prompts, 1)[0]
            print(f"\n[{accepted + 1}] {text}", flush=True)
            print(f"  drawing {args.k} lines in one batch", flush=True)
            seqs = rollout_batch(
                model, [text] * args.k, stoi, device, args.temperature, args.sharpen)
            for i, seq in enumerate(seqs, start=1):
                print(f"  sample {i}/{args.k}: {seq.shape[0]} steps", flush=True)
                log.write({
                    "type": "draw",
                    "update": accepted + 1,
                    "text": text,
                    "draw": i,
                    "k": args.k,
                    "steps": int(seq.shape[0]),
                })

            images = strokes_to_images(seqs, std, height=220, stroke_width=4)
            sheet = choice_sheet(images, text)
            shown += 1
            path = sheet_dir / f"round{shown:04d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(path)
            reveal(path)

            choice = ask_best(args.k)
            if choice == "quit":
                reason = "quit"
                break
            if choice == "skip":
                print("skipped", flush=True)
                log.write({
                    "type": "skip",
                    "update": accepted + 1,
                    "text": text,
                    "sheet": path.name,
                })
                continue

            # Persist the pick before the update. The gradient step can be rerun;
            # the choice cannot.
            append_jsonl(args.prefs, {
                "text": text,
                "winner": choice,
                "temperature": args.temperature,
                "sharpen": args.sharpen,
                "steps": [int(seq.shape[0]) for seq in seqs],
                "seqs": [seq.tolist() for seq in seqs],
            })
            rewards = [1.0 if i == choice else 0.0 for i in range(args.k)]
            advs, _ema = advantages(rewards, args.k, ema=None)
            rollouts = [{"text": text, "seq": seq} for seq in seqs]
            print(f"updating toward sample {choice + 1}", flush=True)
            stats = reinforce_step(
                model, ref, rollouts, advs, stoi, device,
                args.temperature, args.sharpen, args.kl, optimizer, args.clip,
            )
            save_policy(model, args.save)
            accepted += 1
            draws = pack_draws(
                [text] * args.k,
                [int(seq.shape[0]) for seq in seqs],
                rewards, [None] * args.k, [None] * args.k, advs, stats.get("traces"),
            )
            print(f"pick {accepted}  loss {stats['loss']:.4f}  kl {stats['kl']:.4f}  "
                  f"grad {stats['grad']:.2f}  saved {args.save}  logged {args.prefs}",
                  flush=True)
            log.write({
                "type": "update",
                "update": accepted,
                "reward_mean": sum(rewards) / len(rewards),
                "reward_min": min(rewards),
                "reward_max": max(rewards),
                "cer_mean": None,
                "kl": stats["kl"],
                "loss": stats["loss"],
                "grad": stats["grad"],
                "used": stats["used"],
                "stepped": stats["stepped"],
                "steps_mean": sum(item["steps"] for item in draws) / len(draws),
                "winner": choice + 1,
                "sheet": path.name,
                "example": {"text": text, "hyp": f"sample {choice + 1}"},
                "draws": draws,
            })
    except KeyboardInterrupt:
        reason = "interrupt"
        print("\nstopped.", flush=True)
    except Exception:
        reason = "error"
        raise
    finally:
        log.write({"type": "stop", "reason": reason, "update": accepted})
        log.close()

    print(f"session done. {accepted} picks. policy: {args.save}", flush=True)


if __name__ == "__main__":
    main()
