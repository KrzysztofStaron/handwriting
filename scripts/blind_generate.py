"""Pre-generate sheets for the blind model comparison at /blind on the dashboard.

For every prompt and model: 4 tries at one temperature, stacked into one sheet.
Labels A-D are shuffled per round and the mapping goes to samples/blind/key.json.

    uv run --no-sync scripts/blind_generate.py            # skips sheets that exist
    uv run --no-sync scripts/blind_generate.py --fresh    # new shuffle, redraw all
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from generate import load_model, load_std  # noqa: E402
from rl import consistency_sheet, rollout_batch, strokes_to_images  # noqa: E402

MODELS = {
    "consistency1": "models/named/consistency1.pth",
    "consistency1 extended": "models/named/consistency1-extended.pth",
    "extended": "models/named/extended.pth",
    "consistency1 extended smoothed": "models/named/consistency1-extended-smoothed.pth",
}

PROMPTS = [
    "Krzysztof Szymon Serafin Staron",
    "The quick brown fox jumps over the lazy dog.",
    "hello world",
    "Meet me at 7:45 on 21.10.2026, okay?",
    "k.staron314 (at) gmail.com",
    "Zazolc gesla jazn, prosze pana!",
    "\"To be, or not to be\" - Hamlet",
    "3.14159 26535 89793",
    "WARNING: HIGH VOLTAGE",
    "Dear Sean, thanks for the access.",
    "Q4 revenue was +12 (not -12) units.",
    "minimum",
    "Is it 9/11 or 11/9? Who knows...",
    "Pack my box with five dozen liquor jugs",
    "Ksiezyce, Polska; 05-123",
    "lowercase only, no capitals here at all",
    "Buy: 2 eggs, 1 milk, 3 apples #groceries",
    "Wszystko bedzie dobrze, zobaczysz.",
    "A",
    "It was the best of times, it was the worst",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples/blind")
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--sharpen", type=float, default=2.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sheets", type=int, default=2, help="independent sheets per model")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--check", action="store_true", help="only validate prompts")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    stoi = json.loads((ROOT / "stoi.json").read_text())
    for p in PROMPTS:
        bad = sorted({c for c in p if c not in stoi})
        if bad or len(p) > 48:
            raise SystemExit(f"bad prompt {p!r}: chars {bad} len {len(p)}")
    assert len(PROMPTS) == 20 and len(set(PROMPTS)) == 20
    if args.check:
        print("prompts ok")
        return

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    key_path = out / "key.json"
    if key_path.exists() and not args.fresh:
        key = json.loads(key_path.read_text())
    else:
        rng = random.SystemRandom()
        rounds = []
        for i, prompt in enumerate(PROMPTS):
            names = list(MODELS)
            rng.shuffle(names)
            rounds.append({"round": i + 1, "prompt": prompt,
                           "labels": dict(zip("ABCD", names))})
        key = {"temperature": args.temperature, "tries": args.tries, "rounds": rounds}
        key_path.write_text(json.dumps(key, indent=2))
        for f in out.glob("r*.png"):
            f.unlink()
    (out / "prompts.json").write_text(json.dumps(
        {"temperature": args.temperature, "tries": args.tries,
         "rounds": [{"round": r["round"], "prompt": r["prompt"]} for r in key["rounds"]]},
        indent=2))

    std = load_std()
    t0 = time.time()
    for name, ckpt in MODELS.items():
        model, _ = load_model(args.device, ckpt=str(ROOT / ckpt))
        for r in key["rounds"]:
            label = next(k for k, v in r["labels"].items() if v == name)
            text = r["prompt"]
            # Sheet 1 is for the semifinal, sheet 2 (fresh draws) for the final/3rd place.
            for sheet in range(1, args.sheets + 1):
                suffix = "" if sheet == 1 else f"-{sheet}"
                path = out / f"r{r['round']:02d}-{label}{suffix}.png"
                if path.exists():
                    continue
                with torch.no_grad():
                    seqs = rollout_batch(model, [text] * args.tries, stoi, args.device,
                                         [args.temperature] * args.tries, args.sharpen)
                images = strokes_to_images(seqs, std)
                caps = [f"try {i}" for i in range(1, len(images) + 1)]
                consistency_sheet(images, text, caps).save(path)
                print(f"{time.time() - t0:7.1f}s  round {r['round']:2d} {label} sheet {sheet}",
                      flush=True)
        del model
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
