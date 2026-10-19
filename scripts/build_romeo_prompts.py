"""Cut the whole of Romeo and Juliet into trainer prompts.

The Gutenberg header and footer stay out. Characters outside the 77-character
alphabet become spaces. Lines are 8–32 characters, split on words. Repeated
lines are kept once, in first-seen order. The trainer then samples the file
at random.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("/tmp/romeo_raw.txt")
OUT = ROOT / "data" / "romeo_juliet.txt"

EXTRA = {
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u2014": "-", "\u2013": "-", "\u00a0": " ", "\u00ad": "",
    "\u2026": "...",
    "\u00e6": "ae", "\u00c6": "AE",
    "\ufb01": "fi", "\ufb02": "fl",
}


def play_body(raw):
    start = raw.find("*** START")
    end = raw.find("*** END")
    if start < 0 or end < 0 or end <= start:
        raise SystemExit("Romeo and Juliet markers were not found in the source text")
    body = raw[start:end]
    newline = body.find("\n")
    return body[newline + 1:] if newline >= 0 else body


def to_alphabet(text, alphabet):
    for src, dst in EXTRA.items():
        text = text.replace(src, dst)
    chars = []
    for ch in text:
        chars.append(ch if ch in alphabet else " ")
    return "".join(chars)


def chunk(text, min_chars=8, max_chars=32):
    lines = []
    buf = ""
    for word in text.split():
        if len(word) > max_chars:
            if len(buf) >= min_chars:
                lines.append(buf)
            buf = ""
            continue
        candidate = word if not buf else buf + " " + word
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if len(buf) >= min_chars:
                lines.append(buf)
            buf = word
    if len(buf) >= min_chars:
        lines.append(buf)
    seen = set()
    unique = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return lines, unique


def main():
    if not RAW.is_file():
        raise SystemExit(f"missing {RAW}; download Gutenberg ebook 1513 there first")
    alphabet = set(json.loads((ROOT / "stoi.json").read_text(encoding="utf-8")))
    raw = RAW.read_text(encoding="utf-8", errors="replace")
    text = to_alphabet(play_body(raw), alphabet)
    all_lines, unique = chunk(text)
    OUT.write_text("\n".join(unique) + "\n", encoding="utf-8")
    print(f"chunks {len(all_lines)}  unique {len(unique)}  -> {OUT}")
    print("sample:", unique[0])
    print("sample:", unique[len(unique) // 2])
    print("sample:", unique[-1])


if __name__ == "__main__":
    main()
