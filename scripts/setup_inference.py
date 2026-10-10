"""Download model artifacts and verify local inference works.

Run from the repo root:
  uv sync --extra inference
  uv run python scripts/setup_inference.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "PanzerBread/handwriting"

ARTIFACTS = {
    "weights.npz": "weights.npz",
    "stoi.json": "stoi.json",
    "std.json": "std.json",
    "models/post-trained.pth": "best-4.pth",
}


def repo_path(name: str) -> Path:
    return ROOT / name


def download_missing() -> list[str]:
    missing = [path for path in ARTIFACTS if not repo_path(path).exists()]
    if not missing:
        return []

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing artifacts and huggingface_hub is not installed.\n"
            "Run: uv sync --extra inference\n"
            "Then re-run this script."
        ) from exc

    fetched = []
    for local_path in missing:
        remote_name = ARTIFACTS[local_path]
        dest = repo_path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {remote_name} -> {local_path}")
        downloaded = Path(
            hf_hub_download(HF_REPO, remote_name, local_dir=str(dest.parent))
        )
        if downloaded.name != dest.name:
            downloaded.replace(dest)
        elif downloaded != dest:
            downloaded.replace(dest)
        fetched.append(local_path)
    return fetched


def verify_numpy() -> int:
    sys.path.insert(0, str(ROOT))
    import numpy as np

    from model_np import NumpyHandwritingModel

    stoi = json.loads(repo_path("stoi.json").read_text(encoding="utf-8"))
    std = json.loads(repo_path("std.json").read_text(encoding="utf-8"))["std"]
    model = NumpyHandwritingModel.from_npz(str(repo_path("weights.npz")))

    text = "hello"
    c = np.zeros((len(text), len(stoi)), np.float32)
    c[np.arange(len(text)), [stoi[ch] for ch in text]] = 1.0
    pts = list(model.sample_iter(c, np.ones(len(text), np.float32), temperature=0.5))
    print(f"numpy ok: {len(pts)} points for '{text}' (std={std:.4f})")
    return len(pts)


def verify_torch() -> None:
    sys.path.insert(0, str(ROOT))
    import torch

    from generate import default_device, load_model, load_std, sample

    ckpt = repo_path("models/post-trained.pth")
    if not ckpt.exists():
        print("torch: skipped (models/post-trained.pth missing)")
        return

    device = default_device()
    model, stoi = load_model(device, ckpt=str(ckpt))
    std = load_std()
    seq, _ = sample(model, "hello", stoi, device, temperature=0.5)
    print(f"torch ok: {len(seq)} steps on {device} (std={std:.4f})")


def print_next_steps() -> None:
    print()
    print("Local inference is ready.")
    print()
    print("Numpy API (matches production):")
    print("  uv run app_np.py")
    print("  curl 'http://127.0.0.1:5000/generate?text=hello&temperature=0.5'")
    print("  curl -sN 'http://127.0.0.1:5000/generate/stream?text=hello&temperature=0.5'")
    print()
    print("Torch PNG generation:")
    print("  uv run generate.py 'hello world' -t 0.5 -o hello.png")
    print("  uv run python scripts/grid_samples.py")


def main() -> None:
    os.chdir(ROOT)
    fetched = download_missing()
    if fetched:
        print(f"fetched {len(fetched)} artifact(s) from {HF_REPO}")

    missing = [path for path in ARTIFACTS if not repo_path(path).exists()]
    if missing:
        raise SystemExit(f"still missing: {', '.join(missing)}")

    verify_numpy()
    verify_torch()
    print_next_steps()


if __name__ == "__main__":
    main()
