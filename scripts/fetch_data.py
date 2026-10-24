"""Download training data from Hugging Face into data/.

Run from the repo root:
  uv sync --extra inference
  uv run python scripts/fetch_data.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from hf_artifacts import DATA_ARTIFACTS, HF_REPO, download_artifacts, missing_paths


def main() -> None:
    os.chdir(ROOT)
    fetched = download_artifacts(DATA_ARTIFACTS, label="data")
    if fetched:
        print(f"fetched {len(fetched)} file(s) from {HF_REPO}")

    still_missing = missing_paths(DATA_ARTIFACTS)
    if still_missing:
        raise SystemExit(f"still missing: {', '.join(still_missing)}")

    print("data/ is ready for training.train and training.train_rl")


if __name__ == "__main__":
    main()
