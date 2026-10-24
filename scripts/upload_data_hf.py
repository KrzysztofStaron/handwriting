"""One-time (or refresh) upload of local data/ files to the Hugging Face model repo.

Run from the repo root after you have write access to PanzerBread/handwriting:
  uv sync --extra inference
  uv run python scripts/upload_data_hf.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from hf_artifacts import DATA_ARTIFACTS, HF_REPO, upload_artifacts


def main() -> None:
    os.chdir(ROOT)
    uploaded = upload_artifacts(DATA_ARTIFACTS, label="data")
    print(f"uploaded {len(uploaded)} file(s) to {HF_REPO}")


if __name__ == "__main__":
    main()
