"""Download (and upload) repo artifacts from Hugging Face."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "PanzerBread/handwriting"

# local path -> path in the HF repo
INFERENCE_ARTIFACTS: dict[str, str] = {
    "weights.npz": "weights.npz",
    "stoi.json": "stoi.json",
    "std.json": "std.json",
    "models/post-trained.pth": "best-4.pth",
}

DATA_ARTIFACTS: dict[str, str] = {
    "data/strokes.npy": "data/strokes.npy",
    "data/sentences.txt": "data/sentences.txt",
    "data/romeo_juliet.txt": "data/romeo_juliet.txt",
    "data/rejected_review.json": "data/rejected_review.json",
}


def repo_path(relative: str) -> Path:
    return ROOT / relative


def missing_paths(artifacts: dict[str, str]) -> list[str]:
    return [path for path in artifacts if not repo_path(path).exists()]


def download_artifacts(artifacts: dict[str, str], *, label: str) -> list[str]:
    missing = missing_paths(artifacts)
    if not missing:
        return []

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            f"Missing {label} and huggingface_hub is not installed.\n"
            "Run: uv sync --extra inference\n"
            "Then re-run the fetch script."
        ) from exc

    fetched: list[str] = []
    for local_path in missing:
        remote_name = artifacts[local_path]
        dest = repo_path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {remote_name} -> {local_path}")
        downloaded = Path(
            hf_hub_download(HF_REPO, remote_name, local_dir=str(ROOT))
        )
        if downloaded != dest:
            downloaded.replace(dest)
        fetched.append(local_path)
    return fetched


def upload_artifacts(artifacts: dict[str, str], *, label: str) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    uploaded: list[str] = []
    for local_path, remote_name in artifacts.items():
        src = repo_path(local_path)
        if not src.exists():
            raise SystemExit(f"cannot upload {label}: missing {src}")
        print(f"uploading {local_path} -> {HF_REPO}:{remote_name}")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=remote_name,
            repo_id=HF_REPO,
            repo_type="model",
        )
        uploaded.append(local_path)
    return uploaded
