# Data

```bash
uv sync --extra inference
uv run python scripts/fetch_data.py
```

`data/dataset.npz` is built locally with `scripts/export_dataset.py` or `scripts/rebuild_dataset.py` after fetch.
