"""Load the hand-curated confusion list.

The file is a small YAML subset (scalars and `[ref, hyp, cost]` rows) so the
package does not depend on PyYAML.
"""

from pathlib import Path

SEED_PATH = Path(__file__).resolve().parent / "hand_seed.yaml"


def load_hand_seed(path=None):
    path = Path(path) if path else SEED_PATH
    meta = {}
    subs = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("- ["):
            body = stripped[2:].strip()
            if not (body.startswith("[") and body.endswith("]")):
                raise ValueError(f"bad pair row: {raw}")
            parts = [part.strip() for part in body[1:-1].split(",")]
            if len(parts) != 3:
                raise ValueError(f"bad pair row: {raw}")
            subs[(parts[0], parts[1])] = float(parts[2])
            continue
        if ":" not in line or line.startswith(" "):
            raise ValueError(f"bad seed line: {raw}")
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    if not subs:
        raise ValueError(f"no pairs in {path}")
    return subs, meta
