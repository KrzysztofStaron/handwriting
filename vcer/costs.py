"""Cost tables for the visual edit distance.

Lookup is directional: `sub_cost(ref_span, hyp_span)`. An exact span match
costs 0. Anything else is at least `min_sub_cost`, and an insertion or
deletion is at least `min_indel_cost`.
"""

import hashlib
import json
from pathlib import Path

INF = float("inf")


def _canon(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def table_hash(payload):
    body = dict(payload)
    body.pop("hash", None)
    return hashlib.sha256(_canon(body).encode("utf-8")).hexdigest()[:16]


class CostTable:
    def __init__(self, subs=None, ins=1.0, delete=1.0, min_sub_cost=0.15,
                 min_indel_cost=0.6, case_cost=0.2, k=2, version=1, source="memory"):
        self.ins = float(ins)
        self.delete = float(delete)
        self.min_sub_cost = float(min_sub_cost)
        self.min_indel_cost = float(min_indel_cost)
        self.case_cost = float(case_cost)
        self.k = int(k)
        self.version = int(version)
        self.source = source
        self.sub = {}
        for (src, dst), cost in (subs or {}).items():
            self.sub[(src, dst)] = self._floor_sub(src, dst, float(cost))

    def _floor_sub(self, src, dst, cost):
        if src == dst:
            return 0.0
        return min(1.0, max(self.min_sub_cost, cost))

    def sub_cost(self, src, dst):
        if src == dst:
            return 0.0
        found = self.sub.get((src, dst))
        if found is not None:
            return found
        if src.casefold() == dst.casefold():
            return self._floor_sub(src, dst, self.case_cost)
        return 1.0

    def ins_cost(self, char):
        return min(1.0, max(self.min_indel_cost, self.ins))

    def del_cost(self, char):
        return min(1.0, max(self.min_indel_cost, self.delete))

    def to_payload(self):
        pairs = [
            {"ref": src, "hyp": dst, "cost": cost}
            for (src, dst), cost in sorted(self.sub.items())
        ]
        payload = {
            "version": self.version,
            "k": self.k,
            "ins": self.ins,
            "del": self.delete,
            "min_sub_cost": self.min_sub_cost,
            "min_indel_cost": self.min_indel_cost,
            "case_cost": self.case_cost,
            "source": self.source,
            "pairs": pairs,
        }
        payload["hash"] = table_hash(payload)
        return payload

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_payload(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError(f"unsupported cost table version {payload.get('version')}")
        digest = payload.get("hash")
        if digest is not None and digest != table_hash(payload):
            raise ValueError(f"cost table hash mismatch for {path}")
        subs = {(item["ref"], item["hyp"]): item["cost"] for item in payload["pairs"]}
        return cls(
            subs,
            ins=payload.get("ins", 1.0),
            delete=payload.get("del", 1.0),
            min_sub_cost=payload.get("min_sub_cost", 0.15),
            min_indel_cost=payload.get("min_indel_cost", 0.6),
            case_cost=payload.get("case_cost", 0.2),
            k=payload.get("k", 2),
            version=payload["version"],
            source=payload.get("source", str(path)),
        )


def unit_table():
    """Every edit costs 1 and an exact character costs 0. This is Levenshtein at K=1."""
    return CostTable(ins=1.0, delete=1.0, min_sub_cost=1.0, min_indel_cost=1.0, case_cost=1.0, k=1)
