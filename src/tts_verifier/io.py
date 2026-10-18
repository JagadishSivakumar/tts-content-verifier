from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
LABELS = ROOT / "labels"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_labels(path: Path) -> dict[str, dict]:
    """Label CSV -> {clip_key: {"error": 0/1, "types": [...], "notes": str}}."""
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if r.get("error", "").strip() == "":
                continue  # unlabelled row
            out[r["clip_key"]] = {"error": int(r["error"]),
                                  "types": [t for t in r.get("types", "").split(";") if t],
                                  "notes": r.get("notes", "")}
    return out
