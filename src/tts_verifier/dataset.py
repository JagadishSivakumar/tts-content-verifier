"""Joins prompts, generations, transcripts and human labels into evaluation rows."""
from __future__ import annotations

import hashlib

from .io import DATA, LABELS, OUT, read_jsonl, read_labels


def clip_key(clip_id: str) -> str:
    """Opaque key shown to listeners, so the TTS model and seed are hidden while labelling."""
    return hashlib.sha256(("blind-v1:" + clip_id).encode()).hexdigest()[:10]


def load_rows(generations_file: str = "generations.jsonl", labels_file: str = "primary.csv",
              require_labels: bool = True) -> list[dict]:
    prompts = {p["id"]: p for p in read_jsonl(DATA / "prompts.jsonl")}
    gens = read_jsonl(OUT / generations_file)
    trans = {t["clip_id"]: t for t in read_jsonl(OUT / "transcripts.jsonl")}
    labels = read_labels(LABELS / labels_file)
    rows = []
    for g in gens:
        p = prompts[g["prompt_id"]]
        t = trans.get(g["clip_id"])
        lab = labels.get(clip_key(g["clip_id"]))
        if t is None or (require_labels and lab is None):
            continue
        rows.append({
            **g, "prompt": p["text"], "category": p["category"], "group": p["group"],
            "split": p["split"], "whisper_text": t["whisper_text"], "ctc_text": t["ctc_text"],
            "human_error": lab["error"] if lab else None,
            "human_types": lab["types"] if lab else [],
        })
    return rows
