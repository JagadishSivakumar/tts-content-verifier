"""Blind human labelling inside a Colab/Jupyter notebook, for a small validation sample.

The sample is stratified by (prompt category, TTS model) and drawn with a fixed seed, so it is
chosen by rule, not by hand, and never by looking at detector or Gemini output. `show` displays
only the prompt text and the audio; `save` writes one answer per clip to labels/primary.csv.
"""
from __future__ import annotations

import csv
import random

from .dataset import clip_key
from .io import DATA, LABELS, OUT, read_jsonl

TYPES = {"o": "omission", "r": "repetition", "s": "substitution", "t": "truncation", "i": "insertion"}


def validation_sample(n: int = 40) -> list[dict]:
    prompts = {p["id"]: p for p in read_jsonl(DATA / "prompts.jsonl")}
    cells: dict = {}
    for g in read_jsonl(OUT / "generations.jsonl"):
        cells.setdefault((prompts[g["prompt_id"]]["category"], g["tts_model"]), []).append(g)
    per = max(1, round(n / len(cells)))
    sample = []
    for key in sorted(cells):
        clips = sorted(cells[key], key=lambda g: g["clip_id"])
        random.Random(f"validation-{key}").shuffle(clips)   # raising n extends, never reshuffles
        sample += clips[:per]
    random.Random("validation-order").shuffle(sample)
    for g in sample:
        g["text"] = prompts[g["prompt_id"]]["text"]
    return sample


def show(start: int = 1, end: int = 10, n: int = 40):
    """Display clips start..end (1-based) of the validation sample: prompt + audio player only."""
    from IPython.display import Audio, Markdown, display
    sample = validation_sample(n)
    for i in range(start, min(end, len(sample)) + 1):
        g = sample[i - 1]
        display(Markdown(f"**{i}.** {g['text']}"))
        display(Audio(str(OUT / g["wav"])))


def save(answers: str, n: int = 40):
    """answers: one token per clip in order, e.g. "1 1 2o 1 2r? 1".
    1 = clean, 2 = error, letters o r s t i = types, ? = unsure. Fewer tokens than clips is fine."""
    toks = answers.split()
    if not toks:
        print("No answers given; existing labels left unchanged.")
        return
    bad = [t for t in toks if t[0] not in "12"]
    if bad:
        raise ValueError(f"Each answer must start with 1 or 2: {bad}")
    sample = validation_sample(n)
    path = LABELS / "primary.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_key", "error", "types", "notes"])
        for g, t in zip(sample, toks):
            w.writerow([clip_key(g["clip_id"]), int(t[0] == "2"),
                        ";".join(TYPES[c] for c in t[1:] if c in TYPES), "unsure" if "?" in t else ""])
    print(f"Saved {min(len(toks), len(sample))} of {len(sample)} labels to {path}")
