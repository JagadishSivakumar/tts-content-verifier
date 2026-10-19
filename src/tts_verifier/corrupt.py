"""Controlled text corruptions for the synthetic benchmark.

The corrupted text is what the TTS model is asked to speak; the ORIGINAL prompt stays the reference.
So the intended error is known by construction, with no listening needed. This tests whether the
detectors notice deliberate content changes. It does NOT measure how often a TTS model makes
unintended errors, and the TTS model may add its own errors on top of the intended one.
"""
from __future__ import annotations

import random
import re

from .normalize import normalize

KINDS = ("omission", "repetition", "substitution")


def _is_word(tok: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z']{3,}[,.;:!?]*", tok))


def corrupt(text: str, kind: str, rng: random.Random, vocab: list[str] | None = None) -> str | None:
    """Return corrupted text, or None if this prompt cannot take the corruption.

    omission     : delete 1-2 consecutive mid-sentence words
    repetition   : say one word twice in a row
    substitution : replace one word with an unrelated word from `vocab`
    """
    toks = text.split()
    if len(toks) < 6:
        return None
    mid = [i for i in range(1, len(toks) - 2) if _is_word(toks[i])]   # keep first and last words
    if not mid:
        return None
    if kind == "omission":
        runs = [i for i in mid if i + 1 in mid]
        i = rng.choice(runs) if runs and rng.random() < 0.5 else rng.choice(mid)
        n = 2 if i in runs and rng.random() < 0.5 else 1
        out = toks[:i] + toks[i + n:]
    elif kind == "repetition":
        i = rng.choice(mid)
        out = toks[: i + 1] + [re.sub(r"[,.;:!?]+$", "", toks[i])] + toks[i + 1:]
    elif kind == "substitution":
        if not vocab:
            return None
        i = rng.choice(mid)
        old = re.sub(r"[,.;:!?]+$", "", toks[i]).lower()
        pool = [w for w in vocab if w.lower() != old and abs(len(w) - len(old)) <= 3]
        if not pool:
            return None
        new = rng.choice(pool)
        out = toks[:i] + [new + re.sub(r"^[A-Za-z']+", "", toks[i])] + toks[i + 1:]
    else:
        raise ValueError(kind)
    result = " ".join(out)
    return result if normalize(result) != normalize(text) else None
