"""Word-level alignment between the prompt and an ASR transcript, with TTS-oriented error tags.

Tags are *suspected* TTS errors as seen through one recogniser. They are never treated as ground
truth: human listening labels are the reference (see labels/LABELING.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

OMISSION, REPETITION, INSERTION, SUBSTITUTION, TRUNCATION = (
    "omission", "repetition", "insertion", "substitution", "truncation")


@dataclass
class Edit:
    op: str               # "match" | "sub" | "del" | "ins"
    ref: str | None
    hyp: str | None
    ref_idx: int | None   # position in reference (for del/sub/match)


@dataclass
class AlignmentResult:
    edits: list[Edit]
    errors: list[dict] = field(default_factory=list)   # tagged spans
    n_ref: int = 0

    @property
    def wer(self) -> float:
        n = sum(e.op != "match" for e in self.edits)
        return n / max(self.n_ref, 1)

    def count(self, tag: str) -> int:
        return sum(1 for e in self.errors if e["type"] == tag)

    @property
    def max_deletion_run(self) -> int:
        run = best = 0
        for e in self.edits:
            if e.op == "del":
                run += 1
                best = max(best, run)
            elif e.op != "ins":
                run = 0
        return best


def levenshtein_align(ref: list[str], hyp: list[str]) -> list[Edit]:
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    edits: list[Edit] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            op = "match" if ref[i - 1] == hyp[j - 1] else "sub"
            edits.append(Edit(op, ref[i - 1], hyp[j - 1], i - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            edits.append(Edit("del", ref[i - 1], None, i - 1))
            i -= 1
        else:
            edits.append(Edit("ins", None, hyp[j - 1], i - 1 if i > 0 else None))
            j -= 1
    edits.reverse()
    return edits


def _group_runs(edits: list[Edit], op: str) -> list[list[int]]:
    runs, cur = [], []
    for k, e in enumerate(edits):
        if e.op == op:
            cur.append(k)
        elif cur and e.op != op:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _char_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def spelling_equivalent(a: str, b: str) -> bool:
    """Words that differ only by a spelling variant or recogniser typo are not content errors.

    Allowed character edits: none under 4 letters (so "red" vs "bed" stays a substitution),
    one for 4-7 letters (gray/grey, flavor/flavour, chili/chilli), two for 8+ letters (rare names).
    Trade-off: a real one-letter substitution in a 5+ letter word ("bridge" vs "ridge") is missed.
    """
    n = max(len(a), len(b))
    if n < 4:
        return False
    return _char_distance(a, b) <= (1 if n < 8 else 2)


def _merge_compounds(ref: list[str], hyp: list[str]) -> tuple[list[str], list[str]]:
    """Join adjacent tokens that one side writes as one word and the other as two
    ("wrist watches" / "wristwatches", "some one" / "someone", "post code" / "postcode")."""
    def merge(a: list[str], other: list[str]) -> list[str]:
        vocab, out, i = set(other), [], 0
        while i < len(a):
            if i + 1 < len(a) and a[i] + a[i + 1] in vocab and (a[i] not in vocab or a[i + 1] not in vocab):
                out.append(a[i] + a[i + 1]); i += 2
            else:
                out.append(a[i]); i += 1
        return out
    return merge(ref, hyp), merge(hyp, ref)


def tag_errors(ref: list[str], hyp: list[str]) -> AlignmentResult:
    ref, hyp = _merge_compounds(ref, hyp)
    edits = levenshtein_align(ref, hyp)
    for e in edits:   # spelling-variant substitutions count as matches
        if e.op == "sub" and spelling_equivalent(e.ref, e.hyp):
            e.op = "match"
    res = AlignmentResult(edits=edits, n_ref=len(ref))

    # Truncation: a deletion run that reaches the end of the reference.
    trailing = 0
    for e in reversed(edits):
        if e.op == "del":
            trailing += 1
        elif e.op == "ins":
            continue
        else:
            break
    truncated_from = len(ref) - trailing if trailing >= 2 else None
    if truncated_from is not None:
        res.errors.append({"type": TRUNCATION, "ref_span": [truncated_from, len(ref)],
                           "words": ref[truncated_from:]})

    for run in _group_runs(edits, "del"):
        idxs = [edits[k].ref_idx for k in run]
        if truncated_from is not None and min(idxs) >= truncated_from:
            continue
        res.errors.append({"type": OMISSION, "ref_span": [min(idxs), max(idxs) + 1],
                           "words": [edits[k].ref for k in run]})

    for run in _group_runs(edits, "ins"):
        words = [edits[k].hyp for k in run]
        # Repetition if the inserted words duplicate the reference words immediately before
        # the insertion point (an n-gram the model said twice).
        before = edits[run[0]].ref_idx
        anchor = before if before is not None else -1
        span = ref[max(0, anchor - len(words) + 1): anchor + 1]
        after = ref[anchor + 1: anchor + 1 + len(words)]
        is_rep = words == span or words == after or (len(words) == 1 and words[0] in span + after)
        res.errors.append({"type": REPETITION if is_rep else INSERTION,
                           "after_ref_idx": anchor, "words": words})

    for k, e in enumerate(edits):
        if e.op == "sub":
            res.errors.append({"type": SUBSTITUTION, "ref_span": [e.ref_idx, e.ref_idx + 1],
                               "ref": e.ref, "hyp": e.hyp})
    return res
