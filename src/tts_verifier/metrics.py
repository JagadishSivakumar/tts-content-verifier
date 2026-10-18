"""Evaluation metrics with uncertainty.

Bootstrap resamples *prompt template groups*, not clips, because clips from one template are
correlated. Resampling clips would understate the confidence interval width.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Sequence


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def detection_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    c = confusion(y_true, y_pred)
    prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else float("nan")
    rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")
    fpr = c["fp"] / (c["fp"] + c["tn"]) if c["fp"] + c["tn"] else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan")
    return {**c, "precision": prec, "recall": rec, "f1": f1, "false_alarm_rate": fpr}


def grouped_bootstrap_ci(rows: list[dict], group_key: str, stat: Callable[[list[dict]], float],
                         n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and percentile CI of `stat`, resampling whole groups with replacement."""
    by_group: dict = defaultdict(list)
    for r in rows:
        by_group[r[group_key]].append(r)
    groups = list(by_group)
    rng = random.Random(seed)
    point = stat(rows)
    samples = []
    for _ in range(n_boot):
        draw = [r for g in rng.choices(groups, k=len(groups)) for r in by_group[g]]
        v = stat(draw)
        if v == v:  # drop NaN draws (e.g. no positives)
            samples.append(v)
    samples.sort()
    if not samples:
        return point, float("nan"), float("nan")
    lo = samples[int(alpha / 2 * len(samples))]
    hi = samples[min(int((1 - alpha / 2) * len(samples)), len(samples) - 1)]
    return point, lo, hi


def cohens_kappa(a: Sequence, b: Sequence) -> float:
    assert len(a) == len(b) and len(a) > 0
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((list(a).count(l) / n) * (list(b).count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Paired comparison of two detectors' correctness on the same clips (exact binomial)."""
    from math import comb
    b = sum((pa == t) and (pb != t) for t, pa, pb in zip(y_true, pred_a, pred_b))
    c = sum((pa != t) and (pb == t) for t, pa, pb in zip(y_true, pred_a, pred_b))
    n = b + c
    if n == 0:
        return {"a_only_correct": b, "b_only_correct": c, "p_value": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only_correct": b, "b_only_correct": c, "p_value": p}
