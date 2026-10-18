"""Clip-level detectors.

Baseline  : Whisper transcript only -> flag if any tagged content error.
Combined  : features from Whisper + a CTC recogniser (wav2vec2, no language model) + duration
            ratio, scored by logistic regression fitted on the DEV split only.

Why a CTC recogniser: Whisper's decoder is a strong language model and may "repair" a skipped or
garbled word, hiding the TTS error. A greedy CTC model has almost no LM, so it reports more of
what was actually said. Duration ratio is independent of ASR entirely.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .align import OMISSION, REPETITION, SUBSTITUTION, TRUNCATION, INSERTION, tag_errors
from .normalize import normalize

CONTENT_TAGS = (OMISSION, REPETITION, SUBSTITUTION, TRUNCATION, INSERTION)


def asr_features(prompt: str, transcript: str, prefix: str) -> dict:
    ref, hyp = normalize(prompt), normalize(transcript)
    res = tag_errors(ref, hyp)
    return {
        f"{prefix}_wer": res.wer,
        f"{prefix}_n_omission": res.count(OMISSION) + res.count(TRUNCATION),
        f"{prefix}_n_repetition": res.count(REPETITION),
        f"{prefix}_n_substitution": res.count(SUBSTITUTION),
        f"{prefix}_max_del_run": res.max_deletion_run,
        f"{prefix}_any_error": int(any(res.count(t) for t in CONTENT_TAGS)),
    }


@dataclass
class DurationModel:
    """Expected seconds per normalised character, fitted per TTS model on human-clean dev clips."""
    sec_per_char: dict

    @classmethod
    def fit(cls, rows: list[dict]) -> "DurationModel":
        acc: dict = {}
        for r in rows:
            if r["human_error"] == 0:
                n_chars = len(" ".join(normalize(r["prompt"])))
                acc.setdefault(r["tts_model"], []).append(r["duration_s"] / max(n_chars, 1))
        return cls({m: sorted(v)[len(v) // 2] for m, v in acc.items()})  # median: robust

    def log_ratio(self, row: dict) -> float:
        n_chars = len(" ".join(normalize(row["prompt"])))
        expected = self.sec_per_char[row["tts_model"]] * max(n_chars, 1)
        return math.log(row["duration_s"] / expected)


def whisper_only_predict(row: dict) -> int:
    return asr_features(row["prompt"], row["whisper_text"], "w")["w_any_error"]


FEATURES = ["w_wer", "w_n_omission", "w_n_repetition", "w_n_substitution", "w_max_del_run",
            "c_wer", "c_n_omission", "c_n_repetition", "c_max_del_run",
            "abs_log_dur_ratio", "both_flag", "disagree"]


def build_features(row: dict, dur: DurationModel) -> dict:
    f = asr_features(row["prompt"], row["whisper_text"], "w")
    f.update(asr_features(row["prompt"], row["ctc_text"], "c"))
    f["abs_log_dur_ratio"] = abs(dur.log_ratio(row))
    f["both_flag"] = int(f["w_any_error"] and f["c_any_error"])
    f["disagree"] = int(f["w_any_error"] != f["c_any_error"])
    return f


class CombinedDetector:
    """Standardised logistic regression; threshold chosen on dev to hit a target recall."""

    def __init__(self, target_recall: float = 0.9, C: float = 1.0):
        self.target_recall, self.C = target_recall, C

    def fit(self, dev_rows: list[dict]):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.dur = DurationModel.fit(dev_rows)
        X = [[build_features(r, self.dur)[k] for k in FEATURES] for r in dev_rows]
        y = [r["human_error"] for r in dev_rows]
        self.scaler = StandardScaler().fit(X)
        self.clf = LogisticRegression(C=self.C, class_weight="balanced", max_iter=1000)
        self.clf.fit(self.scaler.transform(X), y)
        scores = self.score_rows(dev_rows)
        pos = sorted((s for s, t in zip(scores, y) if t), reverse=True)
        if not pos:
            raise ValueError("No human-labelled errors in dev split; cannot set a threshold.")
        k = max(0, math.ceil(self.target_recall * len(pos)) - 1)
        self.threshold = pos[k]
        return self

    def score_rows(self, rows: list[dict]) -> list[float]:
        X = [[build_features(r, self.dur)[k] for k in FEATURES] for r in rows]
        return list(self.clf.predict_proba(self.scaler.transform(X))[:, 1])

    def predict(self, rows: list[dict]) -> list[int]:
        return [int(s >= self.threshold) for s in self.score_rows(rows)]

    def coefficients(self) -> dict:
        return dict(zip(FEATURES, self.clf.coef_[0].round(3).tolist()))
