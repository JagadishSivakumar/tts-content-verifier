"""Evaluate detectors against blind human labels and write outputs/results.{json,md}.

Protocol
- Combined detector is fitted and its threshold chosen on DEV only; all headline numbers are TEST.
- 95% CIs are grouped bootstrap over prompt template groups.
- Baseline vs combined compared with an exact McNemar test on the same test clips.
- The "Whisper hides errors" analysis is descriptive, reported on all labelled clips per error type.
"""
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tts_verifier.dataset import load_rows  # noqa: E402
from tts_verifier.detectors import CombinedDetector, asr_features, whisper_only_predict  # noqa: E402
from tts_verifier.io import LABELS, OUT, read_labels  # noqa: E402
from tts_verifier.metrics import (cohens_kappa, detection_metrics, grouped_bootstrap_ci,  # noqa: E402
                                  mcnemar_exact)

rows = load_rows()
if not rows:
    sys.exit("No labelled rows. Generate, transcribe, label, then save labels/primary.csv.")
dev = [r for r in rows if r["split"] == "dev"]
test = [r for r in rows if r["split"] == "test"]
results: dict = {"n_clips": len(rows), "n_dev": len(dev), "n_test": len(test)}
md = [f"# Results\n\nLabelled clips: {len(rows)} (dev {len(dev)}, test {len(test)})\n"]


def rate(rs):
    return sum(r["human_error"] for r in rs) / len(rs) if rs else float("nan")


def fmt(p, lo, hi, pct=True):
    f = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"{x:.2f}")
    return f"{f(p)} [{f(lo)}, {f(hi)}]"


# 1. Inter-annotator agreement --------------------------------------------------------------------
primary, second = read_labels(LABELS / "primary.csv"), read_labels(LABELS / "second.csv")
both = sorted(set(primary) & set(second))
if both:
    k = cohens_kappa([primary[c]["error"] for c in both], [second[c]["error"] for c in both])
    agree = sum(primary[c]["error"] == second[c]["error"] for c in both) / len(both)
    results["agreement"] = {"n": len(both), "kappa": k, "raw": agree}
    md.append(f"## Label reliability\n\nSecond listener on {len(both)} clips: Cohen's κ = {k:.2f}, "
              f"raw agreement {100 * agree:.0f}%.\n")

# 2. Human-verified error rates by model x category -----------------------------------------------
md.append("## Human-verified error rate (all labelled clips)\n\n| Model | Category | n | Error rate [95% CI] |\n|---|---|---|---|")
results["error_rates"] = {}
for model in sorted({r["tts_model"] for r in rows}):
    for cat in ["all"] + sorted({r["category"] for r in rows}):
        rs = [r for r in rows if r["tts_model"] == model and (cat == "all" or r["category"] == cat)]
        if not rs:
            continue
        p, lo, hi = grouped_bootstrap_ci(rs, "group", rate)
        results["error_rates"][f"{model}/{cat}"] = {"n": len(rs), "rate": p, "ci": [lo, hi]}
        md.append(f"| {model} | {cat} | {len(rs)} | {fmt(p, lo, hi)} |")
types = Counter(t for r in rows for t in r["human_types"])
results["human_error_types"] = dict(types)
md.append(f"\nHuman-labelled error types: {dict(types)}\n")

# 3. Detection on TEST ----------------------------------------------------------------------------
if sum(r["human_error"] for r in dev) == 0:
    sys.exit("Dev split has no human-labelled errors; cannot fit the combined detector.")
det = CombinedDetector(target_recall=0.9).fit(dev)
for r in rows:
    r["pred_whisper"] = whisper_only_predict(r)
    r["pred_ctc"] = asr_features(r["prompt"], r["ctc_text"], "c")["c_any_error"]
for r, s, p in zip(rows, det.score_rows(rows), det.predict(rows)):
    r["score_combined"], r["pred_combined"] = s, p

md.append("## Detection on held-out test prompts\n\n| Detector | Precision | Recall | False-alarm rate |\n|---|---|---|---|")
results["detection_test"] = {}
for name in ["pred_whisper", "pred_ctc", "pred_combined"]:
    m = detection_metrics([r["human_error"] for r in test], [r[name] for r in test])
    cis = {}
    for metric in ["precision", "recall", "false_alarm_rate"]:
        cis[metric] = grouped_bootstrap_ci(
            test, "group", lambda rs, metric=metric, name=name: detection_metrics(
                [r["human_error"] for r in rs], [r[name] for r in rs])[metric])
    results["detection_test"][name] = {**m, "ci": {k: list(v) for k, v in cis.items()}}
    md.append(f"| {name.replace('pred_', '')} | {fmt(*cis['precision'])} | {fmt(*cis['recall'])} | "
              f"{fmt(*cis['false_alarm_rate'])} |")
mc = mcnemar_exact([r["human_error"] for r in test], [r["pred_whisper"] for r in test],
                   [r["pred_combined"] for r in test])
results["mcnemar_whisper_vs_combined"] = mc
results["combined_coefficients"] = det.coefficients()
results["combined_threshold"] = det.threshold
md.append(f"\nMcNemar (Whisper-only vs combined, test): p = {mc['p_value']:.3f} "
          f"({mc['a_only_correct']} clips only baseline right, {mc['b_only_correct']} only combined right).\n")

# 4. Does Whisper hide TTS errors? ----------------------------------------------------------------
md.append("## Does Whisper's language model hide TTS errors?\n\nRecall on human-labelled errors, by error type "
          "(all labelled clips; descriptive).\n\n| Error type | n | Whisper recall | CTC recall |\n|---|---|---|---|")
results["recall_by_type"] = {}
by_type = defaultdict(list)
for r in rows:
    for t in r["human_types"] or (["unspecified"] if r["human_error"] else []):
        by_type[t].append(r)
for t, rs in sorted(by_type.items()):
    w = sum(r["pred_whisper"] for r in rs) / len(rs)
    c = sum(r["pred_ctc"] for r in rs) / len(rs)
    results["recall_by_type"][t] = {"n": len(rs), "whisper": w, "ctc": c}
    md.append(f"| {t} | {len(rs)} | {100 * w:.0f}% | {100 * c:.0f}% |")
hidden = [r for r in rows if r["human_error"] and not r["pred_whisper"] and r["pred_ctc"]]
results["hidden_by_whisper_caught_by_ctc"] = [
    {"clip_id": r["clip_id"], "prompt": r["prompt"], "whisper": r["whisper_text"], "ctc": r["ctc_text"]}
    for r in hidden]
md.append(f"\n{len(hidden)} human-verified errors were missed by Whisper but caught by CTC "
          "(examples in results.json → hidden_by_whisper_caught_by_ctc).\n")

# 5. Error analysis: false alarms of the combined detector on clean test clips ----------------
fa = [r for r in test if r["pred_combined"] and not r["human_error"]]
results["combined_false_alarms_test"] = [
    {"clip_id": r["clip_id"], "prompt": r["prompt"], "whisper": r["whisper_text"], "ctc": r["ctc_text"]}
    for r in fa]

(OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
(OUT / "results.md").write_text("\n".join(md) + "\n")
with (OUT / "detector.pkl").open("wb") as f:
    pickle.dump(det, f)
# Per-clip predictions for inspection (no audio paths needed).
(OUT / "predictions.jsonl").write_text("\n".join(json.dumps(
    {k: r[k] for k in ["clip_id", "split", "category", "tts_model", "human_error", "human_types",
                       "pred_whisper", "pred_ctc", "pred_combined", "score_combined"]}) for r in rows))
print("\n".join(md))
