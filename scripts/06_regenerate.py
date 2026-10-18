"""Intervention: detector-guided regeneration on TEST prompts, then its evaluation.

Policy per (model, prompt): take the original clip; if the combined detector flags it, resample with
a new seed and re-check, up to --max-retries times; keep the first unflagged clip (or the last try).

Step 1 (GPU):   python scripts/06_regenerate.py --run
Step 2 (human): python scripts/04_build_labeler.py --source regen   -> label final clips blind
Step 3 (CPU):   python scripts/06_regenerate.py --evaluate

Outcome: human-verified error rate of original clips vs policy-selected clips on the same prompts,
with the latency cost (verification on every clip + regeneration on flagged clips).
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tts_verifier.dataset import clip_key  # noqa: E402
from tts_verifier.io import DATA, LABELS, OUT, append_jsonl, read_jsonl, read_labels  # noqa: E402
from tts_verifier.metrics import grouped_bootstrap_ci  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="store_true")
ap.add_argument("--evaluate", action="store_true")
ap.add_argument("--max-retries", type=int, default=3)
args = ap.parse_args()

prompts = {p["id"]: p for p in read_jsonl(DATA / "prompts.jsonl")}
policy_file = OUT / "regen_policy.jsonl"

if args.run:
    from tts_verifier.models import duration_seconds, load_tts, transcribe_ctc, transcribe_whisper

    with (OUT / "detector.pkl").open("rb") as f:
        det = pickle.load(f)
    gens = [g for g in read_jsonl(OUT / "generations.jsonl") if prompts[g["prompt_id"]]["split"] == "test"]
    done = {(p["tts_model"], p["prompt_id"]) for p in read_jsonl(policy_file)}

    def check(g: dict) -> tuple[int, float]:
        t0 = time.perf_counter()
        wav = str(OUT / g["wav"])
        row = {**g, "prompt": prompts[g["prompt_id"]]["text"],
               "whisper_text": transcribe_whisper(wav), "ctc_text": transcribe_ctc(wav)}
        flagged = det.predict([row])[0]
        if g.get("attempt", 0) > 0:   # store transcripts of new clips for later analysis
            append_jsonl(OUT / "transcripts.jsonl", {"clip_id": g["clip_id"],
                         "whisper_text": row["whisper_text"], "ctc_text": row["ctc_text"]})
        return flagged, time.perf_counter() - t0

    for name in sorted({g["tts_model"] for g in gens}):
        tts = load_tts(name)
        for g in [g for g in gens if g["tts_model"] == name]:
            if (name, g["prompt_id"]) in done:
                continue
            flagged, verify_s = check(g)
            cur, attempts, regen_s = g, 0, 0.0
            while flagged and attempts < args.max_retries:
                attempts += 1
                seed = 1000 + attempts
                wav = OUT / "clips" / name / f"{g['prompt_id']}__s{seed}.wav"
                lat = tts.synthesize(prompts[g["prompt_id"]]["text"], wav, seed=seed)
                cur = {"clip_id": f"{name}__{g['prompt_id']}__s{seed}", "prompt_id": g["prompt_id"],
                       "tts_model": name, "seed": seed, "attempt": attempts,
                       "wav": str(wav.relative_to(OUT)), "duration_s": duration_seconds(str(wav)),
                       "gen_latency_s": lat}
                append_jsonl(OUT / "regen.jsonl", cur)
                flagged, v = check(cur)
                regen_s += lat
                verify_s += v
            append_jsonl(policy_file, {"tts_model": name, "prompt_id": g["prompt_id"],
                                       "original_clip": g["clip_id"], "final_clip": cur["clip_id"],
                                       "retries": attempts, "still_flagged": int(flagged),
                                       "orig_gen_s": g["gen_latency_s"], "verify_s": verify_s,
                                       "regen_s": regen_s})
            print(f"{name} {g['prompt_id']}: retries={attempts} still_flagged={flagged}")

if args.evaluate:
    labels = read_labels(LABELS / "primary.csv")
    rows = []
    for p in read_jsonl(policy_file):
        o, fnl = labels.get(clip_key(p["original_clip"])), labels.get(clip_key(p["final_clip"]))
        if o is None or fnl is None:
            continue
        rows.append({**p, "group": prompts[p["prompt_id"]]["group"],
                     "orig_err": o["error"], "final_err": fnl["error"]})
    if not rows:
        sys.exit("No policy rows with labels for both original and final clips yet.")

    def err(key):
        return lambda rs: sum(r[key] for r in rs) / len(rs)

    out = {"n": len(rows)}
    md = [f"# Detector-guided regeneration (test prompts, n={len(rows)})\n",
          "| Model | n | Original error rate | After policy | Change [95% CI] | Mean retries | Latency overhead |",
          "|---|---|---|---|---|---|---|"]
    for model in ["all"] + sorted({r["tts_model"] for r in rows}):
        rs = [r for r in rows if model == "all" or r["tts_model"] == model]
        o = grouped_bootstrap_ci(rs, "group", err("orig_err"))
        f = grouped_bootstrap_ci(rs, "group", err("final_err"))
        d = grouped_bootstrap_ci(rs, "group", lambda x: err("final_err")(x) - err("orig_err")(x))
        retries = sum(r["retries"] for r in rs) / len(rs)
        base = sum(r["orig_gen_s"] for r in rs)
        overhead = sum(r["verify_s"] + r["regen_s"] for r in rs) / base if base else float("nan")
        out[model] = {"orig": o, "final": f, "delta": d, "mean_retries": retries,
                      "latency_overhead_x": overhead}
        md.append(f"| {model} | {len(rs)} | {100*o[0]:.0f}% | {100*f[0]:.0f}% | "
                  f"{100*d[0]:+.0f} pts [{100*d[1]:+.0f}, {100*d[2]:+.0f}] | {retries:.2f} | +{100*overhead:.0f}% |")
    md.append("\nLatency overhead = (ASR verification on every clip + regeneration time) / original generation time.")
    (OUT / "regen_results.json").write_text(json.dumps(out, indent=2))
    (OUT / "regen_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
