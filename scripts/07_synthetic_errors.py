"""Synthetic benchmark: deliberately corrupt the TEXT given to the TTS model, keep the original
prompt as the reference, and check whether the detectors notice. Ground truth is the intended
corruption, known by construction. Resumable. This measures sensitivity to deliberate content
changes, not the rate of naturally occurring TTS errors.

  python scripts/07_synthetic_errors.py --generate --n-prompts 40   # TTS + transcribe (GPU)
  python scripts/07_synthetic_errors.py --evaluate                  # writes outputs/synthetic_results.md
"""
import argparse
import json
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tts_verifier.align import tag_errors  # noqa: E402
from tts_verifier.corrupt import KINDS, corrupt  # noqa: E402
from tts_verifier.detectors import asr_features, whisper_only_predict  # noqa: E402
from tts_verifier.io import DATA, OUT, append_jsonl, read_jsonl  # noqa: E402
from tts_verifier.metrics import grouped_bootstrap_ci  # noqa: E402
from tts_verifier.normalize import normalize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--generate", action="store_true")
ap.add_argument("--evaluate", action="store_true")
ap.add_argument("--n-prompts", type=int, default=40)
ap.add_argument("--models", nargs="+", default=["xtts_v2", "f5_tts"])
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

SYN = OUT / "synthetic"
LOG = SYN / "generations.jsonl"
TR = SYN / "transcripts.jsonl"

if args.generate:
    from tts_verifier.models import duration_seconds, load_tts, transcribe_ctc, transcribe_whisper
    prompts = read_jsonl(DATA / "prompts.jsonl")
    rng = random.Random(2024)
    vocab = sorted({w.strip(".,;:!?").lower() for p in prompts for w in p["text"].split()
                    if w.strip(".,;:!?").isalpha() and len(w) >= 4})
    chosen = rng.sample(prompts, min(args.n_prompts, len(prompts)))
    jobs = []   # (prompt, kind, spoken_text); every prompt gets a clean control + each applicable kind
    for p in chosen:
        jobs.append((p, "clean", p["text"]))
        for kind in KINDS:
            spoken = corrupt(p["text"], kind, random.Random(f"{p['id']}-{kind}"), vocab)
            if spoken:
                jobs.append((p, kind, spoken))
    done = {g["clip_id"] for g in read_jsonl(LOG)}
    for name in args.models:
        tts = load_tts(name)
        for p, kind, spoken in jobs:
            clip_id = f"syn__{name}__{p['id']}__{kind}"
            if clip_id in done:
                continue
            wav = SYN / "clips" / name / f"{p['id']}__{kind}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            lat = tts.synthesize(spoken, wav, seed=args.seed)
            append_jsonl(LOG, {"clip_id": clip_id, "prompt_id": p["id"], "tts_model": name,
                               "corruption": kind, "spoken_text": spoken,
                               "wav": str(wav.relative_to(OUT)),
                               "duration_s": duration_seconds(str(wav)), "gen_latency_s": lat})
            print(clip_id)
    tdone = {t["clip_id"] for t in read_jsonl(TR)}
    for g in read_jsonl(LOG):
        if g["clip_id"] not in tdone:
            w = str(OUT / g["wav"])
            append_jsonl(TR, {"clip_id": g["clip_id"], "whisper_text": transcribe_whisper(w),
                              "ctc_text": transcribe_ctc(w), "ctc_pad_s": 0.5})

if args.evaluate:
    prompts = {p["id"]: p for p in read_jsonl(DATA / "prompts.jsonl")}
    tr = {t["clip_id"]: t for t in read_jsonl(TR)}
    rows = []
    for g in read_jsonl(LOG):
        t = tr.get(g["clip_id"])
        if t is None:
            continue
        p = prompts[g["prompt_id"]]
        r = {**g, "prompt": p["text"], "group": p["group"], "category": p["category"],
             "whisper_text": t["whisper_text"], "ctc_text": t["ctc_text"], "ctc_text_nopad": t.get("ctc_text_nopad")}
        r["intended_error"] = int(g["corruption"] != "clean")
        r["pred_whisper"] = whisper_only_predict(r)
        r["pred_ctc"] = asr_features(r["prompt"], r["ctc_text"], "c")["c_any_error"]
        r["pred_either"] = int(r["pred_whisper"] or r["pred_ctc"])
        # Did the TTS actually say the corrupted text? (CTC vs the text it was given.)
        r["read_faithfully"] = int(not tag_errors(normalize(g["spoken_text"]),
                                                  normalize(t["ctc_text"])).wer > 0.15)
        rows.append(r)
    if not rows:
        sys.exit("No synthetic clips. Run with --generate first.")
    dets = ["pred_whisper", "pred_ctc", "pred_either"]
    pkl = OUT / "detector.pkl"
    if pkl.exists():   # combined detector trained on HUMAN labels, applied here unchanged
        det = pickle.load(pkl.open("rb"))
        for r, p in zip(rows, det.predict(rows)):
            r["pred_combined"] = p
        dets.append("pred_combined")

    def rate(rs, k):
        return sum(r[k] for r in rs) / len(rs) if rs else float("nan")

    def cell(rs, k):
        if not rs:
            return "n/a"
        p, lo, hi = grouped_bootstrap_ci(rs, "group", lambda x: rate(x, k))
        return f"{100 * p:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]"

    md = ["# Synthetic benchmark (NOT human-verified)\n",
          "Text was deliberately corrupted before synthesis; the original prompt is the reference. "
          "Ground truth is the intended corruption. This tests sensitivity to deliberate content "
          "changes, not performance on naturally occurring TTS errors. The TTS model may add its own errors.\n",
          f"Clips: {len(rows)}; corrupted: {sum(r['intended_error'] for r in rows)}; clean controls: "
          f"{sum(1 - r['intended_error'] for r in rows)}\n",
          "## Detection rate by corruption type (95% CI over prompt groups)\n",
          "| TTS model | Corruption | n | " + " | ".join(d.replace("pred_", "") for d in dets) + " |",
          "|---|---|---|" + "---|" * len(dets)]
    for m in sorted({r["tts_model"] for r in rows}):
        for kind in ("clean",) + KINDS:
            rs = [r for r in rows if r["tts_model"] == m and r["corruption"] == kind]
            if rs:
                label = "clean (false-alarm rate)" if kind == "clean" else f"{kind} (recall)"
                md.append(f"| {m} | {label} | {len(rs)} | " + " | ".join(cell(rs, d) for d in dets) + " |")
    corr = [r for r in rows if r["intended_error"]]
    md.append(f"\nCorrupted clips the TTS read as given (CTC WER <= 15% vs spoken text): "
              f"{100 * rate(corr, 'read_faithfully'):.0f}%. The rest contain extra TTS errors, "
              "so recall on them is not purely about the intended corruption.\n")
    clean_np = [r for r in rows if r["corruption"] == "clean" and r["ctc_text_nopad"]]
    if clean_np:
        before = sum(asr_features(r["prompt"], r["ctc_text_nopad"], "c")["c_any_error"] for r in clean_np) / len(clean_np)
        after = sum(r["pred_ctc"] for r in clean_np) / len(clean_np)
        md.append(f"CTC false-alarm rate on clean clips without vs with 0.5 s leading silence: "
                  f"{100 * before:.0f}% -> {100 * after:.0f}% (n={len(clean_np)}).\n")
    (OUT / "synthetic_results.md").write_text("\n".join(md) + "\n")
    (OUT / "synthetic_predictions.json").write_text(json.dumps(
        [{k: r[k] for k in ["clip_id", "corruption", "spoken_text", "whisper_text", "ctc_text"] + dets}
         for r in rows], indent=1))
    print("\n".join(md))
