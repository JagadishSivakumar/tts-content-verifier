"""An audio LLM (Gemini) as an automated judge of the natural clips.

  --judge : Gemini hears each clip with its target text (no TTS model name, no ASR output),
            transcribes it verbatim and labels it clean or error -> labels/gemini.csv.
  --agree : agreement (Cohen's kappa, recall) between Gemini and the human labels on the clips that
            have both, plus a direct check of the detectors on the human sample
            -> outputs/gemini_agreement.md.
Then: python scripts/05_evaluate.py --labels gemini.csv --tag _gemini

Judge labels are automated and are reported separately from human labels. Requires GEMINI_API_KEY.
The free tier allows 20 requests per model per day; --gap sets the pause between calls. Resumable.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tts_verifier.dataset import clip_key, load_rows  # noqa: E402
from tts_verifier.detectors import asr_features, whisper_only_predict  # noqa: E402
from tts_verifier.io import DATA, LABELS, OUT, read_jsonl, read_labels  # noqa: E402
from tts_verifier.metrics import cohens_kappa  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--judge", action="store_true")
ap.add_argument("--agree", action="store_true")
ap.add_argument("--model", default="gemini-3.8-flash")
ap.add_argument("--gap", type=float, default=4.5, help="seconds between calls (free-tier rate limit)")
ap.add_argument("--only-labelled", action="store_true",
                help="judge only clips with a human label (cheap, enough to measure agreement)")
args = ap.parse_args()

PROMPT = """You are checking a text-to-speech clip. Listen to the audio and write down EXACTLY what is
spoken, word for word. Do not correct, complete or smooth anything: if a word is missing, repeated,
replaced or cut off, write it as you hear it.

Target text: "{text}"

Then compare what you heard with the target text. Return JSON only:
{{"heard": "<verbatim words you heard>",
  "error": 0 or 1,   // 1 if any target word is missing, repeated, replaced, cut off, or an extra word is spoken
  "types": [subset of "omission","repetition","substitution","truncation","insertion"]}}
Different but valid readings of numbers, times or codes are NOT errors. Odd stress or accent is NOT an error."""

if args.judge:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompts = {p["id"]: p for p in read_jsonl(DATA / "prompts.jsonl")}
    human = read_labels(LABELS / "primary.csv")
    out_path, raw_path = LABELS / "gemini.csv", OUT / "gemini_raw.jsonl"
    done = {r["clip_key"] for r in read_jsonl(raw_path)}
    fails = 0
    for g in read_jsonl(OUT / "generations.jsonl"):
        k = clip_key(g["clip_id"])
        if k in done or (args.only_labelled and k not in human):
            continue
        wav = (OUT / g["wav"]).read_bytes()
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=args.model,
                    contents=[types.Part.from_bytes(data=wav, mime_type="audio/wav"),
                              PROMPT.format(text=prompts[g["prompt_id"]]["text"])],
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
                j = json.loads(resp.text)
                if isinstance(j, list):
                    j = j[0]
                j["error"] = int(j["error"])   # malformed answer -> retry
                break
            except Exception as e:   # rate limits / bad JSON: back off and retry
                last_err = e
                print("retry", k, repr(e)[:120]); time.sleep(5 * (attempt + 1)); j = None
        if j is None:
            fails += 1
            if fails >= 3:
                sys.exit(f"Stopping: 3 clips in a row failed. Last error: {last_err!r}\n"
                         "Check the API key and --model name, or wait if it is a rate limit, then re-run (it resumes).")
            continue
        fails = 0
        with raw_path.open("a") as f:
            f.write(json.dumps({"clip_key": k, "model": args.model, **j}) + "\n")
        print(k, j["error"])
        time.sleep(args.gap)
    rows = read_jsonl(raw_path)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["clip_key", "error", "types", "notes"])
        for r in rows:
            w.writerow([r["clip_key"], int(r["error"]), ";".join(r.get("types", [])), "gemini: " + r.get("heard", "")])
    print(f"wrote {out_path} ({len(rows)} clips)")

if args.agree:
    h, gm = read_labels(LABELS / "primary.csv"), read_labels(LABELS / "gemini.csv")
    both = sorted(set(h) & set(gm))
    if not both:
        sys.exit("No overlap between human and Gemini labels.")
    a, b = [h[k]["error"] for k in both], [gm[k]["error"] for k in both]
    tp = sum(x == 1 and y == 1 for x, y in zip(a, b)); fn = sum(x == 1 and y == 0 for x, y in zip(a, b))
    fp = sum(x == 0 and y == 1 for x, y in zip(a, b))
    msg = (f"# Gemini judge vs human labels\n\nOverlap: {len(both)} clips. Cohen's kappa = {cohens_kappa(a, b):.2f}; "
           f"raw agreement {100 * sum(x == y for x, y in zip(a, b)) / len(both):.0f}%.\n\n"
           f"Of {tp + fn} human-labelled errors Gemini found {tp} (recall {tp / max(tp + fn, 1):.0%}); "
           f"it flagged {fp} clips humans called clean.\n")
    # The same human sample also gives a small, direct check of the detectors (descriptive only).
    hr = load_rows(labels_file="primary.csv")
    errs, clean = [r for r in hr if r["human_error"]], [r for r in hr if not r["human_error"]]
    msg += "\n## Detectors vs the same human labels (small sample, descriptive)\n\n| Detector | Recall on human errors | False alarms on human-clean |\n|---|---|---|\n"
    for name, f in [("whisper", whisper_only_predict),
                    ("ctc", lambda r: asr_features(r["prompt"], r["ctc_text"], "c")["c_any_error"])]:
        rec = sum(f(r) for r in errs) / len(errs) if errs else float("nan")
        fa = sum(f(r) for r in clean) / len(clean) if clean else float("nan")
        msg += f"| {name} | {100 * rec:.0f}% ({len(errs)} errors) | {100 * fa:.0f}% ({len(clean)} clean) |\n"
    (OUT / "gemini_agreement.md").write_text(msg); print(msg)
