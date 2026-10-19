"""Re-run the CTC recogniser on every clip with 0.5 s of leading silence.

wav2vec2 often garbles or drops the first word of a clip that starts abruptly, which inflated the
false-alarm rate on clean clips. Transcripts made before padding was added are re-transcribed; the
original output is kept as `ctc_text_nopad` so the effect can be reported. Clips already
transcribed with padding are skipped. Resumable.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tts_verifier.io import OUT, read_jsonl, write_jsonl  # noqa: E402
from tts_verifier.models import transcribe_ctc  # noqa: E402

wavs = {}
for f in ["generations.jsonl", "synthetic/generations.jsonl"]:
    for g in read_jsonl(OUT / f):
        wavs[g["clip_id"]] = g["wav"]

for tf in ["transcripts.jsonl", "synthetic/transcripts.jsonl"]:
    rows = read_jsonl(OUT / tf)
    changed = 0
    for r in rows:
        if "ctc_text_nopad" in r or "ctc_pad_s" in r or r["clip_id"] not in wavs:
            continue
        r["ctc_text_nopad"] = r["ctc_text"]
        r["ctc_text"] = transcribe_ctc(str(OUT / wavs[r["clip_id"]]), lead_silence_s=0.5)
        r["ctc_pad_s"] = 0.5
        changed += 1
        if changed % 50 == 0:
            write_jsonl(OUT / tf, rows); print(tf, changed)
    write_jsonl(OUT / tf, rows)
    print(f"{tf}: re-transcribed {changed} of {len(rows)}")
