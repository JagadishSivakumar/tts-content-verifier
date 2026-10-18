"""Transcribe every generated clip with Whisper and with greedy CTC. Resumable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tts_verifier.io import OUT, append_jsonl, read_jsonl  # noqa: E402
from tts_verifier.models import transcribe_ctc, transcribe_whisper  # noqa: E402

out = OUT / "transcripts.jsonl"
done = {t["clip_id"] for t in read_jsonl(out)}
gens = read_jsonl(OUT / "generations.jsonl") + read_jsonl(OUT / "regen.jsonl")
for g in gens:
    if g["clip_id"] in done:
        continue
    wav = str(OUT / g["wav"])
    append_jsonl(out, {"clip_id": g["clip_id"], "whisper_text": transcribe_whisper(wav),
                       "ctc_text": transcribe_ctc(wav)})
    done.add(g["clip_id"])
    print(g["clip_id"])
