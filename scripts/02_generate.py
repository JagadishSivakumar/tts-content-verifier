"""Generate one clip per (prompt, TTS model) at a fixed seed. Resumable: skips existing clips.

Usage: python scripts/02_generate.py --models xtts_v2 f5_tts --seed 0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tts_verifier.io import DATA, OUT, append_jsonl, read_jsonl  # noqa: E402
from tts_verifier.models import duration_seconds, load_tts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["xtts_v2", "f5_tts"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=None, help="first N prompts only (smoke test)")
args = ap.parse_args()

prompts = read_jsonl(DATA / "prompts.jsonl")[: args.limit]
log = OUT / "generations.jsonl"
done = {g["clip_id"] for g in read_jsonl(log)}

for name in args.models:
    tts = load_tts(name)
    for p in prompts:
        clip_id = f"{name}__{p['id']}__s{args.seed}"
        if clip_id in done:
            continue
        wav = OUT / "clips" / name / f"{p['id']}__s{args.seed}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        latency = tts.synthesize(p["text"], wav, seed=args.seed)
        append_jsonl(log, {"clip_id": clip_id, "prompt_id": p["id"], "tts_model": name,
                           "seed": args.seed, "attempt": 0, "wav": str(wav.relative_to(OUT)),
                           "duration_s": duration_seconds(str(wav)), "gen_latency_s": latency})
        print(f"{clip_id}  {latency:.1f}s")
