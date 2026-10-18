"""Write data/prompts.jsonl (100 prompts, group-level stratified dev/test split)."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tts_verifier.io import DATA  # noqa: E402
from tts_verifier.prompts import write_prompts  # noqa: E402

prompts = write_prompts(DATA / "prompts.jsonl")
print(f"Wrote {len(prompts)} prompts to {DATA / 'prompts.jsonl'}")
for (cat, split), n in sorted(Counter((p["category"], p["split"]) for p in prompts).items()):
    print(f"  {cat:18s} {split:4s} {n}")
