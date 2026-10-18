# Catching content errors in generated speech

A verifier for text-to-speech **omissions, repetitions and substitutions**, evaluated against blind
human listening labels, plus a test of whether using it to guide regeneration reduces errors.

> **Status:** pipeline complete and tested on synthetic data. Real results pending
> (`outputs/results.md`, `outputs/regen_results.md`).

## Why this matters

Modern TTS sounds natural but can still say the wrong *content*: skip a phrase, loop a word, cut off
the ending or swap a name. In audiobooks, voice agents and dubbing, those errors are worse than a
slightly unnatural voice, and they are easy to miss at scale.

The standard automatic check is "transcribe with ASR, compute WER". That conflates two things:

1. errors the **TTS model** made, and
2. errors the **ASR model** made, or *hid*. Whisper's decoder is a strong language model and can
   "repair" a skipped or garbled word back into the expected sentence.

This project separates the two, using human listening labels as the reference.

## Research questions

| | Question | How it is measured |
|---|---|---|
| RQ1 | How well does Whisper alone detect TTS content errors? | Precision, recall and false-alarm rate vs blind human labels, on held-out prompts |
| RQ2 | Does adding a weak-LM recogniser (greedy CTC wav2vec2) and an ASR-independent duration signal improve detection, especially of omissions? | Same metrics; exact McNemar test vs baseline; recall by error type |
| RQ3 | Does detector-guided regeneration reduce human-verified errors, and at what latency cost? | Error rate before vs after the policy on the same prompts; latency overhead |

"Whisper hides TTS errors" is a **hypothesis, not an assumption**. The analysis is set up so it can be
rejected.

## Design

```
prompts (100, 4 categories, template groups)
   │  group-level stratified dev/test split (no near-duplicates across splits)
   ▼
TTS: XTTS-v2 (autoregressive)  ·  F5-TTS (non-autoregressive)   — same reference voice
   ▼
ASR: Whisper large-v3-turbo (strong LM)  ·  wav2vec2 greedy CTC (no LM)
   ▼
normalise both sides (explicit rules) → word alignment → tagged suspected errors
   ▼
detectors: Whisper-only baseline  vs  combined (Whisper + CTC + duration ratio, fitted on dev)
   ▼
evaluate vs blind human labels (primary + second listener on 20%, Cohen's κ)
   ▼
intervention: flag → resample (≤3 retries) → re-check;  human-label the kept clips
```

**Measurement decisions**
- **Ground truth = human listening**, recorded blind: clips are renamed to opaque keys, shuffled,
  and shown with the prompt text only (no model name, transcript or prediction). See
  [`labels/LABELING.md`](labels/LABELING.md).
- **Forced alignment and ASR are never ground truth.** They locate *suspected* errors only.
- **Normalisation is explicit and tested** (`£25` = "twenty-five pounds", `QX7` = "Q X 7",
  "two hundred (and) fifty"). Rules: [`docs/NORMALISATION.md`](docs/NORMALISATION.md).
- **No leakage:** the combined detector's weights, duration model and threshold are fitted on dev;
  headline numbers are test-only. The split is by template group and stratified by category.
- **Uncertainty:** 95% CIs by bootstrap over template groups (clips within a group are correlated).

**Prompt categories:** `ordinary` (control), `long` (multi-sentence passages), `numbers_names`
(currency, times, years, codes, uncommon names) and `repetition_prone` (repeated words, lists,
tongue-twisters). All prompts are original text.

## Results

_To be filled from `outputs/results.md` and `outputs/regen_results.md`:_

- Human-verified error rate by model × category
- Detection on test prompts: Whisper-only vs CTC-only vs combined
- Recall by error type: does Whisper miss omissions that CTC catches?
- Regeneration: error-rate change [95% CI] and latency overhead
- 3–5 audio examples of the most instructive failures

## Run it

GPU steps run on a free Colab T4 (`notebooks/run_colab.ipynb`). Labelling and evaluation run on a laptop.

```bash
pip install -r requirements.txt            # core + evaluation (CPU)
pip install -r requirements-gpu.txt        # TTS + ASR (GPU)

python scripts/01_make_prompts.py          # data/prompts.jsonl
python scripts/02_generate.py --limit 3    # smoke test: listen before the full run
python scripts/02_generate.py              # 200 clips
python scripts/03_transcribe.py            # Whisper + CTC transcripts
python scripts/04_build_labeler.py         # outputs/labeler/{primary,second}.html  → label blind
#   save exported CSVs to labels/primary.csv and labels/second.csv
python scripts/05_evaluate.py              # outputs/results.{md,json}, detector.pkl
python scripts/06_regenerate.py --run      # intervention on test prompts (GPU)
python scripts/04_build_labeler.py --source regen   # label the kept clips, append to primary.csv
python scripts/06_regenerate.py --evaluate # outputs/regen_results.md

python -m unittest discover tests          # normalisation, alignment, metrics, prompts
```

## Limitations

- **Small scale:** 100 prompts × 2 models, one reference voice, English only. CIs are wide by design;
  read effect sizes, not point estimates.
- **One primary annotator** (the author); a second listener covers ~20% for reliability.
- **Pronunciation quality is out of scope:** a recognisable but oddly pronounced word is not a content error.
- **The regeneration policy is a sampling intervention, not training.** It shows the verifier is
  useful as a filter or reranker, but does not change the model. A natural next step is using the
  verifier to filter synthetic training data, then fine-tuning and measuring the effect.
- Model licences: XTTS-v2 is released under the Coqui Public Model License (non-commercial).
  Check each model's licence before reusing generated audio.

## Repository layout

```
src/tts_verifier/   normalize · align · detectors · metrics · prompts · models · dataset
scripts/            01–06 pipeline steps
labels/             LABELING.md, primary.csv, second.csv
docs/               NORMALISATION.md
tests/              unit tests (no GPU needed)
outputs/            generated clips, transcripts, results (git-ignored except results.md)
```
