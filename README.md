# Catching content errors in generated speech

How well do ASR-based checks catch text-to-speech **content errors** (skipped, repeated, swapped or
cut-off words), and does Whisper's built-in language model hide some of them? Evaluated on two
open TTS models with three sources of ground truth: deliberate corruptions with known answers, blind
human listening, and an audio-LLM judge validated against the human labels.

## Why this matters

Modern TTS sounds natural but can still say the wrong *content*. In audiobooks, voice agents and
dubbing those errors matter more than a slightly unnatural voice, and they are easy to miss at scale.
The usual automatic check is "transcribe with ASR, compare with the script". That mixes up errors the
TTS model made with errors the ASR model made, or *hid*: Whisper's decoder is a strong language model
and can "repair" a garbled word back into the expected sentence. This project measures that effect
against a recogniser with no language model (greedy CTC wav2vec2).

## Key findings

1. **The two TTS architectures fail in different places.** Grouping the 40 judge-labelled errors by
   where they occur (manual categorisation of the judge's verbatim transcripts):
   - **XTTS-v2 (autoregressive): 13 of 23 errors sit at sentence boundaries.** It drops the first word
     of a sentence (7 clips: "The forecast for the coast…" → "forecast for the coast…", mid-passage
     "A conductor walked…" → "Conductor walked…") or cuts off the last word (6 clips: "double booked"
     → "double book", "agree on the reason." → "agree on the reas."). This is consistent with XTTS
     synthesising long text sentence by sentence; that mechanism is a hypothesis, not tested here.
   - **F5-TTS (non-autoregressive): 15 of 17 errors are misread symbols, abbreviations and digits**:
     currency symbols dropped ("£25" → "25", "$3.50" → "3.50"), "St" → "sate", "ml" → "mel",
     "65,012" → "65 zed or 12". These are text-normalisation failures rather than acoustic ones.
     XTTS-v2 has some too ("St" → "Saint", a final "." read aloud as "dot").
   - Human listening confirmed one example of each pattern (XTTS dropping "The"; F5 dropping "dollars"
     and misreading "€480").
2. **Whisper hides TTS misreadings of numbers and abbreviations by writing them back in canonical
   form.** The TTS said "thirty mel" (CTC heard "thirty mell"); Whisper wrote "30 ml". CTC heard
   "forty eight three hundred" where Whisper wrote "48,300". All 4 judge-labelled errors that Whisper missed
   but CTC caught are of this kind, and Whisper's recall on substitutions was 78% against CTC's 94%.
   Whisper's inverse text normalisation, a feature for ASR users, is a blind spot for TTS checking.
3. **On deliberate errors, Whisper misses more repetitions.** On 240 deliberately corrupted clips both
   recognisers caught 100% of omissions and substitutions. For repeated words on XTTS-v2, Whisper
   caught 85% [70, 96] and CTC 98% [93, 100]; flagging when either fires reached 100%. Consistent with
   Whisper's language model collapsing "the the" into "the". Small sample: read as a direction.
4. **Both recognisers miss cut-off words.** They caught only 22% (Whisper) and 33% (CTC) of the 9
   judge-labelled truncations: both complete "double book" to "double booked". An ASR-free check on
   word endings or clip ends is the obvious next step.
5. **Combining the recognisers did not beat Whisper alone on natural clips** (McNemar p = 1.0;
   precision about 45% at 56–62% recall for all detectors). A negative result, reported as such.
   Precision is probably understated: 6 of the combined detector's 11 test "false alarms" are clips
   with unusual names, where the judge appears to echo the target text (see below).
6. **Fixing the measurement mattered as much as the detector.** Spelling variants (grey/gray), split
   compounds (wrist watches) and CTC garbling the first word of abruptly-starting clips produced most
   early false alarms. Adding 0.5 s of leading silence before CTC cut its false-alarm rate on clean
   synthetic clips from 32% to 20%, while omission recall stayed at 100%.

## Evaluation design: where the ground truth comes from

Hand-labelling every clip was not feasible, so three sources are used, each covering a weakness of
another.

| Source | Clips | Ground truth | Used for |
|---|---|---|---|
| **A. Synthetic corruptions** | 320 (40 prompts × clean/omission/repetition/substitution × 2 models) | The text is corrupted *before* synthesis; the original prompt stays the reference. Known by construction. | Detector sensitivity per error type; false-alarm rate on clean controls |
| **B. Blind human labels** | 40 natural clips, stratified by category × model, fixed seed | Labelled by ear by the author, seeing only the prompt and audio | Validating the judge; small direct check of detectors |
| **C. Gemini as audio judge** | All 200 natural clips | Gemini hears the clip with the target text, transcribes verbatim, labels clean/error. **Automated, not human.** | Error rates and detector evaluation on natural clips |

**How far the judge can be trusted.** On the 40 human-labelled clips Gemini agreed 95% of the time
(Cohen's κ = 0.72): it found 3 of the 4 human-found errors and flagged 1 clip the human called clean.
That is substantial agreement, but it rests on only 4 errors, so the recall estimate is very loose
(exact 95% CI 19–99%). Two concrete weaknesses:
- **Echoing the target text on unusual names.** For 7 of the 8 name-heavy clips (e.g. "Siobhan, Nguyen Thi
  Lan and Wojciech"), Gemini's "verbatim" transcript reproduced the prompt's spelling exactly, while
  Whisper heard "Shovan, Anguian, Thilan and Wojcik". Gemini missed the one name-clip error the human
  found (an inserted "and"). Judge-labelled error rates for names are therefore likely too low.
- **One unresolved disagreement:** a long XTTS-v2 clip the human labelled clean, where Gemini heard
  "the reas." (cut-off "reason"). Gemini flagged the same cut-off in two other XTTS-v2 renderings of
  that sentence, so the human label may be the miss. The blind human label is kept as recorded.

Every natural-clip number in this README that depends on the judge is labelled "judge-labelled".

**Human sample.** 4 of 40 clips had errors (10%, exact 95% CI 3–24%): an inserted "and" in a list of
names (XTTS-v2), a misread "€480" and a dropped "dollars" (both F5-TTS), and a dropped initial "The"
(XTTS-v2). Whisper flagged all 4 with 1 false alarm on 36 clean clips; CTC flagged all 4 with 6 false
alarms.

## Results

### A. Synthetic benchmark (known ground truth)

Detection rate; 95% CI by bootstrap over prompt template groups. Clean rows are false-alarm rates.

| TTS model | Corruption | n | Whisper | CTC | Either |
|---|---|---|---|---|---|
| F5-TTS | clean (false alarm) | 40 | 18% [5, 35] | 18% [5, 35] | 22% [7, 42] |
| F5-TTS | omission | 40 | 100% | 100% | 100% |
| F5-TTS | repetition | 40 | 88% [77, 96] | 85% [71, 96] | 90% [81, 98] |
| F5-TTS | substitution | 40 | 100% | 100% | 100% |
| XTTS-v2 | clean (false alarm) | 40 | 20% [5, 39] | 22% [8, 41] | 25% [10, 44] |
| XTTS-v2 | omission | 40 | 100% | 100% | 100% |
| XTTS-v2 | repetition | 40 | 85% [70, 96] | 98% [93, 100] | 100% |
| XTTS-v2 | substitution | 40 | 100% | 100% | 100% |

92% of corrupted clips were read as given (CTC within 15% WER of the corrupted text); the rest
contain extra TTS errors on top of the intended one. The remaining ~20% false alarms on clean clips
are partly real TTS errors and partly hard formats (booking codes, "St" read as "Saint").

### C. Natural clips, judge-labelled (200 clips)

Error rate by model and prompt category (95% CI over template groups):

| Model | All | Ordinary | Long | Numbers & names | Repetition-prone |
|---|---|---|---|---|---|
| F5-TTS | 17% [7, 31] | 0% | 5% [0, 15] | 50% [28, 75] | 0% |
| XTTS-v2 | 23% [9, 38] | 29% [4, 57] | 30% [0, 60] | 28% [6, 56] | 0% |

Detection on held-out test prompts (combined detector fitted on dev only):

| Detector | Precision | Recall | False-alarm rate |
|---|---|---|---|
| Whisper only | 47% [22, 100] | 56% [26, 100] | 14% [0, 35] |
| CTC only | 43% [15, 75] | 62% [27, 100] | 18% [4, 40] |
| Combined (Whisper + CTC + duration) | 45% [12, 73] | 56% [12, 100] | 15% [3, 35] |

Recall by judge-labelled error type (all 200 clips, descriptive):

| Error type | n | Whisper | CTC |
|---|---|---|---|
| substitution | 18 | 78% | 94% |
| omission | 11 | 100% | 100% |
| truncation | 9 | 22% | 33% |
| insertion | 6 | 100% | 100% |
| repetition | 1 | 100% | 100% |

## Method details

- **Models.** TTS: XTTS-v2 (autoregressive) and F5-TTS (non-autoregressive flow matching), both
  cloning the same reference voice, one seed. ASR: Whisper large-v3-turbo; wav2vec2-large-960h-lv60-self
  with greedy CTC decoding (no language model).
- **Prompts.** 100 original prompts in 4 categories (ordinary, long, numbers & names,
  repetition-prone), built from template groups. Dev/test split is by template group, stratified by
  category, so near-duplicates never straddle the split.
- **Normalisation is explicit and unit-tested** (currency, times, years, ordinals, codes, "and" in
  numbers, abbreviations, spelling variants, split compounds). See
  [`docs/NORMALISATION.md`](docs/NORMALISATION.md).
- **Alignment** tags omissions, repetitions, insertions, substitutions and truncations. ASR output is
  only ever a *suspected* error; ground truth comes from sources A–C.
- **Uncertainty:** 95% CIs by bootstrap over template groups (clips in a group are correlated);
  exact McNemar test for paired detector comparison; exact binomial CIs for small human-sample rates.

## Limitations

- **Small scale:** 100 prompts × 2 models, one voice, one seed, English only. Intervals are wide.
- **Human labels: 40 clips, one annotator** (the author), with only 4 errors. They validate the judge
  loosely, not precisely. No second listener.
- **The judge is automated.** κ = 0.72 against the human sample, but it hears the target text and
  may hide errors the same way Whisper does. Judge-labelled numbers are not human-verified.
- **Synthetic errors are deliberate text edits,** not the errors TTS models make on their own.
- **The spelling tolerance can hide a real one-letter substitution** in a word of 5+ letters
  (e.g. "bridge" vs "ridge").
- **Out of scope:** pronunciation quality, and using the detector to guide regeneration (flag, resample,
  re-check), which would need a larger human-labelled set to evaluate.
- Model licences: XTTS-v2 is released under the Coqui Public Model License (non-commercial). Check
  each model's licence before reusing generated audio.

## Run it

Everything runs in [`notebooks/run_colab.ipynb`](notebooks/run_colab.ipynb) on a Colab T4 GPU, top to
bottom, and the saved notebook includes the outputs of the reported run. It expects the repository
as `tts-content-verifier.zip` in the root of Google Drive and keeps `outputs/` and `labels/` on Drive;
every step is resumable. The Gemini step needs an API key (a few pence on a paid tier; the free tier
allows 20 requests per model per day). Package versions used: [`outputs/env.txt`](outputs/env.txt).

```bash
python -m unittest discover tests        # normalisation, alignment, metrics, prompts (no GPU)
```

| Script | Does |
|---|---|
| `01_make_prompts.py` | Writes the 100 prompts and dev/test split |
| `02_generate.py`, `03_transcribe.py` | 200 natural clips; Whisper + CTC transcripts |
| `07_synthetic_errors.py` | Synthetic benchmark: generate (`--generate`) and score (`--evaluate`) |
| `09_retranscribe_ctc.py` | Re-runs CTC with 0.5 s leading silence (keeps the old output) |
| `08_gemini_judge.py` | Gemini judge (`--judge`) and agreement with human labels (`--agree`) |
| `05_evaluate.py` | Detector evaluation against a label file (`--labels gemini.csv --tag _gemini`) |

Human labels were collected in the notebook with `tts_verifier.listen` (protocol:
[`labels/LABELING.md`](labels/LABELING.md)).

## Repository layout

```
src/tts_verifier/   normalize · align · detectors · metrics · prompts · models · corrupt · listen · dataset
scripts/            pipeline steps (see table above)
labels/             LABELING.md (listening protocol), primary.csv (human), gemini.csv (judge)
docs/               NORMALISATION.md
notebooks/          run_colab.ipynb
tests/              unit tests (no GPU needed)
outputs/            result tables, per-clip synthetic predictions, env.txt (audio and transcripts not committed)
```
