# Listening protocol

Human labels are the reference against which the automated judge is checked, so they were collected
under fixed rules.

## Sample
40 natural clips: 5 per (prompt category × TTS model), drawn with a fixed seed
(`tts_verifier.listen.validation_sample`). The sample was fixed before any labelling and does not
depend on detector or judge output.

## What the labeller saw
The labelling view (`tts_verifier.listen.show`) shows only the prompt text and the audio, in a fixed
shuffled order, without the TTS model name, seed, detector output or judge output.

The labels are **not blind to ASR transcripts**: an AI-assisted comparison of the clips' transcripts
with the prompts was reviewed before the final labels were confirmed by listening. Errors visible in
ASR output were therefore easier to find, which favours the ASR detectors on this sample.

## Decision rule (one label per clip)

Mark **Error** if any of these happen, and record the type.

| Type | Definition | Example (prompt → spoken) |
|---|---|---|
| omission | One or more prompt words are not spoken | "the old wooden box" → "the box" |
| repetition | A word or phrase is spoken more than once when the prompt has it once | "water and rope" → "water water and rope" |
| substitution | A different word is spoken in place of a prompt word | "red lorry" → "bed lorry" |
| truncation | A word or the clip is cut off before it ends | "double booked" → "double book" |
| insertion | An extra word not in the prompt is spoken (not a repeat) | "send it" → "send it over" |

Mark **Clean** otherwise.

**Not errors:**
- Odd stress, accent or prosody when the word is still recognisable
- A different but valid reading of a number, time or code ("fourteen thirty" vs "two thirty pm",
  "B S three" vs "BS three")
- Noise, breaths or clicks that do not replace words

## File
`primary.csv`, columns `clip_key,error,types,notes`. `clip_key` is an opaque hash of the clip id
(`tts_verifier.dataset.clip_key`), so the file does not reveal the TTS model.

`gemini.csv` has the same format and holds the automated judge's labels for all 200 clips; the
`notes` column contains the judge's verbatim transcript.
