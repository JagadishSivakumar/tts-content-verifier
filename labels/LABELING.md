# Listening protocol

Human labels are the ground truth for every result, so they are collected under fixed rules.

## Before you start
- Use headphones, in a quiet room, at the same volume throughout.
- Label in sessions of at most 40 minutes, and note the date of each session.
- **Do not look at transcripts, detector output or results before labelling is finished.** The
  labelling page hides model names, seeds and transcripts; keep it that way.

## Decision rule (one label per clip)

Mark **Error** if any of these happen. Tick every type that applies.

| Type | Definition | Example (prompt → spoken) |
|---|---|---|
| omission | One or more prompt words are not spoken (mid-sentence) | "the old wooden box" → "the box" |
| repetition | A word or phrase is spoken more than once when the prompt has it once | "water and rope" → "water water and rope" |
| substitution | A different word is spoken in place of a prompt word | "red lorry" → "bed lorry" |
| truncation | The clip stops before the end of the prompt | ending skipped or cut mid-word |
| insertion | An extra word not in the prompt is spoken (not a repeat) | "send it" → "send it over" |

Mark **Clean** otherwise.

**Not errors (write a note instead):**
- Odd stress, accent or prosody when the word is still recognisable
- Reading a number or code in a different but valid form ("fourteen thirty" vs "two thirty pm",
  "B S three" vs "BS three")
- Noise, breaths or clicks that don't replace words

**Unsure?** Listen up to three times. If you still can't tell, mark **Error**, tick the most likely
type and write `unsure` in the notes. These clips are reported separately.

## Second listener
A second person labels `second.html` (about 20% of clips, stratified by category and model) using
this same document and without seeing your labels. Agreement is reported as Cohen's κ.

## Files
Export from the page and save as `labels/primary.csv` and `labels/second.csv`
(columns: `clip_key,error,types,notes`). For regenerated clips, export `regen.csv` and append its
rows to `primary.csv`.
