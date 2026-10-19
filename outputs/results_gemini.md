# Results_gemini

**Labels come from an automated judge, NOT humans.**

Labelled clips: 200 (dev 112, test 88)

## Gemini-judged error rate (all labelled clips)

| Model | Category | n | Error rate [95% CI] |
|---|---|---|---|
| f5_tts | all | 100 | 17% [7%, 31%] |
| f5_tts | long | 20 | 5% [0%, 15%] |
| f5_tts | numbers_names | 32 | 50% [28%, 75%] |
| f5_tts | ordinary | 28 | 0% [0%, 0%] |
| f5_tts | repetition_prone | 20 | 0% [0%, 0%] |
| xtts_v2 | all | 100 | 23% [9%, 38%] |
| xtts_v2 | long | 20 | 30% [0%, 60%] |
| xtts_v2 | numbers_names | 32 | 28% [6%, 56%] |
| xtts_v2 | ordinary | 28 | 29% [4%, 57%] |
| xtts_v2 | repetition_prone | 20 | 0% [0%, 0%] |

Gemini-judged error types: {'omission': 11, 'truncation': 9, 'substitution': 18, 'insertion': 6, 'repetition': 1}

## Detection on held-out test prompts

| Detector | Precision | Recall | False-alarm rate |
|---|---|---|---|
| whisper | 47% [22%, 100%] | 56% [26%, 100%] | 14% [0%, 35%] |
| ctc | 43% [15%, 75%] | 62% [27%, 100%] | 18% [4%, 40%] |
| combined | 45% [12%, 73%] | 56% [12%, 100%] | 15% [3%, 35%] |

McNemar (Whisper-only vs combined, test): p = 1.000 (4 clips only baseline right, 3 only combined right).

## Does Whisper's language model hide TTS errors?

Recall on gemini-judged errors, by error type (all labelled clips; descriptive).

| Error type | n | Whisper recall | CTC recall |
|---|---|---|---|
| insertion | 6 | 100% | 100% |
| omission | 11 | 100% | 100% |
| repetition | 1 | 100% | 100% |
| substitution | 18 | 78% | 94% |
| truncation | 9 | 22% | 33% |

4 gemini-judged errors were missed by Whisper but caught by CTC (examples in results.json → hidden_by_whisper_caught_by_ctc).

