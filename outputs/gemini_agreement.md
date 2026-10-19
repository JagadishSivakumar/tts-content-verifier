# Gemini judge vs human labels

Overlap: 40 clips. Cohen's kappa = 0.72; raw agreement 95%.

Of 4 human-labelled errors Gemini found 3 (recall 75%); it flagged 1 clips humans called clean.

## Detectors vs the same human labels (small sample, descriptive)

| Detector | Recall on human errors | False alarms on human-clean |
|---|---|---|
| whisper | 100% (4 errors) | 3% (36 clean) |
| ctc | 100% (4 errors) | 17% (36 clean) |
