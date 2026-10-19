# Synthetic benchmark (NOT human-verified)

Text was deliberately corrupted before synthesis; the original prompt is the reference. Ground truth is the intended corruption. This tests sensitivity to deliberate content changes, not performance on naturally occurring TTS errors. The TTS model may add its own errors.

Clips: 320; corrupted: 240; clean controls: 80

## Detection rate by corruption type (95% CI over prompt groups)

| TTS model | Corruption | n | whisper | ctc | either |
|---|---|---|---|---|---|
| f5_tts | clean (false-alarm rate) | 40 | 18% [5, 35] | 18% [5, 35] | 22% [7, 42] |
| f5_tts | omission (recall) | 40 | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] |
| f5_tts | repetition (recall) | 40 | 88% [77, 96] | 85% [71, 96] | 90% [81, 98] |
| f5_tts | substitution (recall) | 40 | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] |
| xtts_v2 | clean (false-alarm rate) | 40 | 20% [5, 39] | 22% [8, 41] | 25% [10, 44] |
| xtts_v2 | omission (recall) | 40 | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] |
| xtts_v2 | repetition (recall) | 40 | 85% [70, 96] | 98% [93, 100] | 100% [100, 100] |
| xtts_v2 | substitution (recall) | 40 | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] |

Corrupted clips the TTS read as given (CTC WER <= 15% vs spoken text): 92%. The rest contain extra TTS errors, so recall on them is not purely about the intended corruption.

CTC false-alarm rate on clean clips without vs with 0.5 s leading silence: 32% -> 20% (n=80).

