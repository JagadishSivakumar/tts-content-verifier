"""Thin wrappers around the TTS and ASR models. Heavy imports are lazy so the core library
(normalisation, alignment, metrics) runs and tests without a GPU.

Model choices
- XTTS-v2 (Coqui): autoregressive GPT-style decoder over audio tokens. Autoregressive decoding is
  where skips, loops and early stops are expected to occur.
- F5-TTS: non-autoregressive flow-matching model, with a different expected failure profile.
- Whisper large-v3-turbo: strong encoder-decoder ASR with an implicit language model.
- wav2vec2-large-960h-lv60-self, greedy CTC decoding: almost no language model, so it is less
  able to "repair" what the TTS model actually said.

Both TTS models clone the SAME reference voice so voice is held constant across conditions.
Exact package versions used for the reported results are in outputs/env.txt.
"""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

SAMPLE_RATE_ASR = 16_000


def default_reference() -> tuple[str, str]:
    """A short reference clip + its transcript, bundled with the f5-tts package."""
    from importlib.resources import files
    wav = files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")
    return str(wav), "Some call me nature, others call me mother nature."


def _seed_everything(seed: int):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class XTTS:
    name = "xtts_v2"

    def __init__(self, ref_wav: str):
        from TTS.api import TTS  # pip install coqui-tts
        import torch
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.ref_wav = ref_wav

    def synthesize(self, text: str, out_path: Path, seed: int) -> float:
        _seed_everything(seed)
        t0 = time.perf_counter()
        self.tts.tts_to_file(text=text, speaker_wav=self.ref_wav, language="en",
                             file_path=str(out_path))
        return time.perf_counter() - t0


class F5:
    name = "f5_tts"

    def __init__(self, ref_wav: str, ref_text: str):
        from f5_tts.api import F5TTS  # pip install f5-tts
        self.model = F5TTS()
        self.ref_wav, self.ref_text = ref_wav, ref_text

    def synthesize(self, text: str, out_path: Path, seed: int) -> float:
        t0 = time.perf_counter()
        self.model.infer(ref_file=self.ref_wav, ref_text=self.ref_text, gen_text=text,
                         file_wave=str(out_path), seed=seed)
        return time.perf_counter() - t0


def load_tts(name: str):
    ref_wav, ref_text = default_reference()
    if name == "xtts_v2":
        return XTTS(ref_wav)
    if name == "f5_tts":
        return F5(ref_wav, ref_text)
    raise ValueError(name)


def load_audio_16k(path: str):
    import librosa
    audio, _ = librosa.load(path, sr=SAMPLE_RATE_ASR, mono=True)
    return audio


def duration_seconds(path: str) -> float:
    import soundfile as sf
    info = sf.info(path)
    return info.frames / info.samplerate


@lru_cache(maxsize=1)
def _whisper():
    import torch
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device=0 if torch.cuda.is_available() else -1)


def transcribe_whisper(path: str) -> str:
    out = _whisper()({"raw": load_audio_16k(path), "sampling_rate": SAMPLE_RATE_ASR},
                     generate_kwargs={"language": "en", "task": "transcribe"},
                     return_timestamps=True)  # timestamps enable >30 s audio
    return out["text"].strip()


@lru_cache(maxsize=1)
def _ctc():
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    mid = "facebook/wav2vec2-large-960h-lv60-self"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return Wav2Vec2Processor.from_pretrained(mid), Wav2Vec2ForCTC.from_pretrained(mid).to(dev).eval(), dev


def transcribe_ctc(path: str, lead_silence_s: float = 0.5) -> str:
    """Greedy CTC decode: no beam search, no external LM, by design.

    0.5 s of leading silence is added because wav2vec2 tends to garble or drop the first word of an
    abrupt-start clip ("htrain left...", "hebridge in..."); that is a recogniser artifact, not
    evidence of a TTS error. Set lead_silence_s=0 to reproduce the unpadded behaviour.
    """
    import numpy as np
    import torch
    proc, model, dev = _ctc()
    audio = load_audio_16k(path)
    if lead_silence_s > 0:
        audio = np.concatenate([np.zeros(int(lead_silence_s * SAMPLE_RATE_ASR), dtype=audio.dtype), audio])
    inputs = proc(audio, sampling_rate=SAMPLE_RATE_ASR, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(dev)).logits
    ids = torch.argmax(logits, dim=-1)
    return proc.batch_decode(ids)[0].lower()
