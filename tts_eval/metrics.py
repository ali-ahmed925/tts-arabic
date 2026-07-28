"""Quality metrics.

Two automatable, defensible signals:
  1. Intelligibility  -> ASR round-trip Character Error Rate (CER). Lower = clearer.
  2. Expressiveness    -> F0 (pitch) standard deviation in Hz + voiced ratio.
     Monotone/robotic voices have low pitch variance; expressive ones higher.
"""
import re
import unicodedata

import numpy as np


# ----------------------------- Arabic text normalization -----------------------
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"


def normalize_ar(text: str) -> str:
    """Strip diacritics/punctuation and unify letter forms so CER compares the
    *spoken* content, not tashkeel the ASR will never emit."""
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = (text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
                .replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و")
                .replace("ئ", "ي"))
    text = re.sub(r"[^ء-غف-ي\s]", " ", text)  # keep Arabic letters
    return re.sub(r"\s+", " ", text).strip()


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = normalize_ar(reference).replace(" ", "")
    hyp = normalize_ar(hypothesis).replace(" ", "")
    if not ref:
        return float("nan")
    return _levenshtein(ref, hyp) / len(ref)


# ----------------------------- pitch / expressiveness --------------------------
def pitch_stats(wav: np.ndarray, sr: int) -> dict:
    """F0 statistics via librosa.pyin. Returns Hz mean/std over voiced frames
    and the fraction of voiced frames."""
    try:
        import librosa
        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        f0, voiced, _ = librosa.pyin(
            wav, sr=sr, fmin=70, fmax=400, frame_length=2048,
        )
        vf = f0[~np.isnan(f0)]
        return {
            "f0_mean_hz": round(float(np.mean(vf)), 1) if vf.size else float("nan"),
            "f0_std_hz": round(float(np.std(vf)), 1) if vf.size else float("nan"),
            "voiced_ratio": round(float(np.mean(voiced)), 3) if voiced.size else float("nan"),
        }
    except Exception as e:
        return {"f0_mean_hz": float("nan"), "f0_std_hz": float("nan"),
                "voiced_ratio": float("nan"), "pitch_err": str(e)[:80]}


# ----------------------------- ASR (round-trip) --------------------------------
_ASR = None


def transcribe(wav: np.ndarray, sr: int, model_id: str, device: str) -> str:
    """Lazy-load a Whisper ASR pipeline and transcribe (Arabic)."""
    global _ASR
    import torch
    from transformers import pipeline
    if _ASR is None:
        _ASR = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=0 if device == "cuda" else -1,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            chunk_length_s=30,   # chunk long audio (>30s) instead of failing
        )
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    # return_timestamps=True is required by newer transformers for >30s clips;
    # harmless on older versions. out["text"] is still the full transcription.
    out = _ASR(
        {"array": wav.astype(np.float32), "sampling_rate": sr},
        generate_kwargs={"language": "arabic", "task": "transcribe"},
        return_timestamps=True,
    )
    return out["text"].strip()
