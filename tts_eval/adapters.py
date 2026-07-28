"""Model adapters. Each `load_*` returns a `generate(text) -> (wav_float32_mono, sr)`
callable plus a teardown. Adapters keep model-specific quirks isolated so the
orchestrator stays generic and new models are easy to add."""
import re
import gc
import numpy as np


def _to_mono_np(t):
    import torch
    if isinstance(t, torch.Tensor):
        t = t.detach().float().cpu().numpy()
    t = np.asarray(t, dtype=np.float32)
    if t.ndim > 1:
        t = t.squeeze()
    if t.ndim > 1:
        t = t.mean(axis=0)
    return t


def _sentences_ar(text):
    parts = re.split(r"(?<=[\.\؟\!\?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def free_cuda():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _hf_load(cls, repo, device, dtype=None, direct_ok=True, **kw):
    """Load a transformers model. When TTS_DIRECT_GPU=1 (and direct_ok), place
    weights straight on the GPU via device_map (avoids a full CPU staging copy →
    lower peak RSS). Falls back to low_cpu_mem_usage + .to() if unsupported.

    `direct_ok=False` opts a model out entirely: VitsModel produces silent/garbage
    audio under device_map for some checkpoints, and VITS are tiny so there is no
    RSS benefit — so they always take the normal load path."""
    import os
    if dtype is not None:
        kw["torch_dtype"] = dtype
    direct = direct_ok and os.environ.get("TTS_DIRECT_GPU") == "1" and device == "cuda"
    if direct:
        try:
            return cls.from_pretrained(repo, low_cpu_mem_usage=True,
                                       device_map={"": 0}, **kw).eval()
        except Exception as e:
            print(f"  [direct-gpu] device_map unavailable for {cls.__name__} "
                  f"({str(e)[:70]}); using low_cpu_mem_usage + .to()", flush=True)
            return cls.from_pretrained(repo, low_cpu_mem_usage=True,
                                       **kw).to(device).eval()
    return cls.from_pretrained(repo, **kw).to(device).eval()


# --------------------------------- VITS ---------------------------------------
def load_vits(repo, device):
    import torch
    from transformers import VitsModel, AutoTokenizer
    model = _hf_load(VitsModel, repo, device, direct_ok=False)
    tok = AutoTokenizer.from_pretrained(repo)
    sr = model.config.sampling_rate

    def generate(text):
        chunks = _sentences_ar(text) or [text]
        waves = []
        for ch in chunks:
            inputs = tok(ch, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs).waveform
            waves.append(_to_mono_np(out))
            waves.append(np.zeros(int(sr * 0.15), dtype=np.float32))  # short pause
        return np.concatenate(waves), sr

    def teardown():
        nonlocal model
        del model
        free_cuda()

    return generate, sr, teardown


# ------------------------------- SpeechT5 -------------------------------------
_XVECTOR = None


# CMU-Arctic speakers by gender (used for the SpeechT5 voice guardrail).
_XVEC_SPEAKER = {"male": "bdl", "female": "slt"}
_XVECTOR = {}


def _speaker_xvector(device, gender="male"):
    """A single 512-dim CMU-Arctic x-vector for SpeechT5 speaker conditioning,
    chosen by gender (bdl=male, slt=female). Pulled straight from the dataset's
    zip of .npy files (avoids the heavy `datasets` dependency)."""
    import io
    import zipfile
    import torch
    from huggingface_hub import hf_hub_download
    spk_id = _XVEC_SPEAKER.get(gender, "bdl")
    if spk_id not in _XVECTOR:
        z = hf_hub_download("Matthijs/cmu-arctic-xvectors", "spkrec-xvect.zip",
                            repo_type="dataset")
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if n.endswith(".npy")]
            pick = next((n for n in names if spk_id in n), names[0])
            vec = np.load(io.BytesIO(zf.read(pick)))
        _XVECTOR[spk_id] = torch.tensor(vec).float().unsqueeze(0)
    return _XVECTOR[spk_id].to(device)


def load_speecht5(repo, device, gender="male"):
    import torch
    from transformers import (SpeechT5Processor, SpeechT5ForTextToSpeech,
                              SpeechT5HifiGan)
    # fp16 on GPU keeps this ~144M model + HiFi-GAN vocoder under the VRAM budget
    # (fp32 lands ~920MB, over 700MB). CPU stays fp32 for numerical stability.
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = SpeechT5Processor.from_pretrained(repo)
    model = _hf_load(SpeechT5ForTextToSpeech, repo, device, dtype=dtype)
    vocoder = _hf_load(SpeechT5HifiGan, "microsoft/speecht5_hifigan", device, dtype=dtype)
    spk = _speaker_xvector(device, gender).to(dtype)
    sr = 16000

    def generate(text):
        waves = []
        for ch in _sentences_ar(text) or [text]:
            inputs = processor(text=ch, return_tensors="pt").to(device)
            with torch.no_grad():
                sp = model.generate_speech(inputs["input_ids"], spk, vocoder=vocoder)
            waves.append(_to_mono_np(sp))
            waves.append(np.zeros(int(sr * 0.15), dtype=np.float32))
        return np.concatenate(waves), sr

    def teardown():
        nonlocal model, vocoder
        del model, vocoder
        free_cuda()

    return generate, sr, teardown

# ------------------------------- Supertonic (ONNX) ----------------------------
def load_supertonic(device, voice="F1", model="supertonic-3"):
    """Supertone Supertonic-3: efficient multilingual ONNX TTS (runs on CPU via
    onnxruntime). 10 built-in voices (M1..M5 / F1..F5); Arabic supported."""
    import supertonic
    tts = supertonic.TTS(model=model, auto_download=True)
    sr = int(getattr(tts, "sample_rate", None) or tts.model.sample_rate)
    style = tts.get_voice_style(voice)

    def generate(text):
        wav, _ = tts.synthesize(text, style, lang="ar")
        return _to_mono_np(wav), sr

    def teardown():
        pass

    return generate, sr, teardown


# ------------------------------- XTTS-v2 (coqui) ------------------------------
def load_xtts(device, speaker=None):
    """Coqui XTTS-v2: expressive multilingual TTS with voice cloning; Arabic
    (`ar`) is officially supported. Uses a built-in speaker so no reference clip
    is needed. `tts()` auto-splits long text on sentences."""
    import os
    os.environ.setdefault("COQUI_TOS_AGREED", "1")  # accept CPML non-interactively
    from TTS.api import TTS as CoquiTTS
    model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    sr = 24000
    spk = speaker
    if spk is None:  # first built-in speaker
        try:
            spk = list(model.synthesizer.tts_model.speaker_manager.name_to_id.keys())[0]
        except Exception:
            spk = None

    def generate(text):
        wav = model.tts(text=text, language="ar", speaker=spk, split_sentences=True)
        return _to_mono_np(wav), sr

    def teardown():
        nonlocal model
        del model
        free_cuda()

    return generate, sr, teardown


# ------------------------------- OmniVoice ------------------------------------
def load_omnivoice(repo, device, instruct="male", language="arb"):
    """OmniVoice-architecture models (k2-fsa/OmniVoice, VoiceTut-TTS, ...):
    massively-multilingual expressive TTS. Standard Arabic = `arb`. Voice is
    designed via `instruct` from a controlled vocabulary that includes gender
    tags (`male`/`female`, `low pitch`, ...), so no reference clip is needed and
    the male-voice guardrail applies."""
    import torch
    from omnivoice import OmniVoice
    # fp16 on GPU roughly halves VRAM (~4GB -> ~2.1GB) to stay near the ~2GB tier.
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = _hf_load(OmniVoice, repo, device, dtype=dtype)
    sr = int(getattr(model, "sampling_rate", 24000))

    def generate(text):
        outs = model.generate(text=text, language=language, instruct=instruct,
                              normalize_text=True)
        wav = outs[0] if isinstance(outs, (list, tuple)) else outs
        return _to_mono_np(wav), sr

    def teardown():
        nonlocal model
        del model
        free_cuda()

    return generate, sr, teardown


# ------------------------------- Audar GGUF -----------------------------------
def load_audar_gguf(repo, device, gguf_file):
    """Best-effort. The Audar family decodes LLM-emitted NeuCodec tokens; running
    the GGUF needs a bespoke stack. If unavailable we raise so the orchestrator
    records a clean 'skipped' row rather than crashing the run."""
    raise NotImplementedError(
        "Audar GGUF requires llama.cpp + NeuCodec decoder tooling not present; "
        "handled as a documented stretch attempt."
    )

def _gender():
    """The configured voice guardrail (defaults to male if config is unavailable)."""
    try:
        import config
        return getattr(config, "VOICE_GENDER", "male")
    except Exception:
        return "male"


def _pick(mapping, default):
    """Resolve a per-gender voice/speaker map to the configured gender."""
    if not isinstance(mapping, dict):
        return default
    g = _gender()
    return mapping.get(g) or mapping.get("male") or default


ADAPTERS = {
    "vits": lambda m, dev: load_vits(m["hf_repo"], dev),  # fixed single speaker
    "speecht5": lambda m, dev: load_speecht5(m["hf_repo"], dev, gender=_gender()),
    "supertonic": lambda m, dev: load_supertonic(dev, voice=_pick(m.get("voices"), "M1")),
    "xtts": lambda m, dev: load_xtts(dev, speaker=_pick(m.get("speakers"), None)),
    "omnivoice": lambda m, dev: load_omnivoice(m["hf_repo"], dev, instruct=_pick(m.get("instructs"), "male")),
    "audar_gguf": lambda m, dev: load_audar_gguf(m["hf_repo"], dev, m["gguf_file"]),
}

# Which architectures actually let you choose the voice (vs. a fixed checkpoint).
VOICE_SELECTABLE = {"speecht5", "supertonic", "xtts", "omnivoice"}
