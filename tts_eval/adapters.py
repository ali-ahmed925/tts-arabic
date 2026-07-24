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


# --------------------------------- VITS ---------------------------------------
def load_vits(repo, device):
    import torch
    from transformers import VitsModel, AutoTokenizer
    model = VitsModel.from_pretrained(repo).to(device).eval()
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


def _speaker_xvector(device):
    """A single 512-dim CMU-Arctic x-vector for SpeechT5 speaker conditioning.
    Pulled straight from the dataset's zip of .npy files (avoids the heavy
    `datasets` dependency). Any valid speaker vector works for our comparison."""
    global _XVECTOR
    import io
    import zipfile
    import torch
    from huggingface_hub import hf_hub_download
    if _XVECTOR is None:
        z = hf_hub_download("Matthijs/cmu-arctic-xvectors", "spkrec-xvect.zip",
                            repo_type="dataset")
        with zipfile.ZipFile(z) as zf:
            # pick a stable speaker (slt = US female) if present, else first .npy
            names = [n for n in zf.namelist() if n.endswith(".npy")]
            pick = next((n for n in names if "slt" in n), names[0])
            vec = np.load(io.BytesIO(zf.read(pick)))
        _XVECTOR = torch.tensor(vec).float().unsqueeze(0)
    return _XVECTOR.to(device)


def load_speecht5(repo, device):
    import torch
    from transformers import (SpeechT5Processor, SpeechT5ForTextToSpeech,
                              SpeechT5HifiGan)
    # fp16 on GPU keeps this ~144M model + HiFi-GAN vocoder under the VRAM budget
    # (fp32 lands ~920MB, over 700MB). CPU stays fp32 for numerical stability.
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = SpeechT5Processor.from_pretrained(repo)
    model = SpeechT5ForTextToSpeech.from_pretrained(repo, torch_dtype=dtype).to(device).eval()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan",
                                             torch_dtype=dtype).to(device).eval()
    spk = _speaker_xvector(device).to(dtype)
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


# ------------------------------- Audar GGUF -----------------------------------
def load_audar_gguf(repo, device, gguf_file):
    """Best-effort. The Audar family decodes LLM-emitted NeuCodec tokens; running
    the GGUF needs a bespoke stack. If unavailable we raise so the orchestrator
    records a clean 'skipped' row rather than crashing the run."""
    raise NotImplementedError(
        "Audar GGUF requires llama.cpp + NeuCodec decoder tooling not present; "
        "handled as a documented stretch attempt."
    )


ADAPTERS = {
    "vits": lambda m, dev: load_vits(m["hf_repo"], dev),
    "speecht5": lambda m, dev: load_speecht5(m["hf_repo"], dev),
    "audar_gguf": lambda m, dev: load_audar_gguf(m["hf_repo"], dev, m["gguf_file"]),
}
