"""Central config: the test text, paths, and the model registry."""
import os
from pathlib import Path

# The fixed Arabic test sentence (interview-style, needs prosody/expressiveness).
TEST_TEXT = (
    "صباح الخير، وشكرًا لك على وقتك للتحدّث معي اليوم. أتطلّع إلى فهم أسلوبك في العمل "
    "وشخصيتك بشكل أفضل. لِنبدأ. هل يمكنك أن تُحدّثني عن موقفٍ اضطررت فيه إلى التكيّف "
    "مع مشروع أو مبادرة جديدة في العمل؟ كيف كان شعورك وأنت تتعلّم شيئًا جديدًا تمامًا؟"
)

ROOT = Path(__file__).resolve().parent.parent
# Output dir is overridable so a second run (e.g. direct-GPU load) can write to a
# separate folder for side-by-side comparison: TTS_RESULTS_DIR=results_gpu
RESULTS = ROOT / os.environ.get("TTS_RESULTS_DIR", "results")
AUDIO_DIR = RESULTS / "audio"
CSV_PATH = RESULTS / "results.csv"
REPORT_PATH = RESULTS / "report.md"

# When set, transformers models load weights straight onto the GPU (device_map /
# low_cpu_mem_usage) instead of staging a full copy in CPU RAM first — lets you
# compare peak_rss_mb with vs. without the CPU staging copy.
DIRECT_GPU = os.environ.get("TTS_DIRECT_GPU") == "1"

# ASR model used for the round-trip intelligibility (CER) metric. It is the
# *evaluator*, run after the TTS model is unloaded, so its footprint is NOT
# counted against a model's VRAM budget.
ASR_MODEL = "openai/whisper-small"

VRAM_BUDGET_MB = 700  # the shortlisting constraint

# Voice guardrail. Selectable models (SpeechT5 x-vector, Supertonic named voices,
# XTTS named speakers) are forced to this gender. Single-speaker VITS models have
# a fixed voice baked into the weights — they can't be changed, only DETECTED
# (via F0), and are flagged if the detected gender disagrees.
VOICE_GENDER = "male"       # "male" | "female"
# F0 mean (Hz) below this reads as male, above as female (rough heuristic).
GENDER_F0_THRESHOLD_HZ = 160.0

# The shortlist. Each entry names an adapter "kind" (see adapters.py) plus
# whatever that adapter needs. `in_budget` records the pre-run expectation.
MODELS = [
    {
        "id": "mms-tts-ara",
        "hf_repo": "facebook/mms-tts-ara",
        "kind": "vits",
        "lang": "Arabic (MSA)",
        "params": "~36M",
        "in_budget": True,
        "notes": "Meta MMS VITS baseline, single deterministic speaker.",
    },
    {
        "id": "lahja-sa-huba-v1",
        "hf_repo": "wasmdashai/lahja-sa-huba-v1",
        "kind": "vits",
        "lang": "Arabic (Saudi dialect)",
        "params": "~36M",
        "in_budget": True,
        "notes": "VITS fine-tuned on Saudi dialectal data (expressive target).",
    },
    {
        "id": "speecht5-clartts-ar",
        "hf_repo": "MBZUAI/speecht5_tts_clartts_ar",
        "kind": "speecht5",
        "lang": "Arabic (MSA)",
        "params": "~144M",
        "in_budget": True,
        "notes": "SpeechT5 + HiFi-GAN vocoder, x-vector speaker embedding.",
    },
    {
        "id": "seyedali-mms-ar",
        "hf_repo": "SeyedAli/Arabic-Speech-synthesis-MMS",
        "kind": "vits",
        "lang": "Arabic",
        "params": "~36M",
        "in_budget": True,
        "notes": "MMS-derived VITS variant.",
    },
    # ---- Additional Arabic VITS (zero new code: reuse the `vits` adapter,
    #      run in the base env; single fixed speaker each, gender is detected) ----
    {
        "id": "vits-ar-sa-ahmad",
        "hf_repo": "wasmdashai/lahja-sa-ahmad-v1",
        "kind": "vits",
        "lang": "Arabic (Saudi dialect)",
        "params": "~36M",
        "in_budget": True,
        "notes": "Popular wasmdashai Saudi VITS (Ahmad speaker).",
    },
    {
        "id": "vits-ar-sa-huba-v2",
        "hf_repo": "wasmdashai/vits-ar-sa-huba-v2",
        "kind": "vits",
        "lang": "Arabic (Saudi dialect)",
        "params": "~83M",
        "in_budget": True,
        "notes": "Base VITS that lahja-sa-huba-v1 was fine-tuned from.",
    },
    {
        "id": "vits-ar",
        "hf_repo": "wasmdashai/vits-ar",
        "kind": "vits",
        "lang": "Arabic (MSA)",
        "params": "~36M",
        "in_budget": True,
        "notes": "General Arabic VITS.",
    },
    {
        "id": "vits-ar-sa-A",
        "hf_repo": "wasmdashai/vits-ar-sa-A",
        "kind": "vits",
        "lang": "Arabic (Saudi dialect)",
        "params": "~83M",
        "in_budget": True,
        "notes": "wasmdashai Saudi VITS variant 'A'.",
    },
    {
        "id": "vits-ar-ye-sa",
        "hf_repo": "wasmdashai/vits-ar-ye-sa",
        "kind": "vits",
        "lang": "Arabic (Yemeni dialect)",
        "params": "~36M",
        "in_budget": True,
        "notes": "wasmdashai Yemeni-dialect VITS.",
    },
    # ---- Expressive tier (≤~2 GB), each in its own isolated venv ----
    {
        "id": "supertonic-3",
        "hf_repo": "Supertone/supertonic-3",
        "kind": "supertonic",
        "python": str(ROOT / "envs/supertonic/bin/python"),
        "voices": {"male": "M1", "female": "F1"},  # selectable
        "lang": "Arabic (multilingual)",
        "params": "ONNX (small)",
        "in_budget": True,
        "notes": "Efficient multilingual ONNX TTS (CPU/onnxruntime), 10 built-in voices.",
    },
    {
        "id": "omnivoice",
        "hf_repo": "k2-fsa/OmniVoice",
        "kind": "omnivoice",
        "python": str(ROOT / "envs/omnivoice/bin/python"),
        "instructs": {"male": "male", "female": "female"},  # selectable via voice-design
        "lang": "Arabic (arb/MSA + dialects)",
        "params": "~0.6B",
        "in_budget": False,
        "notes": "k2-fsa OmniVoice, expressive multilingual; text voice-design (gender tags). ~2 GB tier.",
    },
    {
        "id": "omnivoice-int8",
        "hf_repo": "k2-fsa/OmniVoice",
        "kind": "omnivoice",
        "python": str(ROOT / "envs/omnivoice/bin/python"),
        "instructs": {"male": "male", "female": "female"},  # selectable via voice-design
        "quantization": "int8",  # bitsandbytes INT8 LLM backbone; audio_heads kept fp16
        "lang": "Arabic (arb/MSA + dialects)",
        "params": "~0.6B (INT8 LLM backbone)",
        "in_budget": False,
        "notes": "k2-fsa OmniVoice, INT8-quantized LLM backbone via bitsandbytes "
                 "(audio_heads skipped); compare VRAM/quality vs. the fp16 'omnivoice' row.",
    },
    {
        "id": "xtts-v2",
        "hf_repo": "coqui/XTTS-v2",
        "kind": "xtts",
        "python": str(ROOT / "envs/xtts/bin/python"),
        "speakers": {"male": "Viktor Eka", "female": "Ana Florence"},  # selectable
        "lang": "Arabic (multilingual)",
        "params": "~460M",
        "in_budget": False,
        "notes": "Coqui XTTS-v2, expressive + voice cloning; Arabic official. ~2 GB tier.",
    },
    {
        # Stretch: only fits the budget as a 4-bit GGUF. Handled best-effort.
        "id": "audar-tts-v1-flash-q4",
        "hf_repo": "audarai/Audar-TTS-V1-Flash",
        "kind": "audar_gguf",
        "gguf_file": "Audar-TTS-V1-Flash-Q4_K_M.gguf",
        "lang": "Arabic (expressive)",
        "params": "~0.6B (Q4)",
        "in_budget": True,
        "notes": "Expressive LLM-codec TTS; fits budget only when 4-bit quantized.",
    },

    {
    "id": "voicetut-tts",
    "hf_repo": "mohammedaly22/VoiceTut-TTS",
    "kind": "omnivoice",
    "python": str(ROOT / "envs/omnivoice/bin/python"),  # shares the OmniVoice env
    "lang": "Arabic (Egyptian)",
    "params": "~612M",
    "in_budget": False,
    "notes": "OmniVoice Arabic TTS with voice cloning support. Manual test used ~4.64GB memory.",
},
]

# Documented exclusions — why they fail the 700MB filter (for the report).
EXCLUSIONS = [
    ("oddadmix/chatterbox-egyptian-v0", "Chatterbox stack ~5.3GB weights, ~4GB VRAM"),
    ("openbmb/VoxCPM2", "2B params, >4GB VRAM"),
    ("bosonai/higgs-tts-3-4b", "5B params, well over budget"),
    ("mistralai/Voxtral-4B-TTS-2603", "4B params, over budget"),
    ("OpenMOSS-Team/MOSS-TTS-v1.5", "8B params, over budget"),
    ("nvidia/magpie_tts_multilingual_357m", "gated + NeMo runtime; 357M borderline & fragile on py3.13"),
]
