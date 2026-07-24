"""Central config: the test text, paths, and the model registry."""
from pathlib import Path

# The fixed Arabic test sentence (interview-style, needs prosody/expressiveness).
TEST_TEXT = (
    "صباح الخير، وشكرًا لك على وقتك للتحدّث معي اليوم. أتطلّع إلى فهم أسلوبك في العمل "
    "وشخصيتك بشكل أفضل. لِنبدأ. هل يمكنك أن تُحدّثني عن موقفٍ اضطررت فيه إلى التكيّف "
    "مع مشروع أو مبادرة جديدة في العمل؟ كيف كان شعورك وأنت تتعلّم شيئًا جديدًا تمامًا؟"
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
AUDIO_DIR = RESULTS / "audio"
CSV_PATH = RESULTS / "results.csv"
REPORT_PATH = RESULTS / "report.md"

# ASR model used for the round-trip intelligibility (CER) metric. It is the
# *evaluator*, run after the TTS model is unloaded, so its footprint is NOT
# counted against a model's VRAM budget.
ASR_MODEL = "openai/whisper-small"

VRAM_BUDGET_MB = 700  # the shortlisting constraint

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
