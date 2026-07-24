# Arabic TTS Evaluation Pipeline

Benchmarks open-source **Arabic** text-to-speech models that fit a strict **≤700 MB VRAM** budget, on a fixed interview-style sentence. For each model it generates audio, measures compute (peak VRAM + CPU RAM), scores quality, and streams results to a CSV + Markdown report.

The goal: find a model that sounds **natural/expressive**, not robotic, while staying inside a small memory budget.

## Results (current)

| Model | Arch | Peak VRAM | ASR-CER ↓ | F0 std ↑ | RTF ↓ |
|---|---|---|---|---|---|
| `facebook/mms-tts-ara` | VITS 36M | 524 MB | 0.094 | 42.1 Hz | 0.057 |
| `wasmdashai/lahja-sa-huba-v1` | VITS 36M (Saudi) | 414 MB | 0.11 | 20.3 Hz | 0.095 |
| `SeyedAli/Arabic-Speech-synthesis-MMS` | VITS 36M | 524 MB | 0.127 | 41.4 Hz | 0.059 |
| `MBZUAI/speecht5_tts_clartts_ar` | SpeechT5 (fp16) | 546 MB | 0.066 | 16.4 Hz | 0.166 |

**Takeaway:** all four are highly intelligible but fairly monotone (low F0 variance) — the truly expressive models (Chatterbox, VoxCPM2, Higgs…) don't fit 700 MB. See the "Excluded models" section of the generated [report](results/report.md).

## Metrics

| Column | Meaning |
|---|---|
| **Peak VRAM** | Process GPU memory (nvidia-smi, incl. CUDA context), measured in an isolated subprocess per model |
| **ASR-CER** | Whisper-small round-trip character error rate — intelligibility (**lower = better**) |
| **F0 std** | Pitch standard deviation in Hz — expressiveness proxy (**higher = less monotone**) |
| **RTF** | Inference time ÷ audio duration (**lower = faster**) |

## Requirements

- Linux, Python **3.10–3.12** (tested on 3.12)
- NVIDIA GPU with CUDA (optional — falls back to CPU)
- **ffmpeg** on PATH (for MP3 export): `sudo apt-get install ffmpeg`

## Setup

### Option A — fresh virtualenv (portable, recommended for a clean checkout)

```bash
python3 -m venv .venv
source .venv/bin/activate

# 1) install torch:
#    GPU (CUDA 12.1 example):
pip install torch --index-url https://download.pytorch.org/whl/cu121
#    CPU-only (no NVIDIA GPU) — plain build, no index-url:
pip install torch

# 2) install the rest:
pip install -r requirements.txt
```

### No GPU? It still works.

The pipeline auto-detects hardware (`device = "cuda" if available else "cpu"`), so no
code or flags change. On a CPU-only machine:

- Install the **CPU torch** build (above).
- Read the **CPU RAM** column instead of VRAM (the VRAM column reports `0`, since no
  GPU is used).
- Expect slower runs, but the small VITS models are still faster than real-time on CPU
  (measured RTF ≈ 0.58, ~0.8 GB RAM). SpeechT5 runs in fp32 and the Whisper ASR scorer
  is the slowest step.

### Option B — reuse an existing conda env that already has CUDA torch

Point `PYTHON` at that env and put any missing extras in a target dir:

```bash
ENV_PY=/path/to/conda/envs/<name>/bin/python
$ENV_PY -m pip install --target=./deps sentencepiece   # only what the env lacks
```

Then pass `PYTHON` and `EXTRA_DEPS` to `run.sh` (see below).

## Usage

```bash
# Run ALL models in the registry (wipes results.csv, writes fresh):
./run.sh

# Run only specific models (appends/updates their rows; does NOT wipe):
./run.sh mms-tts-ara speecht5-clartts-ar

# Option B invocation (reuse an existing env + extra deps dir):
PYTHON=$ENV_PY EXTRA_DEPS=./deps ./run.sh
```

Model ids are the "id" fields in tts_eval/config.py:
mms-tts-ara, lahja-sa-huba-v1, speecht5-clartts-ar, seyedali-mms-ar, audar-tts-v1-flash-q4, voicetut-tts.

Re-running a model **overwrites its own CSV row** (upsert) — no duplicates.

## Outputs (`results/`)

```
results/
├── results.csv        # one row per model: VRAM, RAM, CER, F0, RTF, mp3 path, ASR transcript
├── report.md          # human-readable report (deliverable template)
└── audio/<id>.mp3     # generated speech per model
```

## Adding a new model

1. Add an entry to `MODELS` in [tts_eval/config.py](tts_eval/config.py):
   ```python
   {"id": "my-model", "hf_repo": "org/model", "kind": "vits",
    "lang": "Arabic", "params": "~36M", "in_budget": True, "notes": "..."},
   ```
2. Run it: `./run.sh my-model`

`"kind"` selects an **adapter** in [tts_eval/adapters.py](tts_eval/adapters.py). Same architecture as an existing model → reuse its `kind` (no code). New architecture/runtime → add one small adapter that returns `generate(text) -> (wav_float32_mono, sr)`; everything else (metrics, VRAM measurement, CSV, MP3, report) is architecture-agnostic and unchanged.

## Project structure

```
tts_eval/
├── config.py       # test text, paths, model registry, exclusions
├── adapters.py     # per-architecture load+generate (vits, speecht5, audar_gguf)
├── monitor.py      # peak VRAM (nvidia-smi) + CPU RSS sampler
├── metrics.py      # ASR round-trip CER + F0/pitch stats
├── run.py          # orchestrator: subprocess-isolated runs, streaming CSV
└── make_report.py  # results.csv -> report.md
run.sh              # convenience wrapper
requirements.txt
```

## Notes / gotchas

- **VRAM isolation:** each model runs in its own subprocess so its peak VRAM isn't contaminated by other models or the Whisper ASR evaluator.
- **SpeechT5 runs in fp16** on GPU to stay under budget (fp32 needs ~920 MB).
- **Keep heavy stacks out of `EXTRA_DEPS`:** installing `neucodec`/`llama-cpp-python` there drags in a newer `transformers`/`torch` that shadows the working ones via `PYTHONPATH`. Give the Audar GGUF stretch model its own separate env.
