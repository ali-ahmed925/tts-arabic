# Arabic TTS Evaluation Pipeline

Benchmarks open-source **Arabic** text-to-speech models on a fixed interview-style sentence. For each model it generates audio, measures compute (peak VRAM + CPU RAM), scores quality, and streams results to a CSV + Markdown report.

The goal: find a model that sounds **natural/expressive**, not robotic. Models are grouped in two tiers:
- **Strict tier — ≤700 MB VRAM** (small VITS / SpeechT5 / ONNX models)
- **Expressive tier — ≤~2 GB VRAM** (e.g. XTTS-v2), each isolated in its own venv

## Results (current)

| Model | Arch | Peak VRAM | CPU RAM | ASR-CER ↓ | F0 std ↑ |
|---|---|---|---|---|---|
| `facebook/mms-tts-ara` | VITS 36M | 524 MB | 1393 MB | 0.094 | 42.1 Hz |
| `wasmdashai/lahja-sa-huba-v1` | VITS 36M (Saudi) | 414 MB | 1387 MB | 0.11 | 20.3 Hz |
| `SeyedAli/Arabic-Speech-synthesis-MMS` | VITS 36M | 524 MB | 1396 MB | 0.127 | 41.4 Hz |
| `MBZUAI/speecht5_tts_clartts_ar` | SpeechT5 fp16 | 546 MB | 1294 MB | 0.066 | 16.4 Hz |
| `Supertone/supertonic-3` | ONNX (CPU) | 0 MB | 939 MB | 0.017 | 35.0 Hz |
| `coqui/XTTS-v2` | GPT+HiFiGAN | 2466 MB | — | 0.315* | 15.7* Hz |
| `k2-fsa/OmniVoice` (fp16) | LLM+codec | 2736 MB | — | **0.0** | 34.2 Hz |

All voices are forced **male** by the guardrail where selectable (see below); fixed VITS
voices are detected — `lahja-sa-huba-v1` is female and flagged. `*` XTTS is stochastic and
high-variance run-to-run (a smoke run hit CER 0.033 / F0 std 50). Regenerate with `./run.sh`.

**Takeaway:** strict-tier VITS/SpeechT5 are intelligible but monotone. **OmniVoice** (CER 0.0)
and **Supertonic** (CER 0.017, CPU/0 VRAM) are the standouts for intelligibility; OmniVoice and
XTTS carry the most prosody but need the ~2 GB+ expressive tier.

## Metrics

| Column | Meaning |
|---|---|
| **Peak VRAM** | Process GPU memory (nvidia-smi, incl. CUDA context), measured in an isolated subprocess per model |
| **ASR-CER** | Whisper-small round-trip character error rate — intelligibility (**lower = better**) |
| **F0 std** | Pitch standard deviation in Hz — expressiveness proxy (**higher = less monotone**) |
| **RTF** | Inference time ÷ audio duration (**lower = faster**) |

## Requirements

- Linux, Python **3.10–3.13** (fully tested on 3.12; on 3.13 the deps install and the
  numba-based metric was verified — the `transformers<5` pin is required there)
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

Model ids are the `"id"` fields in [tts_eval/config.py](tts_eval/config.py):
`mms-tts-ara`, `lahja-sa-huba-v1`, `speecht5-clartts-ar`, `seyedali-mms-ar`, `supertonic-3`, `xtts-v2`, `omnivoice`, `voicetut-tts`, `vits-ar-sa-ahmad`, `vits-ar-sa-huba-v2`, `vits-ar`, `vits-ar-sa-A`, `vits-ar-ye-sa`, `audar-tts-v1-flash-q4`. Deferred models: see [TODO_MODELS.md](TODO_MODELS.md).

> The 5 `wasmdashai` Arabic dialect VITS (`vits-ar*`) are **zero-code** additions (reuse the `vits` adapter, run in the base env). They're easily integrable and in-budget, but measured **worse** (CER 0.18–0.31) than the top models — added for dialect breadth, not to beat the best.

Re-running a model **overwrites its own CSV row** (upsert) — no duplicates.

## Direct-to-GPU variant (`run_gpu.sh`)

`run_gpu.sh` runs the same evaluation but loads transformers models **straight onto
the GPU** (`device_map` / `low_cpu_mem_usage`) instead of staging a full weight copy
in CPU RAM, and writes to **`results_gpu/`** so you can compare `peak_rss_mb`.

```bash
PYTHON=<env-py> EXTRA_DEPS=./deps ./run_gpu.sh
```

Effect (measured): it slashes CPU RAM for **heavy** models — OmniVoice `peak_rss_mb`
dropped **4707 → 2774 MB (−1.9 GB)** with identical VRAM and quality. Small models
(VITS/SpeechT5) barely change (the ~1.3 GB Python/CUDA-library baseline dominates), and
XTTS (coqui loader) / Supertonic (CPU/ONNX) are unaffected. **VITS are opted out** of the
direct path — `device_map` produced silent output for some VITS checkpoints, and they
gain no RSS benefit anyway.

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

## Expressive-tier models (isolated envs)

Heavyweight runtimes (coqui-tts, chatterbox, …) pull their own `transformers`/`torch`
and **must not share** the base env — mixing them breaks the VITS stack. So each gets its
own venv, and its `MODELS` entry sets a `"python"` field pointing at that venv's
interpreter; the runner launches that model's subprocess with it. Trick: build the venv
with `--system-site-packages` from the base CUDA env so it **inherits torch/transformers/
librosa** (no 2.5 GB torch re-download) and only layers the model package on top.

```bash
BASE=/home/owais/miniconda3/envs/avatar-gen/bin/python   # env with CUDA torch

# Supertonic (ONNX, CPU — tiny):
$BASE -m venv --system-site-packages envs/supertonic
envs/supertonic/bin/python -m pip install supertonic onnxruntime

# XTTS-v2 (coqui-tts). Pin transformers<5 — 5.x removed a symbol coqui imports:
$BASE -m venv --system-site-packages envs/xtts
envs/xtts/bin/python -m pip install coqui-tts "torchaudio==2.5.1" "transformers>=4.44,<5" \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

Then `./run.sh supertonic-3 xtts-v2`. Envs live under `envs/` (git-ignored).

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
