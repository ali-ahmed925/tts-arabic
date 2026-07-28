#!/usr/bin/env bash
# Same evaluation as run.sh, but transformers models load weights DIRECTLY onto the
# GPU (device_map / low_cpu_mem_usage) instead of staging a full copy in CPU RAM.
# Results go to results_gpu/ so you can compare peak_rss_mb / peak_vram_mb against
# the default run in results/.
#
#   PYTHON=<env-py> EXTRA_DEPS=./deps ./run_gpu.sh            # all models
#   PYTHON=<env-py> EXTRA_DEPS=./deps ./run_gpu.sh omnivoice  # subset
#
# Note: only affects transformers models (VITS / SpeechT5 / OmniVoice). XTTS uses
# coqui's own loader and Supertonic is CPU/ONNX, so those are unchanged.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
export PYTHONPATH="${EXTRA_DEPS:+$EXTRA_DEPS:}${PYTHONPATH}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export TTS_DIRECT_GPU=1
export TTS_RESULTS_DIR=results_gpu
"$PY" tts_eval/run.py "$@"
"$PY" tts_eval/make_report.py
