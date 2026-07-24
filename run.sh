#!/usr/bin/env bash
# Run the Arabic TTS evaluation, then build the report.
#
#   ./run.sh                         # all models in the registry
#   ./run.sh mms-tts-ara speecht5-clartts-ar   # only these ids
#
# Python: defaults to `python3`. Override with PYTHON to point at a specific
# interpreter (e.g. a conda env that already has CUDA torch):
#   PYTHON=/path/to/env/bin/python ./run.sh
# EXTRA_DEPS: optional dir of extra packages to prepend to PYTHONPATH (used when
# reusing an existing env that is missing a few deps):
#   EXTRA_DEPS=./deps PYTHON=/path/to/env/bin/python ./run.sh
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
export PYTHONPATH="${EXTRA_DEPS:+$EXTRA_DEPS:}${PYTHONPATH}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
"$PY" tts_eval/run.py "$@"
"$PY" tts_eval/make_report.py
