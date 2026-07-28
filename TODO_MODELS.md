# Deferred models — to implement later

Models we shortlisted but haven't integrated yet, with the reason and what each needs.
Add them to `tts_eval/config.py` `MODELS` (each in its own isolated venv under `envs/`)
once the blocker below is handled. See README → "Expressive-tier models (isolated envs)".

## Blocked on cost / hardware budget

### ResembleAI/chatterbox  (multilingual, incl. Arabic)
- **Why deferred:** `chatterbox-tts` pins `torch==2.6.0` and `transformers==5.2.0`, so the
  venv can't reuse the base torch 2.5.1 → forces a ~2.5 GB torch re-download. Runtime VRAM
  is ~2.5–3 GB, i.e. **over the ~2 GB cap** (expressive tier only).
- **Payoff:** the single most expressive Arabic option (23-lang MTL, emotion/exaggeration
  control). Adapter also covers the shelved `oddadmix/chatterbox-egyptian-v0`.
- **To do:**
  ```bash
  BASE=/home/owais/miniconda3/envs/avatar-gen/bin/python
  $BASE -m venv --system-site-packages envs/chatterbox
  envs/chatterbox/bin/python -m pip install chatterbox-tts   # pulls torch 2.6.0 (big)
  ```
  Adapter kind `chatterbox`: `ChatterboxMultilingualTTS.from_pretrained(device)`,
  `model.generate(text, language_id="ar")` → wav at `model.sr`. Add a male speaker/ref if
  the guardrail should apply (check whether it exposes speaker selection).

### nvidia/magpie_tts_multilingual_357m
- **Why deferred:** **gated** on HuggingFace (must accept the license + provide an HF token),
  and needs the heavy **NeMo** runtime (fragile on py3.13; use the py3.12 base env).
- **Payoff:** genuinely Arabic (`ar` in tags) + expressive, ~0.7–1 GB (fits the strict-ish tier).
- **To do:** accept license at the model page, `export HF_TOKEN=...`, then
  `pip install nemo_toolkit[tts]` in an isolated venv; adapter loads the `.nemo` checkpoint
  and calls its `convert_text_to_waveform`-style API.

## Blocked on broken / heavyweight package

### fishaudio/fish-speech-1.5
- **Why deferred:** the `fish-speech` PyPI package (v0.1.0) **fails to install**
  (`metadata-generation-failed`) and pulls a full *training* stack — `lightning`,
  `wandb`, `tensorboard`, `hydra-core`, `datasets==2.18.0`, `pyaudio` (needs system
  portaudio), `einx`, `loralib`. It is not a clean inference package.
- **Payoff:** fish-speech is expressive and lists Arabic; worth it if wired properly.
- **To do:** skip PyPI. Clone the GitHub repo (`fishaudio/fish-speech`), download the
  checkpoints from HF, and drive its two-stage inference (LLAMA text2semantic →
  firefly-GAN / VQGAN decoder) via its `tools`/CLI. Give it a dedicated env with only
  the runtime deps (avoid pyaudio/wandb/lightning). Needs a reference clip for cloning.

## Blocked on GGUF / bespoke runtime

### audarai/Audar-TTS-V1-Flash  (Q4 GGUF)  — already a failing registry row
- **Why deferred:** needs `llama-cpp-python` + `neucodec` decoder + a reference audio clip
  (zero-shot cloning). Must live in its OWN env — installing neucodec into a shared path
  drags in transformers 5.x and breaks the VITS stack (we hit this).
- **Payoff:** the only *in-budget* (≤700 MB via 4-bit) model *designed* to be expressive.
- **To do:** dedicated venv; `pip install llama-cpp-python neucodec`; generate a short Arabic
  reference clip (e.g. from mms-tts-ara) + transcript; feed prompt with
  `<|TARGET_CODES_START|>…` markers; decode codes with NeuCodec → 24 kHz wav.

### Serveurperso/Qwen3-TTS-GGUF  (0.9B, Q4)
- **Why deferred:** hardest integration — GGUF + llama.cpp + audio codec decode. Fits ≤2 GB
  only quantized.
- **Payoff:** Qwen3-TTS is very expressive and Arabic-capable.
- **To do:** similar GGUF pipeline to Audar; share the llama.cpp + codec plumbing once built.

## Notes
- Every new model = 1 isolated venv (`--system-site-packages` from the base CUDA env to
  reuse torch, *unless* the package pins a different torch) + 1 adapter (~15–30 lines) +
  1 `MODELS` entry with a `"python"` field. Harness/metrics/report never change.
- Apply the male-voice guardrail via `"voices"`/`"speakers"` maps where the model supports
  speaker selection (see `VOICE_SELECTABLE` in `adapters.py`).
