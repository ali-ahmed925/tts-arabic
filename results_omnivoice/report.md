# Arabic TTS Evaluation — small/expressive models with a male-voice guardrail

**Test text (fixed):**

> صباح الخير، وشكرًا لك على وقتك للتحدّث معي اليوم. أتطلّع إلى فهم أسلوبك في العمل وشخصيتك بشكل أفضل. لِنبدأ. هل يمكنك أن تُحدّثني عن موقفٍ اضطررت فيه إلى التكيّف مع مشروع أو مبادرة جديدة في العمل؟ كيف كان شعورك وأنت تتعلّم شيئًا جديدًا تمامًا؟

**Hardware:** NVIDIA RTX 4050 Laptop (6 GB VRAM), 12-core CPU, 30 GB RAM.  
**Two tiers evaluated:** a strict **≤700 MB VRAM** tier (small VITS/SpeechT5/ONNX models) and a **≤~2 GB VRAM expressive tier** (XTTS-v2 etc.). CPU/ONNX models report VRAM=0 and should be read on the CPU-RAM column.

**Metrics:** peak VRAM = process GPU memory via nvidia-smi (includes CUDA context). `ASR-CER` = Whisper-small round-trip character error rate (intelligibility, lower is better). `F0 std` = pitch standard deviation in Hz (expressiveness proxy, higher = less monotone). `RTF` = inference time / audio duration (lower is faster).

## Summary

| Model | Arch | Peak VRAM | RTF | ASR-CER | F0 std | Voice | Status |
|---|---|---|---|---|---|---|---|
| omnivoice | omnivoice | 0.0 MB | 9.694 | 0.028 | 35.7 Hz | male (sel) | ok |

## Per-model detail

**Model 1: omnivoice  (`k2-fsa/OmniVoice`)**
*   **Audio Output:** results_omnivoice\audio\omnivoice.mp3
*   **Expressiveness / Quality Notes:** highly intelligible; moderate pitch variation. ASR-CER=0.028, F0 std=35.7 Hz, voiced ratio=0.778. Arch: omnivoice (~0.6B).
*   **Voice (male-guardrail):** selectable → forced to **male** (detected male, F0 mean 142.7 Hz)
*   **Memory Consumed:** 0.0 MB VRAM (peak, process), 2562.4 MB CPU RSS
*   **ASR heard:** صباح الخير وشكرا لك على وقتك للتحدث معي اليوم، أتطلع إلى فهم أسلوبك في العمل وشخصيتك بشكل أفضل، لنبدأ، هل يمكنك أن تحدثني عن موقف اتطررت فيه إلى التكيف مع مشروع أو مبادرة جديدة في العمل؟ كيف كان شعورك وانت تعلمشا جديداً 

## Excluded models (over the 700 MB budget)

These are more expressive but do not fit the constraint:

*   `oddadmix/chatterbox-egyptian-v0` — Chatterbox stack ~5.3GB weights, ~4GB VRAM
*   `openbmb/VoxCPM2` — 2B params, >4GB VRAM
*   `bosonai/higgs-tts-3-4b` — 5B params, well over budget
*   `mistralai/Voxtral-4B-TTS-2603` — 4B params, over budget
*   `OpenMOSS-Team/MOSS-TTS-v1.5` — 8B params, over budget
*   `nvidia/magpie_tts_multilingual_357m` — gated + NeMo runtime; 357M borderline & fragile on py3.13
