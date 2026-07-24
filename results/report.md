# Arabic TTS Evaluation — Models under a 700 MB VRAM budget

**Test text (fixed):**

> صباح الخير، وشكرًا لك على وقتك للتحدّث معي اليوم. أتطلّع إلى فهم أسلوبك في العمل وشخصيتك بشكل أفضل. لِنبدأ. هل يمكنك أن تُحدّثني عن موقفٍ اضطررت فيه إلى التكيّف مع مشروع أو مبادرة جديدة في العمل؟ كيف كان شعورك وأنت تتعلّم شيئًا جديدًا تمامًا؟

**Hardware:** NVIDIA RTX 4050 Laptop (6 GB VRAM), 12-core CPU, 30 GB RAM.  
**Budget filter:** peak VRAM ≤ 700 MB.

**Metrics:** peak VRAM = process GPU memory via nvidia-smi (includes CUDA context). `ASR-CER` = Whisper-small round-trip character error rate (intelligibility, lower is better). `F0 std` = pitch standard deviation in Hz (expressiveness proxy, higher = less monotone). `RTF` = inference time / audio duration (lower is faster).

## Summary

| Model | Peak VRAM | RTF | Audio | ASR-CER | F0 std | Status |
|---|---|---|---|---|---|---|
| lahja-sa-huba-v1 | 414.0 MB | 0.095 | 20.45s | 0.11 | 20.3 Hz | ok |
| seyedali-mms-ar | 524.0 MB | 0.059 | 32.3s | 0.127 | 41.4 Hz | ok |
| speecht5-clartts-ar | 546.0 MB | 0.166 | 25.58s | 0.066 | 16.4 Hz | ok |
| mms-tts-ara | 524.0 MB | 0.057 | 32.7s | 0.094 | 42.1 Hz | ok |

## Per-model detail

**Model 1: lahja-sa-huba-v1  (`wasmdashai/lahja-sa-huba-v1`)**
*   **Audio Output:** results/audio/lahja-sa-huba-v1.mp3
*   **Expressiveness / Quality Notes:** highly intelligible; low pitch variation (flat/monotone). ASR-CER=0.11, F0 std=20.3 Hz, voiced ratio=0.883. Arch: vits (~36M).
*   **Memory Consumed:** 414.0 MB VRAM (peak, process), 1386.8 MB CPU RSS
*   **ASR heard:** صباح الخير وشكرا لك على وقتك للتحدث معي يوم أطلق للمفهم بسلوبك في العمل وشخصيتك بشكل أفضل بلا بدأة هل يمكنك أن تحدثني عن موقع فالتررت فيه لتكيف مع مشوع أو مبادرة جديدة في العمل كيف كان شعورك وإنت تسعلم شيئا جديد انت ماما

**Model 2: seyedali-mms-ar  (`SeyedAli/Arabic-Speech-synthesis-MMS`)**
*   **Audio Output:** results/audio/seyedali-mms-ar.mp3
*   **Expressiveness / Quality Notes:** highly intelligible; moderate pitch variation. ASR-CER=0.127, F0 std=41.4 Hz, voiced ratio=0.896. Arch: vits (~36M).
*   **Memory Consumed:** 524.0 MB VRAM (peak, process), 1396.2 MB CPU RSS
*   **ASR heard:** الصباح الخير وشكرًا لك على وقتك للتحدث معي اليومي الطلع إيل فهم أسلوبك في العمل وشخصيتك بشكل أفضل هل يمكنك أن تحدثني عن موقف الترارد فيه إلى التكيفة مع مشروع أو مبادلة جديدة في العملي كيف كان الشعورك وانت تتعلم شيء جديد 

**Model 3: speecht5-clartts-ar  (`MBZUAI/speecht5_tts_clartts_ar`)**
*   **Audio Output:** results/audio/speecht5-clartts-ar.mp3
*   **Expressiveness / Quality Notes:** highly intelligible; low pitch variation (flat/monotone). ASR-CER=0.066, F0 std=16.4 Hz, voiced ratio=0.805. Arch: speecht5 (~144M).
*   **Memory Consumed:** 546.0 MB VRAM (peak, process), 1293.5 MB CPU RSS
*   **ASR heard:** صباح الخير وشكرا لك على وقتك للتحدث تمع اليوم، التطلع إلى فهم أسلوبك في العمل وشخصيتك بشكل أفضل لنندأ هل يمكن أن تحدثني عن موقف الترارف فيه إلى التكيف مع مشروع أو مبادرة جديدة في العمل كيف كان شعورك وانت تتعلم شيء جديدة 

**Model 4: mms-tts-ara  (`facebook/mms-tts-ara`)**
*   **Audio Output:** results/audio/mms-tts-ara.mp3
*   **Expressiveness / Quality Notes:** highly intelligible; moderate pitch variation. ASR-CER=0.094, F0 std=42.1 Hz, voiced ratio=0.873. Arch: vits (~36M).
*   **Memory Consumed:** 524.0 MB VRAM (peak, process), 1392.8 MB CPU RSS
*   **ASR heard:** الصباح الخير وشكرا لك على وقتك للتحدث معي اليومي أطلعوا إلى فهم أسلوبك في العمل وشخصيتك بشكل أفضل لبدأ هل يمكنك أن تحدثني عن موقف إطرار فيه إلى التكيف مع مشروع أو مبادلة جديدة في العملي كيف كان الشعورك وانت تتعلم شيء جدي

## Excluded models (over the 700 MB budget)

These are more expressive but do not fit the constraint:

*   `oddadmix/chatterbox-egyptian-v0` — Chatterbox stack ~5.3GB weights, ~4GB VRAM
*   `openbmb/VoxCPM2` — 2B params, >4GB VRAM
*   `bosonai/higgs-tts-3-4b` — 5B params, well over budget
*   `mistralai/Voxtral-4B-TTS-2603` — 4B params, over budget
*   `OpenMOSS-Team/MOSS-TTS-v1.5` — 8B params, over budget
*   `nvidia/magpie_tts_multilingual_357m` — gated + NeMo runtime; 357M borderline & fragile on py3.13
