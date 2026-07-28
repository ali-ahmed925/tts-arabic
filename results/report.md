# Arabic TTS Evaluation — small/expressive models with a male-voice guardrail

**Test text (fixed):**

> صباح الخير، وشكرًا لك على وقتك للتحدّث معي اليوم. أتطلّع إلى فهم أسلوبك في العمل وشخصيتك بشكل أفضل. لِنبدأ. هل يمكنك أن تُحدّثني عن موقفٍ اضطررت فيه إلى التكيّف مع مشروع أو مبادرة جديدة في العمل؟ كيف كان شعورك وأنت تتعلّم شيئًا جديدًا تمامًا؟

**Hardware:** NVIDIA RTX 4050 Laptop (6 GB VRAM), 12-core CPU, 30 GB RAM.  
**Two tiers evaluated:** a strict **≤700 MB VRAM** tier (small VITS/SpeechT5/ONNX models) and a **≤~2 GB VRAM expressive tier** (XTTS-v2 etc.). CPU/ONNX models report VRAM=0 and should be read on the CPU-RAM column.

**Metrics:** peak VRAM = process GPU memory via nvidia-smi (includes CUDA context). `ASR-CER` = Whisper-small round-trip character error rate (intelligibility, lower is better). `F0 std` = pitch standard deviation in Hz (expressiveness proxy, higher = less monotone). `RTF` = inference time / audio duration (lower is faster).

## Summary

| Model | Arch | Peak VRAM | RTF | ASR-CER | F0 std | Voice | Status |
|---|---|---|---|---|---|---|---|
| lahja-sa-huba-v1 | vits | 412.0 MB | 0.11 | 0.16 | 19.3 Hz | female (fixed) | ok |
| seyedali-mms-ar | vits | 536.0 MB | 0.075 | 0.11 | 40.6 Hz | male (fixed) | ok |
| xtts-v2 | xtts | 2466.0 MB | 0.439 | 0.315 | 15.7 Hz | male (sel) | ok |
| audar-tts-v1-flash-q4 | audar_gguf |  MB |  |  |  Hz |  | failed |
| omnivoice | omnivoice | 2736.0 MB | 0.251 | 0.0 | 34.2 Hz | male (sel) | ok |
| vits-ar-sa-ahmad | vits | 346.0 MB | 0.103 | 0.254 | 9.9 Hz | male (fixed) | ok |
| vits-ar-sa-huba-v2 | vits | 404.0 MB | 0.065 | 0.177 | 19.8 Hz | female (fixed) | ok |
| vits-ar | vits | 384.0 MB | 0.06 | 0.232 | 29.8 Hz | male (fixed) | ok |
| vits-ar-sa-A | vits | 346.0 MB | 0.07 | 0.221 | 18.9 Hz | male (fixed) | ok |
| vits-ar-ye-sa | vits | 344.0 MB | 0.092 | 0.309 | 15.6 Hz | female (fixed) | ok |
| mms-tts-ara | vits | 496.0 MB | 0.048 | 0.144 | 40.1 Hz | male (fixed) | ok |
| speecht5-clartts-ar | speecht5 | 546.0 MB | 0.111 | 0.144 | 13.4 Hz | male (sel) | ok |
| supertonic-3 | supertonic | 0.0 MB | 0.24 | 0.044 | 33.7 Hz | male (sel) | ok |
| voicetut-tts | omnivoice | 0.0 MB | 9.348 | 0.033 | 17.5 Hz | male (sel) | ok |

## Per-model detail

**Model 1: lahja-sa-huba-v1  (`wasmdashai/lahja-sa-huba-v1`)**
*   **Audio Output:** results/audio/lahja-sa-huba-v1.mp3
*   **Expressiveness / Quality Notes:** intelligible with some errors; low pitch variation (flat/monotone). ASR-CER=0.16, F0 std=19.3 Hz, voiced ratio=0.876. Arch: vits (~36M).
*   **Voice (male-guardrail):** fixed/baked-in voice — detected **female** (F0 mean 187.3 Hz); ⚠️ VIOLATES the male-only guardrail
*   **Memory Consumed:** 412.0 MB VRAM (peak, process), 1404.6 MB CPU RSS
*   **ASR heard:** صباحة الخير وشكرا لك على وقتك للتحدث معي هو أطلع إلا فهم أسلوبك في العمل وشخصيةك بشكل أفضل اللي بطأني هل يمكنك أن تحدثني عن موقع فالطرد في إلتكيف مع مشوع ومبادرة جديدة في العمل كيف كان شعورك وانت تسعلم شيئا جديد انت مممز

**Model 2: seyedali-mms-ar  (`SeyedAli/Arabic-Speech-synthesis-MMS`)**
*   **Audio Output:** results/audio/seyedali-mms-ar.mp3
*   **Expressiveness / Quality Notes:** highly intelligible; moderate pitch variation. ASR-CER=0.11, F0 std=40.6 Hz, voiced ratio=0.898. Arch: vits (~36M).
*   **Voice (male-guardrail):** fixed/baked-in voice — detected **male** (F0 mean 133.6 Hz); ✅ satisfies the male-only guardrail
*   **Memory Consumed:** 536.0 MB VRAM (peak, process), 1381.9 MB CPU RSS
*   **ASR heard:** الصبح الخير وشكرا لك على وقتك لتحدث معي اليومي أطلعوا إياه فهم أسلوبك في العمل وشخصيتك بشكل أفضل لبدأوا هل يمكنك أن تحدثنا عن موقف الترارف فيه إلى التكيف مع مشروع أو مبادرة جديدة في العمل. كيف كان الشعورك وانت تتعلم شيئا

**Model 3: xtts-v2  (`coqui/XTTS-v2`)**
*   **Audio Output:** results/audio/xtts-v2.mp3
*   **Expressiveness / Quality Notes:** intelligible with some errors; low pitch variation (flat/monotone). ASR-CER=0.315, F0 std=15.7 Hz, voiced ratio=0.131. Arch: xtts (~460M).
*   **Voice (male-guardrail):** selectable → forced to **male** (detected male, F0 mean 95.0 Hz)
*   **Memory Consumed:** 2466.0 MB VRAM (peak, process), 4524.3 MB CPU RSS
*   **ASR heard:** صباح الحير وشكرا لك على وقتك للتحدث مع اليوم أتطلع إلى فهم أسلوبك في العمل وشحصيتك بشكل أفضل لنبدأ هل يمكنك أن تحدثني عن موقف ان تررت فيه إلى التكيف مع مشوع أو مبدرة جديدة في العمل كيف؟ موقف أن تررت فيه إلى التكيف مع مشو

**Model 4: audar-tts-v1-flash-q4  (`audarai/Audar-TTS-V1-Flash`)**
*   **Audio Output:** (not generated)
*   **Expressiveness / Quality Notes:** FAILED / skipped — Expressive LLM-codec TTS; fits budget only when 4-bit quantized. | ERROR: Audar GGUF requires llama.cpp + NeuCodec decoder tooling not present; handled as a documented stretch attempt.
*   **Voice (-guardrail):** n/a
*   **Memory Consumed:**  MB VRAM (peak, process),  MB CPU RSS

**Model 5: omnivoice  (`k2-fsa/OmniVoice`)**
*   **Audio Output:** results/audio/omnivoice.mp3
*   **Expressiveness / Quality Notes:** highly intelligible; moderate pitch variation. ASR-CER=0.0, F0 std=34.2 Hz, voiced ratio=0.874. Arch: omnivoice (~0.6B).
*   **Voice (male-guardrail):** selectable → forced to **male** (detected male, F0 mean 139.4 Hz)
*   **Memory Consumed:** 2736.0 MB VRAM (peak, process), 4706.6 MB CPU RSS
*   **ASR heard:** صباح الخير وشكرا لك على وقتك للتحدث معي اليوم أتطلع إلى فهم أسلوبك في العمل وشخصيتك بشكل أفضل لنبدأ هل يمكنك أن تحدثني عن موقف إضطررت فيه إلى التكيف مع مشروع أو مبادرة جديدة في العمل كيف كان شعورك وانت تتعلم شيئا جديدا ت

**Model 6: vits-ar-sa-ahmad  (`wasmdashai/lahja-sa-ahmad-v1`)**
*   **Audio Output:** results/audio/vits-ar-sa-ahmad.mp3
*   **Expressiveness / Quality Notes:** intelligible with some errors; low pitch variation (flat/monotone). ASR-CER=0.254, F0 std=9.9 Hz, voiced ratio=0.17. Arch: vits (~36M).
*   **Voice (male-guardrail):** fixed/baked-in voice — detected **male** (F0 mean 119.9 Hz); ✅ satisfies the male-only guardrail
*   **Memory Consumed:** 346.0 MB VRAM (peak, process), 1428.9 MB CPU RSS
*   **ASR heard:** وشكرا لك على وقت التحدث معي اللهم لا تطلع لهم أسوه في العمل وشخصيت بشكل أثر لابده هل يمكنك أن تحدث عن موقف وطرد فيه للتكيف مع مشروعهم بايدة شديدة في العمل كيف كان شعورك وأنت تعلم شيء شديدة تمامة

**Model 7: vits-ar-sa-huba-v2  (`wasmdashai/vits-ar-sa-huba-v2`)**
*   **Audio Output:** results/audio/vits-ar-sa-huba-v2.mp3
*   **Expressiveness / Quality Notes:** intelligible with some errors; low pitch variation (flat/monotone). ASR-CER=0.177, F0 std=19.8 Hz, voiced ratio=0.88. Arch: vits (~83M).
*   **Voice (male-guardrail):** fixed/baked-in voice — detected **female** (F0 mean 186.9 Hz); ⚠️ VIOLATES the male-only guardrail
*   **Memory Consumed:** 404.0 MB VRAM (peak, process), 1417.3 MB CPU RSS
*   **ASR heard:** صباحة خير وشكرا لك على وقتك للتحدث معي يوم وصباحة طلاقة اللفهم أسلوبك في العمل لشخصيتك بشكل أفضل ليابدان هل يمكنك ان تحدثني عن موقف الطررت في اللي التكيف مع مشوع أو موبادر جديد في العمل كيف كان شعلك وانت تستعلم شيئا جديد

**Model 8: vits-ar  (`wasmdashai/vits-ar`)**
*   **Audio Output:** results/audio/vits-ar.mp3
*   **Expressiveness / Quality Notes:** intelligible with some errors; moderate pitch variation. ASR-CER=0.232, F0 std=29.8 Hz, voiced ratio=0.801. Arch: vits (~36M).
*   **Voice (male-guardrail):** fixed/baked-in voice — detected **male** (F0 mean 144.2 Hz); ✅ satisfies the male-only guardrail
*   **Memory Consumed:** 384.0 MB VRAM (peak, process), 1410.0 MB CPU RSS
*   **ASR heard:** سلاحي خيل وشكل لكال وقتك للتحدث معي لو أطلعوا إلى فهم أسلوب كيف لعمل شخصيتك بالشكل أفضل لا بلا هل يمكنك أن تحدثي عن مخفر؟ تررتوا فيه إلى التكيف مع مشواء أو مبدأة جديلة في فنعمل كيف كان شعورك وانتو تتعلم شيء الجديد وتماما

**Model 9: vits-ar-sa-A  (`wasmdashai/vits-ar-sa-A`)**
*   **Audio Output:** results/audio/vits-ar-sa-A.mp3
*   **Expressiveness / Quality Notes:** intelligible with some errors; low pitch variation (flat/monotone). ASR-CER=0.221, F0 std=18.9 Hz, voiced ratio=0.17. Arch: vits (~83M).
*   **Voice (male-guardrail):** fixed/baked-in voice — detected **male** (F0 mean 125.8 Hz); ✅ satisfies the male-only guardrail
*   **Memory Consumed:** 346.0 MB VRAM (peak, process), 1435.9 MB CPU RSS
*   **ASR heard:** ووحي الخريح وشكرا لك على وقت التحديث معي اللهم أثطل على لفهم أسوح في العمل وشخصيط بشكل أفضل لابده هل يمكنك أن تحدث عن موقف تضرط فيه للتكيف مع مشروعهم باديوه جديد في العمل كيف كان شعورك وانت تعلم شيء جديد التمامة

**Model 10: vits-ar-ye-sa  (`wasmdashai/vits-ar-ye-sa`)**
*   **Audio Output:** results/audio/vits-ar-ye-sa.mp3
*   **Expressiveness / Quality Notes:** intelligible with some errors; low pitch variation (flat/monotone). ASR-CER=0.309, F0 std=15.6 Hz, voiced ratio=0.935. Arch: vits (~36M).
*   **Voice (male-guardrail):** fixed/baked-in voice — detected **female** (F0 mean 161.8 Hz); ⚠️ VIOLATES the male-only guardrail
*   **Memory Consumed:** 344.0 MB VRAM (peak, process), 1414.1 MB CPU RSS
*   **ASR heard:** وصباح الخير وشكرا لك علىه بسكرة التحديث معي لله أفضل على فهم أزوائه في العمل وشخصيته بشكل عفظة للمدى وليه متنك أنت حديثني عن مواة تقدرت في إلا التتيف مع مجوعة وبادة جديدة في العمل ككتان جعلك وانت تعلم شيء جديدة ما ما

**Model 11: mms-tts-ara  (`facebook/mms-tts-ara`)**
*   **Audio Output:** results/audio/mms-tts-ara.mp3
*   **Expressiveness / Quality Notes:** highly intelligible; moderate pitch variation. ASR-CER=0.144, F0 std=40.1 Hz, voiced ratio=0.866. Arch: vits (~36M).
*   **Voice (male-guardrail):** fixed/baked-in voice — detected **male** (F0 mean 133.1 Hz); ✅ satisfies the male-only guardrail
*   **Memory Consumed:** 496.0 MB VRAM (peak, process), 1381.8 MB CPU RSS
*   **ASR heard:** الصباح الخيري وشكر لك على وقتك للتحدث معي يومي أطلعوا إلى فهم أسلوبك في العمل وشخصيتك بشكل أفضل لذلك هل يمكنك أن تحدثنا عن موقف الطرورت فيه إلى تكيف ما مشروع أو مبادرة جديدة في العمني. إيه فكنا شعورك وانت تتعلم شيئا جديد

**Model 12: speecht5-clartts-ar  (`MBZUAI/speecht5_tts_clartts_ar`)**
*   **Audio Output:** results/audio/speecht5-clartts-ar.mp3
*   **Expressiveness / Quality Notes:** highly intelligible; low pitch variation (flat/monotone). ASR-CER=0.144, F0 std=13.4 Hz, voiced ratio=0.633. Arch: speecht5 (~144M).
*   **Voice (male-guardrail):** selectable → forced to **male** (detected male, F0 mean 95.1 Hz)
*   **Memory Consumed:** 546.0 MB VRAM (peak, process), 1296.1 MB CPU RSS
*   **ASR heard:** صباح الخير وشكرا لك على وقتك للتحدث تمعي لهم أطلع إلى فهم أسوبك في العمل وشخصيتك بشكل أفضل هل يمكنك أنت حد ثني عن موقف الطرار فيه إلى التكيه أو مبادرة جديدة في العمل كيف كان شعورك وانت تعلم شيء جديد تماما

**Model 13: supertonic-3  (`Supertone/supertonic-3`)**
*   **Audio Output:** results/audio/supertonic-3.mp3
*   **Expressiveness / Quality Notes:** highly intelligible; moderate pitch variation. ASR-CER=0.044, F0 std=33.7 Hz, voiced ratio=0.501. Arch: supertonic (ONNX (small)).
*   **Voice (male-guardrail):** selectable → forced to **male** (detected male, F0 mean 130.6 Hz)
*   **Memory Consumed:** 0.0 MB VRAM (peak, process), 938.1 MB CPU RSS
*   **ASR heard:** صباح الخير وشكرا لك على وقتك للتحدث معي اليوم أطلع إلى فهم أسلوبك في العمل وشخصيتك بشكل أفضل لنبدأ هل يمكنك أن تحدثني عن موقف إنتررت فيه للتكيف مع مشروع أو مبادرة جديدة في العمل كيف كان شعورك وانت تتعلم شيئ جديدنا تماما

**Model 14: voicetut-tts  (`mohammedaly22/VoiceTut-TTS`)**
*   **Audio Output:** results/audio/voicetut-tts.mp3
*   **Expressiveness / Quality Notes:** highly intelligible; low pitch variation (flat/monotone). ASR-CER=0.033, F0 std=17.5 Hz, voiced ratio=0.657. Arch: omnivoice (~612M).
*   **Voice (male-guardrail):** selectable → forced to **male** (detected male, F0 mean 107.0 Hz)
*   **Memory Consumed:** 0.0 MB VRAM (peak, process), 0.0 MB CPU RSS
*   **ASR heard:** صبح الخير وشكرا لك على وقتك للتحدث مع اليوم أطلع إلى فهم أسلوبك في العمل وشخصيتك بشكل أفضل لنبدأ هل يمكن أن تحدثني عن موقف التررت فيه إلى التكيف مع مشروع أو مبادرة جديدة في العمل كيف كان شعورك وانت تتعلم شيئا جديدا تماما

## Excluded models (over the 700 MB budget)

These are more expressive but do not fit the constraint:

*   `oddadmix/chatterbox-egyptian-v0` — Chatterbox stack ~5.3GB weights, ~4GB VRAM
*   `openbmb/VoxCPM2` — 2B params, >4GB VRAM
*   `bosonai/higgs-tts-3-4b` — 5B params, well over budget
*   `mistralai/Voxtral-4B-TTS-2603` — 4B params, over budget
*   `OpenMOSS-Team/MOSS-TTS-v1.5` — 8B params, over budget
*   `nvidia/magpie_tts_multilingual_357m` — gated + NeMo runtime; 357M borderline & fragile on py3.13
