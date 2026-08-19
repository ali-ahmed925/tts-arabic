# Arabic TTS Model Evaluation & Optimization Study

> **Note on this document's provenance:** this file was empty prior to this audit — there was no pre-existing detailed version to revise. Every claim below was built directly from repository artifacts (CSVs, committed audio, scripts, config, git history), not from prior conversation. Where a number, script, or artifact could not be found, that is stated explicitly rather than filled in.

## 1. Executive Summary

We evaluated a wide range of open-source Arabic-capable TTS models against **Supertonic-3** (`Supertone/supertonic-3`), our resource-efficient baseline, against a practical VRAM target of **~500–700 MB**, stretched at most to **~1 GB**.

- **Supertonic-3** is the strongest all-round resource-efficient baseline in the evidence we have: near-zero measured GPU footprint, fast, intelligible, multilingual, 10 built-in voices.
- Every model that clearly *beat* Supertonic on expressiveness/voice-cloning quality (**XTTS-v2**, **OmniVoice**, **VoiceTut-TTS**) used **2–5.5 GB** of memory — 3–8x over budget — and was rejected on resource grounds.
- A dedicated optimization study on OmniVoice (INT8, CPU offload, `torch.compile`, FlashInfer) did not bring it near the target; every technique either left VRAM in the 2.3–3.5 GB range, made inference much slower, or both.
- Lightweight, in-budget models (MMS Arabic, SeyedAli MMS, SpeechT5, five `wasmdashai` VITS variants) fit comfortably under 700 MB but were judged worse than Supertonic on quality and/or failed the male-voice requirement (fixed female voices in 3 of 6 cases).
- **`audarai/Audar-TTS-V1-Flash` (Q4 GGUF)** looked promising on paper but could not be benchmarked at all — the required `llama.cpp` + NeuCodec decoder tooling was unavailable in the environment.
- **MOSS-TTS-Nano** (`OpenMOSS-Team/MOSS-TTS-Nano`) is documented, on the strength of one committed CSV row and one committed audio file, as **781 MB peak VRAM** with voice cloning — within the stretched ~1 GB ceiling. This document recommends it as the current best-supported candidate for the voice-cloning use case, **not** as a definitive best-in-class result — see §9 and §15 for the exact basis and its limits.
- **This is a scoped conclusion, not an exhaustive one.** Several names raised during the search (Piper Arabic, `VITS-OpenBible-Arabic-Standard`, and a handful of other Hugging Face listings) have **no corroborating artifact anywhere in this repository** — no CSV row, no audio, no config entry, no script. They are listed in §11 as uncorroborated, not as tested-and-rejected.

## 2. Objective and Constraints

Find an Arabic TTS model that can outperform, or at least credibly compete with, Supertonic while staying inside a tight memory budget, balancing:

- Arabic speech quality and intelligibility
- Latency / real-time factor (RTF)
- Peak VRAM usage
- Ability to actually run on the target hardware
- Voice quality and voice control (fixed vs. selectable vs. cloning)
- Practical deployability

**Budget:** ~500–700 MB VRAM as the original target, stretched to ~1 GB as a hard outer limit. Meeting the VRAM number alone is **not** sufficient grounds for acceptance — a model must also be intelligible, fast enough to be practical, and (where relevant) satisfy the male-voice requirement. §15 applies this explicitly rather than accepting on VRAM alone.

## 3. Evaluation Methodology

This was **not** a single uniform benchmark run. Results come from at least four distinct harness invocations, on at least two different machines, and this is reflected throughout rather than papered over.

**What actually produced each artifact, verified against git history and the scripts themselves:**

| Source | Script present in repo? | Hardware (as documented) | Covers |
|---|---|---|---|
| `results/results.csv` / `results/report.md` | Yes — `tts_eval/run.py` + `tts_eval/adapters.py`, orchestrated per-model in an isolated subprocess | RTX 4050 Laptop, 6 GB VRAM (per `results/report.md` header) | Supertonic (main row), XTTS-v2 (aggregate), OmniVoice (baseline + a cpu-offload row), MMS, SeyedAli MMS, SpeechT5, 5x `wasmdashai` VITS, VoiceTut-TTS, Audar (failed) |
| `supertonic_results/results.csv` | Yes — `tts_eval/supertonic_male_voices.py`, explicitly reuses the same `ResourceSampler`/`monitor.py`/adapter code as the main harness | Same machine as above (script imports the shared `config.py`) | Supertonic M1–M5 male-voice sweep |
| `xtts_results/results.csv` | **No** — the commit that added this CSV (`547627d`) only touched `monitor.py`/`run.py`/`make_report.py` plus the CSV+WAVs themselves; no standalone "sweep 10 speakers" script was committed | Not stated in the CSV or commit; presumed same environment as the main harness given shared code paths, not confirmed | XTTS-v2, 10 individual speakers |
| `results/omnivoice/csv/omnivoice_optimization_results.csv` | **Partially** — `omnivoice`, `omnivoice-int8`, and `omnivoice-cpu-offload` are registered in `tts_eval/config.py`'s `MODELS` list and are reproducible from committed code. **`omnivoice-torch-compile` and `omnivoice-flashinfer` are not in `config.py` at all** — their CSV rows and audio exist, but the code that produced them is not present in this repository. | Not stated in the CSV | OmniVoice FP16 / INT8 / CPU-offload / torch-compile / FlashInfer |
| `results/moss-tts-nano/csv/moss_tts_nano_result.csv` | **No** — no benchmarking script for MOSS-TTS-Nano exists anywhere in this repository (the `MOSS-TTS-Nano/` directory contains only the vendor's own `app.py`/`infer.py`/runtime files, not a benchmark harness) | Tesla T4 (per the CSV's own `notes` field) | MOSS-TTS-Nano |
| `results_omnivoice/results.csv` | Same harness as row 1, appears to be an earlier/exploratory invocation | Not stated | One anomalous OmniVoice row (see below) — not used for conclusions |

**Practical effect of the "no script" gaps:** the CSV rows and audio files for XTTS's 10-speaker sweep, OmniVoice's torch-compile/FlashInfer variants, and all of MOSS-TTS-Nano are genuine, committed artifacts and are reported as such below — but their exact measurement methodology cannot be independently re-verified from this repository's code the way the main-harness and Supertonic-sweep rows can. This is noted per-section, not treated as disqualifying.

**Known numeric discrepancies between sources** (kept, not silently resolved):
- `README.md`'s inline results table (mms-tts-ara, seyedali-mms-ar, lahja-sa-huba-v1, speecht5-clartts-ar, supertonic-3) shows different VRAM/CER/F0 numbers than the current `results/results.csv` / `results/report.md`. Git history shows the CSV was regenerated (`a9bd0e2 "results: regenerate CSV, report, and audio for 12 working models"`) after the README table was last written. **This document treats `results/results.csv` + `results/report.md` as authoritative**; the README table is an earlier, superseded snapshot and should be updated or removed to avoid confusion.
- **OmniVoice FP16 baseline** exists with different numbers in three places: `results/results.csv` (`omnivoice`: 2736 MB VRAM / 4706.6 MB RSS / RTF 0.251 / CER 0.0), `omnivoice_optimization_results.csv` (`omnivoice-fp16`: 2776 MB VRAM / 4639.1 MB RSS / RTF 0.275 / CER 0.028), and `results_omnivoice/results.csv` (`omnivoice`: an anomalous 0 MB VRAM / 2562.4 MB RSS / 192.5 s inference / RTF 9.694 / 1866 s load time). §8 uses the `omnivoice-fp16` row from the optimization CSV as the reference point, since it's the row measured alongside INT8/CPU-offload/torch-compile/FlashInfer for direct comparison. The `results/results.csv` row is a consistent, independent corroboration at the same order of magnitude. The `results_omnivoice/results.csv` row (nearly half an hour just to load, 0 MB VRAM for a GPU-resident model) looks like a broken/exploratory measurement and is excluded from all conclusions.
- **OmniVoice CPU-offload** likewise has two versions: `omnivoice_optimization_results.csv` (2522 MB VRAM / RTF 0.358) vs. a much worse `results/results.csv` row (0 MB VRAM / RTF 8.235). The optimization CSV's version is used in §8 as representative.
- **XTTS-v2 peak VRAM**: the aggregate run in `results/results.csv` reports a real GPU measurement of 2466 MB. The 10-speaker sweep (`xtts_results/results.csv`) reports **`peak_vram_mb = 0.0` for all 10 rows** — see §5.1 for why this is treated as a measurement gap, not a real zero.

**Parameter-count caveat:** every "~Nparams" figure in this document (e.g. ~460M for XTTS-v2, ~0.6B for OmniVoice, ~36M for the VITS models, ~0.1B for MOSS-TTS-Nano) is a model-card/config/vendor-README figure carried through from `tts_eval/config.py` or the model's own documentation. None of them were independently computed by summing loaded model weights in this repository's harness. They are reported as descriptive metadata, not as a benchmarked measurement.

## 4. Baseline: Supertonic-3

**HF repo:** `Supertone/supertonic-3` · **Architecture:** ONNX, small multilingual TTS · **Voices:** 10 built-in · **Arabic:** supported (multilingual)

Peak VRAM is reported as **0.0 MB**. The monitor (`tts_eval/monitor.py`) measures GPU memory by querying `nvidia-smi --query-compute-apps=pid,used_memory` for this process's PID. Supertonic runs on CPU via ONNX Runtime and never opens a CUDA context, so `nvidia-smi` legitimately has nothing to report for it — this is a genuine (not merely a measurement-limitation) near-zero GPU number. It is **not** the same situation as XTTS's 0.0 VRAM in §5.1, which is a real GPU-resident model with a measurement gap. CPU RSS is the meaningful memory figure for Supertonic.

| Metric | Value |
|---|---|
| Peak VRAM (GPU) | 0.0 MB (CPU/ONNX — no CUDA context opened) |
| Peak RSS (CPU) | 938.1 MB |
| Load time | 0.91 s |
| Inference time | 5.62 s |
| Audio duration | 23.41 s |
| RTF | 0.24 |
| ASR CER | 0.044 |
| Voice control | selectable (male voice forced/tested) |

Source: `results/results.csv` (`supertonic-3` row) · Audio: [`results/audio/supertonic-3.mp3`](results/audio/supertonic-3.mp3) — **verified present.**

Supertonic is the strongest baseline in this evidence set on: low resource usage, fast inference, reasonable Arabic quality, multilingual support, multiple built-in voices, and simple ONNX deployment.

### 4.1 Supertonic male-voice variation study

A separate, fully-reproducible run (`tts_eval/supertonic_male_voices.py` → `supertonic_results/results.csv`) tested 5 of Supertonic's built-in male voices to check voice-to-voice consistency. **This is not representative of all 10 voices** — it targeted variance specifically, and several voices showed severe repetition artifacts, visible directly in the ASR transcripts (the same word/phrase repeated for hundreds of characters):

| Voice | Infer (s) | RTF | ASR CER | Notes |
|---|---|---|---|---|
| M1 | 7.33 | 0.313 | 0.79 | Degraded but not looping |
| M2 | 7.42 | 0.322 | 3.166 | Severe repetition — unusable |
| M3 | 5.27 | 0.263 | 2.536 | Severe repetition — unusable |
| M4 | 5.87 | 0.264 | 8.21 | Severe repetition — unusable |
| M5 | 6.41 | 0.274 | 5.138 | Severe repetition — unusable |

Source: `supertonic_results/results.csv` · Audio (all verified present): [`M1.wav`](supertonic_results/M1.wav) · [`M2.wav`](supertonic_results/M2.wav) · [`M3.wav`](supertonic_results/M3.wav) · [`M4.wav`](supertonic_results/M4.wav) · [`M5.wav`](supertonic_results/M5.wav)

## 5. Voice-Cloning / Expressive Models

### 5.1 XTTS-v2 — `coqui/XTTS-v2` (~460M params)

Aggregate/reference run (male voice forced, from `results/results.csv`, produced by the standard `tts_eval` harness):

| Metric | Value |
|---|---|
| Peak VRAM | 2466 MB |
| Peak RSS | 4524.3 MB |
| Inference time | 10.57 s |
| Audio duration | 24.08 s |
| RTF | 0.439 |
| ASR CER | 0.315 |

Audio: [`results/audio/xtts-v2.mp3`](results/audio/xtts-v2.mp3) — verified present.

**Separate 10-speaker benchmark** (`xtts_results/results.csv`) — do not confuse with the row above:

| Speaker | Peak VRAM (CSV) | RSS (MB) | Infer (s) | Audio (s) | RTF | ASR CER |
|---|---|---|---|---|---|---|
| Baldur Sanjin | **0.0 MB** — see note below | 4185.4 | 73.82 | 25.61 | 2.882 | 0.072 |
| Aaron Dreschner | 0.0 MB | 5466.5 | 72.94 | 24.69 | 2.954 | 0.055 |
| Dionisio Schuyler | 0.0 MB | 5516.0 | 78.12 | 26.68 | 2.928 | 0.033 |
| Damien Black | 0.0 MB | 5493.6 | 64.46 | 22.97 | 2.806 | 0.061 |
| Royston Min | 0.0 MB | 5406.2 | 74.77 | 24.40 | 3.064 | 0.039 |
| Filip Traverse | 0.0 MB | 5418.9 | 81.58 | 29.05 | 2.808 | 0.028 |
| Adde Michal | 0.0 MB | 5397.4 | 65.43 | 23.29 | 2.809 | 0.050 |
| Abrahan Mack | 0.0 MB | 5429.2 | 60.58 | 22.68 | 2.671 | 0.022 |
| Ilkin Urbano | 0.0 MB | 5388.4 | 82.08 | 27.94 | 2.938 | 0.044 |
| Damjan Chapman | 0.0 MB | 5406.6 | 72.22 | 25.23 | 2.862 | 0.033 |

Source: `xtts_results/results.csv` · Audio (all verified present): [`speaker_1.wav`](xtts_results/speaker_1.wav) … [`speaker_10.wav`](xtts_results/speaker_10.wav) (Baldur Sanjin → Damjan Chapman, in table order)

> **`0.0 MB` here is `UNVERIFIED` as a real VRAM measurement, not accepted as a true zero.** XTTS-v2 is a GPU-resident PyTorch model (unlike Supertonic's ONNX/CPU case), and the aggregate run above measured real GPU usage of 2466 MB for the same architecture. `monitor.py`'s `_proc_gpu_mb()` depends on `nvidia-smi --query-compute-apps` successfully matching this process's PID, which is a known-unreliable query on some driver configurations (notably Windows WDDM mode) and can silently return nothing. No script for this specific sweep is committed to the repo (§3), so the exact cause can't be confirmed — but treating these 10 rows as "0 MB VRAM used" would contradict the aggregate measurement of the same model and is not adopted here. RSS (4.2–5.5 GB) is the reliable memory signal for this run.

**Conclusion: REJECTED.** Voice quality was consistently reasonable (CER mostly 0.02–0.07), but RSS of 4.2–5.5 GB, confirmed GPU VRAM of ~2.5 GB from the aggregate run, and RTF ~2.7–3.1x real-time are all far outside the resource target.

### 5.2 OmniVoice — `k2-fsa/OmniVoice` (~0.6B params)

Baseline (fp16, male voice forced, `omnivoice-fp16` row from `results/omnivoice/csv/omnivoice_optimization_results.csv`, registered in `tts_eval/config.py` and reproducible):

| Metric | Value |
|---|---|
| Peak VRAM | 2776 MB |
| Peak RSS | 4639.1 MB |
| Load time | 11.39 s |
| Inference time | 5.4 s |
| Audio duration | 19.66 s |
| RTF | 0.275 |
| ASR CER | 0.028 |

An earlier, independent run in `results/results.csv` recorded a consistent (not identical) result: 2736 MB VRAM / 4706.6 MB RSS / RTF 0.251 / CER 0.0. Audio: [`results/audio/omnivoice.mp3`](results/audio/omnivoice.mp3) · [`results/omnivoice/audio/omnivoice-fp16.mp3`](results/omnivoice/audio/omnivoice-fp16.mp3) — both verified present.

Quality is reasonable but resource usage is roughly 3.5–4x the stretched budget.

**Conclusion: REJECTED as a directly-deployable model** — too heavy. A dedicated optimization attempt follows in §8.

### 5.3 VoiceTut-TTS — `mohammedaly22/VoiceTut-TTS` (OmniVoice architecture, ~612M params, Arabic Egyptian)

| Metric | Value |
|---|---|
| Peak VRAM (CSV) | 0.0 MB — same class of measurement gap as §5.1, this is a real GPU-loaded OmniVoice-architecture model |
| Manual memory observation | ~4.64 GB (recorded directly in the CSV's own `notes` field, not reconstructed here) |
| Load time | 3558.56 s |
| Inference time | 187.52 s |
| Audio duration | 20.06 s |
| RTF | 9.348 |
| ASR CER | 0.033 |

Source: `results/results.csv` (`voicetut-tts` row, notes: *"OmniVoice Arabic TTS with voice cloning support. Manual test used ~4.64GB memory."*) · Audio: [`results/audio/voicetut-tts.mp3`](results/audio/voicetut-tts.mp3) — verified present.

**Conclusion: REJECTED.** RTF of 9.3x real-time and a manually-observed ~4.6 GB memory footprint make this clearly unusable for the target deployment, independent of the VRAM-column measurement gap.

## 6. Lightweight Arabic Models (MMS / SpeechT5)

Closest models to Supertonic on resource usage, compared directly against it for that reason. All numbers from `results/results.csv`, produced by the standard, reproducible `tts_eval` harness.

| Model | HF repo | Params | Peak VRAM | Peak RSS | Infer (s) | RTF | ASR CER |
|---|---|---|---|---|---|---|---|
| MMS TTS Arabic | `facebook/mms-tts-ara` | ~36M | 496 MB | 1381.8 MB | 1.53 | 0.048 | 0.144 |
| SeyedAli Arabic MMS | `SeyedAli/Arabic-Speech-synthesis-MMS` | ~36M | 536 MB | 1381.9 MB | 2.39 | 0.075 | 0.11 |
| SpeechT5 (ClArTTS) | `MBZUAI/speecht5_tts_clartts_ar` | ~144M | 546 MB | 1296.1 MB | 2.92 | 0.111 | 0.144 |

Audio (all verified present): [`mms-tts-ara.mp3`](results/audio/mms-tts-ara.mp3) · [`seyedali-mms-ar.mp3`](results/audio/seyedali-mms-ar.mp3) · [`speecht5-clartts-ar.mp3`](results/audio/speecht5-clartts-ar.mp3)

All three comfortably meet the VRAM budget. Their CER (0.11–0.144) is worse than Supertonic's (0.044); qualitative review judged Supertonic better overall.

**Conclusion: REJECTED** — resource-efficient, but did not beat Supertonic on overall Arabic quality.

## 7. `wasmdashai` VITS Family (Lahja / dialect VITS)

Small (~36–83M param) single-speaker Arabic VITS checkpoints, all fixed-voice (gender cannot be selected — only detected via F0 mean vs. a 160 Hz threshold, per `config.py`'s `GENDER_F0_THRESHOLD_HZ`).

| Model | HF repo | Params | Peak VRAM | Infer (s) | RTF | ASR CER | Requested / Detected gender |
|---|---|---|---|---|---|---|---|
| lahja-sa-huba-v1 | `wasmdashai/lahja-sa-huba-v1` | ~36M | 412 MB | 2.25 | 0.11 | 0.16 | male / **female** ❌ |
| vits-ar-sa-ahmad | `wasmdashai/lahja-sa-ahmad-v1` | ~36M | 346 MB | 1.6 | 0.103 | 0.254 | male / male ✅ |
| vits-ar-sa-huba-v2 | `wasmdashai/vits-ar-sa-huba-v2` | ~83M | 404 MB | 1.34 | 0.065 | 0.177 | male / **female** ❌ |
| vits-ar | `wasmdashai/vits-ar` | ~36M | 384 MB | 1.15 | 0.06 | 0.232 | male / male ✅ |
| vits-ar-sa-A | `wasmdashai/vits-ar-sa-A` | ~83M | 346 MB | 1.11 | 0.07 | 0.221 | male / male ✅ |
| vits-ar-ye-sa | `wasmdashai/vits-ar-ye-sa` | ~36M | 344 MB | 1.19 | 0.092 | 0.309 | male / **female** ❌ |

Source: `results/results.csv` and `tts_eval/config.py` · Audio (all verified present, under [`results/audio/`](results/audio/)): `lahja-sa-huba-v1.mp3`, `vits-ar-sa-ahmad.mp3`, `vits-ar-sa-huba-v2.mp3`, `vits-ar.mp3`, `vits-ar-sa-A.mp3`, `vits-ar-ye-sa.mp3`

> The model registered as `vits-ar-sa-ahmad` points at HF repo `wasmdashai/lahja-sa-ahmad-v1` (per `config.py` and the CSV `hf_repo` column) — **not** `wasmdashai/vits-ar-sa-ahmad-v1`. Corrected here from the actual config/CSV.

**Conclusion:**
- `lahja-sa-huba-v1`, `vits-ar-sa-huba-v2`, `vits-ar-ye-sa` — **REJECTED — voice control** (fixed female voice despite a male requirement).
- `vits-ar-sa-ahmad`, `vits-ar`, `vits-ar-sa-A` — **REJECTED — quality** (CER 0.221–0.254, well above Supertonic's 0.044; correct gender but not competitive on intelligibility).

All six are attractive on VRAM alone (344–412 MB) but none beat Supertonic on the combination of quality and voice control.

## 8. OmniVoice Optimization Study

The base OmniVoice model (§5.2) was ~3–4x over budget, so a dedicated optimization study was run. **These are optimization experiments on one model, not separate TTS models.**

| Variant | Registered in `config.py`? | Peak VRAM | Peak RSS | Load (s) | Infer (s) | RTF | ASR CER | Outcome |
|---|---|---|---|---|---|---|---|---|
| FP16 (baseline) | Yes | 2776 MB | 4639.1 MB | 11.39 | 5.4 | 0.275 | 0.028 | Reference point |
| INT8 | Yes | 2352 MB | 4474.3 MB | 37.78 | 9.59 | 0.489 | 0.006 | VRAM down ~424 MB but still ~2.35 GB; load and inference both slower |
| CPU offload | Yes | 2522 MB | 4720.6 MB | 11.49 | 7.22 | 0.358 | 0.011 | VRAM barely moved (−254 MB vs. FP16); RTF got worse |
| `torch.compile` | **No — CSV/audio only, not reproducible from committed code** | 2736 MB | 3586.2 MB | 40.17 | 7.75 | 0.397 | 0.022 | No meaningful VRAM reduction; slower inference; much longer load time |
| FlashInfer | **No — CSV/audio only, not reproducible from committed code** | 3452 MB | 3610.2 MB | 20.31 | 361.3 | 18.368 | 0.017 | VRAM went **up**; inference collapsed to 18x real-time |

Source: `results/omnivoice/csv/omnivoice_optimization_results.csv` · Audio (all verified present): [`omnivoice-fp16.mp3`](results/omnivoice/audio/omnivoice-fp16.mp3) · [`omnivoice-int8.mp3`](results/omnivoice/audio/omnivoice-int8.mp3) · [`omnivoice-cpu-offload.mp3`](results/omnivoice/audio/omnivoice-cpu-offload.mp3) · [`omnivoice-torch-compile.mp3`](results/omnivoice/audio/omnivoice-torch-compile.mp3) · [`omnivoice-flashinfer.mp3`](results/omnivoice/audio/omnivoice-flashinfer.mp3)

> **Gender inconsistency across runs.** In `omnivoice_optimization_results.csv`, `requested_gender = male` for every row, but `detected_gender = female` for the **FP16**, **`torch.compile`**, and **FlashInfer** rows, while `detected_gender = male` for the **INT8** and **CPU-offload** rows. Separately, the main-harness `omnivoice` row in `results/results.csv` — nominally the same FP16 configuration — reports `detected_gender = male`. So the same model, at what should be the same settings, produced male in one committed run and female in another. No cause for this is recorded anywhere in the repository, and none is inferred here. Given the evaluation's male-voice requirement, this means the OmniVoice FP16/torch-compile/FlashInfer results should not be read as having cleanly satisfied that requirement — but it should equally not be read as OmniVoice definitively failing it, since other same-architecture runs detected male. It is reported here as an unresolved inconsistency in the available artifacts.

**Per-technique conclusions:**
- **INT8** reduced VRAM but not nearly enough — still ~2.35 GB against a ~1 GB ceiling — and both load and inference time regressed.
- **CPU offload** barely reduced VRAM (the model is small enough that little needed offloading) and made inference markedly slower.
- **`torch.compile`** did not solve the VRAM problem at all — VRAM was effectively unchanged from FP16, and load time nearly quadrupled from compilation overhead.
- **FlashInfer** was unsuccessful in this setup: VRAM increased and inference time exploded to RTF 18.37 (single run).

**Overall conclusion: none of the tested, practical optimizations brought OmniVoice anywhere close to the 500 MB – 1 GB target.** The lowest VRAM achieved (INT8, 2352 MB) is still roughly 2.3x the stretched ceiling.

> **Not present in this repository at all — do not treat as benchmarked:** 4-bit quantization, a TensorRT-LLM + FP8 + Triton pipeline, and a general ONNX export attempt. No CSV row, config entry, script, or audio artifact for any of the three exists anywhere in this repository. They are excluded from the table above entirely (not even as `UNVERIFIED` rows, since there is no artifact to anchor a row to) — see §11.
>
> **Orphaned artifact:** [`results/omnivoice/audio/omnivoice-4bit.mp3`](results/omnivoice/audio/omnivoice-4bit.mp3) exists on disk (verified) but has **no corresponding row in either `omnivoice_optimization_results.csv` or its backup**. This is the one piece of physical evidence that a 4-bit attempt happened, but with no metrics, no config, and no script attached to it, no numbers can be reported — it is listed here as an unexplained artifact, not a benchmarked result.

## 9. FINAL ACCEPTED MODEL — MOSS-TTS-Nano

**HF repo:** `OpenMOSS-Team/MOSS-TTS-Nano` · **Params:** ~0.1B, per the vendor's own README (`MOSS-TTS-Nano/README.md`) — **not** independently confirmed by our benchmark, since no local script inspected the loaded model · **Language:** Arabic · **Voice control:** voice cloning · **Hardware:** Tesla T4 (per the CSV's `notes` field — different GPU than the RTX 4050 used for the rest of the harness, see §3)

| Metric | Value |
|---|---|
| Peak VRAM | **781 MB** |
| Peak RSS | 872.86 MB |
| Inference time | 32.814 s |
| Audio duration | 27.44 s |
| RTF | 1.196 |
| ASR CER | 0.1736 |
| F0 std | 88.27 Hz |
| F0 mean | 159.87 Hz |
| Voiced ratio | 1.0 |
| Requested gender | male |
| Output sample rate | 48 kHz |

Source: `results/moss-tts-nano/csv/moss_tts_nano_result.csv` (notes: *"MOSS-TTS-Nano voice cloning; Tesla T4; 48 kHz output."*) · Audio: [`results/moss-tts-nano/audio/moss-tts-nano-voice-clone.wav`](results/moss-tts-nano/audio/moss-tts-nano-voice-clone.wav) — verified present (4,976,718 bytes). A second, larger, uncommitted take also exists at the repo root: `MOSS-TTS-Nano_clone_male.wav` (6,021,164 bytes, different file, different date — not the same recording, both are real).

**As noted in §3, no benchmarking script for MOSS-TTS-Nano exists in this repository.** This CSV row and audio file are genuine committed artifacts, but the exact code path that produced the VRAM/RTF/CER figures (e.g. whether VRAM was sampled the same way as `tts_eval/monitor.py` or via a different method entirely) cannot be independently re-verified from this repo. The figures are reported as-is, from the one artifact that exists, with this caveat attached rather than silently assumed equivalent to the main harness's methodology.

**Status: current best-supported candidate for the voice-cloning use case — not a definitive "best Arabic TTS model" claim.**

MOSS-TTS-Nano does **not** beat Supertonic on every metric: its ASR CER (0.1736) is markedly worse than Supertonic's (0.044), and its RTF (1.196, i.e. slower than real-time) is worse than Supertonic's (0.24). Acceptance here is not "it fits under 1 GB" alone — it is that, of the models with an actual committed artifact in this repository, it is the **only** one that combines (a) voice-cloning capability, (b) a peak VRAM figure inside the stretched ~1 GB ceiling, and (c) qualitatively acceptable — not best-in-class — Arabic male-voice output. Qualitative review judged Arabic clarity weaker than the best reference/Colab result seen during the broader search (that reference result itself has no corroborating artifact in this repository — see §11).

## 10. Models Blocked by Environment / Dependencies

### 10.1 Audar-TTS-V1-Flash — `audarai/Audar-TTS-V1-Flash` (Q4 GGUF, ~0.6B)

Investigated because it looked promising on paper: Arabic-first, expressive, voice cloning, ~0.6B class, available in Q4/Q5/Q8 GGUF variants. **Only the Q4 variant was ever registered or attempted** (`tts_eval/config.py`, id `audar-tts-v1-flash-q4`) — there is no config entry, script, or artifact anywhere in this repository for Q5 or Q8.

**Result: FAILED / NOT BENCHMARKED.** No VRAM, RTF, or CER figures exist — the run never got past model loading, and none are reported.

> Error recorded verbatim in the CSV: *"Audar GGUF requires llama.cpp + NeuCodec decoder tooling not present; handled as a documented stretch attempt."*

Source: `results/results.csv` (`audar-tts-v1-flash-q4` row, `status=failed`, all metric columns empty) · corroborated by `TODO_MODELS.md`, which independently describes the same blocker (needs a dedicated venv with `llama-cpp-python` + `neucodec`, kept isolated because installing `neucodec` elsewhere breaks the shared VITS environment).

**Status: BLOCKED — DEPENDENCY.** No VRAM/RTF/CER values are assigned, and none should be inferred.

## 11. Models Referenced But Not Corroborated in This Repository

A repository-wide search (config files, every CSV, `report.md` files, README, `TODO_MODELS.md`, git history, and the filesystem under the project directory) found **no supporting artifact** for the items below. Per the no-invention constraint on this document, none of their claimed figures are presented as verified results, and none are given fabricated `UNVERIFIED` numbers — they are listed as topics only.

| Claimed item | What was claimed (per prior discussion) | Search result |
|---|---|---|
| Piper Arabic ("Kareem Low") | ~383 MB peak VRAM; already integrated in a production pipeline with known-bad quality | No config entry, CSV row, script, or audio artifact found anywhere in the repository. |
| `VITS-OpenBible-Arabic-Standard` (`multilingual-tts/VITS-OpenBible-Arabic-Standard`) | ~998 MB checkpoint downloaded and inspected (`model_last.pth`, `speakers.pth`, `config.json`); Coqui `TTS` package install blocked by a Python-version incompatibility | No checkpoint files, config entry, or notes about this model found in the repository or on disk under the project directory. |
| A dedicated "Lahja-SA-Ahmad" benchmark at ~4.6 GB VRAM, 2.27 s inference, 13.36 s audio, RTF 0.170, 16 kHz | Standalone high-VRAM run | The only recorded result for this model family is `vits-ar-sa-ahmad` (`wasmdashai/lahja-sa-ahmad-v1`) in `results/results.csv` (§7): **346 MB VRAM**, RTF 0.103, CER 0.254 — an entirely different profile. No second, separate benchmark of this model exists. |
| `NightPrince/Fasih-TTS-V1`, `shangeth/Wren-TTS-0.5B-multi-expressive`, KaniTTS Arabic 400M, "Nipponjo" (~1.5B, exact identity unclear) | Discovered via HF search filters (`pipeline_tag=text-to-speech`, `language=ar`, `≤1B` params) | None of these names appear anywhere in the repository (code, docs, CSVs, or git history). No parameter count, VRAM figure, or rejection reason for any of them could be verified. |
| OmniVoice 4-bit quantization, TensorRT-LLM + FP8 + Triton, ONNX export | See §8 | No CSV row, script, or artifact found, other than the orphaned `omnivoice-4bit.mp3` noted in §8. |

These are **not** included in the master table (§13) or the rejection summary (§14) as benchmarked or even confirmed-investigated candidates, since there is no evidence in this repository that they were tested here. If work on them happened outside this repository, it left no trace here and would need its own artifacts committed before it can be documented as evaluated.

## 12. Models Investigated But Not Fully Benchmarked (evidenced)

Unlike §11, the items below **are** corroborated by repository content — `TODO_MODELS.md` and `tts_eval/config.py`'s `EXCLUSIONS` list — as genuinely considered and explicitly deferred or excluded, even though no benchmark run exists for any of them.

**Deferred (shortlisted, not yet integrated) — from `TODO_MODELS.md`:**

| Model | Reason deferred |
|---|---|
| `ResembleAI/chatterbox` (multilingual, incl. Arabic) | Pins `torch==2.6.0`/`transformers==5.2.0`, incompatible with the shared base env; runtime VRAM estimated ~2.5–3 GB, over the expressive-tier cap. |
| `nvidia/magpie_tts_multilingual_357m` | Gated on HuggingFace (license + token required) and needs the NeMo runtime, fragile on Python 3.13. |
| `fishaudio/fish-speech-1.5` | The PyPI package fails to install (`metadata-generation-failed`) and pulls a full training stack (`lightning`, `wandb`, etc.). |
| `Serveurperso/Qwen3-TTS-GGUF` (0.9B, Q4) | Hardest integration — GGUF + llama.cpp + audio codec decode; fits ≤2 GB only when quantized. |

**Rejected immediately (already known unsuitable) — from `tts_eval/config.py`'s `EXCLUSIONS` list:**

| Model | Reason |
|---|---|
| `oddadmix/chatterbox-egyptian-v0` | Chatterbox stack, ~5.3 GB weights, ~4 GB VRAM. |
| `openbmb/VoxCPM2` | 2B params, >4 GB VRAM. |
| `bosonai/higgs-tts-3-4b` | 5B params, well over budget. |
| `mistralai/Voxtral-4B-TTS-2603` | 4B params, over budget. |
| `OpenMOSS-Team/MOSS-TTS-v1.5` | 8B params, over budget. |
| `nvidia/magpie_tts_multilingual_357m` | Gated + NeMo runtime; 357M borderline and fragile on Python 3.13 (listed in both tables). |

None of these produced VRAM/RTF/CER numbers — they were excluded on parameter-count or dependency grounds before any run was attempted, and are documented here as such rather than given fabricated metrics.

## 13. Master Comparison Table

VRAM/RSS are peak figures. `N/A` = not measured/not applicable. `UNVERIFIED` = a number exists in a source but could not be corroborated as a genuine measurement (used only for the XTTS 10-speaker VRAM column, §5.1). `0 MB` is shown only where the source genuinely reports it **and** is explained per-row.

| Model | Parameters | Arabic | Voice Control | Peak VRAM | Inference (RTF) | Quality / Notes | Status |
|---|---|---|---|---|---|---|---|
| MOSS-TTS-Nano | ~0.1B (vendor-claimed) | Yes | Voice cloning | 781 MB | RTF 1.196 | CER 0.1736; acceptable but not best-in-class clarity; 48 kHz; benchmark script not in repo (§9) | **Current recommended candidate** |
| Supertonic-3 | ONNX (small) | Yes (multilingual) | Selectable (10 voices) | 0 MB GPU (genuine — CPU/ONNX, no CUDA context) / 938 MB RSS | RTF 0.24 | CER 0.044 — best intelligibility/speed tradeoff | **BASELINE** |
| XTTS-v2 (aggregate) | ~460M | Yes (multilingual) | Selectable + cloning | 2466 MB (real GPU measurement) | RTF 0.439 | CER 0.315 (aggregate run) | REJECTED — VRAM/SPEED |
| XTTS-v2 (10 speakers) | ~460M | Yes (multilingual) | Selectable + cloning | `UNVERIFIED` (CSV says 0 MB; contradicted by aggregate row, §5.1) — RSS 4.2–5.5 GB is the reliable figure | RTF 2.67–3.06 | CER 0.022–0.072; reasonable quality, far too slow/heavy | REJECTED — VRAM/SPEED |
| OmniVoice (FP16) | ~0.6B | Yes (MSA + dialects) | Selectable (voice design) | 2776 MB | RTF 0.275 | CER 0.028 | REJECTED — VRAM |
| OmniVoice INT8 | ~0.6B | Yes | Selectable | 2352 MB | RTF 0.489 | CER 0.006; still 2.3x over budget | OPTIMIZATION EXPERIMENT |
| OmniVoice CPU offload | ~0.6B | Yes | Selectable | 2522 MB | RTF 0.358 | Minimal VRAM gain, slower | OPTIMIZATION EXPERIMENT |
| OmniVoice `torch.compile` | ~0.6B | Yes | Selectable | 2736 MB (artifact only, not reproducible from committed code) | RTF 0.397 | No VRAM benefit | OPTIMIZATION EXPERIMENT |
| OmniVoice FlashInfer | ~0.6B | Yes | Selectable | 3452 MB (artifact only, not reproducible from committed code) | RTF 18.368 | VRAM up, inference collapsed | OPTIMIZATION EXPERIMENT |
| OmniVoice 4-bit / TensorRT-LLM / ONNX | — | — | — | N/A — no artifact at all except one orphaned audio file (§8) | N/A | Not benchmarked | NOT BENCHMARKED |
| VoiceTut-TTS | ~612M | Yes (Egyptian) | Selectable + cloning | ~4.64 GB (manual observation, CSV VRAM column reads 0 MB — measurement gap) | RTF 9.348 | CER 0.033; unusably slow | REJECTED — SPEED |
| MMS TTS Arabic | ~36M | Yes (MSA) | Fixed | 496 MB | RTF 0.048 | CER 0.144; below Supertonic quality | REJECTED — QUALITY |
| SeyedAli Arabic MMS | ~36M | Yes | Fixed | 536 MB | RTF 0.075 | CER 0.11; below Supertonic quality | REJECTED — QUALITY |
| SpeechT5 (ClArTTS) | ~144M | Yes (MSA) | Selectable | 546 MB | RTF 0.111 | CER 0.144; below Supertonic quality | REJECTED — QUALITY |
| lahja-sa-huba-v1 | ~36M | Yes (Saudi) | Fixed | 412 MB | RTF 0.11 | CER 0.16; fixed **female** voice | REJECTED — VOICE CONTROL |
| vits-ar-sa-ahmad | ~36M | Yes (Saudi) | Fixed | 346 MB | RTF 0.103 | CER 0.254; correct gender, weak quality | REJECTED — QUALITY |
| vits-ar-sa-huba-v2 | ~83M | Yes (Saudi) | Fixed | 404 MB | RTF 0.065 | CER 0.177; fixed **female** voice | REJECTED — VOICE CONTROL |
| vits-ar | ~36M | Yes (MSA) | Fixed | 384 MB | RTF 0.06 | CER 0.232; below Supertonic quality | REJECTED — QUALITY |
| vits-ar-sa-A | ~83M | Yes (Saudi) | Fixed | 346 MB | RTF 0.07 | CER 0.221; below Supertonic quality | REJECTED — QUALITY |
| vits-ar-ye-sa | ~36M | Yes (Yemeni) | Fixed | 344 MB | RTF 0.092 | CER 0.309; fixed **female** voice | REJECTED — VOICE CONTROL |
| Audar-TTS-V1-Flash (Q4) | ~0.6B (Q4 GGUF) | Yes (expressive) | Cloning (designed) | N/A — never loaded | N/A | Blocked by missing llama.cpp/NeuCodec tooling | BLOCKED — DEPENDENCY |
| Piper Arabic / OpenBible / Fasih / Wren-TTS / KaniTTS / "Nipponjo" | — | — | — | N/A | N/A | No repository evidence of any benchmark run (§11) | NOT BENCHMARKED |

## 14. Rejection Summary

| Model | Reason for rejection / status |
|---|---|
| MOSS-TTS-Nano | **Current recommended candidate** — within stretched VRAM ceiling, only in-budget model with voice cloning; not best-in-class on CER/RTF |
| Supertonic-3 | **BASELINE** |
| XTTS-v2 | REJECTED — VRAM (2.5 GB confirmed, RSS up to 5.5 GB) and speed (RTF 2.7–3.1x on the 10-speaker run) |
| OmniVoice (FP16) | REJECTED — VRAM (~2.8 GB) |
| OmniVoice INT8 / CPU offload / `torch.compile` / FlashInfer | OPTIMIZATION EXPERIMENT — none reached the target; FlashInfer regressed both VRAM and latency |
| VoiceTut-TTS | REJECTED — SPEED (RTF 9.35) and memory (~4.6 GB, manually observed) |
| MMS TTS Arabic | REJECTED — QUALITY (vs. Supertonic) |
| SeyedAli Arabic MMS | REJECTED — QUALITY |
| SpeechT5 (ClArTTS) | REJECTED — QUALITY |
| lahja-sa-huba-v1 | REJECTED — VOICE CONTROL (fixed female) |
| vits-ar-sa-ahmad | REJECTED — QUALITY |
| vits-ar-sa-huba-v2 | REJECTED — VOICE CONTROL (fixed female) |
| vits-ar | REJECTED — QUALITY |
| vits-ar-sa-A | REJECTED — QUALITY |
| vits-ar-ye-sa | REJECTED — VOICE CONTROL (fixed female) |
| Audar-TTS-V1-Flash (Q4) | BLOCKED — DEPENDENCY (llama.cpp + NeuCodec unavailable) |
| chatterbox (multilingual), chatterbox-egyptian-v0, magpie_tts, fish-speech-1.5, Qwen3-TTS-GGUF, VoxCPM2, higgs-tts-3-4b, Voxtral-4B-TTS-2603, MOSS-TTS-v1.5 | NOT BENCHMARKED — deferred/excluded before any run, on parameter-count or dependency grounds (§12) |
| Piper Arabic, VITS-OpenBible-Arabic-Standard, "Lahja-SA-Ahmad" ~4.6GB run, OmniVoice 4-bit/TensorRT-LLM/ONNX experiments, Fasih-TTS-V1, Wren-TTS, KaniTTS, "Nipponjo" | NOT BENCHMARKED — no corroborating evidence found in this repository (§11) |

## 15. Final Recommendation

**Based on the models actually benchmarked and corroborated in this repository, MOSS-TTS-Nano is the current best-supported candidate for the voice-cloning use case** — not a claim that it is definitively the best Arabic TTS model in existence, and not a claim earned by VRAM alone.

Applying the full criteria set (§2) rather than VRAM in isolation:
- **VRAM:** 781 MB — inside the stretched ~1 GB ceiling. ✅
- **Arabic / male voice:** present, requested male voice, qualitatively acceptable but not the clearest Arabic output seen during the broader search. Partial ✅
- **Intelligibility:** CER 0.1736 — worse than Supertonic (0.044) and worse than most of the in-budget VITS/MMS models. ⚠️
- **Speed:** RTF 1.196 (slower than real-time) — worse than Supertonic (0.24) and every in-budget VITS/MMS model. ⚠️
- **Voice control:** genuine voice cloning — a capability no other in-budget model in this repository offers. ✅ (this is the deciding factor)
- **Deployability:** demonstrated by one committed run; the benchmarking methodology itself is not reproducible from this repository (§3, §9) — a real gap to close before treating this as production-validated.

Given that, the recommendation is qualified, not absolute:
- **If voice cloning is not a hard requirement, Supertonic-3 remains the better default** on this evidence — it is faster, more intelligible, and effectively free in GPU memory.
- **MOSS-TTS-Nano earns its place only because it is the sole in-budget model offering voice cloning at acceptable quality** — every model that clearly beat it on cloning/expressiveness quality (XTTS-v2, OmniVoice, VoiceTut-TTS) needed 2.3–5.5 GB VRAM, even after a dedicated OmniVoice optimization pass (§8).
- **Two leads remain genuinely open**, not resolved by this repository's evidence: Audar-TTS-V1-Flash (Q4) never got a real benchmark due to missing tooling, and the Piper Arabic / VITS-OpenBible claims in §11 could not be verified one way or the other here. Either is worth a follow-up pass with proper environment setup before being ruled in or out.
- **Before relying on MOSS-TTS-Nano's numbers for a deployment decision**, the missing benchmark script (§3, §9) should be reconstructed and re-run through the same harness used for everything else in this document, so its VRAM/RTF figures are measured the same way as the models it's being compared against.

## 16. Appendix — Raw Benchmark Data / Artifact References

All paths below were checked to exist on disk; none are inferred.

**CSV sources (repository-relative paths):**
- [`results/results.csv`](results/results.csv) — main harness: Supertonic, XTTS-v2 (aggregate), OmniVoice (baseline + a cpu-offload row), MMS, SeyedAli MMS, SpeechT5, 5x `wasmdashai` VITS, VoiceTut-TTS, Audar (failed)
- [`results/moss-tts-nano/csv/moss_tts_nano_result.csv`](results/moss-tts-nano/csv/moss_tts_nano_result.csv) — MOSS-TTS-Nano (primary; `moss_tts_nano_result_backup.csv` in the same folder holds identical data with a mangled UTF-8/Latin-1 encoding of the Arabic text — a re-save artifact, not a data conflict)
- [`results/omnivoice/csv/omnivoice_optimization_results.csv`](results/omnivoice/csv/omnivoice_optimization_results.csv) — OmniVoice FP16/INT8/CPU-offload/torch-compile/FlashInfer (`omnivoice_optimization_results_backup.csv` in the same folder is missing the INT8 row — an earlier snapshot)
- [`supertonic_results/results.csv`](supertonic_results/results.csv) — Supertonic M1–M5 voice variants, produced by `tts_eval/supertonic_male_voices.py`
- [`xtts_results/results.csv`](xtts_results/results.csv) — XTTS-v2 10-speaker benchmark (no script committed, §3)
- [`results_omnivoice/results.csv`](results_omnivoice/results.csv) — anomalous/exploratory OmniVoice run, excluded from all conclusions (§3)

**Unexplained artifact:** [`results/omnivoice/audio/omnivoice-4bit.mp3`](results/omnivoice/audio/omnivoice-4bit.mp3) exists on disk but has no corresponding row in either OmniVoice optimization CSV. No metrics are reported for it.

**Report/summary files:** [`results/report.md`](results/report.md), [`results_omnivoice/report.md`](results_omnivoice/report.md) — auto-generated from the CSVs above by `tts_eval/make_report.py`.

**Registry / config / scripts:** [`tts_eval/config.py`](tts_eval/config.py) (model registry + `EXCLUSIONS`), [`tts_eval/monitor.py`](tts_eval/monitor.py) (VRAM/RSS measurement method — see §5.1 for its known limitation), [`tts_eval/supertonic_male_voices.py`](tts_eval/supertonic_male_voices.py) (the M1–M5 sweep script), [`TODO_MODELS.md`](TODO_MODELS.md) (deferred models), [`README.md`](README.md) (harness documentation — contains a superseded results table, §3).
