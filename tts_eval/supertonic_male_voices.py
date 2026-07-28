"""Standalone script: synthesize the fixed Arabic evaluation sentence with every
built-in MALE voice of Supertonic-3, saving one WAV per voice under
supertonic_results/ plus a results.csv in the same format as the main eval
pipeline's output (tts_eval/run.py).

Does NOT modify the shared pipeline (config.py / adapters.py / run.py) -- it
only imports and reuses it: `load_supertonic` for model loading + synthesis,
and `metrics`/`monitor` for the same CER / pitch / resource metrics.

Male voices are discovered automatically from `TTS.voice_style_names` (the
list the `supertonic` package itself reports for the loaded model) rather than
assuming a fixed count -- no hardcoded "M1..M5".

Must run with the isolated Supertonic interpreter (see README.md):
    envs/supertonic/Scripts/python.exe tts_eval/supertonic_male_voices.py
    envs/supertonic/Scripts/python.exe tts_eval/supertonic_male_voices.py M1 M2   # subset
"""
import csv
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from adapters import load_supertonic, free_cuda
from metrics import cer, pitch_stats, transcribe
from monitor import ResourceSampler

MODEL_ID = "supertonic-3"
MODEL_SPEC = next(m for m in C.MODELS if m["id"] == MODEL_ID)

OUT_DIR = C.ROOT / "supertonic_results"
CSV_PATH = OUT_DIR / "results.csv"

# Same shape as tts_eval/run.py's CSV_FIELDS; "mp3" -> "wav" since this script
# saves WAVs (not MP3s) and there is one row per *voice* instead of per model.
CSV_FIELDS = [
    "model_id", "hf_repo", "arch", "params", "lang", "status",
    "peak_vram_mb", "peak_rss_mb", "load_s", "infer_s", "audio_s", "rtf",
    "asr_cer", "f0_std_hz", "f0_mean_hz", "voiced_ratio",
    "voice_control", "requested_gender", "detected_gender",
    "wav", "asr_hypothesis", "notes",
]


def discover_male_voices(model=MODEL_ID):
    """Ask the `supertonic` package itself which voice styles the model ships
    with, and keep only the male ones ('M' + digits, e.g. M1, M2, ...)."""
    import supertonic
    tts = supertonic.TTS(model=model, auto_download=True)
    male = [v for v in tts.voice_style_names if re.fullmatch(r"M\d+", v)]
    return sorted(male, key=lambda v: int(v[1:]))


def append_row(row):
    """Upsert by model_id, mirroring run.py's append_row."""
    rows = []
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("model_id") != row["model_id"]]
    rows.append({k: str(row.get(k, "")) for k in CSV_FIELDS})
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def run_voice(voice, device):
    row = {
        "model_id": f"{MODEL_ID}-{voice}", "hf_repo": MODEL_SPEC["hf_repo"],
        "arch": MODEL_SPEC["kind"], "params": MODEL_SPEC["params"],
        "lang": MODEL_SPEC["lang"], "notes": MODEL_SPEC["notes"],
        "voice_control": "selectable", "requested_gender": "male",
    }
    try:
        with ResourceSampler() as sampler:
            t0 = time.time()
            generate, sr, teardown = load_supertonic(device, voice=voice, model=MODEL_ID)
            row["load_s"] = round(time.time() - t0, 2)
            t1 = time.time()
            wav, sr = generate(C.TEST_TEXT)
            row["infer_s"] = round(time.time() - t1, 2)
        row["peak_vram_mb"] = round(sampler.peak_gpu_mb, 1)
        row["peak_rss_mb"] = round(sampler.peak_rss_mb, 1)
        row["audio_s"] = round(len(wav) / sr, 2)
        row["rtf"] = round(row["infer_s"] / max(row["audio_s"], 1e-6), 3)

        import soundfile as sf
        wav_path = OUT_DIR / f"{voice}.wav"
        sf.write(wav_path, wav, sr)
        row["wav"] = str(wav_path.relative_to(C.ROOT))

        row.update(pitch_stats(wav, sr))
        f0m = row.get("f0_mean_hz")
        try:
            row["detected_gender"] = "male" if float(f0m) < C.GENDER_F0_THRESHOLD_HZ else "female"
        except (TypeError, ValueError):
            row["detected_gender"] = "?"

        try:
            teardown()
        except Exception:
            pass
        free_cuda()

        try:
            hyp = transcribe(wav, sr, C.ASR_MODEL, device)
            row["asr_hypothesis"] = hyp
            row["asr_cer"] = round(cer(C.TEST_TEXT, hyp), 3)
        except Exception as e:
            row["asr_cer"] = ""
            row["asr_hypothesis"] = f"ASR_FAILED: {e}"[:120]

        row["status"] = "ok"
        print(f"  [{voice}] ok | dur={row['audio_s']}s rtf={row['rtf']} "
              f"cer={row.get('asr_cer')} f0={row.get('f0_mean_hz')}Hz", flush=True)
    except Exception as e:
        row["status"] = "failed"
        row["notes"] = (row.get("notes", "") + f" | ERROR: {e}")[:300]
        print(f"  [{voice}] FAILED: {e}", flush=True)
        traceback.print_exc()
    finally:
        free_cuda()
    return row


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    only = sys.argv[1:] or None
    voices = discover_male_voices()
    if only:
        voices = [v for v in voices if v in only]
    print(f"Discovered male voices: {voices}")

    for voice in voices:
        print(f"\n{'='*70}\n[{MODEL_ID}] voice={voice}", flush=True)
        row = run_voice(voice, device)
        append_row(row)

    print(f"\nDone. {len(voices)} voice(s) -> {CSV_PATH}")


if __name__ == "__main__":
    main()
