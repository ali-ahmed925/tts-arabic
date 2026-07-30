"""Orchestrator: runs each shortlisted model on the fixed Arabic text, records
compute + quality metrics, streams rows to results.csv, and saves an .mp3 per model."""
import csv
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from adapters import ADAPTERS, free_cuda
from metrics import cer, pitch_stats, transcribe
from monitor import ResourceSampler

CSV_FIELDS = [
    "model_id", "hf_repo", "arch", "params", "lang", "status",
    "peak_vram_mb", "peak_rss_mb", "load_s", "infer_s", "audio_s", "rtf",
    "asr_cer", "f0_std_hz", "f0_mean_hz", "voiced_ratio",
    "voice_control", "requested_gender", "detected_gender",
    "mp3", "asr_hypothesis", "notes",
]


def wav_to_mp3(wav, sr, mp3_path):
    wav = np.asarray(wav, dtype=np.float32)
    peak = float(np.max(np.abs(wav))) or 1.0
    wav = (wav / peak) * 0.97  # normalize to avoid clipping on export
    tmp = mp3_path.with_suffix(".wav")
    sf.write(tmp, wav, sr)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
         "-codec:a", "libmp3lame", "-qscale:a", "2", str(mp3_path)],
        check=True,
    )
    tmp.unlink(missing_ok=True)


def append_row(row):
    """Upsert by model_id: re-running a model overwrites its own row instead of
    appending a duplicate. Rewrites the file each time (fine for a small table)."""
    rows = []
    if C.CSV_PATH.exists():
        with open(C.CSV_PATH, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("model_id") != row["model_id"]]
    rows.append({k: str(row.get(k, "")) for k in CSV_FIELDS})
    with open(C.CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def run_one(spec, device):
    import torch
    mid = spec["id"]
    print(f"\n{'='*70}\n[{mid}] {spec['hf_repo']}  ({spec['kind']})", flush=True)
    row = {"model_id": mid, "hf_repo": spec["hf_repo"], "arch": spec["kind"],
           "params": spec["params"], "lang": spec["lang"], "notes": spec["notes"]}
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with ResourceSampler() as sampler:
            t0 = time.time()
            generate, sr, teardown = ADAPTERS[spec["kind"]](spec, device)
            row["load_s"] = round(time.time() - t0, 2)
            t1 = time.time()
            wav, sr = generate(C.TEST_TEXT)
            row["infer_s"] = round(time.time() - t1, 2)
        row["peak_vram_mb"] = round(sampler.peak_gpu_mb, 1)
        row["peak_rss_mb"] = round(sampler.peak_rss_mb, 1)
        row["audio_s"] = round(len(wav) / sr, 2)
        row["rtf"] = round(row["infer_s"] / max(row["audio_s"], 1e-6), 3)

        mp3 = C.AUDIO_DIR / f"{mid}.mp3"
        wav_to_mp3(wav, sr, mp3)
        row["mp3"] = str(mp3.relative_to(C.ROOT))

        # expressiveness proxy (on the raw waveform)
        row.update(pitch_stats(wav, sr))

        # ---- voice guardrail bookkeeping ----
        from adapters import VOICE_SELECTABLE
        selectable = spec["kind"] in VOICE_SELECTABLE
        row["voice_control"] = "selectable" if selectable else "fixed"
        row["requested_gender"] = C.VOICE_GENDER
        f0m = row.get("f0_mean_hz")
        try:
            det = "male" if float(f0m) < C.GENDER_F0_THRESHOLD_HZ else "female"
        except (TypeError, ValueError):
            det = "?"
        row["detected_gender"] = det
        # flag a fixed-voice model whose baked-in gender disagrees with the guardrail
        if not selectable and det not in ("?", C.VOICE_GENDER):
            row["notes"] = (row.get("notes", "") +
                            f" | NOTE: fixed voice is {det}, cannot force {C.VOICE_GENDER}").strip(" |")

        # free the TTS model BEFORE loading the ASR evaluator
        try:
            teardown()
        except Exception:
            pass
        free_cuda()

        # intelligibility: ASR round-trip CER
        try:
            hyp = transcribe(wav, sr, C.ASR_MODEL, device)
            row["asr_hypothesis"] = hyp
            row["asr_cer"] = round(cer(C.TEST_TEXT, hyp), 3)
        except Exception as e:
            row["asr_cer"] = ""
            row["asr_hypothesis"] = f"ASR_FAILED: {e}"[:120]

        row["status"] = "ok"
        print(f"  ok | vram={row['peak_vram_mb']}MB rss={row['peak_rss_mb']}MB "
              f"dur={row['audio_s']}s rtf={row['rtf']} cer={row.get('asr_cer')} "
              f"f0std={row.get('f0_std_hz')}", flush=True)
    except Exception as e:
        row["status"] = "failed"
        row["notes"] = (spec.get("notes", "") + f" | ERROR: {e}")[:300]
        print(f"  FAILED: {e}", flush=True)
        traceback.print_exc()
    finally:
        free_cuda()
    append_row(row)
    return row


def run_single(model_id):
    """Runs exactly one model in THIS process (clean CUDA context) and appends
    its row. Invoked as a subprocess by main() so each model's peak VRAM is
    measured in isolation, uncontaminated by other models or the ASR evaluator."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    C.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    spec = next(s for s in C.MODELS if s["id"] == model_id)
    run_one(spec, device)


def main():
    C.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:] or None
    if only is None and C.CSV_PATH.exists():
        C.CSV_PATH.unlink()
    selected = [s for s in C.MODELS if only is None or s["id"] in only]
    print(f"budget={C.VRAM_BUDGET_MB}MB | running {len(selected)} model(s), "
          f"each in an isolated subprocess for clean VRAM measurement")
    for spec in selected:
        # A model may declare its own interpreter (an isolated venv holding a
        # heavy runtime like coqui-tts). Default to this process's interpreter.
        interp = spec.get("python") or sys.executable
        if not Path(interp).exists():
            # venv paths in config.py are written Linux-style (envs/x/bin/python);
            # on Windows the same venv's interpreter lives at envs/x/Scripts/python.exe.
            winterp = Path(interp).parent.parent / "Scripts" / "python.exe"
            if winterp.exists():
                interp = str(winterp)
            else:
                print(f"\n[{spec['id']}] SKIP: interpreter not found: {interp}", flush=True)
                continue
        # fresh process per model -> clean CUDA context, isolated peak VRAM
        subprocess.run([interp, __file__, "--single", spec["id"]], check=False)
    print(f"\nDone. CSV -> {C.CSV_PATH}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--single":
        run_single(sys.argv[2])
    else:
        main()
