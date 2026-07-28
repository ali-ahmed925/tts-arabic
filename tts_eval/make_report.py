"""Read results.csv and render report.md in the requested deliverable template."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def expressiveness_label(row):
    """Turn the F0-std proxy into a plain-language note. Higher pitch variance =
    less monotone. Thresholds are rough, calibrated for these small models."""
    try:
        std = float(row["f0_std_hz"])
    except (ValueError, KeyError, TypeError):
        return "n/a"
    if std >= 45:
        return "notable pitch variation (more expressive)"
    if std >= 25:
        return "moderate pitch variation"
    return "low pitch variation (flat/monotone)"


def intelligibility_label(row):
    try:
        c = float(row["asr_cer"])
    except (ValueError, KeyError, TypeError):
        return "n/a"
    if c <= 0.15:
        return "highly intelligible"
    if c <= 0.35:
        return "intelligible with some errors"
    return "poor intelligibility"


def main():
    rows = []
    if C.CSV_PATH.exists():
        with open(C.CSV_PATH, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    lines = []
    lines.append("# Arabic TTS Evaluation — small/expressive models with a male-voice guardrail\n")
    lines.append(f"**Test text (fixed):**\n\n> {C.TEST_TEXT}\n")
    lines.append(f"**Hardware:** NVIDIA RTX 4050 Laptop (6 GB VRAM), 12-core CPU, 30 GB RAM.  ")
    lines.append(f"**Two tiers evaluated:** a strict **≤{C.VRAM_BUDGET_MB} MB VRAM** tier (small VITS/SpeechT5/ONNX "
                 "models) and a **≤~2 GB VRAM expressive tier** (XTTS-v2 etc.). CPU/ONNX models report VRAM=0 "
                 "and should be read on the CPU-RAM column.\n")
    lines.append("**Metrics:** peak VRAM = process GPU memory via nvidia-smi (includes CUDA context). "
                 "`ASR-CER` = Whisper-small round-trip character error rate (intelligibility, lower is better). "
                 "`F0 std` = pitch standard deviation in Hz (expressiveness proxy, higher = less monotone). "
                 "`RTF` = inference time / audio duration (lower is faster).\n")

    # summary table
    lines.append("## Summary\n")
    lines.append("| Model | Arch | Peak VRAM | RTF | ASR-CER | F0 std | Voice | Status |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        ctrl = r.get("voice_control", "")
        det = r.get("detected_gender", "?")
        voice = (f"{det} ({'sel' if ctrl=='selectable' else 'fixed'})" if ctrl else "")
        lines.append(
            f"| {r['model_id']} | {r.get('arch','')} | {r.get('peak_vram_mb','')} MB | "
            f"{r.get('rtf','')} | {r.get('asr_cer','')} | {r.get('f0_std_hz','')} Hz | "
            f"{voice} | {r.get('status','')} |"
        )
    lines.append("")

    # per-model detail in the requested template
    lines.append("## Per-model detail\n")
    for i, r in enumerate(rows, 1):
        lines.append(f"**Model {i}: {r['model_id']}  (`{r['hf_repo']}`)**")
        mp3 = r.get("mp3", "")
        lines.append(f"*   **Audio Output:** {mp3 or '(not generated)'}")
        if r.get("status") == "ok":
            notes = (f"{intelligibility_label(r)}; {expressiveness_label(r)}. "
                     f"ASR-CER={r.get('asr_cer','?')}, F0 std={r.get('f0_std_hz','?')} Hz, "
                     f"voiced ratio={r.get('voiced_ratio','?')}. Arch: {r.get('arch','')} ({r.get('params','')}).")
        else:
            notes = f"FAILED / skipped — {r.get('notes','')}"
        lines.append(f"*   **Expressiveness / Quality Notes:** {notes}")
        # voice guardrail line
        ctrl = r.get("voice_control", "")
        det = r.get("detected_gender", "?")
        req = r.get("requested_gender", "")
        if ctrl == "selectable":
            voice = f"selectable → forced to **{req}** (detected {det}, F0 mean {r.get('f0_mean_hz','?')} Hz)"
        elif ctrl == "fixed":
            ok = "✅ satisfies" if det == req else "⚠️ VIOLATES"
            voice = f"fixed/baked-in voice — detected **{det}** (F0 mean {r.get('f0_mean_hz','?')} Hz); {ok} the {req}-only guardrail"
        else:
            voice = "n/a"
        lines.append(f"*   **Voice ({req}-guardrail):** {voice}")
        lines.append(f"*   **Memory Consumed:** {r.get('peak_vram_mb','?')} MB VRAM (peak, process), "
                     f"{r.get('peak_rss_mb','?')} MB CPU RSS")
        if r.get("asr_hypothesis"):
            lines.append(f"*   **ASR heard:** {r.get('asr_hypothesis','')[:220]}")
        lines.append("")

    # exclusions
    lines.append("## Excluded models (over the 700 MB budget)\n")
    lines.append("These are more expressive but do not fit the constraint:\n")
    for repo, why in C.EXCLUSIONS:
        lines.append(f"*   `{repo}` — {why}")
    lines.append("")

    C.REPORT_PATH.write_text("\n".join(lines))
    print(f"report -> {C.REPORT_PATH}")


if __name__ == "__main__":
    main()
