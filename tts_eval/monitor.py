"""Resource monitoring: peak VRAM (process-level via nvidia-smi + torch's own
allocator counters) and CPU RAM (RSS)."""
import os
import subprocess
import threading
import time


def _proc_rss_mb() -> float:
    """Resident set size of this process in MB (from /proc; falls back to
    psutil where /proc isn't available, e.g. native Windows)."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def _proc_gpu_mb(pid: int) -> float:
    """VRAM used by `pid` right now, per nvidia-smi (includes CUDA context)."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
                return float(parts[1])
    except Exception:
        pass
    return 0.0


class ResourceSampler:
    """Background thread that tracks peak process VRAM and peak RSS."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.pid = os.getpid()
        self.peak_gpu_mb = 0.0
        self.peak_rss_mb = 0.0
        self._stop = threading.Event()
        self._t = None

    def _loop(self):
        while not self._stop.is_set():
            self.peak_gpu_mb = max(self.peak_gpu_mb, _proc_gpu_mb(self.pid))
            self.peak_rss_mb = max(self.peak_rss_mb, _proc_rss_mb())
            time.sleep(self.interval)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)
        # one final sample in case the peak happened right before exit
        self.peak_gpu_mb = max(self.peak_gpu_mb, _proc_gpu_mb(self.pid))
        self.peak_rss_mb = max(self.peak_rss_mb, _proc_rss_mb())
