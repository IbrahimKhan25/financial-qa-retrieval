"""Describe the machine so every latency number can be reported with its hardware."""
import os
import platform
import subprocess


def _cpu_model() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _ram_gb() -> float | None:
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return round(int(out) / 1024**3, 1)
        pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        return round(pages / 1024**3, 1)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def describe(num_threads: int | None = None) -> dict:
    """Hardware summary. Pass the thread count the benchmark actually used."""
    return {
        "cpu_model": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "threads_used": num_threads,
        "ram_gb": _ram_gb(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
