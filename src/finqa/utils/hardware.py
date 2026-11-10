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
    elif platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except (ImportError, OSError):
            pass
    return platform.processor() or "unknown"


def _ram_gb() -> float | None:
    try:
        if platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return round(int(out) / 1024**3, 1)
        if platform.system() == "Windows":
            import ctypes

            class _Mem(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                            ("page_avail", ctypes.c_ulonglong), ("virt_total", ctypes.c_ulonglong),
                            ("virt_avail", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]

            m = _Mem()
            m.length = ctypes.sizeof(_Mem)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.total / 1024**3, 1)
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
