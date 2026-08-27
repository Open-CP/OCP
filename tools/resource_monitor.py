"""Runtime resource monitoring (CPU / thread / memory) for solver runs, via psutil.

``RuntimeResourceMonitor`` samples a process's CPU usage and thread count in a background
daemon thread between ``start()`` and ``stop()``; everything degrades to no-ops / ``None``
when psutil is not installed.
"""

import os
import platform
import threading

try:
    import psutil
    HAS_PSUTIL = True
    PSUTIL_ERRORS = (psutil.Error, OSError, RuntimeError)
except ImportError:
    HAS_PSUTIL = False
    PSUTIL_ERRORS = (OSError, RuntimeError)


def get_platform_info():
    """Return static platform info (OS, CPU counts, total RAM).

    ``cpu_count_physical`` / ``ram_total_gb`` stay ``None`` when psutil is unavailable.
    """
    info = {
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "platform_processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": None,
        "ram_total_gb": None,
    }

    if HAS_PSUTIL:
        try:
            info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        except PSUTIL_ERRORS:
            pass
        try:
            info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except PSUTIL_ERRORS:
            pass

    return info


class RuntimeResourceMonitor:
    """Sample a process's CPU% and thread count in a background thread between ``start()`` and ``stop()``.

    A no-op when psutil is unavailable. ``stop()`` joins the sampling thread and returns the
    collected stats. ``max_num_threads`` is the peak TOTAL process threads (including the
    sampler and main threads); ``max_added_threads`` reports the peak workload-added threads
    over the start-of-monitoring baseline -- use it for solver-parallelism analysis.
    """

    def __init__(self, pid=None, interval=0.2):
        """Monitor process ``pid`` (default: the current process), sampling every ``interval`` seconds."""
        self.pid = pid or os.getpid()
        self.interval = interval
        self.platform_info = get_platform_info()

        self._stop_event = threading.Event()
        self._thread = None

        self.sample_count = 0
        self.cpu_percent_sum = 0.0
        self.max_cpu_percent = 0.0
        self.baseline_num_threads = None  # process thread count when monitoring starts (main + sampler + pre-existing)
        self.max_num_threads = 0          # peak TOTAL process threads

    def _run(self):
        """Sampling loop (daemon thread): accumulate CPU% and peak thread count until stopped."""
        if not HAS_PSUTIL:
            return

        try:
            proc = psutil.Process(self.pid)
            proc.cpu_percent(interval=None)
            # Baseline captured before the workload ramps up: main thread + this sampler
            # thread + any pre-existing threads. Subtracting it isolates workload-added threads.
            self.baseline_num_threads = proc.num_threads()
        except PSUTIL_ERRORS:
            return

        while not self._stop_event.is_set():
            try:
                cpu_percent = proc.cpu_percent(interval=self.interval)
                num_threads = proc.num_threads()

                self.sample_count += 1
                self.cpu_percent_sum += cpu_percent
                self.max_cpu_percent = max(self.max_cpu_percent, cpu_percent)
                self.max_num_threads = max(self.max_num_threads, num_threads)
            except PSUTIL_ERRORS:
                break

    def start(self):
        """Start the background sampling thread (no-op without psutil)."""
        if not HAS_PSUTIL:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop and join the sampling thread, returning ``get_stats()``."""
        if HAS_PSUTIL and self._thread is not None:
            self._stop_event.set()
            self._thread.join()
        return self.get_stats()

    def get_runtime_usage(self):
        """Return aggregated usage, or a disabled record if nothing was sampled.

        ``max_num_threads`` is the peak TOTAL process thread count; ``max_added_threads`` is
        the peak over the baseline (``baseline_num_threads``), i.e. the threads the workload
        itself spun up -- excluding the main and sampler threads. Use ``max_added_threads`` for
        solver-parallelism analysis.
        """
        if not HAS_PSUTIL or self.sample_count == 0:
            return {
                "monitoring_enabled": False,
                "avg_cpu_percent": None,
                "max_cpu_percent": None,
                "avg_used_cores_estimated": None,
                "max_used_cores_estimated": None,
                "max_num_threads": None,
                "baseline_num_threads": None,
                "max_added_threads": None,
            }

        avg_cpu_percent = self.cpu_percent_sum / self.sample_count
        baseline = self.baseline_num_threads or 0
        return {
            "monitoring_enabled": True,
            "avg_cpu_percent": round(avg_cpu_percent, 2),
            "max_cpu_percent": round(self.max_cpu_percent, 2),
            "avg_used_cores_estimated": round(avg_cpu_percent / 100.0, 2),
            "max_used_cores_estimated": round(self.max_cpu_percent / 100.0, 2),
            "max_num_threads": self.max_num_threads,
            "baseline_num_threads": self.baseline_num_threads,
            "max_added_threads": max(0, self.max_num_threads - baseline),
        }

    def get_stats(self):
        """Return ``{"platform_info": ..., "runtime_usage": ...}``."""
        return {
            "platform_info": self.platform_info,
            "runtime_usage": self.get_runtime_usage(),
        }
