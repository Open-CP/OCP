import os
import platform
import threading

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    print("[WARNING] psutil is not installed. To record runtime information such as CPU usage, used cores, threads, RAM, and platform information, please run: pip install psutil")
    HAS_PSUTIL = False


def get_platform_info():
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
        except Exception:
            pass
        try:
            info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except Exception:
            pass

    return info


class RuntimeResourceMonitor:
    def __init__(self, pid=None, interval=0.2):
        self.pid = pid or os.getpid()
        self.interval = interval
        self.platform_info = get_platform_info()

        self._stop_event = threading.Event()
        self._thread = None

        self.sample_count = 0
        self.cpu_percent_sum = 0.0
        self.max_cpu_percent = 0.0
        self.max_num_threads = 0

    def _run(self):
        if not HAS_PSUTIL:
            return

        try:
            proc = psutil.Process(self.pid)
            proc.cpu_percent(interval=None)
        except Exception:
            return

        while not self._stop_event.is_set():
            try:
                cpu_percent = proc.cpu_percent(interval=self.interval)
                num_threads = proc.num_threads()

                self.sample_count += 1
                self.cpu_percent_sum += cpu_percent
                self.max_cpu_percent = max(self.max_cpu_percent, cpu_percent)
                self.max_num_threads = max(self.max_num_threads, num_threads)
            except Exception:
                break

    def start(self):
        if not HAS_PSUTIL:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if HAS_PSUTIL and self._thread is not None:
            self._stop_event.set()
            self._thread.join()
        return self.get_stats()

    def get_runtime_usage(self):
        if not HAS_PSUTIL or self.sample_count == 0:
            return {
                "monitoring_enabled": False,
                "avg_cpu_percent": None,
                "max_cpu_percent": None,
                "avg_used_cores_estimated": None,
                "max_used_cores_estimated": None,
                "max_num_threads": None,
            }

        avg_cpu_percent = self.cpu_percent_sum / self.sample_count
        return {
            "monitoring_enabled": True,
            "avg_cpu_percent": round(avg_cpu_percent, 2),
            "max_cpu_percent": round(self.max_cpu_percent, 2),
            "avg_used_cores_estimated": round(avg_cpu_percent / 100.0, 2),
            "max_used_cores_estimated": round(self.max_cpu_percent / 100.0, 2),
            "max_num_threads": self.max_num_threads,
        }

    def get_stats(self):
        return {
            "platform_info": self.platform_info,
            "runtime_usage": self.get_runtime_usage(),
        }
