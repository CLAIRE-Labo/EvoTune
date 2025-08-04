import threading
import time
import psutil
import logging


class MemoryWatcher(threading.Thread):
    """
    A thread that periodically checks the memory usage of a target process (by PID).
    If usage exceeds a given threshold, it calls a user-specified 'callback_on_exceed'.
    """

    def __init__(self, task_id, pid, mem_limit_bytes, callback_on_exceed, check_interval=1.0):
        super().__init__()
        self.task_id = task_id
        self.pid = pid
        self.mem_limit_bytes = mem_limit_bytes
        self.callback = callback_on_exceed
        self.check_interval = check_interval
        self._stop_event = threading.Event()

    def run(self):
        try:
            target_proc = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            logging.warning(f"[Task {self.task_id}] MemoryWatcher: No process found with PID={self.pid}")
            return

        while not self._stop_event.is_set():
            try:
                rss = target_proc.memory_info().rss  # Resident Set Size in bytes
                if rss > self.mem_limit_bytes:
                    logging.warning(
                        f"[Task {self.task_id}] MemoryWatcher: PID={self.pid} exceeded {self.mem_limit_bytes / 1024 / 1024 / 1024} bytes (actual={rss / 1024 / 1024 / 1024}). Triggering callback..."
                    )
                    self.callback()
                    break
            except psutil.NoSuchProcess:
                logging.info(f"[Task {self.task_id}] MemoryWatcher: PID={self.pid} is gone, stopping.")
                break
            time.sleep(self.check_interval)

    def stop(self):
        self._stop_event.set()
