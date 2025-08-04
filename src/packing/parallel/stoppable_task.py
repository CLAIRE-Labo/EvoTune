import multiprocessing
import os
import resource
import signal
import time
from threading import Timer
import logging
import traceback
import copy
import threading
import pickle

from packing.parallel.memory_watcher import MemoryWatcher


def assert_picklable(obj, obj_name):
    try:
        pickle.dumps(obj)
        print(f"{obj_name} is picklable ✅")
    except Exception as e:
        print(f"{obj_name} is NOT picklable ❌: {e}")


# Gigabytes which are allocated as virtual memory per subprocess on CSCS
# Set this higher than what `malloc` assigns to the subprocess. In our case, we saw that with `htop --tree`,
# most jobs claim around 80GB of virtual memory, so we set it much higher but not so high that the job would crash
CSCS_CLUSTER_VMEM_ALLOCATION = int(120 * 1024 * 1024 * 1024)


def set_memory_limit():
    """Set a hard memory limit for the process."""
    # Note that RLIMIT_RSS doesn't work which really sucks
    resource.setrlimit(resource.RLIMIT_AS, (CSCS_CLUSTER_VMEM_ALLOCATION, CSCS_CLUSTER_VMEM_ALLOCATION))


# We put this method outside so that it is serializable
def execute_program(cfg, input_struct, function_class, evaluate_func, result_queue, task_id, memory_limit_bytes):
    """
    The target function for the subprocess.
    We wrap the user function in a try/except to handle errors.
    """
    set_memory_limit()

    try:
        result = evaluate_func(cfg, input_struct, function_class)
        # Return a consistent tuple (task_id, result)
        logging.info(
            f"[Task {task_id}] Putting result in self.results_queue, results score is {result.true_score}")
        result_queue.put((task_id, result))
        logging.info(f"[Task {task_id}] Result was successfully put in the self.results_queue.")

    except Exception as e:
        tb_str = traceback.format_exc()
        logging.error(
            f"[Task {task_id}] Exception in evaluate_func: {e}\n{tb_str}"
        )
        # Put a consistent tuple (task_id, exception_string)
        result_queue.put((task_id, f"Exception: {e}\n{tb_str}"))


class StoppableTask:
    def __init__(self, cfg, evaluate_func, input_struct, function_class, task_id, timeout):
        """
        Initialize a stoppable task that will run evaluate_func in a subprocess.
        
        Args:
            cfg: Configuration object
            evaluate_func: Function to run in subprocess
            input_struct: Input data structure
            function_class: Class containing function to evaluate
            task_id: Unique identifier for this task
            timeout: Maximum time to allow task to run
        """
        self.cfg = cfg
        self.evaluate_func = evaluate_func
        self.input_struct = input_struct
        self.function_class = function_class
        self.timeout = timeout
        self.task_id = task_id

        self.process = None
        self.result_queue = multiprocessing.Queue()
        self._process_lock = threading.RLock()
        self._timer_lock = threading.Lock()
        # Shared state for signalling timeouts
        self._timeout_event = threading.Event()
        # This event ensures stop() only runs its shutdown steps once
        self._already_stopped = threading.Event()

        self.timer = None

        # Memory Monitor
        self.mem_limit_bytes = int(
            cfg.task.mem_limit_gb * 1024 * 1024 * 1024)  # getattr(cfg, "memory_limit_bytes", 0.5 * 1024 * 1024 * 1024)
        self._mem_watcher = None

    def __enter__(self):
        """
        Allows usage like:
            with StoppableTask(...) as task:
                task.run()
                # ...
        so that cleanup is guaranteed in __exit__.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Ensures we stop/terminate if this goes out of scope or an exception is raised.
        """
        self.stop()

    def run(self):
        """
        Starts the child process and sets up a timer for time limit
        and a MemoryWatcher for memory limit.
        """
        try:
            with self._process_lock:
                if self.process is None:
                    self.process = multiprocessing.Process(target=execute_program, args=(
                        self.cfg, self.input_struct, self.function_class, self.evaluate_func, self.result_queue,
                        self.task_id, self.mem_limit_bytes,
                    ))
                    self.process.start()
                    logging.info(f"[Task {self.task_id}] Started process PID={self.process.pid}")
                elif self.is_alive():
                    logging.info(f"[Task {self.task_id}] Already running, PID={self.process.pid}")
                    return
                else:
                    # Process exists but not alive - clean it up and retry
                    self.stop()  # This will handle setting process to None safely
                    return self.run()
        except Exception as e:
            logging.error(f"[Task {self.task_id}] Error in process setup: {e}")
            self.stop()
            return

        # Start the memory watcher
        try:
            if self._mem_watcher is None and self.process is not None:
                self._mem_watcher = MemoryWatcher(
                    task_id=self.task_id,
                    pid=self.process.pid,
                    mem_limit_bytes=self.mem_limit_bytes,
                    callback_on_exceed=self.stop,  # If memory is exceeded, self.stop() is called
                )
                self._mem_watcher.start()
        except Exception as e:
            logging.error(f"[Task {self.task_id}] Error in MemoryWatcher setup: {e}")
            self.stop()

        # Start the timer for the CPU/Wall time limit
        try:
            with self._timer_lock:
                if self.timer is None:
                    self.timer = Timer(self.timeout, self._timeout_handler)
                    self.timer.start()
        except Exception as e:
            logging.error(f"[Task {self.task_id}] Error in timer setup: {e}")
            self.stop()

    def stop(self):
        """
        Stops the process, cancels the timer, stops the memory watcher,
        and cleans up the queue. Only executes once per task.
        Subsequent calls are ignored.
        """
        logging.info(f"[Task {self.task_id}] stop() initiated (thr={threading.current_thread().name}).")
        if self._already_stopped.is_set():
            logging.info(f"[Task {self.task_id}] stop() called again; already stopped.")
            return

        self._already_stopped.set()

        # Cancel the memory watcher first
        if self._mem_watcher is not None:
            logging.info(f"[Task {self.task_id}] Stopping MemoryWatcher...")
            self._mem_watcher.stop()
            # If we're inside the MemoryWatcher thread, skip join (to avoid "cannot join current thread")
            if threading.current_thread() is not self._mem_watcher:
                logging.info(f"[Task {self.task_id}] Joining MemoryWatcher thread...")
                self._mem_watcher.join(timeout=5)
            self._mem_watcher = None
            logging.info(f"[Task {self.task_id}] MemoryWatcher stopped.")

        # Cancel the timer
        with self._timer_lock:
            if self.timer is not None:
                logging.info(f"[Task {self.task_id}] Canceling timer...")
                self.timer.cancel()
                self.timer = None

        # Terminate the process
        with self._process_lock:
            if self.process is not None:
                try:
                    if self.process.is_alive():
                        logging.info(f"[Task {self.task_id}] Terminating process PID={self.process.pid}")
                        self.process.terminate()
                        self.process.join(timeout=5)
                        if self.process.is_alive():
                            logging.warning(f"[Task {self.task_id}] Process did not terminate, killing")
                            self.process.kill()
                            self.process.join(timeout=1)
                    else:
                        logging.info(f"[Task {self.task_id}] Process PID={self.process.pid} is not alive")
                except Exception as e:
                    logging.error(f"[Task {self.task_id}] Error during process PID={self.process.pid} cleanup: {e}")
                finally:
                    self.process = None

        logging.info(f"[Task {self.task_id}] stop() completed.")

    def join(self, timeout=5):
        """
        A convenience method to join the underlying process. 
        Returns True if the process ended, False if still alive after timeout.
        """
        with self._process_lock:
            if self.process is not None:
                self.process.join(timeout=timeout)
                return not self.process.is_alive()
            return True  # No process = trivially "joined"

    def is_alive(self):
        """Thread-safe check for process status."""
        with self._process_lock:
            return self.process is not None and self.process.is_alive()

    def is_timed_out(self):
        return self._timeout_event.is_set()

    def _timeout_handler(self):
        """Called by self.timer when time is up."""
        logging.info(f"[Task {self.task_id}] CALLING THE TIMEOUT HANDLER. PID={self.process.pid}")

        # Try acquiring the process lock for up to 10 seconds
        lock_acquired = self._process_lock.acquire(timeout=10)

        if lock_acquired:
            try:
                logging.info(f"[Task {self.task_id}] TIMEOUT HANDLER acquired lock. PID={self.process.pid}")
                if self.process is not None and self.process.is_alive():
                    logging.info(f"[Task {self.task_id}] TIMED OUT. PID={self.process.pid}")
                    self._timeout_event.set()
                    self.stop()
            finally:
                self._process_lock.release()
        else:
            logging.warning(
                f"[Task {self.task_id}] TIMEOUT HANDLER could not acquire lock. Using os.kill() as a last resort.")

            # If the lock could not be acquired, manually kill the process
            if self.process is not None and self.process.is_alive():
                try:
                    os.kill(self.process.pid, signal.SIGKILL)  # Force kill process
                    logging.warning(f"[Task {self.task_id}] Forcefully killed PID={self.process.pid} using os.kill().")
                except Exception as e:
                    logging.error(
                        f"[Task {self.task_id}] Failed to kill process PID={self.process.pid} using os.kill(): {e}")

            # Set timeout event regardless to mark the timeout occurrence
            self._timeout_event.set()

        self.stop()
        logging.info(f"[Task {self.task_id}] Killed due to timeout.")

    def get_result(self):
        """
        Non-blocking retrieval of whatever was put into the queue.
        Returns the item if available, or None if the queue is empty.
        """
        if not self.result_queue.empty():
            try:
                return self.result_queue.get_nowait()
            except Exception as e:
                logging.error(f"[Task {self.task_id}] Error getting result: {e}")
        return None


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # file_handler = logging.FileHandler(f"{cfg.logs_dir}/stdout.log")
    # file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    from packing.logging.function_class import FunctionClass

    function_str = """
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate the utilization ratio for each bin
    utilization_ratios = (bins - np.maximum(item, bins)) / (bins + 1e-6)

    # Introduce a non-linear transformation to prioritize bins with lower utilization ratios
    transformed_utilities = np.power(1 - utilization_ratios, 3)

    # Normalize the transformed utilities to ensure they are within the range [0, 1]
    normalized_utilities = (transformed_utilities - np.min(transformed_utilities)) / (
        np.max(transformed_utilities) - np.min(transformed_utilities) + 1e-6
    )

    return normalized_utilities"""

    function_class = FunctionClass(function_str=function_str, imports_str="")
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "failed_score": -10000,
        "function_str_to_extract": "priority",
        "OR": 1,
        "Weibull": 0,
        "timeout_period": 90,
    })

    from packing.evaluate.bin_packing.task_bin import generate_input, evaluate_func

    input_struct = generate_input(cfg)

    # Evaluate the function
    task = StoppableTask(
        cfg,
        evaluate_func,
        copy.deepcopy(input_struct),
        function_class,
        0,
        timeout=cfg.task.timeout_period,
    )
    task.run()
    # Wait for the task to complete or timeout
    while (task.process is not None) and task.process.is_alive():
        time.sleep(1)

    result = task.get_result()
    print(result[1].true_score)
