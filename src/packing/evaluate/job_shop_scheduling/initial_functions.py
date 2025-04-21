import numpy as np


def shortest_processing_time(
        ops_machine_ids: np.ndarray,
        ops_durations: np.ndarray,
        ops_mask: np.ndarray,
        machines_job_ids: np.ndarray,
        machines_remaining_times: np.ndarray,
        action_mask: np.ndarray,
) -> np.ndarray:
    """
    Assigns priority based on the shortest processing time (SPT) rule.

    Args:
        ops_machine_ids: Matrix (num_jobs, max_num_ops) of machine assignments per operation.
        ops_durations: Matrix (num_jobs, max_num_ops) of processing times for each operation.
        ops_mask: Boolean matrix (num_jobs, max_num_ops) indicating remaining unscheduled operations.
        machines_job_ids: Array (num_machines,) of job indices currently being processed on machines.
        machines_remaining_times: Array (num_machines,) of remaining processing times for each machine.
        action_mask: Boolean array (num_machines, num_jobs + 1) indicating legal actions per machine.

    Returns:
        A 1D array of shape (num_jobs,) with priority scores for each job.
    """
    # Gets the durations of the next task for a job, with infinity if the job completed all tasks
    next_op_durations = np.where(
        ops_mask.any(axis=1),
        ops_durations[np.arange(ops_durations.shape[0]), ops_mask.argmax(axis=1)],
        np.inf,
    )
    return -next_op_durations  # Higher priority for shorter jobs


def equal_priority(
        ops_machine_ids: np.ndarray,
        ops_durations: np.ndarray,
        ops_mask: np.ndarray,
        machines_job_ids: np.ndarray,
        machines_remaining_times: np.ndarray,
        action_mask: np.ndarray,
) -> np.ndarray:
    """
    Default heuristic: assigns equal priority to all jobs.

    Args:
        ops_machine_ids: Matrix (num_jobs, max_num_ops) of machine assignments per operation.
        ops_durations: Matrix (num_jobs, max_num_ops) of processing times for each operation.
        ops_mask: Boolean matrix (num_jobs, max_num_ops) indicating remaining unscheduled operations.
        machines_job_ids: Array (num_machines,) of job indices currently being processed on machines.
        machines_remaining_times: Array (num_machines,) of remaining processing times for each machine.
        action_mask: Boolean array (num_machines, num_jobs + 1) indicating legal actions per machine.

    Returns:
        A 1D array of shape (num_jobs,) with equal priority scores.
    """
    return np.zeros(ops_durations.shape[0])
