import os
import traceback
from typing import Callable

import jax
import numpy as np
from jumanji.environments import JobShop
from tqdm import tqdm

from packing.evaluate.job_shop_scheduling.custom_generators import PreloadedGenerator
from packing.logging.function_class import FunctionClass


# -----------------------------------------------------------------------------
# INITIAL FUNCTION SETUP
# -----------------------------------------------------------------------------
def get_initial_func(cfg) -> tuple[Callable, str]:
    """
    Returns the initial heuristic function and its name based on the configuration.

    Args:
        cfg: Configuration object with attribute 'init_shortest_processing_time'.

    Returns:
        A tuple containing the heuristic function and its name as a string.
    """
    if cfg.init_shortest_processing_time:
        def priority(
                ops_machine_ids: np.ndarray,
                ops_durations: np.ndarray,
                ops_mask: np.ndarray,
                machines_job_ids: np.ndarray,
                machines_remaining_times: np.ndarray,
                action_mask: np.ndarray,
        ) -> np.ndarray:
            """
            Assigns priority to jobs, where jobs with a higher priority will be scheduled first on their current machine

            Args:
                ops_machine_ids: Numpy array (int32) of shape (num_jobs, max_num_ops). For each job, it specifies the machine each op must be processed on. Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
                ops_durations: Numpy array (int32) of shape (num_jobs, max_num_ops). For each job, it specifies the processing time of each operation. Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
                ops_mask: Numpy array (bool) of shape (num_jobs, max_num_ops). For each job, indicates which operations remain to be scheduled. False if the op has been scheduled or if the op was added for padding, True otherwise. The first True in each row (i.e. each job) identifies the next operation for that job.
                machines_job_ids: Numpy array (int32) of shape (num_machines,). For each machine, it specifies the job currently being processed. Note that -1 means no-op in which case the remaining time until available is always 0.
                machines_remaining_times: Numpy array (int32) of shape (num_machines,). For each machine, it specifies the number of time steps until available.
                action_mask: Numpy array (bool) of (num_machines, num_jobs + 1). For each machine, it indicates which jobs (or no-op) can legally be scheduled. The last column corresponds to no-op.

            Returns:
                A 1D numpy array of shape (num_jobs,) with priority scores for each job. Jobs with a higher score will be prioritized to be scheduled first on their machine
            """
            # Gets the durations of the next task for a job, with infinity if the job completed all tasks
            next_op_durations = np.where(
                ops_mask.any(axis=1),
                ops_durations[np.arange(ops_durations.shape[0]), ops_mask.argmax(axis=1)],
                np.inf,
            )
            return -next_op_durations  # Higher priority for shorter jobs
    else:
        def priority(
                ops_machine_ids: np.ndarray,
                ops_durations: np.ndarray,
                ops_mask: np.ndarray,
                machines_job_ids: np.ndarray,
                machines_remaining_times: np.ndarray,
                action_mask: np.ndarray,
        ) -> np.ndarray:
            """
            Assigns priority to jobs, where jobs with a higher priority will be scheduled first on their current machine

            Args:
                ops_machine_ids: Numpy array (int32) of shape (num_jobs, max_num_ops). For each job, it specifies the machine each op must be processed on. Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
                ops_durations: Numpy array (int32) of shape (num_jobs, max_num_ops). For each job, it specifies the processing time of each operation. Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
                ops_mask: Numpy array (bool) of shape (num_jobs, max_num_ops). For each job, indicates which operations remain to be scheduled. False if the op has been scheduled or if the op was added for padding, True otherwise. The first True in each row (i.e. each job) identifies the next operation for that job.
                machines_job_ids: Numpy array (int32) of shape (num_machines,). For each machine, it specifies the job currently being processed. Note that -1 means no-op in which case the remaining time until available is always 0.
                machines_remaining_times: Numpy array (int32) of shape (num_machines,). For each machine, it specifies the number of time steps until available.
                action_mask: Numpy array (bool) of (num_machines, num_jobs + 1). For each machine, it indicates which jobs (or no-op) can legally be scheduled. The last column corresponds to no-op.

            Returns:
                A 1D numpy array of shape (num_jobs,) with priority scores for each job. Jobs with a higher score will be prioritized to be scheduled first on their machine
            """
            return np.zeros(ops_durations.shape[0])

    return priority, "priority"


# -----------------------------------------------------------------------------
# INPUT GENERATION
# -----------------------------------------------------------------------------
def generate_input(cfg, set: str) -> str:
    """
    Generates a FlatPack environment instance from a dataset.

    Args:
        cfg: The OmegaConfig
        set: The dataset name

    Returns:
        An instance of the FlatPack environment.
    """
    if set == "train":
        return cfg.train_set_path_jssp
    elif set == "trainperturbedset":
        return cfg.train_perturbed_set_path_jssp
    elif set == "testset":
        return cfg.test_set_path_jssp
    else:
        raise ValueError(f"Unknown set: {set}")


# -----------------------------------------------------------------------------
# ENVIRONMENT EVALUATION
# -----------------------------------------------------------------------------
def evaluate_jssp_instance(env: JobShop, random_key: jax.random.PRNGKey, priority_func: Callable) -> float:
    """
    Runs one episode of the JSSP simulation using the provided priority function.

    Args:
        env: An instance of the JobShop environment.
        random_key: A random key, needed, for resetting the environment and reproducability
        priority_func: A heuristic function that returns priority scores.

    Returns:
        The makespan (total time to complete all jobs) as a float.
    """
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    state, timestep = reset_fn(random_key)

    obs = timestep.observation
    done = False

    cumulative_reward = 0
    while not done:
        action = np.full(env.num_machines, env.num_jobs)  # Initialize with no-op actions

        # Determine which machines are idle
        idle_machines = np.where(1 - obs.machines_remaining_times)[0]

        if idle_machines.size > 0:
            # Calculate priority scores for each job
            priorities = priority_func(
                obs.ops_machine_ids,
                obs.ops_durations,
                obs.ops_mask,
                obs.machines_job_ids,
                obs.machines_remaining_times,
                obs.action_mask,
            )

            for machine in idle_machines:
                # Identify jobs that can be scheduled on this machine
                valid_jobs = np.where(obs.action_mask[machine, :-1])[0]

                if valid_jobs.size > 0:
                    # Select the job with the highest priority
                    selected_job = valid_jobs[np.argmax(priorities[valid_jobs])]
                    action[machine] = selected_job

        state, timestep = step_fn(state, action)

        obs = timestep.observation
        done = timestep.last()
        cumulative_reward += timestep.reward

    # env.render(state)

    # This is the makespan we obtained (since the environment gives -1 rewards)
    makespan = -cumulative_reward.item()
    return makespan


# -----------------------------------------------------------------------------
# FUNCTION EVALUATION (WRAPPER)
# -----------------------------------------------------------------------------
GENERAL_IMPORTS = '''
import random
import numpy
import numpy as np
from itertools import product
import math
import scipy
import scipy.stats
import scipy.special
import copy
'''


def evaluate_func(cfg, dataset_name: str, function_class) -> FunctionClass:
    """
    Evaluates the heuristic function on a set of JSSP instances.

    Args:
        cfg: Configuration object.
        dataset_name: Name of the dataset to evaluate on.
        function_class: An object containing the function and its imports as strings.

    Returns:
        The updated function_class with evaluation score and failure flags.
    """
    os.environ["JAX_PLATFORM_NAME"] = "cpu"  # We can't use GPU with multiprocessing for evaluation very well yet
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    function_str = function_class.function_str
    imports = function_class.imports_str

    try:
        globals_dict = {}
        exec(GENERAL_IMPORTS, globals_dict)
        exec(imports, globals_dict)
        local_dict = {}
        exec(function_str, globals_dict, local_dict)
        func_from_llm = local_dict.get(cfg.function_str_to_extract)
        assert func_from_llm is not None
    except Exception as e:
        tb_str = traceback.format_exc()
        function_class.fail_flag = 1
        function_class.fail.reason_imports = 1
        function_class.score = cfg.failed_score
        function_class.true_score = cfg.failed_score
        function_class.fail.exception = tb_str
        return function_class

    env_generator = PreloadedGenerator(instance_file=dataset_name)
    env = JobShop(generator=env_generator) # Have to re-initialize to recompute the grid size
    key = jax.random.PRNGKey(seed=0)  # The seed doesn't matter here since our generator and env is deterministic

    optimality_gaps = []
    for instance_idx in range(env.generator.num_instances):
        try:
            cur_key, key = jax.random.split(key, num=2)
            env = JobShop(generator=env_generator)  # Have to re-initialize to recompute the grid size
            makespan = evaluate_jssp_instance(env, cur_key, func_from_llm)
            assert makespan is not None
            assert isinstance(
                makespan,
                (
                    float,
                    int,
                    np.float64,
                    np.float32,
                    np.float16,
                    np.int64,
                    np.int32,
                    np.int16,
                    np.int8
                )
            )

            lower_bound = env_generator.get_lower_bound_for_instance_idx(instance_idx=instance_idx)
            optimality_gap = (makespan - lower_bound) / lower_bound
            optimality_gaps.append(optimality_gap)
        except Exception as e:
            tb_str = traceback.format_exc()
            print(tb_str)
            function_class.fail_flag = 1
            function_class.fail.reason_exception = 1
            function_class.fail.exception = tb_str
            function_class.score = cfg.failed_score
            function_class.true_score = cfg.failed_score
            return function_class

    avg_optimality_gap = np.mean(optimality_gaps)  # We want to minimize this term
    score = round(-avg_optimality_gap * 100, 3)  # Turn it into maximization and make it more interpretable for the LLM

    function_class.score = score
    function_class.true_score = score
    function_class.fail_flag = 0
    function_class.correct_flag = 1

    return function_class


# -----------------------------------------------------------------------------
# MAIN BLOCK (TESTING)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from omegaconf import OmegaConf
    from packing.utils.functions import function_to_string
    from packing.logging.function_class import FunctionClass

    # Rendering example
    import matplotlib

    print(matplotlib.get_backend())
    # matplotlib.use("TkAgg")

    cfg = OmegaConf.create({
        "train_set_path_jssp": "data/job_shop_scheduling/jssp_20j_10m_8ops_5maxdur_100_instances.json",
        "trainperturbedset_jssp": "",
        "test_set_path_jssp": "",
        "init_shortest_processing_time": 1,
        "failed_score": -1e-6,
    })

    # Convert heuristic function to a string
    initial_function, func_str = get_initial_func(cfg)
    cfg.function_str_to_extract = func_str

    function_str = function_to_string(initial_function)
    imports_str = "import numpy as np"

    dataset_name = generate_input(cfg, "train")

    # Create a FunctionClass object
    function_class = FunctionClass(function_str, imports_str)

    print("Evaluating heuristic function on JSSP testset...")
    result = evaluate_func(cfg, dataset_name, function_class)
    print("Score (negative average makespan):", result.true_score)
