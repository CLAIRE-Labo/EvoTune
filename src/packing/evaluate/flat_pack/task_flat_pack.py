import copy
import os
import time
import traceback
from typing import Callable

import jax
import numpy as np
from jumanji.environments import FlatPack
from matplotlib import pyplot as plt

from packing.evaluate.flat_pack.custom_generators import PreloadedGenerator
from packing.logging.function_class import FunctionClass


# ---------------------------------------------------------------------------
# INITIAL FUNCTION SETUP
# ---------------------------------------------------------------------------
def get_initial_func(cfg) -> tuple[Callable, str]:
    """
    Returns the initial heuristic function and its name based on the configuration.

    Args:
        cfg: Configuration object with attribute 'init_shortest_processing_time'.

    Returns:
        A tuple containing the heuristic function and its name as a string.
    """
    if cfg.init_equal:
        def priority(
                current_grid: np.ndarray,
                blocks: np.ndarray,
                action_mask: np.ndarray
        ) -> np.ndarray:
            """
            Computes values for all valid (block, rotation, row-2, col-2) placements.
            Larger values get prioritized.

            Args:
                current_grid: Grid representing the current state of the environment. Numpy array (float32) of shape (num_rows, num_cols).
                blocks: Array containing available blocks. Numpy array (float32) of shape (num_blocks, 3, 3), indicating the 3x3 blocks un-rotated.
                action_mask: Boolean mask indicating valid placements. Numpy array (bool) of shape (num_blocks, 4, num_rows-2, num_cols-2), indicating whether the top left corner of each rotation (4 possible) of each block can be placed in the grid.

            Returns:
                A numpy array of shape (num_blocks, 4, num_rows-2, num_cols-2), indicating the value of placing the top left corner of each rotation (4 possible) of each block at that location in the grid. Higher value is better.
            """
            # Precompute rotated versions of all blocks
            num_blocks = blocks.shape[0]
            rotated_blocks = np.array([[np.rot90(block, k=r) for r in range(4)] for block in blocks])

            # Pad the grid once (for boundary checking)
            padded_grid = np.pad(current_grid, 1, mode='constant', constant_values=0)

            # Initialize Q-value matrix
            values = np.full(action_mask.shape, -np.inf)

            # Vectorized placement loop to
            for block_idx in range(num_blocks):
                for rotation in range(4):
                    block = rotated_blocks[block_idx, rotation]
                    block_rows, block_cols = block.shape

                    # Extract all possible placements using NumPy slicing
                    sub_grids = np.lib.stride_tricks.sliding_window_view(padded_grid, (block_rows, block_cols))

                    # Compute final scores, with the stride trick above it becomes possible to efficiently assign scores over
                    # all possible placements. For now just assigns equal score everywhere.
                    values[block_idx, rotation, ...] = 1

            # Apply action mask in one operation
            values[~action_mask] = -np.inf

            return values
    else:
        def priority(
                current_grid: np.ndarray,
                blocks: np.ndarray,
                action_mask: np.ndarray
        ) -> np.ndarray:
            """
            Computes random values for all valid (block, rotation, row-2, col-2) placements.
            Larger values get prioritized.

            Args:
                current_grid: Grid representing the current state of the environment. Numpy array (float32) of shape (num_rows, num_cols).
                blocks: Array containing available blocks. Numpy array (float32) of shape (num_blocks, 3, 3), indicating the 3x3 blocks un-rotated.
                action_mask: Boolean mask indicating valid placements. Numpy array (bool) of shape (num_blocks, 4, num_rows-2, num_cols-2), indicating whether the top left corner of each rotation (4 possible) of each block can be placed in the grid.

            Returns:
                A numpy array of shape (num_blocks, 4, num_rows-2, num_cols-2), indicating the value of placing the top left corner of each rotation (4 possible) of each block at that location in the grid. Higher value is better.
            """
            # Generate random scores between 0 and 1
            values = np.random.rand(*action_mask.shape)

            return values

    return priority, "priority"


# ---------------------------------------------------------------------------
# INPUT GENERATION
# ---------------------------------------------------------------------------
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
        return cfg.train_set_path
    elif set == "trainperturbedset":
        return cfg.train_perturbed_set_path
    elif set == "testset":
        return cfg.test_set_path
    else:
        raise ValueError(f"Unknown set: {set}")


# ---------------------------------------------------------------------------
# ENVIRONMENT EVALUATION
# ---------------------------------------------------------------------------
def evaluate_flatpack_instance(env: FlatPack, random_key: jax.random.PRNGKey, heuristic_func: Callable) -> float:
    """
    Runs one episode of the FlatPack simulation using the given heuristic function.

    Args:
        env: An instance of the FlatPack environment.
        random_key: A random key for reproducibility.
        heuristic_func: A function that assigns Q-values for valid actions.

    Returns:
        The final score achieved by the heuristic.
    """
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    state, timestep = reset_fn(random_key)

    obs = timestep.observation
    done = False

    while not done:
        # Compute heuristic Q-values
        q_values = heuristic_func(
            obs.grid,
            obs.blocks,
            obs.action_mask
        )

        # Mask invalid actions by setting them to -inf
        masked_q_values = np.where(obs.action_mask, q_values, -np.inf)

        # Select the action with the highest Q-value
        best_action_idx = np.unravel_index(np.argmax(masked_q_values), masked_q_values.shape)
        action = np.array(best_action_idx, dtype=np.int32)

        # Step through environment
        state, timestep = step_fn(state, action)
        obs = timestep.observation
        done = timestep.last()

    num_cells = len(state['grid']) * len(state['grid'][0])
    proportion_cells_occupied = 1 - np.sum(state['grid'] == 0) / num_cells
    return proportion_cells_occupied.item()


# ---------------------------------------------------------------------------
# FUNCTION EVALUATION (WRAPPER)
# ---------------------------------------------------------------------------
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


def evaluate_func(cfg, dataset_name: str, function_class: FunctionClass) -> FunctionClass:
    """
    Evaluates the heuristic function on a set of FlatPack instances.

    Args:
        cfg: The OmegaConf config
        dataset_name: Path to JSON file containing preloaded instances.
        function_class: An object containing the function and its metadata.

    Returns:
        The updated function_class with evaluation score and failure flags.
    """
    os.environ["JAX_PLATFORM_NAME"] = "cpu"  # We can't use GPU with multiprocessing for evaluation very well yet
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    function_str = function_class.function_str
    imports = function_class.imports_str

    try:
        globals_dict = {}
        exec(GENERAL_IMPORTS, globals_dict)  # General imports
        exec(imports, globals_dict)
        local_dict = {}
        exec(function_str, globals_dict, local_dict)
        heuristic_func = local_dict.get(cfg.function_str_to_extract)
        assert heuristic_func is not None
    except Exception as e:
        tb_str = traceback.format_exc()
        function_class.fail_flag = 1
        function_class.fail.reason_imports = 1
        function_class.score = cfg.failed_score
        function_class.true_score = cfg.failed_score
        function_class.fail.exception = tb_str
        return function_class

    env_generator = PreloadedGenerator(instance_file=dataset_name)
    env = FlatPack(generator=env_generator)  # Have to re-initialize to recompute the grid size
    key = jax.random.PRNGKey(0)  # Fixed seed for reproducibility

    scores = []
    for _ in range(env_generator.num_instances):  # Loop through all instances
        try:
            cur_key, key = jax.random.split(key, num=2)
            env = FlatPack(generator=env_generator)  # Have to re-initialize to recompute the grid size
            score = evaluate_flatpack_instance(env, cur_key, heuristic_func)
            assert score is not None
            assert isinstance(
                score,
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

            scores.append(score)
        except Exception as e:
            tb_str = traceback.format_exc()
            print(tb_str)
            function_class.fail_flag = 1
            function_class.fail.reason_exception = 1
            function_class.fail.exception = tb_str
            function_class.score = cfg.failed_score
            function_class.true_score = cfg.failed_score
            return function_class

    avg_score = np.mean(scores)  # This will be the average occupation proportion, which we want to maximize
    avg_score *= 100  # Make it more interpretable for the LLM
    optimality_gap = round(avg_score, 3) - 100
    function_class.score = optimality_gap
    function_class.true_score = optimality_gap
    function_class.fail_flag = 0
    function_class.correct_flag = 1

    return function_class


# ---------------------------------------------------------------------------
# MAIN BLOCK (TESTING)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from omegaconf import OmegaConf
    from packing.utils.functions import function_to_string

    # Rendering example
    import matplotlib

    print(matplotlib.get_backend())
    matplotlib.use("TkAgg")

    cfg = OmegaConf.create({
        "train_set_path": "data/flat_pack/train_flatpack_dynamic_0_seed.json",
        "trainperturbedset": "",
        "test_set_path": "data/flat_pack/test_flatpack_dynamic_0_seed.json",
        "init_adjacency_scores": 1,
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

    print(f"Evaluating heuristic function: {func_str} on FlatPack train set...")
    result = evaluate_func(cfg, dataset_name, function_class)
    print("Score:", result.true_score)
