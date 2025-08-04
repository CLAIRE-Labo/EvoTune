import numpy as np
from omegaconf import DictConfig, OmegaConf
from packing.evaluate.bin_packing.bin_datasets import datasets
import traceback
import copy

from packing.evaluate.registry import TASK_REGISTRY


def get_initial_func(cfg):
    # assert cfg.init_best_fit or cfg.init_const, "Invalid initialization configuration"
    if cfg.init_best_fit:
        def priority(item: float, bins: np.ndarray) -> np.ndarray:
            """Returns priority with which we want to add item to each bin.

            Args:
                item: Size of item to be added to the bin.
                bins: Array of capacities for each bin.

            Return:
                Array of same size as bins with priority score of each bin.
            """
            return -(bins - item)

    else:
        def priority(item: float, bins: np.ndarray) -> np.ndarray:
            """Returns priority with which we want to add item to each bin.

            Args:
                item: Size of item to be added to the bin.
                bins: Array of capacities for each bin.

            Return:
                Array of same size as bins with priority score of each bin.
            """
            return np.ones_like(bins)

    initial_function = priority
    function_str_to_extract = "priority"
    #   else:
    #     raise ValueError("Invalid initialization configuration")

    # For debugging: check the source code of a function
    # inspect.getsource(dummy_function)
    return initial_function, function_str_to_extract


def l1_bound(items: tuple[int, ...], capacity: int) -> float:
    """Computes L1 lower bound on OPT for bin packing.

    Args:
      items: Tuple of items to pack into bins.
      capacity: Capacity of bins.

    Returns:
      Lower bound on number of bins required to pack items.
    """
    return np.ceil(np.sum(items) / capacity)


def l1_bound_dataset(instances: dict) -> float:
    """Computes the mean L1 lower bound across a dataset of bin packing instances.

    Args:
        instances: Dictionary containing a set of bin packing instances.

    Returns:
        Average L1 lower bound on number of bins required to pack items.
    """
    l1_bounds = []
    for name in instances:
        instance = instances[name]
        l1_bounds.append(l1_bound(instance['items'], instance['capacity']))
    return np.mean(l1_bounds)


opt_num_bins = {}
for name, dataset in datasets.items():
    opt_num_bins[name] = l1_bound_dataset(dataset)


## TASK ESSENTIALS
# Function to create the input
def generate_input(cfg, set):
    # 1. The [OR3 dataset](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/binpackinfo.html) containing 20 bin packing instances, each with 500 items.
    # 2. The Weibull 5k dataset containing 5 bin packing instances, each with 5,000 items.
    if set == "train":
        if cfg.Weibull:
            return 'Weibull 5k'
        elif cfg.OR:
            return 'OR3'

    elif set == "trainperturbedset":
        return 'OR3_permuted'

    elif set == "testset":
        return 'OR3_val'
    else:
        raise ValueError("Invalid dataset name")


def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]


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


## TASK ESSENTIALS
def evaluate_func(cfg, dataset_name, function_class):
    priority_func_str = function_class.function_str
    imports = function_class.imports_str

    # Execute imports and the function
    try:
        # Create a shared globals dictionary
        globals_dict = {}

        # Execute general imports
        exec(GENERAL_IMPORTS, globals_dict)

        # Execute the imports into the globals dictionary
        exec(imports, globals_dict)

        # Execute the perturbation function string using the same globals dictionary
        local_dict = {}
        exec(priority_func_str, globals_dict, local_dict)

        # Extract the function from the local dictionary
        func_from_llm = local_dict.get(cfg.function_str_to_extract)

        assert func_from_llm is not None

    except Exception as e:
        tb_str = traceback.format_exc()
        function_class.fail_flag = 1
        function_class.score = cfg.task.failed_score
        function_class.true_score = cfg.task.failed_score
        function_class.fail_exception = tb_str
        return function_class

    instances = datasets[dataset_name]
    ## EVALUATE
    """Evaluate heuristic function on a set of online binpacking instances."""
    # List storing number of bins used for each instance.
    num_bins = []
    # Perform online binpacking for each instance.
    for name in instances:
        instance = instances[name]
        capacity = instance['capacity']
        items = instance['items']
        # Create num_items bins so there will always be space for all items,
        # regardless of packing order. Array has shape (num_items,).
        bins = np.array([capacity for _ in range(instance['num_items'])])
        # Pack items into bins and return remaining capacity in bins_packed, which
        # has shape (num_items,).

        ## ONLINE BINPACK
        """Performs online binpacking of `items` into `bins`."""
        # Track which items are added to each bin.
        packing = [[] for _ in bins]
        # Add items to bins.
        for item in items:
            # Extract bins that have sufficient space to fit item.
            valid_bin_indices = get_valid_bin_indices(item, bins)
            # Score each bin based on heuristic.
            item_original = copy.deepcopy(item)
            bins_original = copy.deepcopy(bins)
            valid_bin_indices_original = copy.deepcopy(valid_bin_indices)
            try:
                priorities = func_from_llm(item, bins[valid_bin_indices])
                assert item == item_original, "Item has been modified"
                assert np.array_equal(bins, bins_original), "Bins have been modified"
                assert np.array_equal(valid_bin_indices,
                                      valid_bin_indices_original), "Valid bin indices have been modified"
                assert priorities is not None, "Priorities is None"
                assert np.__name__ == "numpy", "Do not overwrite np"
                assert isinstance(priorities, np.ndarray), f"Priorities is not a numpy array, but a {type(priorities)}"
                assert len(priorities) > 0, "Priorities is empty"
                assert np.all(np.isfinite(priorities)), "Priorities contains non-finite values"
                assert np.all(np.isreal(priorities)), "Priorities contains complex values"
                best_bin = valid_bin_indices[np.argmax(priorities)]
                # Assert that best_bin is an integer, or numpy integer
                assert isinstance(best_bin, (int, np.int16, np.int32,
                                             np.int64)), f"Best_bin is not an integer but a {type(best_bin)}"
            except Exception as e:
                tb_str = traceback.format_exc()
                function_class.fail_flag = 1
                function_class.fail_exception = tb_str
                function_class.score = cfg.task.failed_score
                function_class.true_score = cfg.task.failed_score
                return function_class
            bins[best_bin] -= item
            packing[best_bin].append(item)
        # Remove unused bins from packing.
        packing = [bin_items for bin_items in packing if bin_items]
        # If remaining capacity in a bin is equal to initial capacity, then it is
        # unused. Count number of used bins.
        num_bins.append((bins != capacity).sum())
    # Score of heuristic function is negative of average number of bins used
    # across instances (as we want to minimize number of bins).

    avg_num_bins = np.mean(num_bins)
    excess = (avg_num_bins - opt_num_bins[dataset_name]) / opt_num_bins[dataset_name]

    score = - excess * 100  # to make it a percentage
    score = score * 100  # multiply again to work better with the prompt, round to 2 decimal places
    score = round(score, 2)
    function_class.score = score
    function_class.true_score = score
    function_class.fail_flag = 0
    function_class.correct_flag = 1

    return function_class

append_prompt = """You are tasked with creating a new function, priority(), that outperforms the other two presented functions. To achieve this, follow these guidelines:
Think Outside the Box: Avoid simply rewriting or rephrasing existing approaches. Prioritize creating novel solutions rather than making superficial tweaks.
Analyze the Score Drivers: Analyze the characteristics of the higher-scoring function. Identify what it is doing differently or more effectively than the lower-scoring function. Determine which specific changes or techniques lead to better performance.
Experiment with Variations: Use the insights to create a new function that builds upon successful ideas but introduces innovative variations. Consider entirely new strategies or optimizations that were not present in the previous attempts.
To summarize, your task is to write a new function named priority() that will perform better than both functions above and achieve a higher score."""

system_prompt = "You are helpful, excellent and innovative problem solver specializing in mathematical optimization and algorithm design. You are an expert in writing Python functions."

TASK_REGISTRY.register(
    "bin",
    generate_input=generate_input,
    evaluate_func=evaluate_func,
    get_initial_func=get_initial_func,
    system_prompt=system_prompt,
    append_prompt=append_prompt,
)

if __name__ == "__main__":

    cfg = OmegaConf.create({
        "Weibull": False,
        "function_str_to_extract": "priority",
        "OR": True,
        "failed_score": -200000
    })


    def priority(item: float, bins: np.ndarray) -> np.ndarray:
        """Returns priority with which we want to add item to each bin.

        Args:
            item: Size of item to be added to the bin.
            bins: Array of capacities for each bin.

        Return:
            Array of same size as bins with priority score of each bin.
        """
        return -(bins - item)


    def priority(item: float, bins: np.ndarray) -> np.ndarray:
        """Returns priority with which we want to add item to each bin.

        Args:
            item: Size of item to be added to the bin.
            bins: Array of capacities for each bin.

        Return:
            Array of same size as bins with priority score of each bin.
        """
        bin_count = len(bins)

        # Initialize DP table
        dp = np.full((bin_count + 1, sum(bins) + 1), -np.inf)
        dp[0][0] = 0

        # Fill the DP table
        for i in range(1, bin_count + 1):
            for j in range(sum(bins) + 1):
                if j >= bins[i - 1]:
                    # Option 1: Place the item in the current bin
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - bins[i - 1]] + item)
                # Option 2: Skip the current bin
                dp[i][j] = max(dp[i][j], dp[i - 1][j])

        # Calculate priorities using the DP table
        priorities = np.zeros(bin_count)
        for j in range(1, sum(bins) + 1):
            # Find the best way to fill bins to reach this capacity
            for i in range(bin_count, 0, -1):
                if dp[i][j] != dp[i - 1][j]:
                    priorities[i - 1] += item
                    j -= bins[i - 1]
                    break

        return priorities


    from packing.utils.functions import string_to_function, extract_functions, extract_imports, function_to_string
    from packing.logging.function_class import FunctionClass

    function_str = function_to_string(priority)
    imports_str = "import numpy as np"

    function_class = FunctionClass(function_str, imports_str)

    dataset_name = generate_input(cfg)

    # Time the function evaluation
    import time

    start = time.time()

    print(f"Evaluating function on {dataset_name} dataset")
    result = evaluate_func(cfg, dataset_name, function_class)

    end = time.time()
    print(f"Time taken: {end - start} seconds")

    print(result.true_score)
