import time

from packing.evaluate.job_shop_scheduling.task_job_shop_scheduling import generate_input, evaluate_func, get_initial_func
import packing.evaluate.job_shop_scheduling.initial_functions as initial_functions
from packing.logging.function_class import FunctionClass
import inspect

from omegaconf import OmegaConf
from packing.utils.functions import function_to_string

if __name__ == "__main__":
    # Get all functions defined in the module `initial_functions`
    methods = inspect.getmembers(initial_functions, inspect.isfunction)

    # Print function names
    for func_name, func in methods:
        cfg = OmegaConf.create({
            "train_set_path_jssp": "data/job_shop_scheduling/train_jssp_dynamic_instances.json",
            "trainperturbedset_jssp": "",
            "test_set_path_jssp": "",
            "init_adjacency_scores": 1,
            "failed_score": -1e-6,
        })

        # Convert heuristic function to a string
        cfg.function_str_to_extract = func_name

        function_str = function_to_string(func)
        imports_str = "import numpy as np"

        # Create a FunctionClass object
        function_class = FunctionClass(function_str, imports_str)
        print(f'Function: {func_name}')
        start_time = time.time()
        train_result = evaluate_func(cfg, generate_input(cfg, "train"), function_class)
        end_time = time.time()
        print("\tTrain Optimality Gap:", train_result.true_score)
        print(f'\tDuration for eval: {round(end_time - start_time, 3)}s')
        # start_time = time.time()
        # test_result = evaluate_func(cfg, generate_input(cfg, "testset"), function_class)
        # end_time = time.time()
        # print("\tTest Optimality Gap:", test_result.true_score)
        # print(f'\tDuration for eval: {round(end_time - start_time, 3)}s')
