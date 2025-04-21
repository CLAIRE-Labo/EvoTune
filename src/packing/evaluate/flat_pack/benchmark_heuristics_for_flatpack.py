import os
import time
import os

from packing.evaluate.flat_pack.task_flat_pack import generate_input, evaluate_func
import packing.evaluate.flat_pack.initial_functions as initial_functions
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
            "train_set_path": "data/flat_pack/train_flatpack_dynamic_0_seed.json",
            "train_perturbed_set_path": "data/flat_pack/train_perturbed_flatpack_0_seed.json",
            "test_set_path": "data/flat_pack/test_flatpack_dynamic_0_seed.json",
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

        if cfg.train_set_path != "":
            start_time = time.time()
            train_result = evaluate_func(cfg, generate_input(cfg, "train"), function_class)
            end_time = time.time()
            print("\tTrain Optimality Gap:", train_result.true_score)
            print(f'\tDuration for eval: {round(end_time - start_time, 3)}s')

        if cfg.train_perturbed_set_path != "":
            start_time = time.time()
            test_result = evaluate_func(cfg, generate_input(cfg, "trainperturbedset"), function_class)
            end_time = time.time()
            print("\tTrain Perturbed Optimality Gap:", test_result.true_score)
            print(f'\tDuration for eval: {round(end_time - start_time, 3)}s')

        if cfg.test_set_path != "":
            start_time = time.time()
            test_result = evaluate_func(cfg, generate_input(cfg, "testset"), function_class)
            end_time = time.time()
            print("\tTest Optimality Gap:", test_result.true_score)
            print(f'\tDuration for eval: {round(end_time - start_time, 3)}s')
