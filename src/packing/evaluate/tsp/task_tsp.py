import logging
# from gen_inst import TSPInstance, load_dataset
# from gls import guided_local_search
import time
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import numba as nb
import pickle

from packing.evaluate.registry import TASK_REGISTRY
from packing.evaluate.tsp.tsp_generate_dataset import TSPInstance, load_dataset, calculate_cost
import traceback
import os

######################
dataset_conf = {
    # 'train': (200,),
    # 'val':   (20, 50, 100, 200),
    'test': (100, 200),
}

batch_sizes = {
    # 20: 400,
    # 50: 20,
    100: 100,
    200: 100, }

perturbation_moves_map = {
    # 20: 5,
    # 50: 30//2,
    100: 40 // 2,
    200: 40 // 2,
}
iter_limit_map = {
    #     20: 73//17,
    #     50: 175//17,
    100: 800 // 50,
    200: 800 // 100,
}

SCALE = 1000000
test_sizes = list(iter_limit_map.keys())



def generate_input(cfg, set):
    # TODO: create the dataset here
    if set == "train":
        return ("train", "0.0")
    elif set == "trainperturbedset":
        return ("train", "0.2")
    elif set == "testset":
        return ("val", "0.0")
    else:
        raise ValueError(f"Unknown set: {set}")


def get_initial_func(cfg):
    if cfg.identity_heuristic:
        def heuristics(distance_matrix):
            return distance_matrix

        initial_function = heuristics
        function_str_to_extract = "heuristics"
    else:

        def heuristics(distance_matrix: np.ndarray) -> np.ndarray:
            # Calculate the average distance for each node
            average_distance = np.mean(distance_matrix, axis=1)

            # Calculate the distance ranking for each node
            distance_ranking = np.argsort(distance_matrix, axis=1)

            # Calculate the mean of the closest distances for each node
            closest_mean_distance = np.mean(
                distance_matrix[np.arange(distance_matrix.shape[0])[:, None], distance_ranking[:, 1:5]], axis=1)

            # Initialize the indicator matrix and calculate ratio of distance to average distance
            indicators = distance_matrix / average_distance[:, np.newaxis]

            # Set diagonal elements to np.inf
            np.fill_diagonal(indicators, np.inf)

            # Adjust the indicator matrix using the statistical measure
            indicators += closest_mean_distance[:, np.newaxis] / np.sum(distance_matrix, axis=1)[:, np.newaxis]

            return indicators

        initial_function = heuristics
        function_str_to_extract = "heuristics"

    return initial_function, function_str_to_extract


FloatArray = npt.NDArray[np.float_]
IntArray = npt.NDArray[np.int_]
usecache = True


@nb.njit(nb.float32(nb.float32[:, :], nb.uint16[:], nb.uint16), nogil=True, cache=usecache)
def _two_opt_once(distmat, tour, fixed_i=0):
    '''in-place operation'''
    n = tour.shape[0]
    p = q = 0
    delta = 0
    for i in range(1, n - 1) if fixed_i == 0 else range(fixed_i, fixed_i + 1):
        for j in range(i + 1, n):
            node_i, node_j = tour[i], tour[j]
            node_prev, node_next = tour[i - 1], tour[(j + 1) % n]
            if node_prev == node_j or node_next == node_i:
                continue
            change = (distmat[node_prev, node_j]
                      + distmat[node_i, node_next]
                      - distmat[node_prev, node_i]
                      - distmat[node_j, node_next])
            if change < delta:
                p, q, delta = i, j, change
    if delta < -1e-6:
        tour[p: q + 1] = np.flip(tour[p: q + 1])
        return delta
    else:
        return 0.0


@nb.njit(nb.float32(nb.float32[:, :], nb.uint16[:], nb.uint16), nogil=True, cache=usecache)
def _relocate_once(distmat, tour, fixed_i=0):
    n = distmat.shape[0]
    delta = p = q = 0
    for i in range(1, n) if fixed_i == 0 else range(fixed_i, fixed_i + 1):
        node = tour[i]
        prev_node = tour[i - 1]
        next_node = tour[(i + 1) % n]
        for j in range(n):
            if j == i or j == i - 1:
                continue
            prev_insert = tour[j]
            next_insert = tour[(j + 1) % n]
            cost = (- distmat[prev_node, node]
                    - distmat[node, next_node]
                    - distmat[prev_insert, next_insert]
                    + distmat[prev_insert, node]
                    + distmat[node, next_insert]
                    + distmat[prev_node, next_node])
            if cost < delta:
                delta, p, q = cost, i, j
    if delta >= 0:
        return 0.0
    if p < q:
        tour[p:q + 1] = np.roll(tour[p:q + 1], -1)
    else:
        tour[q:p + 1] = np.roll(tour[q:p + 1], 1)
    return delta


@nb.njit(nb.float32(nb.float32[:, :], nb.uint16[:]), nogil=True, cache=usecache)
def _calculate_cost(distmat, tour):
    cost = distmat[tour[-1], tour[0]]
    for i in range(len(tour) - 1):
        cost += distmat[tour[i], tour[i + 1]]
    return cost


@nb.njit(nb.float32(nb.float32[:, :], nb.uint16[:], nb.uint16, nb.uint16), nogil=True, cache=usecache)
def _local_search(distmat, cur_tour, fixed_i=0, count=1000):
    sum_delta = 0.0
    delta = -1
    while delta < 0 and count > 0:
        delta = 0
        delta += _two_opt_once(distmat, cur_tour, fixed_i)
        delta += _relocate_once(distmat, cur_tour, fixed_i)
        count -= 1
        sum_delta += delta
    return sum_delta


@nb.njit(nb.void(nb.float32[:, :], nb.float32[:, :], nb.float32[:, :], nb.uint16[:], nb.float32, nb.uint32), nogil=True,
         cache=usecache)
def _perturbation(distmat, guide, penalty, cur_tour, k, perturbation_moves=30):
    moves = 0
    n = distmat.shape[0]
    while moves < perturbation_moves:
        # Step 1: Identify the edge with the highest utility
        # penalize edge
        max_util = 0
        max_util_idx = 0
        # Iterate over all edges in current tour
        for i in range(n - 1):
            j = i + 1
            u, v = cur_tour[i], cur_tour[j]
            # Compute utility of the edge, high utility if high guide and low penalty
            util = guide[u, v] / (1.0 + penalty[u, v])
            # Find the edge with the highest utility
            if util > max_util:
                max_util_idx, max_util = i, util

        # Step 2: Penalize the edge with the highest utility
        penalty[cur_tour[max_util_idx], cur_tour[max_util_idx + 1]] += 1.0
        # Edge with the highest utility is now less likely to be selected

        # Step 3: Update the modified distance matrix
        edge_weight_guided = distmat + k * penalty

        # Step 4: Perform local search using the modified distances
        # Select indices around the penalized edge
        for fixed_i in (max_util_idx, max_util_idx + 1):
            # Skip invalid indices
            if fixed_i == 0 or fixed_i + 1 == n:
                continue
            delta = _local_search(edge_weight_guided, cur_tour, fixed_i, 1)
            if delta < 0:
                moves += 1


@nb.njit(nb.uint16[:](nb.float32[:, :], nb.uint16), nogil=True, cache=usecache)
def _init_nearest_neighbor(distmat, start):
    n = distmat.shape[0]
    tour = np.zeros(n, dtype=np.uint16)
    visited = np.zeros(n, dtype=np.bool_)
    visited[start] = True
    tour[0] = start
    for i in range(1, n):
        min_dist = np.inf
        min_idx = -1
        for j in range(n):
            if not visited[j] and distmat[tour[i - 1], j] < min_dist:
                min_dist = distmat[tour[i - 1], j]
                min_idx = j
        tour[i] = min_idx
        visited[min_idx] = True
    return tour


@nb.njit(nb.uint16[:](nb.float32[:, :], nb.float32[:, :], nb.uint16, nb.int32, nb.uint16), nogil=True, cache=usecache)
def _guided_local_search(
        distmat, guide, start, perturbation_moves=30, iter_limit=1000
) -> npt.NDArray[np.uint16]:
    penalty = np.zeros_like(distmat)

    best_tour = _init_nearest_neighbor(distmat, start)
    _local_search(distmat, best_tour, 0, 1000)
    best_cost = _calculate_cost(distmat, best_tour)
    k = 0.1 * best_cost / distmat.shape[0]
    cur_tour = best_tour.copy()

    for _ in range(iter_limit):
        _perturbation(distmat, guide, penalty, cur_tour, k, perturbation_moves)
        _local_search(distmat, cur_tour, 0, 1000)
        cur_cost = _calculate_cost(distmat, cur_tour)
        if cur_cost < best_cost:
            best_tour, best_cost = cur_tour.copy(), cur_cost
    return best_tour


def guided_local_search(
        distmat: FloatArray,
        guide: FloatArray,
        perturbation_moves: int = 30,
        iter_limit: int = 1000
) -> npt.NDArray[np.uint16]:
    return _guided_local_search(
        distmat=distmat.astype(np.float32),
        guide=guide.astype(np.float32),
        start=0,
        perturbation_moves=perturbation_moves,
        iter_limit=iter_limit,
    )


def solve(inst: TSPInstance, heuristics):
    start_time = time.time()
    heu = heuristics(inst.distmat.copy())
    result = guided_local_search(inst.distmat, heu, perturbation_moves_map[inst.n], iter_limit_map[inst.n])
    duration = time.time() - start_time
    return calculate_cost(inst, result), duration


def evaluate(function_from_llm, dataset_struct, iter_limit_map):
    t0 = time.perf_counter()
    split, drop_rate = dataset_struct

    # Load the optimal solutions
    logging.info(f"{os.getcwd()}")
    logging.info(
        f"[*] Loading optimal_objs_dict_{split}_drop_{drop_rate}_townsizes_100_200_perturbations_20_20_iterlimits_16_8.pkl")
    try:
        preamble = "."
        with open(
                f"{preamble}/dataset/optimal_objs_dict_{split}_drop_{drop_rate}_townsizes_100_200_perturbations_20_20_iterlimits_16_8.pkl",
                "rb") as f:
            optimal_objs_dict = pickle.load(f)
    except:
        preamble = "/claire-rcp-scratch/home/smmansou/packing/run"
        with open(
                f"{preamble}/dataset/optimal_objs_dict_{split}_drop_{drop_rate}_townsizes_100_200_perturbations_20_20_iterlimits_16_8.pkl",
                "rb") as f:
            optimal_objs_dict = pickle.load(f)
    logging.info(f"[*] Time taken to load optimal_objs_dict: {time.perf_counter() - t0:.6f}s")

    mean_gap_per_problem_size = []
    # print(f"Iter limit map{iter_limit_map}")
    for problem_size in iter_limit_map.keys():
        positions_path = f"{preamble}/dataset/{split}{problem_size}_positions.npy"
        dataset = load_dataset(positions_path, drop_rate)
        # print(f"[*] Evaluating {dataset_path}")

        objs = []
        durations = []
        for instance in tqdm(dataset):
            obj, duration = solve(instance, function_from_llm)
            objs.append(obj)
            durations.append(duration)

        # The final objective
        mean_obj = np.mean(objs).item()
        mean_optimal_obj = optimal_objs_dict[problem_size]
        gap = mean_obj / mean_optimal_obj - 1
        gap *= 100
        # print(f"[*] Average for {problem_size}: {mean_obj:.12f} ({mean_optimal_obj:.12f})")
        # print(f"[*] Average for {problem_size}: {mean_obj:.6f}")
        # print(f"[*] Optimality gap: {gap:.6f}%")
        # print(f"[*] Total/Average duration: {sum(durations):.6f}s {sum(durations)/len(durations):.6f}s")
        mean_gap_per_problem_size.append(gap)
    # Average the objective over all problem sizes
    avg_gap = sum(mean_gap_per_problem_size) / len(mean_gap_per_problem_size)
    # print(f"[*] Average across datasets: {avg_gap:.6f}")
    logging.info(f"Total time taken in evaluate: {time.perf_counter() - t0:.6f}s")
    return avg_gap


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
def evaluate_func(cfg, dataset_config, function_class):
    func_str = function_class.function_str
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
        exec(func_str, globals_dict, local_dict)

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

    # Here func_from_vllm can be None, if the function string does not contain the function
    try:
        avg_mean_obj = evaluate(func_from_llm, dataset_config, iter_limit_map)
        assert avg_mean_obj is not None
        # Assert that avg_mean_obj is either a float or an integer
        assert isinstance(avg_mean_obj,
                          (float, int, np.float64, np.float32, np.float16, np.int64, np.int32, np.int16, np.int8))
    except Exception as e:
        tb_str = traceback.format_exc()
        function_class.fail_flag = 1
        function_class.fail_exception = tb_str
        function_class.score = cfg.task.failed_score
        function_class.true_score = cfg.task.failed_score
        return function_class

    # We want to minimize the objective hence the minus sign
    score = - avg_mean_obj
    score = 100 * score
    # Round the score to 3 decimal places
    score = round(score, 3)
    function_class.score = score
    function_class.true_score = score
    function_class.fail_flag = 0
    function_class.correct_flag = 1

    return function_class


append_prompt = """You are tasked with creating a new function, heuristics(), that outperforms the other two presented functions. 
The heuristics() function takes as input a distance matrix, and returns prior indicators of how undesirable it is to include each edge in a solution. The returned matrix should be of the same shape as the input. 
When writing the new function, follow these guidelines:
Think Outside the Box: Avoid simply rewriting or rephrasing existing approaches. Prioritize creating novel solutions rather than making superficial tweaks.
Analyze the Score Drivers: Analyze the characteristics of the higher-scoring function. Identify what it is doing differently or more effectively than the lower-scoring function. Determine which specific changes or techniques lead to better performance.
To summarize, your task is to write a new function named heuristics() that will perform better than both functions above and achieve a higher score."""

system_prompt = "You are helpful, excellent and innovative problem solver specializing in mathematical optimization and algorithm design. You are an expert in writing Python functions."

TASK_REGISTRY.register(
    "tsp",
    generate_input=generate_input,
    evaluate_func=evaluate_func,
    get_initial_func=get_initial_func,
    system_prompt=system_prompt,
    append_prompt=append_prompt,
)
