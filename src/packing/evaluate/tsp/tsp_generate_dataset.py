import numpy as np
import logging
import time
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import concurrent.futures
import torch
import pickle

import numpy as np
import numpy.typing as npt
from scipy.spatial import distance_matrix
import os

######################
dataset_conf = {
    'train': (100, 200),
    'val': (100, 200),
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
sizes = list(iter_limit_map.keys())



class TSPInstance:
    def __init__(self, positions: npt.NDArray[np.float_], adjacency: npt.NDArray[np.float_]) -> None:
        self.positions = positions
        self.adjacency = adjacency
        self.n = positions.shape[0]
        self.distmat = distance_matrix(positions, positions) + np.eye(self.n) * 1e-5
        self.distmat = self.distmat * adjacency + (1 - adjacency) * 10


def generate_dataset(filepath, n, batch_size=64, drop_rate=0.0):
    if not os.path.exists(filepath):
        positions = np.random.random((batch_size, n, 2))
        print(f"Saving dataset to {filepath}")
        np.save(filepath, positions)
    else:
        print(f"Dataset already exists at {filepath}")
        positions = np.load(filepath)
    adjacency_matrix = np.ones((batch_size, n, n))
    for i in range(batch_size):
        for j in range(n):
            for k in range(j):
                if np.random.rand() < drop_rate:
                    adjacency_matrix[i, j, k] = 0
                    adjacency_matrix[i, k, j] = 0
    np.save(filepath.replace("positions", f"adjacency_{drop_rate}"), adjacency_matrix)  #


def generate_datasets(basepath=None, drop_rate=0.0):
    import os
    # TODO: change the directory
    basepath = basepath or os.path.join(os.getcwd(), "dataset")
    os.makedirs(basepath, exist_ok=True)

    for split, problem_sizes in dataset_conf.items():
        np.random.seed(len(split))
        for n in problem_sizes:
            batch_size = batch_sizes[n]
            filepath = os.path.join(basepath, f"{split}{n}_positions.npy")
            # generate_dataset(filepath, n, batch_size=10 if split =='train' else 64)
            generate_dataset(filepath, n, batch_size, drop_rate)


def load_dataset(fp, drop_rate) -> list[TSPInstance]:
    positions = np.load(fp)
    adjacency = np.load(fp.replace("positions", f"adjacency_{drop_rate}"))
    dataset = [TSPInstance(pos, adj) for pos, adj in zip(positions, adjacency)]
    return dataset


def calculate_cost(inst: TSPInstance, path: np.ndarray):
    return inst.distmat[path, np.roll(path, 1)].sum().item()


if __name__ == "__main__":
    import elkai  # for computing the optimal solutions to TSP

    np.random.seed(0)
    drop_rate = 0.0

    for split in dataset_conf.keys():

        generate_datasets(drop_rate=drop_rate)

        # Precompute the optimal solutions
        optimal_objs_dict = {}
        for problem_size in sizes:
            dataset_path = f"dataset/{split}{problem_size}_positions.npy"
            dataset = load_dataset(dataset_path, drop_rate)  # converts position to distance matrix
            n_instances = dataset[0].n
            print(f"[*] Evaluating {dataset_path} with LKH")

            optimal_objs = []
            n_high_cost_edges = []
            for i, instance in enumerate(tqdm(dataset, desc=f"tsp{problem_size}")):
                elkai_dist = elkai.DistanceMatrix(((instance.distmat * SCALE).astype(int)).tolist())
                optimal_route = elkai_dist.solve_tsp()  # e.g. [0, 2, 1, 0]; with proven optimal solutions up to N=315 (https://github.com/fikisipi/elkai)
                optimal_obj = calculate_cost(instance, np.array(optimal_route[:-1]))
                optimal_objs.append(optimal_obj)

                # number of high cost edges in the instance's adjacency matrix
                n_h = (instance.adjacency == 0.).sum() / 2
                n_high_cost_edges.append(n_h)

                for k in range(instance.distmat.shape[0]):
                    for m in range(k):
                        if instance.adjacency[k, m] == 0:
                            # check if k, m are consecutive in optimal_route (or vice versa)
                            if (optimal_route.index(k) == optimal_route.index(m) + 1) or (
                                    optimal_route.index(m) == optimal_route.index(k) + 1):
                                print(f"Found a dropped edge in optimal_route for instance {i}!")

            mean_optimal_obj = np.mean(optimal_objs)
            optimal_objs_dict[problem_size] = mean_optimal_obj
            # use 2 decimal places
            print(f"[*] Mean optimal objective for {problem_size}: {mean_optimal_obj:.2f}")
            print(f"[*] Optimal objectives: {optimal_objs}")
            print(f"[*] Number of high cost edges: {n_high_cost_edges}")

        print(optimal_objs_dict)

        # Save the optimal solutions in folder dataset, optimal_objs_dict is a dictionary with keys as problem sizes and values as optimal solutions
        sizes_str = '_'.join(map(str, sizes))
        name = f"optimal_objs_dict_{split}_drop_{drop_rate}_townsizes_{'_'.join(map(str, sizes))}_perturbations_{'_'.join(map(str, perturbation_moves_map.values()))}_iterlimits_{'_'.join(map(str, iter_limit_map.values()))}.pkl"
        with open(f"dataset/{name}", "wb") as f:
            pickle.dump(optimal_objs_dict, f)
