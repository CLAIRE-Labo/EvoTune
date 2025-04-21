import json
import os
import copy
import numpy as np


def perturb_place_block_in_middle(instance, block_in_middle_size_l, block_in_middle_size_r):
    def _add_block_in_middle(matrix, a, b, block_value):
        n, m = len(matrix), len(matrix[0])

        # Compute the top-left corner ensuring bounds are respected
        start_i = max(0, (n - a) // 2)
        start_j = max(0, (m - b) // 2)

        # Ensure the block fits within bounds
        end_i = min(n, start_i + a)
        end_j = min(m, start_j + b)

        # Apply the block
        for i in range(start_i, end_i):
            for j in range(start_j, end_j):
                matrix[i][j] = block_value

        return matrix

    perturbed_instance = copy.deepcopy(instance)
    n, m = len(perturbed_instance['state']['grid']), len(perturbed_instance['state']['grid'][0])

    perturbed_instance['state']['grid'] = _add_block_in_middle(
        perturbed_instance['state']['grid'],
        block_in_middle_size_l, block_in_middle_size_r,
        block_value=len(perturbed_instance['state']['blocks']) + 42  # For fun upper bound
    )

    # Check the block placement validity (update action mask)
    for block_idx in range(len(perturbed_instance['state']['action_mask'])):
        for rot_idx in range(len(perturbed_instance['state']['action_mask'][block_idx])):
            block_with_rot = np.rot90(perturbed_instance['state']['blocks'][block_idx], k=4 - rot_idx)

            # Ensure the sliding window size matches the block dimensions
            block_h, block_w = block_with_rot.shape
            sub_grids = np.lib.stride_tricks.sliding_window_view(
                perturbed_instance['state']['grid'], (block_h, block_w)
            )

            # Ensure no overlap
            valid_action = np.all((sub_grids == 0) | (block_with_rot == 0), axis=(-2, -1))

            perturbed_instance['state']['action_mask'][block_idx][rot_idx] = valid_action.tolist()

    return perturbed_instance


DATA_DIR = os.path.join("data", "flat_pack")
if __name__ == '__main__':
    dataset_to_perturb_path = os.path.join(DATA_DIR, "train_flatpack_dynamic_0_seed.json")

    with open(dataset_to_perturb_path) as f:
        dataset_to_perturb = json.load(f)

    perturbed_dataset = copy.deepcopy(dataset_to_perturb)
    perturbed_dataset['instances'] = []
    perturbed_dataset['perturbation_block_sizes'] = []

    idx_ctr = 0
    for grid_size, num_instances in zip(dataset_to_perturb['grid_sizes'], dataset_to_perturb['num_instances']):
        block_in_middle_size_l = int(round(grid_size[0] ** 0.5))
        block_in_middle_size_r = int(round(grid_size[1] ** 0.5))
        perturbed_dataset['perturbation_block_sizes'].append([block_in_middle_size_l, block_in_middle_size_r])

        for _ in range(num_instances):
            perturbed_instance = perturb_place_block_in_middle(
                dataset_to_perturb['instances'][idx_ctr],
                block_in_middle_size_l,
                block_in_middle_size_r,
            )
            perturbed_dataset['instances'].append(perturbed_instance)

            idx_ctr += 1

    # Save the perturbed dataset to a new file
    output_file_path = os.path.join(DATA_DIR, "perturbed_train_flatpack.json")
    with open(output_file_path, "w") as f:
        json.dump(perturbed_dataset, f, indent=4)

    print(f"Perturbed dataset saved to {output_file_path}")