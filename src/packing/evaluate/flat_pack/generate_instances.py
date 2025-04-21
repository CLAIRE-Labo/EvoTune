import json
import os

import jax
from jumanji.environments.packing.flat_pack.generator import RandomFlatPackGenerator


# -----------------------------
# Generate a FlatPack instance using RandomFlatPackGenerator
# -----------------------------
def generate_instance(num_row_blocks: int, num_col_blocks: int, seed: int = 42):
    """
    Generates a random FlatPack instance.

    Args:
        num_row_blocks: Number of rows in the grid.
        num_col_blocks: Number of columns in the grid.
        num_blocks: Number of blocks to place.
        seed: Random seed.

    Returns:
        A dictionary representing the FlatPack instance.
    """
    generator = RandomFlatPackGenerator(
        num_row_blocks=num_row_blocks,
        num_col_blocks=num_col_blocks,
    )
    key = jax.random.PRNGKey(seed)
    state = generator(key)

    # Convert JAX arrays to numpy lists for JSON serialization
    instance = {
        "num_row_blocks": num_row_blocks,
        "num_col_blocks": num_col_blocks,
        "state": {key: value if not isinstance(value, jax.Array) else value.tolist()
                  for key, value in vars(state).items()},
        "upper_bound_optimal_score": num_row_blocks * num_col_blocks  # Upper bound
    }
    return instance


# -----------------------------
# Main script
# -----------------------------
WRITE_PATH = "data/flat_pack"

if __name__ == "__main__":
    # Define a FlatPack configuration
    initial_seed = 100

    # THESE ARE NOT THE GRID SIZES (weird I know)
    # num_blocks = num_row_blocks * num_col_blocks
    num_blocks_sizes = [(4, 4), (5, 5), (7, 7)]  # Example sizes
    all_num_instances = (15, 20, 10)

    instance_list = []
    grid_sizes = []

    for num_instances, (num_row_blocks, num_col_blocks) in zip(all_num_instances, num_blocks_sizes):
        total_grid_rows = 3 * num_row_blocks - (num_row_blocks - 1)
        total_grid_cols = 3 * num_col_blocks - (num_col_blocks - 1)
        print(f'Generating instances for grid size {total_grid_rows}x{total_grid_cols}')
        grid_sizes.append((total_grid_rows, total_grid_cols))

        for i in range(num_instances):
            instance = generate_instance(
                num_row_blocks=num_row_blocks,
                num_col_blocks=num_col_blocks,
                seed=initial_seed + i
            )
            instance_list.append(instance)
            print(f"Instance {i + 1}/{num_instances} for {total_grid_rows}x{total_grid_cols} generated")

    # Save all instances in one JSON file
    output_file_name = f"flatpack_dynamic_{initial_seed}_seed.json"
    output_path = os.path.join(WRITE_PATH, output_file_name)

    # Group instances by grid size in the final dataset
    final_dataset = {"num_instances": all_num_instances, "grid_sizes": grid_sizes, "instances": instance_list}

    with open(output_path, "w") as f:
        json.dump(final_dataset, f, indent=4)

    print(f"Successfully saved dataset with multiple grid sizes to {output_path}")
