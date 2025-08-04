import json

import chex
import jax.numpy as jnp
from jumanji.environments.packing.flat_pack.generator import InstanceGenerator
from jumanji.environments.packing.flat_pack.types import State


class PreloadedGenerator(InstanceGenerator):
    """
    A custom generator that loads pre-generated FlatPack instances.

    Each instance includes:
    - `grid_shape`: Shape of the packing grid.
    - `blocks`: The set of blocks to be placed.
    - `optimal_score`: The best-known score for this instance.

    This allows experiments on fixed instances without randomness.
    """

    def __init__(self, instance_file: str) -> None:
        """
        Initialize the generator by loading instances from a file.

        Args:
            instance_file: Path to a JSON file containing pre-generated instances.
        """
        with open(instance_file, "r") as f:
            self.instances = json.load(f)["instances"]

        # from pympler.asizeof import asizeof
        # print(asizeof(self.instances))

        self.num_instances = len(self.instances)
        self.cur_idx = 0

        super().__init__(
            num_row_blocks=self.instances[self.cur_idx]["num_row_blocks"],
            num_col_blocks=self.instances[self.cur_idx]["num_col_blocks"]
        )

        self.num_row_blocks = self.instances[self.cur_idx]["num_row_blocks"]  # We overshadow these
        self.num_col_blocks = self.instances[self.cur_idx]["num_col_blocks"]

    def __call__(self, key: chex.PRNGKey) -> State:
        """
        Generates an environment state from the preloaded dataset.

        Args:
            key: PRNGKey.

        Returns:
            A `State` object representing the environment.
        """
        del key

        # Select instance in a cyclic manner
        instance = self.instances[self.cur_idx]

        self.cur_idx = (self.cur_idx + 1) % self.num_instances
        self.num_row_blocks = self.instances[self.cur_idx]["num_row_blocks"]
        self.num_col_blocks = self.instances[self.cur_idx]["num_col_blocks"]

        state = State(
            blocks=jnp.array(instance["state"]["blocks"], jnp.int32),
            num_blocks=instance["state"]["num_blocks"],
            action_mask=jnp.array(instance["state"]["action_mask"], bool),
            grid=jnp.array(instance["state"]["grid"], jnp.int32),
            step_count=instance["state"]["step_count"],
            key=jnp.array(instance["state"]["key"], jnp.uint32),
            placed_blocks=jnp.array(instance["state"]["placed_blocks"], bool),
        )

        return state
