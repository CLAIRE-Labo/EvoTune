import jax
import jax.numpy as jnp
import numpy as np
import json
from jumanji.environments.packing.job_shop.generator import Generator
from jumanji.environments.packing.job_shop.types import State
import chex


class PreloadedGenerator(Generator):
    """
    A custom generator that loads pre-generated Job Shop Scheduling (JSSP) instances.

    Each instance includes:
    - `ops_machine_ids`: Specifies the machine for each operation.
    - `ops_durations`: Specifies the duration of each operation.
    - `optimal_makespan`: The precomputed optimal makespan.

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

        self.num_instances = len(self.instances)
        self.instances_lower_bounds = [instance["lower_bound_makespan"] for instance in self.instances]
        self.cur_idx = 0

        super().__init__(
            num_jobs=self.instances[self.cur_idx]["num_jobs"],
            num_machines=self.instances[self.cur_idx]["num_machines"],
            max_num_ops=self.instances[self.cur_idx]["max_num_ops"],
            max_op_duration=self.instances[self.cur_idx]["max_op_duration"],
        )

        self.num_jobs = self.instances[self.cur_idx]["num_jobs"]
        self.num_machines = self.instances[self.cur_idx]["num_machines"]
        self.max_num_ops = self.instances[self.cur_idx]["max_num_ops"]
        self.max_op_duration = self.instances[self.cur_idx]["max_op_duration"]

    def get_lower_bound_for_instance_idx(self, instance_idx: int) -> float:
        return self.instances_lower_bounds[instance_idx]

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
        self.num_jobs = self.instances[self.cur_idx]["num_jobs"]
        self.num_machines = self.instances[self.cur_idx]["num_machines"]
        self.max_num_ops = self.instances[self.cur_idx]["max_num_ops"]
        self.max_op_duration = self.instances[self.cur_idx]["max_op_duration"]

        state = State(
            ops_machine_ids=jnp.array(instance["state"]["ops_machine_ids"], jnp.int32),
            ops_durations=jnp.array(instance["state"]["ops_durations"], jnp.int32),
            ops_mask=jnp.array(instance["state"]["ops_mask"], bool),
            machines_job_ids=jnp.array(instance["state"]["machines_job_ids"], jnp.int32),
            machines_remaining_times=jnp.array(instance["state"]["machines_remaining_times"], jnp.int32),
            action_mask=None,  # Will be computed dynamically by the environment
            step_count=jnp.array(instance["state"]["step_count"], jnp.int32),
            scheduled_times=jnp.array(instance["state"]["scheduled_times"], jnp.int32),
            key=jnp.array(instance["state"]["key"], jnp.uint32),
        )

        return state
