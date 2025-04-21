import itertools

import jax
import numpy as np
from jumanji.environments.packing.job_shop.generator import RandomGenerator


# -----------------------------
# Generate a JSSP instance using RandomGenerator
# -----------------------------
def generate_instance(num_jobs: int, num_machines: int, max_num_ops: int, max_op_duration: int, seed: int = 42):
    # Create the RandomGenerator instance and generate a problem instance (state)
    generator = RandomGenerator(
        num_jobs=num_jobs,
        num_machines=num_machines,
        max_num_ops=max_num_ops,
        max_op_duration=max_op_duration,
    )
    key = jax.random.PRNGKey(seed)
    state = generator(key)
    # Convert jax arrays to numpy arrays for convenience
    instance = {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "max_num_ops": max_num_ops,
        "max_op_duration": max_op_duration,
        "state": {key: value if not isinstance(value, jax.Array) else value.tolist()
                  for key, value in vars(state).items()}
    }
    return instance


# -----------------------------
# Build the Gurobi model for JSSP
# -----------------------------
def solve_jssp_with_gurobi(instance) -> float:
    num_jobs = instance["num_jobs"]
    num_machines = instance["num_machines"]
    max_num_ops = instance["max_num_ops"]
    ops_machine_ids = instance["state"]["ops_machine_ids"]
    ops_durations = instance["state"]["ops_durations"]
    ops_mask = instance["state"]["ops_mask"]

    # For each job, collect the indices of its valid (non-padded) operations.
    valid_ops = {}
    for i in range(num_jobs):
        valid_ops[i] = [j for j in range(max_num_ops) if ops_mask[i, j]]

    # Compute a big M: sum over all processing times (a safe upper bound)
    M = np.sum([ops_durations[i, j] for i in range(num_jobs) for j in valid_ops[i]])

    model = Model("JobShop")
    model.Params.OutputFlag = 0  # turn off output for clarity

    # Create start time variables for each valid operation
    x = {}
    for i in range(num_jobs):
        for j in valid_ops[i]:
            x[(i, j)] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")

    # Create variable for makespan
    C_max = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="C_max")

    # -----------------------------
    # Precedence constraints: within each job, operations must follow the given order.
    # -----------------------------
    for i in range(num_jobs):
        ops_list = valid_ops[i]
        for k in range(len(ops_list) - 1):
            j_curr = ops_list[k]
            j_next = ops_list[k + 1]
            processing_time = ops_durations[i, j_curr]
            model.addConstr(x[(i, j_next)] >= x[(i, j_curr)] + processing_time,
                            name=f"prec_{i}_{j_curr}_{j_next}")

    # -----------------------------
    # Machine disjunctive constraints: operations on the same machine cannot overlap.
    # -----------------------------
    # For each machine, list all operations that require it.
    machine_ops = {m: [] for m in range(num_machines)}
    for i in range(num_jobs):
        for j in valid_ops[i]:
            m_required = ops_machine_ids[i, j]
            machine_ops[m_required].append((i, j))

    # For each pair of operations on the same machine, add disjunctive constraints.
    # We introduce binary variable z[(i,j),(i',j')] which is 1 if op (i,j) precedes op (i',j').
    z = {}
    for m in range(num_machines):
        ops_on_m = machine_ops[m]
        # For every distinct pair
        for idx in range(len(ops_on_m)):
            i, j = ops_on_m[idx]
            for idx2 in range(idx + 1, len(ops_on_m)):
                i2, j2 = ops_on_m[idx2]
                z[(i, j, i2, j2)] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}_{i2}_{j2}")
                # If (i,j) comes before (i2,j2):
                model.addConstr(x[(i, j)] + ops_durations[i, j] <= x[(i2, j2)] + M * (1 - z[(i, j, i2, j2)]),
                                name=f"mach_{m}_{i}_{j}_before_{i2}_{j2}")
                # If (i2,j2) comes before (i,j):
                model.addConstr(x[(i2, j2)] + ops_durations[i2, j2] <= x[(i, j)] + M * z[(i, j, i2, j2)],
                                name=f"mach_{m}_{i2}_{j2}_before_{i}_{j}")

    # -----------------------------
    # Makespan constraints: C_max must be at least the completion time of every operation.
    # -----------------------------
    for i in range(num_jobs):
        for j in valid_ops[i]:
            model.addConstr(C_max >= x[(i, j)] + ops_durations[i, j],
                            name=f"makespan_{i}_{j}")

    # -----------------------------
    # Objective: minimize the makespan
    # -----------------------------
    model.setObjective(C_max, GRB.MINIMIZE)

    # Optimize the model
    model.optimize()

    if model.status == GRB.OPTIMAL:
        optimal_makespan = model.ObjVal
    else:
        optimal_makespan = None

    return optimal_makespan


def calculate_naive_lower_bound(instance) -> float:
    ops_machine_ids = np.array(instance["state"]["ops_machine_ids"])
    ops_durations = np.array(instance["state"]["ops_durations"])

    # Mask to exclude -1 values in ops_machine_ids
    valid_mask = ops_machine_ids != -1

    # Create an array to store the summed durations for each machine
    max_machine_id = ops_machine_ids[valid_mask].max()  # Get the max machine ID
    machine_sums = np.zeros(max_machine_id + 1)  # Create an array to hold sums

    # Sum durations per machine
    np.add.at(machine_sums, ops_machine_ids[valid_mask], ops_durations[valid_mask])

    # Return the maximum over all machine sums, as it is a lower bound on our makespan
    return machine_sums.max()


# -----------------------------
# Main script
# -----------------------------
import json
import os

WRITE_PATH = 'data/job_shop_scheduling'

USE_GUROBI = False
if USE_GUROBI:
    from gurobipy import GRB
    from gurobipy import Model

    print('Gurobi loaded!')

if __name__ == "__main__":
    # Define a JSSP configuration
    nums_instances = (20, 20, 20, 20, 20)
    nums_jobs = (20, 30, 20, 30, 15)
    nums_machines = (10, 15, 10, 15, 5)
    max_nums_ops = (8, 16, 8, 12, 7)
    max_op_durations = (5, 7, 9, 7, 6)

    instance_list = []

    i = 0
    for num_instances, num_jobs, num_machines, max_num_ops, max_op_duration in zip(nums_instances, nums_jobs,
                                                                                   nums_machines, max_nums_ops,
                                                                                   max_op_durations):
        print(
            f'Configuration: {num_jobs} jobs, {num_machines} machines, max {max_num_ops} ops, max {max_op_duration} dur')
        for j in range(num_instances):
            seed = 42 + i  # Ensure unique seeds for different instances
            instance = generate_instance(
                num_jobs=num_jobs,
                num_machines=num_machines,
                max_num_ops=max_num_ops,
                max_op_duration=max_op_duration,
                seed=seed
            )

            # Solve the instance using Gurobi to get the optimal makespan
            lower_bound_makespan = None
            if USE_GUROBI:
                optimal_makespan = solve_jssp_with_gurobi(instance)
                lower_bound_makespan = optimal_makespan
            else:
                print('\tUsing alternative way to compute a lower bound')
                lower_bound_makespan = calculate_naive_lower_bound(instance)
            if USE_GUROBI and lower_bound_makespan is None:
                print(f"\tWarning: Instance {i} did not solve to optimality. Skipping...")
                continue  # Skip instances that could not be solved optimally

            # Store the instance with makespan
            instance_data = {
                "seed": seed,
                "lower_bound_makespan": lower_bound_makespan,
                **instance,
            }
            instance_list.append(instance_data)

            print(f"Instance {j}/{num_instances} generated with lower bound: {lower_bound_makespan}")

            i += 1

    # Save all instances to a JSON file
    output_file_name = f"jssp_dynamic_instances.json"
    output_path = os.path.join(WRITE_PATH, output_file_name)

    # Group instances by grid size in the final dataset
    final_dataset = {"num_instances": nums_instances,
                     "nums_jobs": nums_jobs,
                     "nums_machines": nums_machines,
                     "max_nums_ops": max_nums_ops,
                     "max_op_durations": max_op_durations,
                     "instances": instance_list}

    with open(output_path, "w") as f:
        json.dump(final_dataset, f, indent=4)

    print(f"Successfully saved {len(instance_list)} instances to {output_path}")
