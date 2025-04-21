# Adding a New Task to the Codebase

Adding a new task requires modifying key components to ensure seamless integration into the pipeline. The process follows these essential steps:

1. **Create a Task Evaluation File**: Implement task-specific functions (`generate_input`, `evaluate_func`, `get_initial_func`).
2. **Create a Config File**: Create a new configuration file which will be loaded when executing the code.
3. **Modify `main.py`**: Register the new task, ensure configuration compatibility, and set up task-specific logic.
4. **Modify `prompt.py`**: Add the prompt template for the corresponding task.

---

## Step 1: Implementing Task-Specific Evaluation

Each task requires an evaluation file in `packing/evaluate` that implements:

1. **Defining an Initial Heuristic Function** (`get_initial_func`)  
2. **Generating Input Data** (`generate_input`)  
4. **Evaluating and Scoring a Function** (`evaluate_func`)  

The final evaluation **returns a score that is maximized**, meaning better solutions receive higher scores.

### 1.1 File Structure

Create a new file: `packing/evaluate/new_task/task_newtask.py`

It should contain:

+ **`get_initial_func(cfg) -> tuple[Callable, str]`**  
  - Defines an initial function for scheduling, selection, or optimization.  
  - Returns the function and its name as a string.  

+ **`generate_input(cfg, set: str) -> str`**  
  - Loads the appropriate dataset based on the task split (`train`, `test`, etc.).  
  - Returns a dataset path or structured input data.

+ **`evaluate_func(cfg, dataset_name, function_class) -> FunctionClass`**  
  - Executes the function across multiple test cases, computes an aggregated score, and ensures compatibility with the existing framework.  
  - **Higher scores indicate better performance.**  

### 1.2 Key Implementation Notes

- **Score transformation**: If the metric is a minimization objective (e.g., makespan, cost), it should be negated and rescaled to ensure **higher is better**.  
- **Error handling**: If execution fails, assign a **default failure score** (`cfg.failed_score`) and store the exception.  

---

## Step 2: Defining Task-Specific Configurations

To add a new task:
1. **Create a YAML file** inside `configs/configs_base/` (e.g., `configs/configs_base/config_newtask.yaml`).
2. **Define the task-specific settings** by following the structure of existing tasks.

For example, to add a `newtask` task:

```yaml
# Task settings
task_newtask: 1
train_set_path_newtask: "data/newtask/train_instances.json"
train_perturbed_set_path_newtask: ""
test_set_path_newtask: ""
init_some_heuristic: 1  # Task-specific initialization flag
# Disable other tasks
task_bin: 0
task_code: 0
task_tsp: 0
task_flatpack: 0
task_jssp: 0
```

Make sure **only one `task_` flag is set to `1`** to ensure mutual exclusivity. 

Also ensure that `train_set_path_newtask` is correctly set in the configuration, as `generate_input` retrieves the dataset path from it.

### 2.1 Copy Task Parameters to Other Configurations

Each base configuration (e.g., `configs/configs_base/config_task_bin.yaml`, `configs/configs_base/config_task_tsp.yaml`) includes **all possible task parameters**, even if they are not active in that file.  

To maintain consistency, **copy the new task’s parameters** into every other config file and set its flag to `0`.  

For example, in `configs/configs_base/config_task_bin.yaml`:

```yaml
# Task bin
task_bin: 1
Weibull: 0
OR: 1
init_best_fit: 1

# Task newtask
task_newtask: 0
train_set_path_newtask: ""
train_perturbed_set_path_newtask: ""
test_set_path_newtask: ""
init_some_heuristic: 0
+ ```

This ensures that **all configurations remain compatible** and that switching tasks only requires setting a **single flag to `1`**.

---

## 3. Modifying `main.py`

To register a new task in `main.py`, follow these steps:

### 3.1 Define a Task Flag in the Configuration
Each task is controlled via boolean flags in the configuration (`cfg`). Update the task selection assertion:

```python
assert cfg.task_bin or cfg.task_code or cfg.task_tsp or cfg.task_flatpack or cfg.task_jssp or cfg.task_newtask, "One task must be selected"
assert cfg.task_bin + cfg.task_code + cfg.task_tsp + cfg.task_flatpack + cfg.task_jssp + cfg.task_newtask == 1, "Only one task can be selected"
if cfg.task_newtask:
    cfg.task_name = "newtask"
```

### 3.2 Add Task-Specific Imports
Each task needs specific evaluation functions. Modify the task import section:

```python
if cfg.task_code:
    from packing.evaluate.task_code import generate_input, evaluate_func, get_initial_func
elif cfg.task_bin:
    from packing.evaluate.bin_packing.task_bin import generate_input, evaluate_func, get_initial_func
elif cfg.task_tsp:
    from packing.evaluate.tsp.task_tsp import generate_input, evaluate_func, get_initial_func
elif cfg.task_flatpack:
    from packing.evaluate.flat_pack.task_flat_pack import generate_input, evaluate_func, get_initial_func
elif cfg.task_jssp:
    from packing.evaluate.job_shop_scheduling.task_job_shop_scheduling import generate_input, evaluate_func, get_initial_func
elif cfg.task_newtask:
    from packing.evaluate.new_task.task_newtask import generate_input, evaluate_func, get_initial_func
```

## 4. Updating `prompt.py` to Support the New Task

The prompt generation process for different tasks is handled in `src/packing/model/prompt.py`. When adding a new task, it’s essential to ensure that prompts are correctly generated for it.

`prompt.py` ensures that the model receives properly formatted input for each task. Modify generate_batch_prompts to define how the prompt is structured for your new task.

```python
if cfg.task_bin:
    prompt_template = "Solve this bin packing problem: {problem_data}"
elif cfg.task_tsp:
    prompt_template = "Find the optimal route for this traveling salesman problem: {problem_data}"
elif cfg.task_flatpack:
    prompt_template = "Determine the best way to assemble this flatpack structure: {problem_data}"
elif cfg.task_jssp:
    prompt_template = "Schedule the jobs efficiently to minimize makespan: {problem_data}"
elif cfg.task_newtask:
    prompt_template = "Solve this new task efficiently: {problem_data}"
```