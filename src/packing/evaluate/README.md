# Adding a New Task to the Codebase

Adding a new task requires implementing a small task module and registering it using the `task_registry`. The system is modular and automatically loads all tasks on import. The process follows these steps:

1. **Create a Task Evaluation File**: Implement task-specific functions and register them with the task registry.
2. **Create a Config File**: Define task-specific config parameters in a YAML file.
3. **You're done!** The registry handles dynamic loading automatically.

---

## Step 1: Implementing Task-Specific Evaluation

Each task requires an evaluation file in `packing/evaluate` that implements:

1. **Defining an Initial Heuristic Function** (`get_initial_func`)  
2. **Generating Input Data** (`generate_input`)  
3. **Evaluating and Scoring a Function** (`evaluate_func`)  

The final evaluation **returns a score that is maximized**, meaning better solutions receive higher scores.

### 1.1 File Structure

Create a new file: `packing/evaluate/new_task/task_newtask.py`

It should contain:

```python
from packing.registry import task_registry

def get_initial_func(cfg) -> tuple[Callable, str]:
    ...

def generate_input(cfg, set: str) -> Any:
    ...

def evaluate_func(cfg, dataset, function_class) -> FunctionClass:
    ...

task_registry.register(
    "newtask",
    generate_input=generate_input,
    evaluate_func=evaluate_func,
    get_initial_func=get_initial_func,
    system_prompt="Solve the following new task.",
    append_prompt="Return the optimal solution.",
)
```

You can look at other tasks for more inspiration.

## Step 2: Defining Task-Specific Configurations

To add a new task:
1. Create a YAML file under ```configs/task/``` (e.g., ```newtask.yaml```)
2. Define the task-specific settings using your own naming convention.

Example ```config_newtask.yaml```:

```yaml
# Task flatpack
task_name: newtask
function_str_to_extract: ...
...

failed_score: ...
timeout_period: ...
mem_limit_gb: ...

programdatabaseConfig:
  temp: ...
```

---
