from packing.evaluate import import_all_tasks


class TaskRegistry:
    def __init__(self):
        self.is_initialized = False
        self._registry = {}

    def register(self, name, generate_input, evaluate_func, get_initial_func, system_prompt, append_prompt):
        self._registry[name] = {
            "generate_input": generate_input,
            "evaluate_func": evaluate_func,
            "get_initial_func": get_initial_func,
            "system_prompt": system_prompt,
            "append_prompt": append_prompt,
        }

    def get(self, name):
        if name not in self._registry:
            raise ValueError(f"Task '{name}' is not registered")
        return self._registry[name]


# Global registry instance
TASK_REGISTRY = TaskRegistry()

if not TASK_REGISTRY.is_initialized:
    import_all_tasks()
    TASK_REGISTRY.is_initialized = True
