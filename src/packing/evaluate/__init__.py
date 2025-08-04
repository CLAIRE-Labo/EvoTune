import importlib
import pathlib


def import_all_tasks():
    base_path = pathlib.Path(__file__).parent
    for path in base_path.rglob("task_*.py"):
        # Convert to dotted module path
        rel_path = path.relative_to(base_path.parent).with_suffix("")
        module = ".".join(['packing'] + list(rel_path.parts))
        importlib.import_module(module)
