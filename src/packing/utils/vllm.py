from types import SimpleNamespace


def dict_to_namespace(data):
    """Recursively convert a dict (and any nested dicts/lists) into SimpleNamespace objects."""
    if isinstance(data, dict):
        ns = SimpleNamespace()
        for key, value in data.items():
            setattr(ns, key, dict_to_namespace(value))
        return ns
    elif isinstance(data, list):
        return [dict_to_namespace(item) for item in data]
    else:
        return data