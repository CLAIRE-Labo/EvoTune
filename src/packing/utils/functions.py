import inspect
import textwrap


def function_to_string(func):
    """
    Takes a function and returns its source code as a string.

    args:
    func (function): The function whose source code is to be extracted.

    Returns:
    str: The source code of the function.
    """
    try:
        source = inspect.getsource(func)
        dedented_source = textwrap.dedent(source)
        return dedented_source
    except TypeError:
        return "The provided input is not a function."
    except OSError:
        return "The source code for the function could not be retrieved."


def string_to_function(cfg, func_str, imports):
    """
    Convert a string representation of a Python function into an actual Python function.

    Parameters:
    func_str (str): The string representation of the Python function.
    func_name (str): The name of the function to retrieve after executing the string.

    Returns:
    function: The actual Python function.
    """
    # Define a local dictionary to execute the function string
    local_dict = {}

    # Here I want to execute the imports
    exec(imports, globals(), local_dict)

    # Execute the function string
    exec(func_str, globals(), local_dict)

    # Retrieve the function from the local dictionary
    func = local_dict.get(cfg.function_str_to_extract)

    # Ensure the function exists in the local dictionary
    if func is None:
        raise ValueError(f"Function '{cfg.function_str_to_extract}' not found in the provided string.")

    return func


def extract_imports(code_output):
    """
    Extracts the import statements from the given code output.

    Parameters:
    code_output (str): The output string containing the code.

    Returns:
    str: The extracted import statements, or an empty string if no imports are found.
    """
    lines = code_output.split("\n")
    import_lines = []

    for line in lines:
        stripped_line = line.strip()

        # Check if the line is an import statement
        if stripped_line.startswith("import ") or stripped_line.startswith("from "):
            import_lines.append(line)

    if import_lines:
        return "\n".join(import_lines)
    else:
        return ""


def extract_functions(code_output):
    """
    Extracts the outer function definition from the given code output,
    including any nested functions within it.

    Parameters:
    code_output (str): The output string containing the code.

    Returns:
    str: The extracted outer function definition, or an empty string if no function is found.
    """
    lines = code_output.split("\n")
    in_function = False
    nested_level = 0
    function_lines = []

    for line in lines:
        stripped_line = line.strip()

        # Check if the line is the start of a function definition
        if stripped_line.startswith("def "):
            if not in_function:
                in_function = True
            nested_level += 1

        # If inside a function, add the line to the function_lines list
        if in_function:
            function_lines.append(line)

        # Check for the return statement to end the current function scope
        if in_function and stripped_line.startswith("return "):
            nested_level -= 1
            if nested_level == 0:
                break

    if function_lines:
        return "\n".join(function_lines)
    else:
        return ""


def separate_imports_from_func(func_str):
    lines = func_str.splitlines()
    import_lines = []
    func_lines = []
    found_def = False
    for line in lines:
        if line.strip().startswith("def "):
            found_def = True
            func_lines.append(line)
        elif not found_def and (line.strip().startswith("import ") or line.strip().startswith("from ")):
            import_lines.append(line)
        else:
            if found_def:
                func_lines.append(line)
    return "\n".join(import_lines), "\n".join(func_lines)
