import re


def extract_function_name(code_body: str) -> str:
    """
    Extracts the function name from a code body string.
    If the function name can't be found, it defaults to a short placeholder.

    Args:
        code_body (str): Contains a whole function definition.

    Returns:
        str: The name of the function.
    """
    # Search for Python function definition: 'def name('
    match = re.search(r"def\s+(\w+)\s*\(", code_body)
    if match:
        return match.group(1)

    # Search for common Java/C# method signature: 'methodName(' after type/access modifiers
    match = re.search(
        r"(public|private|static|\s)\s+[\w<>,\[\]]+\s+(\w+)\s*\(", code_body
    )
    if match:
        # The second group captured is the function name
        return match.group(2)

    # If a name can't be found, return a placeholder
    return "unknown_function"
