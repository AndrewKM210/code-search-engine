from pathlib import Path

MAX_FILE_CHARS = 10_000  # cap how much of a file gets returned to the LLM
MAX_DIR_ENTRIES = 200  # cap how many directory entries get returned to the LLM


def read_file(path: str, base_dir: str = ".") -> str:
    """
    Reads a text file's contents, scoped to base_dir.

    Args:
        path (str): File path to read, relative to base_dir.
        base_dir (str): Root directory the read is restricted to (prevents
            escaping the project via "../" or absolute paths).

    Returns:
        str: The file's contents (truncated if very large), or a
            human-readable error message starting with "Error:".
    """
    base = Path(base_dir).resolve()
    target = (base / path).resolve()

    if not target.is_relative_to(base):
        return f"Error: '{path}' is outside the allowed directory."
    if not target.exists():
        return f"Error: '{path}' does not exist."
    if not target.is_file():
        return f"Error: '{path}' is not a file."

    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: '{path}' is not a text file."

    if len(content) > MAX_FILE_CHARS:
        content = content[:MAX_FILE_CHARS] + "\n... [truncated]"

    return content


def list_directory(path: str = ".", base_dir: str = ".") -> str:
    """
    Lists the entries of a directory, scoped to base_dir.

    Args:
        path (str): Directory path to list, relative to base_dir.
        base_dir (str): Root directory the listing is restricted to (prevents
            escaping the project via "../" or absolute paths).

    Returns:
        str: One entry per line with directories suffixed by "/" and listed
            first, an "(empty directory)" note, or a human-readable error
            message starting with "Error:".
    """
    base = Path(base_dir).resolve()
    target = (base / path).resolve()

    if not target.is_relative_to(base):
        return f"Error: '{path}' is outside the allowed directory."
    if not target.exists():
        return f"Error: '{path}' does not exist."
    if not target.is_dir():
        return f"Error: '{path}' is not a directory."

    # Sort directories first, then files, both alphabetically
    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    if not entries:
        return "(empty directory)"

    lines = [f"{e.name}/" if e.is_dir() else e.name for e in entries]

    truncated = len(lines) > MAX_DIR_ENTRIES
    if truncated:
        lines = lines[:MAX_DIR_ENTRIES]

    listing = "\n".join(lines)
    if truncated:
        listing += "\n... [truncated]"

    return listing
