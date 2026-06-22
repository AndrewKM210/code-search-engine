import re
from pathlib import Path

from cse.search_engine.engine import CodeSearchEngine

MAX_FILE_CHARS = 10_000  # cap how much of a file gets returned to the LLM
MAX_DIR_ENTRIES = 200  # cap how many directory entries get returned to the LLM
MAX_GREP_MATCHES = 100  # cap how many grep matches get returned to the LLM
MAX_GREP_LINE_CHARS = 200  # cap the length of each matched line shown
MAX_SEARCH_CHARS = (
    10_000  # cap how much search_code output gets returned to the LLM
)

SEARCH_CODE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_code",
        "description": (
            "Searches the indexed codebase for snippets relevant to a "
            "natural language query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of the code to find.",
                }
            },
            "required": ["query"],
        },
    },
}

READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Reads the full contents of a file in the repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the repository root.",
                }
            },
            "required": ["path"],
        },
    },
}

LIST_DIRECTORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_directory",
        "description": (
            "Lists the files and subdirectories of a directory in the repository."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to the repository root.",
                }
            },
            "required": [],
        },
    },
}

GREP_SCHEMA = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "Searches file contents in the repository for a regular expression."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search, relative to the repository root.",
                },
            },
            "required": ["pattern"],
        },
    },
}

# All tool schemas, in the OpenAI-style dict shape used by LLMClient.call_with_tools*
TOOL_SPECS = [
    SEARCH_CODE_SCHEMA,
    READ_FILE_SCHEMA,
    LIST_DIRECTORY_SCHEMA,
    GREP_SCHEMA,
]


def _resolve_within(path: str, base_dir: str) -> Path | str:
    """
    Resolves path under base_dir and verifies it stays inside and exists.

    Args:
        path (str): Path relative to base_dir.
        base_dir (str): Root directory the path is restricted to (prevents
            escaping via "../" or absolute paths).

    Returns:
        Path | str: The resolved Path on success, or a human-readable error
            message starting with "Error:".
    """
    base = Path(base_dir).resolve()
    target = (base / path).resolve()

    if not target.is_relative_to(base):
        return f"Error: '{path}' is outside the allowed directory."
    if not target.exists():
        return f"Error: '{path}' does not exist."

    return target


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
    target = _resolve_within(path, base_dir)
    if isinstance(target, str):
        return target
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
    target = _resolve_within(path, base_dir)
    if isinstance(target, str):
        return target
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


def grep(pattern: str, path: str = ".", base_dir: str = ".") -> str:
    """
    Searches file contents for a regex pattern, scoped to base_dir.

    Args:
        pattern (str): Regular expression to search for, line by line.
        path (str): File or directory to search, relative to base_dir. A
            directory is searched recursively.
        base_dir (str): Root directory the search is restricted to (prevents
            escaping the project via "../" or absolute paths).

    Returns:
        str: One match per line as "relative/path:lineno: line", a
            "No matches found." note, or a human-readable error message
            starting with "Error:".
    """
    target = _resolve_within(path, base_dir)
    if isinstance(target, str):
        return target

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid pattern '{pattern}': {e}."

    base = Path(base_dir).resolve()

    # Search a single file directly or every file under a directory
    files = (
        [target]
        if target.is_file()
        else sorted(p for p in target.rglob("*") if p.is_file())
    )

    matches: list[str] = []
    truncated = False
    for file in files:
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            # Skip binary or unreadable files
            continue

        rel = file.relative_to(base)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                snippet = line.strip()
                if len(snippet) > MAX_GREP_LINE_CHARS:
                    snippet = snippet[:MAX_GREP_LINE_CHARS] + "..."
                matches.append(f"{rel}:{lineno}: {snippet}")
                if len(matches) >= MAX_GREP_MATCHES:
                    truncated = True
                    break
        if truncated:
            break

    if not matches:
        return "No matches found."

    result = "\n".join(matches)
    if truncated:
        result += "\n... [truncated]"

    return result


def search_code(query: str, engine: CodeSearchEngine, k: int = 5) -> str:
    """
    Searches the indexed codebase for snippets relevant to a query.

    Args:
        query (str): Natural language description of the code to find.
        engine (CodeSearchEngine): Search engine bound to the indexed collection.
        k (int): Maximum number of results to return.

    Returns:
        str: Ranked snippets with their source file and similarity score,
            or a message if there are no results.
    """
    results = engine.search(query, k=k)
    if not results:
        return "No matching code found."

    blocks = []
    for i, res in enumerate(results, start=1):
        payload = res["payload"]
        content = payload.get("content") or payload.get("code_content", "")
        source = payload.get("source", "unknown")
        blocks.append(
            f"--- Result {i} (source: {source}, score: {res['score']:.3f}) ---\n"
            f"{content}"
        )

    listing = "\n\n".join(blocks)
    if len(listing) > MAX_SEARCH_CHARS:
        listing = listing[:MAX_SEARCH_CHARS] + "\n... [truncated]"

    return listing
