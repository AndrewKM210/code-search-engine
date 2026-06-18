from unittest.mock import MagicMock

from cse.agent.tools import (
    MAX_DIR_ENTRIES,
    MAX_FILE_CHARS,
    MAX_GREP_MATCHES,
    MAX_SEARCH_CHARS,
    grep,
    list_directory,
    read_file,
    search_code,
)


def test_read_file_returns_contents(tmp_path):
    (tmp_path / "hello.py").write_text("print('hi')")

    result = read_file("hello.py", base_dir=str(tmp_path))

    assert result == "print('hi')"


def test_read_file_reads_nested_path(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x = 1")

    result = read_file("pkg/mod.py", base_dir=str(tmp_path))

    assert result == "x = 1"


def test_read_file_missing_file_returns_error(tmp_path):
    result = read_file("missing.py", base_dir=str(tmp_path))

    assert result.startswith("Error:")


def test_read_file_rejects_path_outside_base_dir(tmp_path):
    (tmp_path / "project").mkdir()
    outside_secret = tmp_path / "secret.txt"
    outside_secret.write_text("top secret")

    result = read_file("../secret.txt", base_dir=str(tmp_path / "project"))

    assert result.startswith("Error:")
    assert "top secret" not in result


def test_read_file_rejects_directory(tmp_path):
    (tmp_path / "subdir").mkdir()

    result = read_file("subdir", base_dir=str(tmp_path))

    assert result.startswith("Error:")


def test_read_file_truncates_large_files(tmp_path):
    (tmp_path / "big.txt").write_text("a" * (MAX_FILE_CHARS + 500))

    result = read_file("big.txt", base_dir=str(tmp_path))

    assert len(result) < MAX_FILE_CHARS + 500
    assert result.endswith("[truncated]")


def test_list_directory_lists_dirs_first_then_files(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "mod.py").write_text("x = 1")
    (tmp_path / "README.md").write_text("# hi")

    result = list_directory(".", base_dir=str(tmp_path))

    assert result.splitlines() == ["pkg/", "README.md", "mod.py"]


def test_list_directory_lists_nested_path(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x = 1")

    result = list_directory("pkg", base_dir=str(tmp_path))

    assert result == "mod.py"


def test_list_directory_empty_directory(tmp_path):
    (tmp_path / "empty").mkdir()

    result = list_directory("empty", base_dir=str(tmp_path))

    assert result == "(empty directory)"


def test_list_directory_missing_path_returns_error(tmp_path):
    result = list_directory("missing", base_dir=str(tmp_path))

    assert result.startswith("Error:")


def test_list_directory_rejects_path_outside_base_dir(tmp_path):
    (tmp_path / "project").mkdir()
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "passwords.txt").write_text("hunter2")

    result = list_directory("../secret", base_dir=str(tmp_path / "project"))

    assert result.startswith("Error:")
    assert "passwords.txt" not in result


def test_list_directory_rejects_file(tmp_path):
    (tmp_path / "mod.py").write_text("x = 1")

    result = list_directory("mod.py", base_dir=str(tmp_path))

    assert result.startswith("Error:")


def test_list_directory_truncates_many_entries(tmp_path):
    for i in range(MAX_DIR_ENTRIES + 10):
        (tmp_path / f"file_{i:04d}.txt").write_text("x")

    result = list_directory(".", base_dir=str(tmp_path))

    assert result.endswith("[truncated]")
    assert len(result.splitlines()) == MAX_DIR_ENTRIES + 1


def test_grep_finds_matching_lines(tmp_path):
    (tmp_path / "mod.py").write_text("a = 1\ndef foo():\n    return 2\n")

    result = grep("def foo", base_dir=str(tmp_path))

    assert result == "mod.py:2: def foo():"


def test_grep_returns_multiple_matches_across_files(tmp_path):
    (tmp_path / "a.py").write_text("hit one\nskip\nhit two\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "b.py").write_text("hit three\n")

    result = grep("hit", base_dir=str(tmp_path))

    assert result.splitlines() == [
        "a.py:1: hit one",
        "a.py:3: hit two",
        "pkg/b.py:1: hit three",
    ]


def test_grep_searches_recursively(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "deep.py").write_text("x = 1\ntarget = 2\n")

    result = grep("target", base_dir=str(tmp_path))

    assert result == "pkg/deep.py:2: target = 2"


def test_grep_supports_regex(tmp_path):
    (tmp_path / "mod.py").write_text("foo123\nfoobar\nbaz\n")

    result = grep(r"foo\d+", base_dir=str(tmp_path))

    assert result == "mod.py:1: foo123"


def test_grep_can_target_a_single_file(tmp_path):
    (tmp_path / "a.py").write_text("hit\n")
    (tmp_path / "b.py").write_text("hit\n")

    result = grep("hit", path="a.py", base_dir=str(tmp_path))

    assert result == "a.py:1: hit"


def test_grep_no_matches_returns_message(tmp_path):
    (tmp_path / "mod.py").write_text("nothing here\n")

    result = grep("absent", base_dir=str(tmp_path))

    assert result == "No matches found."


def test_grep_invalid_pattern_returns_error(tmp_path):
    (tmp_path / "mod.py").write_text("anything\n")

    result = grep("(unclosed", base_dir=str(tmp_path))

    assert result.startswith("Error:")


def test_grep_skips_binary_files(tmp_path):
    (tmp_path / "data.bin").write_bytes(b"\xff\xfe\x00match\x00")
    (tmp_path / "code.py").write_text("match\n")

    result = grep("match", base_dir=str(tmp_path))

    assert result == "code.py:1: match"


def test_grep_missing_path_returns_error(tmp_path):
    result = grep("anything", path="missing", base_dir=str(tmp_path))

    assert result.startswith("Error:")


def test_grep_rejects_path_outside_base_dir(tmp_path):
    (tmp_path / "project").mkdir()
    (tmp_path / "secret.txt").write_text("hunter2")

    result = grep("hunter2", path="..", base_dir=str(tmp_path / "project"))

    assert result.startswith("Error:")
    assert "hunter2" not in result


def test_grep_truncates_many_matches(tmp_path):
    lines = "\n".join("match" for _ in range(MAX_GREP_MATCHES + 10))
    (tmp_path / "many.py").write_text(lines)

    result = grep("match", base_dir=str(tmp_path))

    assert result.endswith("[truncated]")
    assert len(result.splitlines()) == MAX_GREP_MATCHES + 1


def test_search_code_formats_ranked_results():
    engine = MagicMock()
    engine.search.return_value = [
        {
            "code_id": 1,
            "score": 0.876,
            "payload": {"content": "def foo(): pass", "source": "a.py"},
        },
        {
            "code_id": 2,
            "score": 0.5,
            "payload": {"content": "def bar(): pass", "source": "b.py"},
        },
    ]

    result = search_code("find foo", engine)

    assert result == (
        "--- Result 1 (source: a.py, score: 0.876) ---\ndef foo(): pass"
        "\n\n"
        "--- Result 2 (source: b.py, score: 0.500) ---\ndef bar(): pass"
    )


def test_search_code_passes_query_and_k_to_engine():
    engine = MagicMock()
    engine.search.return_value = []

    search_code("a query", engine, k=7)

    engine.search.assert_called_once_with("a query", k=7)


def test_search_code_no_results_returns_message():
    engine = MagicMock()
    engine.search.return_value = []

    result = search_code("nothing relevant", engine)

    assert result == "No matching code found."


def test_search_code_falls_back_to_code_content_payload_key():
    engine = MagicMock()
    engine.search.return_value = [
        {
            "code_id": 1,
            "score": 0.9,
            "payload": {"code_content": "x = 1"},
        }
    ]

    result = search_code("anything", engine)

    assert "x = 1" in result
    assert "source: unknown" in result


def test_search_code_truncates_long_output():
    engine = MagicMock()
    engine.search.return_value = [
        {
            "code_id": 1,
            "score": 0.9,
            "payload": {
                "content": "a" * (MAX_SEARCH_CHARS + 500),
                "source": "a.py",
            },
        }
    ]

    result = search_code("anything", engine)

    assert len(result) < MAX_SEARCH_CHARS + 500
    assert result.endswith("[truncated]")
