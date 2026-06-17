from cse.agent.tools import (
    MAX_DIR_ENTRIES,
    MAX_FILE_CHARS,
    list_directory,
    read_file,
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
