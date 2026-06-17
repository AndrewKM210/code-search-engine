from cse.agent.tools import MAX_FILE_CHARS, read_file


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
