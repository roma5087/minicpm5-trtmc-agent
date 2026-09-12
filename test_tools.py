import pytest

from tools import WORKSPACE, calculator, read_file, write_file


def test_calculator_basic_arithmetic():
    assert calculator("2 + 2") == "4"
    assert calculator("150 * 3.7") == "555.0"


def test_calculator_supports_comparisons():
    assert calculator("4200 <= 5000") == "True"
    assert calculator("4200 * 2 <= 5000") == "False"
    assert calculator("10 == 10") == "True"


def test_calculator_rejects_bool_as_numeric_literal():
    # bool is a subclass of int in Python; "True + 1" must not silently
    # evaluate to "2" just because isinstance(True, int) is True.
    result = calculator("True + 1")
    assert result.startswith("error:")


def test_calculator_huge_result_returns_error_not_crash():
    # str() on a sufficiently large int raises (Python's own int-to-str
    # conversion limit) -- this must come back as a normal "error: ..."
    # string, never an uncaught exception, since the agent's safety net
    # only recognizes failures that come back that way.
    result = calculator("10**200000")
    assert result.startswith("error:")


def test_calculator_rejects_non_arithmetic():
    result = calculator("__import__('os').system('echo pwned')")
    assert result.startswith("error:")


def test_calculator_rejects_name_lookup():
    result = calculator("os.getcwd()")
    assert result.startswith("error:")


def test_write_then_read_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.WORKSPACE", tmp_path)
    write_file("note.txt", "hello workspace")
    assert read_file("note.txt") == "hello workspace"


def test_read_missing_file_reports_error(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.WORKSPACE", tmp_path)
    result = read_file("does_not_exist.txt")
    assert result.startswith("error:")


def test_write_file_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.WORKSPACE", tmp_path)
    result = write_file("../escape.txt", "malicious")
    assert result.startswith("error:")
    assert not (tmp_path.parent / "escape.txt").exists()


def test_write_file_rejects_the_workspace_directory_itself(tmp_path, monkeypatch):
    # filename="." resolves to the workspace directory itself, not a file
    # within it -- writing to a directory path raises IsADirectoryError if
    # this isn't rejected at the validation layer first.
    monkeypatch.setattr("tools.WORKSPACE", tmp_path)
    result = write_file(".", "malicious")
    assert result.startswith("error:")


def test_write_file_reports_os_errors_instead_of_crashing(tmp_path, monkeypatch):
    # A filename that resolves to an existing directory *within* the
    # workspace (not the workspace root itself, which is covered above)
    # must still come back as an error string, not an uncaught exception.
    monkeypatch.setattr("tools.WORKSPACE", tmp_path)
    (tmp_path / "a_directory").mkdir()
    result = write_file("a_directory", "content")
    assert result.startswith("error:")
