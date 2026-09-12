import pytest

from tools import WORKSPACE, calculator, read_file, write_file


def test_calculator_basic_arithmetic():
    assert calculator("2 + 2") == "4"
    assert calculator("150 * 3.7") == "555.0"


def test_calculator_supports_comparisons():
    assert calculator("4200 <= 5000") == "True"
    assert calculator("4200 * 2 <= 5000") == "False"
    assert calculator("10 == 10") == "True"


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
