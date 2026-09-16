import pytest

from tools import WORKSPACE, calculator, read_file, web_search, write_file


class _FakeDDGS:
    """Stand-in for ddgs.DDGS -- a context manager whose .text() either
    returns canned results or whose __enter__ raises, to exercise
    web_search()'s formatting and error-handling paths without a real
    network call."""

    def __init__(self, results=None, error=None):
        self._results = results or []
        self._error = error

    def __enter__(self):
        if self._error is not None:
            raise self._error
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text(self, query, max_results=4):
        return self._results


def test_web_search_formats_title_and_body_from_each_result(monkeypatch):
    monkeypatch.setattr(
        "tools.DDGS",
        lambda: _FakeDDGS(
            [
                {"title": "NVIDIA A40", "body": "48GB VRAM workstation GPU"},
                {"title": "NVIDIA L40S", "body": "Ada Lovelace architecture"},
            ]
        ),
    )
    result = web_search("NVIDIA A40 vs L40S")
    assert result == (
        "- NVIDIA A40: 48GB VRAM workstation GPU\n"
        "- NVIDIA L40S: Ada Lovelace architecture"
    )


def test_web_search_reports_errors_instead_of_crashing(monkeypatch):
    # web_search is a tool: like every other tool, a failure (network
    # error, DDGS internals raising, etc.) must come back as an
    # "error: ..." string, never an uncaught exception, or the agent's
    # safety net never sees it and the process crashes mid-run.
    monkeypatch.setattr("tools.DDGS", lambda: _FakeDDGS(error=RuntimeError("network down")))
    result = web_search("anything")
    assert result.startswith("error:")


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


def test_calculator_rejects_exponent_before_computing_it():
    # The exponent magnitude is checked *before* pow() runs, not just after
    # via the int-to-str guard above -- a huge exponent must be rejected
    # cheaply rather than actually computed (a real DoS surface: computing
    # e.g. 99**99999999 is expensive in itself, independent of whether its
    # result would ever reach str()).
    result = calculator("2**999999999999")
    assert result.startswith("error:")


def test_calculator_still_allows_reasonably_large_exponents():
    # The guard must not be so tight it breaks ordinary arithmetic a real
    # comparison task would use.
    assert calculator("2**10") == "1024"


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
