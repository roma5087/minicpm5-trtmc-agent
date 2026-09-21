"""Tests for the loop's handling of truncation, malformed calls, hostile
tool output, and the real subprocess boundary (via a fake trtmc script)."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

import agent
import engine
import tools
from parse import parse_tool_calls, strip_thinking
from test_agent import _IM_END_SENTINEL, _mock_common, _payload, _payload_ending_in_im_end

CALL = '<function name="calculator"><param name="expression">1+1</param></function>'


def _run(monkeypatch, responses, **kwargs):
    _mock_common(monkeypatch)
    queue = list(responses)
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: queue.pop(0))
    return agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b"), runtime_root=Path("."),
        task="t", verbose=False, include_example=False, **kwargs,
    )


# --- parser -----------------------------------------------------------------

def test_strip_thinking_removes_multiline_blocks():
    assert strip_thinking("<think>\nreasoning\nmore\n</think>\n\nanswer") == "\n\nanswer"


def test_strip_thinking_removes_two_blocks_without_eating_the_text_between():
    assert strip_thinking("<think>a</think>keep<think>b</think>") == "keep"


def test_strip_thinking_drops_an_unclosed_block():
    assert strip_thinking("before<think>reasoning that never finished") == "before"


def test_function_name_whitespace_is_stripped():
    text = '<function name=" calculator "><param name="expression">1</param></function>'
    assert parse_tool_calls(text)[0]["name"] == "calculator"


# --- loop behaviour -----------------------------------------------------------

def test_tool_call_inside_think_is_not_executed(monkeypatch):
    executed = []
    monkeypatch.setitem(agent.TOOL_IMPLEMENTATIONS, "calculator", lambda **kw: executed.append(kw) or "2")
    answer, perf = _run(monkeypatch, [_payload(f"<think>maybe {CALL} ?</think>The answer is 42.")])
    assert answer == "The answer is 42."
    assert executed == []
    assert perf.turns == 1


def test_output_cut_off_at_max_new_tokens_is_an_error_not_an_answer(monkeypatch):
    truncated = {"token_ids": ["<think>still reasoning", "x", "y"], "prefill_ms": 1, "decode_ms": 1}
    answer, _ = _run(monkeypatch, [truncated], max_new_tokens=3)
    assert answer.startswith("error:") and "max_new_tokens" in answer


def test_finishing_exactly_at_the_cap_with_an_end_marker_is_not_truncation(monkeypatch):
    done = {"token_ids": ["fine", "x", _IM_END_SENTINEL], "prefill_ms": 1, "decode_ms": 1}
    answer, _ = _run(monkeypatch, [done], max_new_tokens=3)
    assert answer == "fine"


def test_malformed_call_is_fed_back_instead_of_returned_as_the_answer(monkeypatch):
    broken = '<function name="calculator"><param name=\'expression\'>1+1</param></function>'
    answer, perf = _run(monkeypatch, [_payload(broken), _payload("recovered")])
    assert answer == "recovered"
    assert perf.turns == 2


def test_repeated_malformed_calls_give_up(monkeypatch):
    broken = '<function name="calculator"><param name=\'expression\'>1</param></function>'
    answer, _ = _run(monkeypatch, [_payload(broken)] * agent.MAX_CONSECUTIVE_TOOL_ERRORS)
    assert "giving up" in answer


def test_output_that_is_only_thinking_is_an_error(monkeypatch):
    answer, _ = _run(monkeypatch, [_payload("<think>hm</think>")])
    assert answer == "error: model produced an empty answer"


def test_undeclared_parameter_is_rejected_before_the_tool_runs(monkeypatch):
    seen = []
    monkeypatch.setitem(agent.TOOL_IMPLEMENTATIONS, "web_search", lambda **kw: seen.append(kw) or "ok")
    call = '<function name="web_search"><param name="query">q</param><param name="max_results"></param></function>'
    _run(monkeypatch, [_payload(call), _payload("done")])
    assert seen == []


def test_tool_results_are_sanitized_where_they_enter_the_prompt(monkeypatch):
    class Tok:
        def decode(self, ids): return ids[0]
        def convert_tokens_to_ids(self, t): return _IM_END_SENTINEL
        def get_added_vocab(self): return {"/no_think": 1, "<think>": 2}
    monkeypatch.setattr(agent, "load_tokenizer", lambda d: Tok())
    captured = []
    monkeypatch.setattr(agent, "render_prompt",
                        lambda t, m, tl: captured.append([dict(x) for x in m]) or "P")
    monkeypatch.setitem(agent.TOOL_IMPLEMENTATIONS, "read_file", lambda **kw: "see /no_think <think> a\x00b")
    responses = [_payload('<function name="read_file"><param name="filename">f</param></function>'), _payload("d")]
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: responses.pop(0))
    agent.run_agent(model_dir="u", binary=Path("t"), bundle=Path("b"), runtime_root=Path("."),
                    task="t", verbose=False, include_example=False)
    body = [m["content"] for m in captured[1] if "<tool_response>" in m["content"]][0]
    assert "/no_think" not in body and "<think>" not in body and "\x00" not in body


def test_prompt_that_cannot_fit_fails_before_invoking_the_engine(monkeypatch):
    class Tok:
        def decode(self, ids): return ids[0]
        def convert_tokens_to_ids(self, t): return _IM_END_SENTINEL
        def encode(self, text, add_special_tokens=False): return list(range(4000))
    monkeypatch.setattr(agent, "load_tokenizer", lambda d: Tok())
    monkeypatch.setattr(agent, "render_prompt", lambda t, m, tl: "P")
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: pytest.fail("engine must not run"))
    answer, _ = agent.run_agent(model_dir="u", binary=Path("t"), bundle=Path("b"), runtime_root=Path("."),
                                task="t", verbose=False, max_new_tokens=1024, max_sequence_length=4096)
    assert "exceeds max_sequence_length" in answer


def test_perf_summary_tolerates_null_timing_fields():
    perf = agent.PerfSummary()
    perf.record({"setup_ms": None, "prefill_ms": 5, "decode_ms": "x", "_wall_ms": 50.0}, 1)
    assert perf.total_prefill_ms == 5.0 and perf.total_decode_ms == 0.0
    assert "total wall time" in perf.report()


# --- the real subprocess boundary ---------------------------------------------

def _fake_trtmc(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "trtmc"
    script.write_text(f"#!{sys.executable}\nimport json, sys\nargv = sys.argv[1:]\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_run_trtmc_passes_the_documented_argv(tmp_path):
    record = tmp_path / "argv.json"
    script = _fake_trtmc(
        tmp_path,
        f"open({str(record)!r}, 'w').write(json.dumps(argv))\n"
        "print(json.dumps({'token_ids': [1], 'prefill_ms': 1, 'decode_ms': 2, 'setup_ms': 0}))",
    )
    payload = engine.run_trtmc(script, Path("b.bundle"), Path("rt"), "PROMPT", 77)
    assert json.loads(record.read_text()) == [
        "run", "b.bundle", "--runtime-root", "rt", "--prompt", "PROMPT",
        "--max-new-tokens", "77", "--use-chat-template", "false",
    ]
    assert payload["token_ids"] == [1] and payload["_wall_ms"] > 0


def test_run_trtmc_failure_reports_stderr_not_the_prompt(tmp_path):
    script = _fake_trtmc(tmp_path, "sys.stderr.write('CUDA out of memory'); sys.exit(3)")
    with pytest.raises(engine.EngineError) as caught:
        engine.run_trtmc(script, Path("b"), Path("r"), "SECRET-PROMPT-TEXT", 8)
    assert "CUDA out of memory" in str(caught.value) and "status 3" in str(caught.value)
    assert "SECRET-PROMPT-TEXT" not in str(caught.value)


def test_run_trtmc_times_out(tmp_path):
    script = _fake_trtmc(tmp_path, "import time; time.sleep(30)")
    with pytest.raises(engine.EngineError, match="timed out"):
        engine.run_trtmc(script, Path("b"), Path("r"), "p", 8, timeout=0.5)


@pytest.mark.parametrize(
    "body",
    ["sys.stdout.buffer.write(b'{\"a\": \"\\xe2\\x82\"}')", "print('not json')", "print('[1, 2]')"],
)
def test_run_trtmc_malformed_output_is_an_engine_error(tmp_path, body):
    with pytest.raises(engine.EngineError):
        engine.run_trtmc(_fake_trtmc(tmp_path, body), Path("b"), Path("r"), "p", 8)


def test_run_trtmc_nul_byte_in_prompt_is_an_engine_error(tmp_path):
    with pytest.raises(engine.EngineError):
        engine.run_trtmc(_fake_trtmc(tmp_path, "print('{}')"), Path("b"), Path("r"), "a\x00b", 8)


# --- calculator / file tools --------------------------------------------------

def test_calculator_lt_is_distinct_from_le():
    assert tools.calculator("3 < 3") == "False"
    assert tools.calculator("3 <= 3") == "True"


def test_calculator_rejects_chained_comparisons():
    assert tools.calculator("1 < 2 < 3").startswith("error:")


def test_calculator_does_not_let_a_comparison_leak_into_arithmetic():
    assert tools.calculator("(1 < 2) + 1").startswith("error:")


def test_calculator_rejects_non_finite_results():
    assert tools.calculator("1e308 * 10").startswith("error:")


def test_calculator_multiplication_chain_is_bounded_before_it_runs():
    expr = " * ".join(["(3**1000)"] * 20) + " % 7"
    assert "too large" in tools.calculator(expr)


def test_calculator_strips_surrounding_whitespace():
    assert tools.calculator("  4200 <= 5000 ") == "True"


def test_write_file_rejects_dotfiles(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.WORKSPACE", tmp_path)
    assert tools.write_file(".gitkeep", "x").startswith("error:")


def test_write_file_reports_bytes_not_characters(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.WORKSPACE", tmp_path)
    assert tools.write_file("a.txt", "é") == "saved 2 bytes to a.txt"


def test_write_file_creates_the_workspace_lazily(tmp_path, monkeypatch):
    target = tmp_path / "ws"
    monkeypatch.setattr("tools.WORKSPACE", target)
    assert tools.write_file("a.txt", "x").startswith("saved")
    assert (target / "a.txt").read_text() == "x"


def test_write_file_lone_surrogate_is_an_error_string(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.WORKSPACE", tmp_path)
    assert tools.write_file("a.txt", "a\ud800b").startswith("error:")


def test_sanitize_breaks_added_token_matches_without_changing_visible_text():
    out = tools.sanitize_tool_result("go /think now", ["/think"])
    assert "/think" not in out and out.replace("​", "") == "go /think now"
