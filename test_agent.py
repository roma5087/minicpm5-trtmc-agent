"""Orchestration-logic tests for run_agent(), with trtmc/the tokenizer
mocked out -- these don't need a GPU, only the loop's control flow."""

from __future__ import annotations

from pathlib import Path

import agent


def _mock_common(monkeypatch):
    monkeypatch.setattr(agent, "load_tokenizer", lambda model_dir: object())
    monkeypatch.setattr(agent, "render_prompt", lambda tokenizer, messages, tools: "RENDERED_PROMPT")


def _payload(text: str) -> dict:
    return {
        "text": text,
        "token_ids": list(range(5)),
        "setup_ms": 0.1,
        "prefill_ms": 10.0,
        "decode_ms": 20.0,
    }


def test_final_answer_with_no_tool_calls(monkeypatch):
    _mock_common(monkeypatch)
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: _payload("just a plain final answer"))

    answer, perf = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert answer == "just a plain final answer"
    assert perf.turns == 1


def test_single_tool_call_then_final_answer(monkeypatch):
    _mock_common(monkeypatch)
    responses = [
        _payload('<function name="calculator"><param name="expression">2+2</param></function>'),
        _payload("the answer is 4"),
    ]
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: responses.pop(0))

    answer, perf = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="what is 2+2", verbose=False,
    )
    assert answer == "the answer is 4"
    assert perf.turns == 2


def test_max_consecutive_tool_errors_is_still_three():
    # A tripwire, not a redundant check: test_gives_up_after_consecutive_
    # tool_errors below derives its expected turn count FROM this same
    # constant, so it stays trivially true even if the constant's value
    # changes. This test independently pins the intended value, so changing
    # it is a deliberate, visible decision rather than a silent drift.
    assert agent.MAX_CONSECUTIVE_TOOL_ERRORS == 3


def test_gives_up_after_consecutive_tool_errors(monkeypatch):
    _mock_common(monkeypatch)
    # Always calls an unknown tool -> always errors -> should bail, not hit MAX_TURNS.
    monkeypatch.setattr(
        agent, "run_trtmc",
        lambda *a, **k: _payload('<function name="not_a_real_tool"><param name="x">y</param></function>'),
    )

    answer, perf = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something impossible", verbose=False,
    )
    assert "giving up" in answer
    assert perf.turns == agent.MAX_CONSECUTIVE_TOOL_ERRORS


def test_one_failing_call_among_successes_in_the_same_turn_still_counts_as_an_error_turn(monkeypatch):
    # A turn with TWO tool calls, one unknown (errors) and one known
    # (succeeds), must still count as an "error turn" for the consecutive-
    # error cap -- the OR-across-calls aggregation, previously untested for
    # the multi-call case.
    _mock_common(monkeypatch)
    bad_turn = _payload(
        '<function name="not_a_real_tool"><param name="x">y</param></function>'
        '<function name="calculator"><param name="expression">1+1</param></function>'
    )
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: bad_turn)

    answer, perf = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert "giving up" in answer
    assert perf.turns == agent.MAX_CONSECUTIVE_TOOL_ERRORS


def test_all_calls_succeeding_in_a_turn_resets_the_error_counter(monkeypatch):
    # One bad turn, then two turns where every call succeeds, then two more
    # bad turns -- the counter must reset on the all-successful turns, so
    # this should NOT give up (only 2 consecutive errors at the end, never
    # reaching MAX_CONSECUTIVE_TOOL_ERRORS=3).
    _mock_common(monkeypatch)
    ok_call = '<function name="calculator"><param name="expression">1+1</param></function>'
    bad_call = '<function name="not_a_real_tool"><param name="x">y</param></function>'
    responses = [
        _payload(bad_call),
        _payload(ok_call),
        _payload(ok_call),
        _payload(bad_call),
        _payload(bad_call),
        _payload("final answer after recovering twice"),
    ]
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: responses.pop(0))

    answer, perf = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert answer == "final answer after recovering twice"
    assert perf.turns == 6


def test_exceeded_max_turns_path(monkeypatch):
    # A model that keeps calling tools successfully forever, never
    # returning a plain final answer, must hit the MAX_TURNS ceiling with
    # its own distinct error message -- previously untested.
    _mock_common(monkeypatch)
    monkeypatch.setattr(
        agent, "run_trtmc",
        lambda *a, **k: _payload(
            '<function name="calculator"><param name="expression">1+1</param></function>'
        ),
    )

    answer, perf = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="never stop calling tools", verbose=False,
    )
    assert answer == "error: exceeded max turns without a final answer"
    assert perf.turns == agent.MAX_TURNS


def test_think_block_stripped_before_entering_history(monkeypatch):
    # <think> blocks must not compound in history across turns -- only the
    # final answer's own thinking is stripped by strip_thinking() at
    # return time; history entries need the same treatment as they're
    # appended, or thinking re-expands into every subsequent render.
    _mock_common(monkeypatch)
    responses = [
        _payload('<think>reasoning</think><function name="calculator"><param name="expression">1+1</param></function>'),
        _payload("final answer"),
    ]
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: responses.pop(0))
    captured_messages = []
    monkeypatch.setattr(
        agent, "render_prompt",
        lambda tokenizer, messages, tools: captured_messages.append([dict(m) for m in messages]) or "RENDERED_PROMPT",
    )

    agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    # The second render call sees the first turn's assistant message in
    # history -- it must not contain the <think> block.
    second_call_messages = captured_messages[1]
    assistant_messages = [m for m in second_call_messages if m["role"] == "assistant"]
    assert len(assistant_messages) == 1
    assert "<think>" not in assistant_messages[0]["content"]


def test_recovers_after_one_error_then_succeeds(monkeypatch):
    _mock_common(monkeypatch)
    responses = [
        _payload('<function name="not_a_real_tool"><param name="x">y</param></function>'),
        _payload('<function name="calculator"><param name="expression">1+1</param></function>'),
        _payload("recovered and answered"),
    ]
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: responses.pop(0))

    answer, perf = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert answer == "recovered and answered"
    assert perf.turns == 3
