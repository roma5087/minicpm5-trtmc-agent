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
