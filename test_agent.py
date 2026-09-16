"""Orchestration-logic tests for run_agent(), with trtmc/the tokenizer
mocked out -- these don't need a GPU, only the loop's control flow."""

from __future__ import annotations

from pathlib import Path

import agent


# A sentinel standing in for the real <|im_end|> token id. agent.py strips
# the turn-end marker at the token-id level (slicing it off token_ids before
# ever calling decode()), not by matching a decoded string's suffix -- so
# tests simulate it by appending this sentinel as the last token_ids entry,
# not by putting the literal "<|im_end|>" substring in the payload text.
_IM_END_SENTINEL = "STUB_IM_END_TOKEN_ID"


class _StubTokenizer:
    """agent.py decodes token_ids itself now (trtmc's own text field silently
    drops MiniCPM5's <function>/<param>/</function> special tokens -- see
    README "Results"). This stub carries the intended text as the first
    token_ids entry and unwraps it, so tests keep controlling output via
    plain strings without needing real token ids."""

    def decode(self, token_ids):
        return token_ids[0]

    def convert_tokens_to_ids(self, token_str):
        assert token_str == "<|im_end|>", token_str
        return _IM_END_SENTINEL


def _mock_common(monkeypatch):
    monkeypatch.setattr(agent, "load_tokenizer", lambda model_dir: _StubTokenizer())
    monkeypatch.setattr(agent, "render_prompt", lambda tokenizer, messages, tools: "RENDERED_PROMPT")


def _payload(text: str) -> dict:
    return {
        "token_ids": [text],
        "setup_ms": 0.1,
        "prefill_ms": 10.0,
        "decode_ms": 20.0,
    }


def _payload_ending_in_im_end(text: str) -> dict:
    """A payload whose last token_ids entry is the turn-end marker, the way
    a real trtmc response would end one -- text is everything decode()
    should produce *after* agent.py slices the marker off."""
    return {
        "token_ids": [text, _IM_END_SENTINEL],
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


def test_decodes_token_ids_itself_ignoring_any_text_field(monkeypatch):
    # Regression test for the real bug found on GPU hardware: trtmc's own
    # "text" field silently drops MiniCPM5's <function>/<param>/</function>
    # special tokens (verified against the real HF tokenizer decoding the
    # same token_ids). agent.py must decode token_ids itself and must NOT
    # read payload["text"] at all -- a payload with no "text" key must still
    # work, and a wrong/absent "text" value must not affect the outcome.
    _mock_common(monkeypatch)
    payload_without_text_field = {
        "token_ids": ["a well-formed final answer"],
        "setup_ms": 0.0,
        "prefill_ms": 1.0,
        "decode_ms": 1.0,
    }
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: payload_without_text_field)

    answer, _ = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert answer == "a well-formed final answer"


def test_include_example_prepends_the_one_shot_before_the_real_task(monkeypatch):
    _mock_common(monkeypatch)
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: _payload("final answer"))
    captured_messages = []
    monkeypatch.setattr(
        agent, "render_prompt",
        lambda tokenizer, messages, tools: captured_messages.append([dict(m) for m in messages]) or "RENDERED_PROMPT",
    )

    agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="the real task", verbose=False, include_example=True,
    )
    assert captured_messages[0] == [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        *agent.ONE_SHOT_EXAMPLE,
        {"role": "user", "content": "the real task"},
    ]


def test_no_example_by_default(monkeypatch):
    _mock_common(monkeypatch)
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: _payload("final answer"))
    captured_messages = []
    monkeypatch.setattr(
        agent, "render_prompt",
        lambda tokenizer, messages, tools: captured_messages.append([dict(m) for m in messages]) or "RENDERED_PROMPT",
    )

    agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="the real task", verbose=False,
    )
    assert captured_messages[0] == [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "the real task"},
    ]


def test_trailing_im_end_marker_is_stripped_from_final_answer(monkeypatch):
    # trtmc's own (buggy) text field used to implicitly swallow this
    # template artifact; agent.py now strips it itself by slicing the
    # matching token id off the *end of token_ids*, before ever calling
    # decode() -- not by matching a decoded string's suffix, which a
    # trailing newline/whitespace after the token would silently defeat.
    _mock_common(monkeypatch)
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: _payload_ending_in_im_end("the final answer"))

    answer, _ = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert answer == "the final answer"


def test_trailing_im_end_marker_does_not_break_tool_call_parsing_or_leak_into_history(monkeypatch):
    # The marker token is sliced off BEFORE decode(), so a tool call whose
    # last real token is immediately followed by the turn-end marker must
    # still decode to well-formed XML and be recognized/executed, and the
    # assistant turn stored in history must not carry the marker forward
    # into later prompt renders.
    _mock_common(monkeypatch)
    tool_call = '<function name="calculator"><param name="expression">2+2</param></function>'
    responses = [_payload_ending_in_im_end(tool_call), _payload("done")]
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: responses.pop(0))
    captured_messages = []
    monkeypatch.setattr(
        agent, "render_prompt",
        lambda tokenizer, messages, tools: captured_messages.append([dict(m) for m in messages]) or "RENDERED_PROMPT",
    )

    answer, perf = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="what is 2+2", verbose=False,
    )
    assert answer == "done"
    assert perf.turns == 2
    second_call_messages = captured_messages[1]
    assistant_messages = [m for m in second_call_messages if m["role"] == "assistant"]
    assert len(assistant_messages) == 1
    assert "<|im_end|>" not in assistant_messages[0]["content"]
    tool_responses = [
        m for m in second_call_messages
        if m["role"] == "user" and "<tool_response>" in m["content"]
    ]
    assert any("4" in m["content"] for m in tool_responses)


def test_embedded_im_end_text_with_no_trailing_marker_token_is_untouched(monkeypatch):
    # agent.py never does any string-level matching on "<|im_end|>" -- it
    # only ever inspects the last *token id*. So literal text that happens
    # to contain the substring "<|im_end|>" (describing the token itself,
    # say), with no turn-end marker token actually present, must survive
    # completely unchanged -- there's nothing here for the token-id check
    # to even look at.
    _mock_common(monkeypatch)
    text = "the token <|im_end|> marks end of turn."
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: _payload(text))

    answer, _ = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert answer == text


def test_perf_total_tokens_reflects_real_token_ids_length_not_turn_count(monkeypatch):
    # PerfSummary.total_tokens is reported to the user as genuine
    # engineering signal -- it must come from the actual decoded
    # len(payload["token_ids"]) each turn, not e.g. a fixed one-per-turn
    # count or len(raw_output). Padding token_ids with extra entries (the
    # stub tokenizer only ever reads index 0 to get the intended text)
    # lets this be checked without a real tokenizer.
    _mock_common(monkeypatch)

    def _payload_with_token_count(text, n_tokens):
        return {
            "token_ids": [text] + [0] * (n_tokens - 1),
            "setup_ms": 0.0,
            "prefill_ms": 0.0,
            "decode_ms": 0.0,
        }

    responses = [
        _payload_with_token_count(
            '<function name="calculator"><param name="expression">1+1</param></function>', 7
        ),
        _payload_with_token_count("final answer", 3),
    ]
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: responses.pop(0))

    _, perf = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert perf.total_tokens == 10


def test_one_shot_example_persists_across_subsequent_turns_alongside_tool_calls(monkeypatch):
    # ONE_SHOT_EXAMPLE must remain in history for every turn's render, not
    # just the first prompt -- and a later turn's tool-call round trip must
    # not duplicate or otherwise mutate it.
    _mock_common(monkeypatch)
    responses = [
        _payload('<function name="calculator"><param name="expression">1+1</param></function>'),
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
        runtime_root=Path("."), task="the real task", verbose=False, include_example=True,
    )
    assert len(captured_messages) == 2
    example_len = len(agent.ONE_SHOT_EXAMPLE)
    for messages in captured_messages:
        assert messages[1 : 1 + example_len] == agent.ONE_SHOT_EXAMPLE
    second_turn = captured_messages[1]
    assert second_turn.count(agent.ONE_SHOT_EXAMPLE[0]) == 1


def test_trtmc_invocation_failure_returns_a_clear_error_instead_of_crashing(monkeypatch):
    # run_trtmc failing (subprocess crash, timeout, malformed JSON, or a
    # payload missing token_ids) is an infrastructure failure, not something
    # another model turn can reformulate around -- it must come back as a
    # normal "error: ..." return, not an uncaught exception that takes down
    # the whole agent process.
    import subprocess

    _mock_common(monkeypatch)

    def _raise(*a, **k):
        raise subprocess.CalledProcessError(returncode=1, cmd=["trtmc"])

    monkeypatch.setattr(agent, "run_trtmc", _raise)

    answer, perf = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert answer.startswith("error: trtmc invocation failed")
    assert perf.turns == 0


def test_payload_missing_token_ids_returns_a_clear_error_instead_of_crashing(monkeypatch):
    # A malformed/degenerate trtmc response (e.g. an error payload with no
    # token_ids at all) must not surface as a bare uncaught KeyError.
    _mock_common(monkeypatch)
    monkeypatch.setattr(
        agent, "run_trtmc",
        lambda *a, **k: {"setup_ms": 0.0, "prefill_ms": 0.0, "decode_ms": 0.0},
    )

    answer, perf = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert answer.startswith("error: trtmc invocation failed")
    assert perf.turns == 0


def test_payload_that_is_not_a_dict_returns_a_clear_error_instead_of_crashing(monkeypatch):
    # Valid JSON of the wrong *type* (a bare list here, standing in for
    # trtmc emitting e.g. a raw array instead of the expected object) makes
    # payload["token_ids"] raise TypeError, not KeyError -- confirmed this
    # was NOT caught by the first version of this error handling, which
    # only listed KeyError.
    _mock_common(monkeypatch)
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: [1, 2, 3])

    answer, perf = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert answer.startswith("error: trtmc invocation failed")
    assert perf.turns == 0


def test_payload_with_non_list_token_ids_returns_a_clear_error_instead_of_crashing(monkeypatch):
    # token_ids present but not a list (e.g. a bare int) passes the
    # KeyError-based check fine, then crashes later at token_ids[-1] with
    # an uncaught TypeError -- confirmed this second crash site existed
    # even after the first "missing token_ids" case was fixed.
    _mock_common(monkeypatch)
    monkeypatch.setattr(
        agent, "run_trtmc",
        lambda *a, **k: {"token_ids": 5, "setup_ms": 0.0, "prefill_ms": 0.0, "decode_ms": 0.0},
    )

    answer, perf = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert answer.startswith("error: trtmc invocation failed")
    assert perf.turns == 0


class _StubTokenizerDetectsUnstrippedImEnd:
    """Unlike _StubTokenizer above, decode() here inspects the FULL
    token_ids list it's handed, not just index 0 -- so it can actually tell
    whether the im_end marker token was sliced off before decode() was
    called, which is the real contract agent.py claims to implement.
    (_StubTokenizer's decode() only ever reads token_ids[0], which is
    identical whether or not the trailing marker was sliced off index -1 --
    so none of the tests built on it can actually distinguish "sliced
    correctly" from "slicing silently deleted".)"""

    def decode(self, token_ids):
        assert _IM_END_SENTINEL not in token_ids, (
            "the im_end marker token reached decode() -- it must be sliced "
            "off token_ids before decode() is ever called, not left for "
            "string-level cleanup afterward"
        )
        return token_ids[0]

    def convert_tokens_to_ids(self, token_str):
        assert token_str == "<|im_end|>", token_str
        return _IM_END_SENTINEL


def test_im_end_marker_token_is_actually_removed_before_decode_is_called(monkeypatch):
    # Regression test for a real gap: every existing im_end test (above)
    # builds on _StubTokenizer, whose decode() only reads token_ids[0] --
    # so those tests still pass even if the token_ids[:-1] slice is deleted
    # entirely (verified directly: it is). This uses a tokenizer that
    # inspects the whole list handed to decode() and fails loudly if the
    # marker is still in it.
    monkeypatch.setattr(agent, "load_tokenizer", lambda model_dir: _StubTokenizerDetectsUnstrippedImEnd())
    monkeypatch.setattr(agent, "render_prompt", lambda tokenizer, messages, tools: "RENDERED_PROMPT")
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: _payload_ending_in_im_end("the final answer"))

    answer, _ = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert answer == "the final answer"


class _StubTokenizerHandlesEmptyTokenIds:
    """decode() here tolerates an empty token_ids list (returns ""), unlike
    the shared _StubTokenizer (whose token_ids[0] would itself raise
    IndexError on empty input for an unrelated reason) -- so this isolates
    whether run_agent()'s own "token_ids and token_ids[-1] == im_end_id"
    guard protects against a degenerate zero-token trtmc response."""

    def decode(self, token_ids):
        return token_ids[0] if token_ids else ""

    def convert_tokens_to_ids(self, token_str):
        assert token_str == "<|im_end|>", token_str
        return _IM_END_SENTINEL


def test_empty_token_ids_list_does_not_index_error_checking_for_im_end(monkeypatch):
    # A degenerate trtmc response that generated zero tokens (token_ids: [],
    # present but empty -- not missing, so this doesn't hit the KeyError
    # path) must not crash on token_ids[-1] while checking for the trailing
    # im_end marker.
    monkeypatch.setattr(agent, "load_tokenizer", lambda model_dir: _StubTokenizerHandlesEmptyTokenIds())
    monkeypatch.setattr(agent, "render_prompt", lambda tokenizer, messages, tools: "RENDERED_PROMPT")
    monkeypatch.setattr(
        agent, "run_trtmc",
        lambda *a, **k: {"token_ids": [], "setup_ms": 0.0, "prefill_ms": 0.0, "decode_ms": 0.0},
    )

    answer, perf = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert answer == ""
    assert perf.turns == 1


def test_tool_returning_a_non_string_result_does_not_crash_the_error_check(monkeypatch):
    # Every real tool in tools.py always returns a string, but the
    # isinstance(result, str) guard in run_agent()'s error detection is
    # explicit defense against a future/misbehaving tool that doesn't --
    # result.startswith("error:") would raise AttributeError on a non-str
    # result without it. No existing test registers a tool that returns
    # anything but a string.
    _mock_common(monkeypatch)
    monkeypatch.setitem(agent.TOOL_IMPLEMENTATIONS, "returns_none", lambda: None)
    responses = [
        _payload('<function name="returns_none"></function>'),
        _payload("final answer"),
    ]
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: responses.pop(0))

    answer, perf = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert answer == "final answer"
    assert perf.turns == 2


def test_final_answer_strips_surrounding_whitespace(monkeypatch):
    # The returned final answer goes through strip_thinking(...).strip() --
    # every existing "no tool calls" test happens to use text with no
    # leading/trailing whitespace, so none of them would notice if .strip()
    # were dropped.
    _mock_common(monkeypatch)
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: _payload("  \n  final answer with padding  \n  "))

    answer, _ = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert answer == "final answer with padding"


def test_unexpected_exception_from_a_tool_impl_is_caught_not_propagated(monkeypatch):
    # Every tool in tools.py is written to catch its own failures and
    # return an "error: ..." string, never raise -- run_agent()'s own
    # `except Exception` around impl(**arguments) is defense in depth for
    # when that discipline doesn't hold. No existing test actually makes a
    # tool raise; this forces an unrelated exception type (TypeError, not
    # something a narrower except clause might still happen to catch) out
    # of a tool impl and checks it becomes a normal recoverable
    # tool-response error instead of crashing run_agent().
    _mock_common(monkeypatch)

    def _broken_tool(**kwargs):
        raise TypeError("boom")

    monkeypatch.setitem(agent.TOOL_IMPLEMENTATIONS, "broken_tool", _broken_tool)
    responses = [
        _payload('<function name="broken_tool"><param name="x">y</param></function>'),
        _payload("recovered"),
    ]
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: responses.pop(0))

    answer, perf = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert answer == "recovered"
    assert perf.turns == 2


def test_one_shot_example_messages_appended_are_copies_not_shared_references(monkeypatch):
    # ONE_SHOT_EXAMPLE is a shared module-level list reused across every
    # run_agent() call -- messages.extend() must copy each dict (dict(m)),
    # not extend by reference, or an in-place mutation of a message dict in
    # some future code path would silently corrupt the shared example for
    # every subsequent call. Nothing in this run mutates a message dict in
    # place, so an object-identity check is the only way to tell "copied"
    # from "referenced" apart.
    _mock_common(monkeypatch)
    monkeypatch.setattr(agent, "run_trtmc", lambda *a, **k: _payload("final answer"))
    captured_messages = []
    monkeypatch.setattr(
        agent, "render_prompt",
        lambda tokenizer, messages, tools: captured_messages.append(messages) or "RENDERED_PROMPT",
    )

    agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="the real task", verbose=False, include_example=True,
    )
    example_len = len(agent.ONE_SHOT_EXAMPLE)
    appended = captured_messages[0][1 : 1 + example_len]
    for appended_msg, original_msg in zip(appended, agent.ONE_SHOT_EXAMPLE):
        assert appended_msg is not original_msg


def test_perfsummary_record_accumulates_each_field_from_the_payload():
    # PerfSummary.record()/report() have no direct unit test at all today --
    # every existing test only checks .turns or .total_tokens via a full
    # run_agent() call. This pins down setup/prefill/decode accumulation
    # per field directly, so e.g. swapping which payload key feeds which
    # running total wouldn't go unnoticed.
    perf = agent.PerfSummary()
    perf.record({"setup_ms": 1.0, "prefill_ms": 2.0, "decode_ms": 3.0}, token_count=5)
    perf.record({"setup_ms": 10.0, "prefill_ms": 20.0, "decode_ms": 30.0}, token_count=7)
    assert perf.turns == 2
    assert perf.total_setup_ms == 11.0
    assert perf.total_prefill_ms == 22.0
    assert perf.total_decode_ms == 33.0
    assert perf.total_tokens == 12


def test_perfsummary_report_does_not_divide_by_zero_with_no_recorded_tokens():
    # report()'s avg_decode_per_token guards total_tokens == 0 -- untested
    # anywhere today (report() is never even called in the test suite).
    perf = agent.PerfSummary()
    report = perf.report()
    assert "0.000" in report


def test_trtmc_timeout_returns_a_clear_error_instead_of_crashing(monkeypatch):
    # The caught-exception tuple in run_agent() lists four distinct types;
    # only CalledProcessError and the missing-token_ids KeyError path are
    # exercised elsewhere. TimeoutExpired specifically was untested --
    # removing it from the tuple left the full suite green.
    import subprocess

    _mock_common(monkeypatch)

    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd=["trtmc"], timeout=120)

    monkeypatch.setattr(agent, "run_trtmc", _raise)

    answer, perf = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert answer.startswith("error: trtmc invocation failed")
    assert perf.turns == 0


def test_trtmc_malformed_json_returns_a_clear_error_instead_of_crashing(monkeypatch):
    # Same gap as the timeout test above, for json.JSONDecodeError:
    # untested on its own, removing it from the caught tuple left the full
    # suite green.
    import json

    _mock_common(monkeypatch)

    def _raise(*a, **k):
        raise json.JSONDecodeError("bad json", "not json", 0)

    monkeypatch.setattr(agent, "run_trtmc", _raise)

    answer, perf = agent.run_agent(
        model_dir="unused", binary=Path("trtmc"), bundle=Path("b.bundle"),
        runtime_root=Path("."), task="do something", verbose=False,
    )
    assert answer.startswith("error: trtmc invocation failed")
    assert perf.turns == 0


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
