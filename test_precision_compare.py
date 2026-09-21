"""Tests for precision_compare.measure()'s own arithmetic and guards, with
run_once() mocked out -- these don't need trtmc or a GPU."""

from __future__ import annotations

from pathlib import Path

import pytest

import precision_compare


def _payload(decode_ms: float, prefill_ms: float = 10.0, n_tokens: int = 5) -> dict:
    return {"prefill_ms": prefill_ms, "decode_ms": decode_ms, "token_ids": list(range(n_tokens))}


def test_measure_rejects_repeats_below_one(monkeypatch):
    with pytest.raises(ValueError):
        precision_compare.measure(Path("trtmc"), Path("b.bundle"), Path("."), "prompt", 64, repeats=0)


def test_measure_discards_first_run_as_warmup_when_repeats_greater_than_one(monkeypatch):
    # First call pays a distinct one-time cost -- it must not appear in the
    # averaged numbers when repeats > 1.
    responses = [
        _payload(decode_ms=999.0),  # warmup: should be discarded
        _payload(decode_ms=20.0),
        _payload(decode_ms=20.0),
    ]
    monkeypatch.setattr(precision_compare, "run_once", lambda *a, **k: responses.pop(0))

    result = precision_compare.measure(Path("trtmc"), Path("b.bundle"), Path("."), "prompt", 64, repeats=2)
    assert result["decode_ms_avg"] == 20.0
    assert result["runs"] == 2


def test_measure_skips_warmup_discard_when_repeats_is_one(monkeypatch):
    # With repeats=1 there's nothing left to average after a discard, so
    # the single run must be used directly rather than thrown away.
    monkeypatch.setattr(precision_compare, "run_once", lambda *a, **k: _payload(decode_ms=42.0))
    result = precision_compare.measure(Path("trtmc"), Path("b.bundle"), Path("."), "prompt", 64, repeats=1)
    assert result["decode_ms_avg"] == 42.0
    assert result["runs"] == 1
    assert result["ms_per_token_stdev"] == 0.0


def test_measure_computes_stdev_only_when_repeats_greater_than_one(monkeypatch):
    responses = [
        _payload(decode_ms=100.0),  # warmup
        _payload(decode_ms=10.0, n_tokens=5),   # 2.0 ms/token
        _payload(decode_ms=20.0, n_tokens=5),   # 4.0 ms/token
    ]
    monkeypatch.setattr(precision_compare, "run_once", lambda *a, **k: responses.pop(0))
    result = precision_compare.measure(Path("trtmc"), Path("b.bundle"), Path("."), "prompt", 64, repeats=2)
    assert result["ms_per_token_stdev"] > 0.0


def test_measure_does_not_run_an_extra_warmup_invocation_when_repeats_is_one(monkeypatch):
    # test_measure_skips_warmup_discard_when_repeats_is_one (above) only
    # checks the resulting averages, and its mock always returns the same
    # constant payload -- so it can't tell an accidental extra warmup call
    # apart from none at all. This counts actual invocations of run_once():
    # an extra "wasted" trtmc subprocess call at repeats=1 would double the
    # real cost of that measurement in production, invisibly to the
    # existing test.
    call_count = {"n": 0}

    def _counting_run_once(*a, **k):
        call_count["n"] += 1
        return _payload(decode_ms=42.0)

    monkeypatch.setattr(precision_compare, "run_once", _counting_run_once)
    precision_compare.measure(Path("trtmc"), Path("b.bundle"), Path("."), "prompt", 64, repeats=1)
    assert call_count["n"] == 1


def test_measure_wraps_malformed_payload_keyerror_with_run_index_context(monkeypatch):
    # No existing test ever triggers the "except KeyError" wrapper at all --
    # deleting the whole try/except (letting a bare KeyError propagate) left
    # the full suite green. This forces a malformed mid-run payload (missing
    # prefill_ms/decode_ms) and checks it surfaces as the documented
    # RuntimeError, naming which of the `repeats` invocations it was.
    responses = [
        _payload(decode_ms=999.0),  # warmup
        {"token_ids": [1, 2, 3]},  # malformed: missing prefill_ms/decode_ms
    ]
    monkeypatch.setattr(precision_compare, "run_once", lambda *a, **k: responses.pop(0))
    with pytest.raises(RuntimeError, match=r"run 1/2.*prefill_ms"):
        precision_compare.measure(Path("trtmc"), Path("b.bundle"), Path("."), "prompt", 64, repeats=2)


def test_main_rejects_repeats_below_one(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "precision_compare.py",
            "--model-dir", "unused",
            "--binary", "trtmc",
            "--runtime-root", ".",
            "--bf16-bundle", "a.bundle",
            "--fp16-bundle", "b.bundle",
            "--repeats", "0",
        ],
    )
    with pytest.raises(SystemExit):
        precision_compare.main()


def test_main_handles_zero_ms_per_token_without_dividing_by_zero(monkeypatch, capsys):
    # A degenerate run (e.g. max_new_tokens so low decode reports ~0) must
    # not raise ZeroDivisionError computing the bf16/fp16 speedup ratio.
    monkeypatch.setattr(precision_compare, "load_tokenizer", lambda model_dir: object())
    monkeypatch.setattr(precision_compare, "render_prompt", lambda tokenizer, messages, tools: "PROMPT")
    monkeypatch.setattr(
        precision_compare,
        "measure",
        lambda *a, **k: {
            "prefill_ms_avg": 1.0,
            "decode_ms_avg": 0.0,
            "ms_per_token_avg": 0.0,
            "ms_per_token_stdev": 0.0,
            "ms_per_token_median": 0.0,
            "ms_per_token_min": 0.0,
            "ms_per_token_runs": [0.0],
            "token_counts": [1],
            "runs": 1,
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "precision_compare.py",
            "--model-dir", "unused",
            "--binary", "trtmc",
            "--runtime-root", ".",
            "--bf16-bundle", "a.bundle",
            "--fp16-bundle", "b.bundle",
            "--repeats", "1",
        ],
    )
    precision_compare.main()
    out = capsys.readouterr().out
    assert "can't compute a speedup ratio" in out
