#!/usr/bin/env python3
"""Tool-calling agent loop: MiniCPM5-2B served via TensorRT-Model-Connect's
native `trtmc` runtime, with tool orchestration done entirely in Python.

trtmc has no persistent/server mode -- every turn re-invokes `trtmc run` as a
fresh subprocess, re-loading the compiled engine each time. That's fine for a
demo; a long-running deployment would need a persistent server instead.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from engine import EngineError, run_trtmc
from parse import parse_tool_calls, strip_thinking
from render import load_tokenizer, render_prompt
from tools import TOOL_IMPLEMENTATIONS, TOOL_SCHEMAS, sanitize_tool_result

MAX_TURNS = 12
# A single failing tool call gets fed back to the model so it can recover on
# its own (reformulate a query, fix a bad argument, etc.) -- this cap only
# stops it from burning every remaining turn retrying a tool that is stuck.
MAX_CONSECUTIVE_TOOL_ERRORS = 3
# Room for reasoning plus a tool call carrying a file body; 300 truncates the
# default task's final write_file turn.
DEFAULT_MAX_NEW_TOKENS = 1024
# Must match the max_sequence_length the bundle was built with.
DEFAULT_MAX_SEQUENCE_LENGTH = 4096

DEFAULT_TASK = (
    "I have a $10,000 budget and need at least 40GB of GPU VRAM for a workload. "
    "Research the NVIDIA A40, L40S, and RTX 6000 Ada online for background, then "
    "use these figures as given for your comparison: FP16 TFLOPS of 149.7 for the "
    "A40, 183 for the L40S, and 91.1 for the RTX 6000 Ada; all three have 48GB "
    "VRAM. Using approximate list prices of $5,500 for the A40, $8,600 for the "
    "L40S, and $6,800 for the RTX 6000 Ada, use the calculator to compute FP16 "
    "TFLOPS per dollar for each. Then use the calculator again to directly check "
    "which ratio is greatest (e.g. a '>' comparison between two of the ratios at "
    "a time) rather than judging by eye, since the highest ratio is the one to "
    "recommend. Save the comparison to a file, including the recommendation and "
    "why."
)

# A worked example of one full tool-call round trip, in the exact XML syntax
# parse.py expects, as an in-context aid for a small model. The missing tag
# names seen in early runs turned out to come from trtmc's decoder, not the
# model (see README "The trtmc finding"), so this example's own effect on
# reliability was never measured separately; --no-example turns it off.
ONE_SHOT_EXAMPLE = [
    {"role": "user", "content": "What is 15 times 23?"},
    {
        "role": "assistant",
        "content": '<function name="calculator"><param name="expression">15 * 23</param></function>',
    },
    {"role": "user", "content": "<tool_response>345</tool_response>"},
    {"role": "assistant", "content": "15 times 23 is 345."},
]


def _number(value) -> float:
    """A timing field trtmc reported, or 0.0 if it is missing or not a number."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


class PerfSummary:
    """Per-turn timing from trtmc's own JSON output (prefill_ms, decode_ms,
    setup_ms), plus the wall-clock time of each subprocess call, so the cost
    of reloading the engine every turn is measured rather than just stated."""

    def __init__(self) -> None:
        self.turns = 0
        self.total_setup_ms = 0.0
        self.total_prefill_ms = 0.0
        self.total_decode_ms = 0.0
        self.total_wall_ms = 0.0
        self.total_tokens = 0

    def record(self, payload: dict, token_count: int) -> None:
        self.turns += 1
        self.total_setup_ms += _number(payload.get("setup_ms"))
        self.total_prefill_ms += _number(payload.get("prefill_ms"))
        self.total_decode_ms += _number(payload.get("decode_ms"))
        self.total_wall_ms += _number(payload.get("_wall_ms"))
        self.total_tokens += token_count

    def report(self) -> str:
        avg_decode_per_token = self.total_decode_ms / self.total_tokens if self.total_tokens else 0.0
        lines = [
            f"engine invocations : {self.turns} (one per turn -- trtmc has no server mode)",
            f"total setup time   : {self.total_setup_ms:.2f} ms",
            f"total prefill time : {self.total_prefill_ms:.2f} ms",
            f"total decode time  : {self.total_decode_ms:.2f} ms across {self.total_tokens} tokens",
            f"avg decode/token   : {avg_decode_per_token:.3f} ms",
        ]
        if self.total_wall_ms:
            outside = self.total_wall_ms - (self.total_setup_ms + self.total_prefill_ms + self.total_decode_ms)
            lines.append(f"total wall time    : {self.total_wall_ms:.2f} ms")
            lines.append(f"outside prefill/decode : {outside:.2f} ms (process start, engine load, JSON)")
        return "\n".join(lines)


# Parameters each tool declares, from its schema. Model-supplied arguments are
# checked against this before the call, so the model cannot pass a parameter
# the schema doesn't advertise (e.g. web_search's internal max_results).
_ALLOWED_PARAMS = {
    schema["function"]["name"]: set(schema["function"]["parameters"]["properties"])
    for schema in TOOL_SCHEMAS
}

_MALFORMED_CALL_MESSAGE = (
    "error: could not parse your tool call. Use exactly "
    '<function name="TOOL"><param name="ARG">VALUE</param></function>.'
)


def _count_tokens(tokenizer, text: str) -> int | None:
    encode = getattr(tokenizer, "encode", None)
    return len(encode(text, add_special_tokens=False)) if encode else None


def run_agent(
    *,
    model_dir: str,
    binary: Path,
    bundle: Path,
    runtime_root: Path,
    task: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    verbose: bool = True,
    include_example: bool = True,
) -> tuple[str, PerfSummary]:
    tokenizer = load_tokenizer(model_dir)
    # Turn-end markers are stripped at the token-id level rather than by
    # matching the decoded string's suffix, which a trailing newline after the
    # token would defeat.
    eos_ids = {tokenizer.convert_tokens_to_ids("<|im_end|>")}
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.add(tokenizer.eos_token_id)
    # Every added token can be matched as a substring by trtmc's encoder; the
    # list drives sanitize_tool_result() below.
    added_tokens = list(getattr(tokenizer, "get_added_vocab", lambda: {})())
    messages = [{"role": "system", "content": "You are a helpful assistant with access to tools."}]
    if include_example:
        # Copied, not referenced: ONE_SHOT_EXAMPLE is a shared module-level
        # list reused across run_agent() calls.
        messages.extend(dict(m) for m in ONE_SHOT_EXAMPLE)
    messages.append({"role": "user", "content": task})
    perf = PerfSummary()
    consecutive_tool_errors = 0

    for turn in range(1, MAX_TURNS + 1):
        prompt = render_prompt(tokenizer, messages, TOOL_SCHEMAS)
        prompt_tokens = _count_tokens(tokenizer, prompt)
        if prompt_tokens is not None:
            if verbose:
                print(f"\n--- turn {turn}: prompt is {prompt_tokens} tokens ---")
            if prompt_tokens + max_new_tokens > max_sequence_length:
                return (
                    f"error: prompt ({prompt_tokens} tokens) + max_new_tokens ({max_new_tokens}) "
                    f"exceeds max_sequence_length ({max_sequence_length})",
                    perf,
                )
        # Decode token_ids ourselves rather than use payload["text"]: trtmc's
        # decoder drops every token the tokenizer flags `special` (see
        # README "The trtmc finding"), and MiniCPM5's tool-call tags are
        # among them.
        try:
            payload = run_trtmc(binary, bundle, runtime_root, prompt, max_new_tokens)
            token_ids = payload["token_ids"]
            if not isinstance(token_ids, list):
                raise TypeError(f"expected a list for token_ids, got {type(token_ids).__name__}")
            hit_length_cap = bool(token_ids) and len(token_ids) >= max_new_tokens and token_ids[-1] not in eos_ids
            if token_ids and token_ids[-1] in eos_ids:
                token_ids = token_ids[:-1]
            raw_output = tokenizer.decode(token_ids)
        except (
            EngineError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            OSError,
        ) as error:
            # A trtmc failure isn't something another model turn can
            # reformulate around, so stop with a clear error.
            return f"error: trtmc invocation failed: {type(error).__name__}: {error}", perf
        perf.record(payload, len(payload["token_ids"]))
        if verbose:
            print(f"\n--- turn {turn}: model output ---\n{raw_output}")
        if hit_length_cap:
            return (
                f"error: model output hit max_new_tokens ({max_new_tokens}) without finishing; "
                "raise --max-new-tokens",
                perf,
            )

        # Only the visible text counts: reasoning inside <think> must neither
        # execute tool calls nor be stored (the template re-renders history).
        visible = strip_thinking(raw_output)
        messages.append({"role": "assistant", "content": visible})
        calls = parse_tool_calls(visible)
        turn_had_error = False
        if not calls:
            if "<function" not in visible:
                answer = visible.strip()
                return answer or "error: model produced an empty answer", perf
            # The model attempted a call the parser rejected. Say so instead of
            # returning the broken markup as if it were the answer.
            turn_had_error = True
            messages.append({"role": "user", "content": f"<tool_response>{_MALFORMED_CALL_MESSAGE}</tool_response>"})

        for call in calls:
            impl = TOOL_IMPLEMENTATIONS.get(call["name"])
            allowed = _ALLOWED_PARAMS.get(call["name"])
            unexpected = set(call["arguments"]) - allowed if allowed is not None else set()
            if impl is None:
                result = f"error: unknown tool {call['name']!r}"
            elif unexpected:
                result = (
                    f"error: {call['name']} got unexpected parameter(s) {sorted(unexpected)}; "
                    f"expected {sorted(allowed)}"
                )
            else:
                try:
                    result = impl(**call["arguments"])
                except Exception as error:
                    # Tools return "error: ..." strings rather than raising; this
                    # keeps a tool that breaks that rule from crashing the loop.
                    result = f"error: {call['name']} raised {type(error).__name__}: {error}"
            if isinstance(result, str) and result.startswith("error:"):
                turn_had_error = True
            if verbose:
                print(f"[tool] {call['name']}({call['arguments']}) -> {result}")
            # The one place tool output enters the prompt: sanitize everything
            # (search results, file contents, error strings that echo model input).
            safe = sanitize_tool_result(str(result), added_tokens)
            messages.append({"role": "user", "content": f"<tool_response>{safe}</tool_response>"})

        consecutive_tool_errors = consecutive_tool_errors + 1 if turn_had_error else 0
        if consecutive_tool_errors >= MAX_CONSECUTIVE_TOOL_ERRORS:
            return (
                f"error: giving up after {consecutive_tool_errors} consecutive tool "
                "errors -- the model kept retrying a tool that isn't recovering",
                perf,
            )

    return "error: exceeded max turns without a final answer", perf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniCPM5-2B tool-calling agent, served via TensorRT-Model-Connect"
    )
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK, help="The task to give the agent")
    parser.add_argument(
        "--model-dir", required=True, help="Local HF checkpoint dir (for tokenizer + chat template)"
    )
    parser.add_argument("--binary", required=True, help="Path to the trtmc CLI")
    parser.add_argument("--bundle", required=True, help="Path to the built .bundle file")
    parser.add_argument(
        "--runtime-root", required=True, help="Directory containing the compiled runtime .so files"
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=DEFAULT_MAX_SEQUENCE_LENGTH,
        help="The max_sequence_length the bundle was built with; a turn whose prompt plus "
        "--max-new-tokens would exceed it fails before invoking trtmc",
    )
    parser.add_argument(
        "--no-example",
        action="store_true",
        help="Skip the one-shot tool-call example (on by default).",
    )
    args = parser.parse_args()

    answer, perf = run_agent(
        model_dir=args.model_dir,
        binary=Path(args.binary),
        bundle=Path(args.bundle),
        runtime_root=Path(args.runtime_root),
        task=args.task,
        max_new_tokens=args.max_new_tokens,
        max_sequence_length=args.max_sequence_length,
        include_example=not args.no_example,
    )
    print("\n=== final answer ===")
    print(answer)
    print("\n=== performance (real, measured from trtmc) ===")
    print(perf.report())


if __name__ == "__main__":
    main()
