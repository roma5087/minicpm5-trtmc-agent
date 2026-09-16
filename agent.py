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

from parse import parse_tool_calls, strip_thinking
from render import load_tokenizer, render_prompt
from tools import TOOL_IMPLEMENTATIONS, TOOL_SCHEMAS

MAX_TURNS = 12
# A single failing tool call gets fed back to the model so it can recover on
# its own (reformulate a query, fix a bad argument, etc.) -- this cap only
# stops it from burning every remaining turn retrying a tool that is stuck.
MAX_CONSECUTIVE_TOOL_ERRORS = 3

DEFAULT_TASK = (
    "I have a $5,000 budget and need at least 40GB of GPU VRAM for a workload. "
    "Research the NVIDIA A40, L40S, and RTX 6000 Ada online for background, then "
    "use these figures for your comparison: dense FP16 TFLOPS are 149.7 for the "
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
# parse.py expects. Measured directly (see README "Results"): on a genuine
# multi-entity task, MiniCPM5-2B reasoned through the right plan but then
# dropped the literal <function>/<param> tag names on its first real attempt
# -- it knew *what* to do, not how to spell the syntax for it. One in-context
# example of the correct spelling is a standard, well-established technique
# for this exact failure mode in small models; it does not change what the
# model decides to do, only whether it reproduces the syntax correctly.
ONE_SHOT_EXAMPLE = [
    {"role": "user", "content": "What is 15 times 23?"},
    {
        "role": "assistant",
        "content": '<function name="calculator"><param name="expression">15 * 23</param></function>',
    },
    {"role": "user", "content": "<tool_response>345</tool_response>"},
    {"role": "assistant", "content": "15 times 23 is 345."},
]


def run_trtmc(
    binary: Path, bundle: Path, runtime_root: Path, prompt: str, max_new_tokens: int
) -> dict:
    command = [
        str(binary),
        "run",
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--use-chat-template",
        "false",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=True)
    return json.loads(completed.stdout)


class PerfSummary:
    """Real per-turn timing from trtmc's own JSON output (prefill_ms,
    decode_ms, setup_ms) -- accumulated instead of discarded, since it's
    genuine engineering signal about the compiled engine, not the model's
    text output."""

    def __init__(self) -> None:
        self.turns = 0
        self.total_setup_ms = 0.0
        self.total_prefill_ms = 0.0
        self.total_decode_ms = 0.0
        self.total_tokens = 0

    def record(self, payload: dict, token_count: int) -> None:
        self.turns += 1
        self.total_setup_ms += payload.get("setup_ms", 0.0)
        self.total_prefill_ms += payload.get("prefill_ms", 0.0)
        self.total_decode_ms += payload.get("decode_ms", 0.0)
        self.total_tokens += token_count

    def report(self) -> str:
        avg_decode_per_token = self.total_decode_ms / self.total_tokens if self.total_tokens else 0.0
        return (
            f"engine invocations : {self.turns} (one per turn -- trtmc has no server mode)\n"
            f"total setup time   : {self.total_setup_ms:.2f} ms\n"
            f"total prefill time : {self.total_prefill_ms:.2f} ms\n"
            f"total decode time  : {self.total_decode_ms:.2f} ms across {self.total_tokens} tokens\n"
            f"avg decode/token   : {avg_decode_per_token:.3f} ms"
        )


def run_agent(
    *,
    model_dir: str,
    binary: Path,
    bundle: Path,
    runtime_root: Path,
    task: str,
    max_new_tokens: int = 300,
    verbose: bool = True,
    include_example: bool = False,
) -> tuple[str, PerfSummary]:
    tokenizer = load_tokenizer(model_dir)
    # Resolved once: used to strip a trailing turn-end marker at the token-id
    # level (see below) rather than by matching the decoded string's suffix,
    # which a trailing newline/whitespace after the token would silently
    # defeat.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    messages = [{"role": "system", "content": "You are a helpful assistant with access to tools."}]
    if include_example:
        # Copied, not referenced: ONE_SHOT_EXAMPLE is a shared module-level
        # list reused across every run_agent() call in the same process --
        # nothing here mutates a message dict in place today, but extending
        # by reference would silently corrupt it across calls the moment
        # something does.
        messages.extend(dict(m) for m in ONE_SHOT_EXAMPLE)
    messages.append({"role": "user", "content": task})
    perf = PerfSummary()
    consecutive_tool_errors = 0

    for turn in range(1, MAX_TURNS + 1):
        prompt = render_prompt(tokenizer, messages, TOOL_SCHEMAS)
        try:
            payload = run_trtmc(binary, bundle, runtime_root, prompt, max_new_tokens)
            token_ids = payload["token_ids"]
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
            KeyError,
        ) as error:
            # trtmc itself failing (crash, timeout, malformed output) isn't
            # something another model turn can reformulate around -- unlike
            # a tool error, there's nothing to feed back and retry against,
            # so stop now with a clear error rather than loop or crash raw.
            return f"error: trtmc invocation failed: {type(error).__name__}: {error}", perf
        # Decode token_ids ourselves rather than trust payload["text"]: trtmc's
        # native detokenizer silently drops MiniCPM5's added special tokens
        # (verified directly -- token ids 18/20/21/19 decode via the real HF
        # tokenizer to <function>/<param>/</param>/</function>, but trtmc's own
        # "text" field omits all four). The model has been emitting well-formed
        # tool calls the whole time; only trtmc's own text rendering was wrong.
        # See README "Results" for the full diagnostic.
        if token_ids and token_ids[-1] == im_end_id:
            token_ids = token_ids[:-1]
        raw_output = tokenizer.decode(token_ids)
        perf.record(payload, len(payload.get("token_ids", [])))
        if verbose:
            print(f"\n--- turn {turn}: model output ---\n{raw_output}")

        # Strip <think> before persisting to history: MiniCPM5's template only
        # needs the current turn's thinking, not every prior turn's -- storing
        # it verbatim would let it compound and re-expand into the prompt on
        # every subsequent render across a long multi-turn run.
        messages.append({"role": "assistant", "content": strip_thinking(raw_output)})
        calls = parse_tool_calls(raw_output)
        if not calls:
            return strip_thinking(raw_output).strip(), perf

        turn_had_error = False
        for call in calls:
            impl = TOOL_IMPLEMENTATIONS.get(call["name"])
            if impl is None:
                result = f"error: unknown tool {call['name']!r}"
            else:
                try:
                    result = impl(**call["arguments"])
                except Exception as error:
                    # Defense in depth: every tool in tools.py is written to
                    # catch its own failures and return an "error: ..."
                    # string, never raise -- but the safety net below only
                    # works if that discipline holds for every tool, present
                    # and future, so an unexpected exception here is also
                    # turned into a normal (recoverable) tool-response error
                    # instead of crashing the whole agent process.
                    result = f"error: {call['name']} raised {type(error).__name__}: {error}"
            if isinstance(result, str) and result.startswith("error:"):
                turn_had_error = True
            if verbose:
                print(f"[tool] {call['name']}({call['arguments']}) -> {result}")
            messages.append({"role": "user", "content": f"<tool_response>{result}</tool_response>"})

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
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument(
        "--no-example",
        action="store_true",
        help=(
            "Skip the one-shot tool-call example (see README Results for why it's on by "
            "default: this project's own zero-shot run showed the raw model reasoning "
            "correctly but dropping the <function>/<param> tag names)."
        ),
    )
    args = parser.parse_args()

    answer, perf = run_agent(
        model_dir=args.model_dir,
        binary=Path(args.binary),
        bundle=Path(args.bundle),
        runtime_root=Path(args.runtime_root),
        task=args.task,
        max_new_tokens=args.max_new_tokens,
        include_example=not args.no_example,
    )
    print("\n=== final answer ===")
    print(answer)
    print("\n=== performance (real, measured from trtmc) ===")
    print(perf.report())


if __name__ == "__main__":
    main()
