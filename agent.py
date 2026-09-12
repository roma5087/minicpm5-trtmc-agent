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
    "Research the NVIDIA A40, L40S, and RTX 6000 Ada, compare FP16 TFLOPS per "
    "dollar for each, and recommend one with justification. Save the comparison "
    "to a file."
)


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
) -> tuple[str, PerfSummary]:
    tokenizer = load_tokenizer(model_dir)
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": task},
    ]
    perf = PerfSummary()
    consecutive_tool_errors = 0

    for turn in range(1, MAX_TURNS + 1):
        prompt = render_prompt(tokenizer, messages, TOOL_SCHEMAS)
        payload = run_trtmc(binary, bundle, runtime_root, prompt, max_new_tokens)
        raw_output = payload["text"]
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
    args = parser.parse_args()

    answer, perf = run_agent(
        model_dir=args.model_dir,
        binary=Path(args.binary),
        bundle=Path(args.bundle),
        runtime_root=Path(args.runtime_root),
        task=args.task,
        max_new_tokens=args.max_new_tokens,
    )
    print("\n=== final answer ===")
    print(answer)
    print("\n=== performance (real, measured from trtmc) ===")
    print(perf.report())


if __name__ == "__main__":
    main()
