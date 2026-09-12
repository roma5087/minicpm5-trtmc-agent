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

MAX_TURNS = 6


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


def run_agent(
    *,
    model_dir: str,
    binary: Path,
    bundle: Path,
    runtime_root: Path,
    task: str,
    max_new_tokens: int = 300,
    verbose: bool = True,
) -> str:
    tokenizer = load_tokenizer(model_dir)
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": task},
    ]

    for turn in range(1, MAX_TURNS + 1):
        prompt = render_prompt(tokenizer, messages, TOOL_SCHEMAS)
        payload = run_trtmc(binary, bundle, runtime_root, prompt, max_new_tokens)
        raw_output = payload["text"]
        if verbose:
            print(f"\n--- turn {turn}: model output ---\n{raw_output}")

        messages.append({"role": "assistant", "content": raw_output})
        calls = parse_tool_calls(raw_output)
        if not calls:
            return strip_thinking(raw_output).strip()

        for call in calls:
            impl = TOOL_IMPLEMENTATIONS.get(call["name"])
            if impl is None:
                result = f"error: unknown tool {call['name']!r}"
            else:
                try:
                    result = impl(**call["arguments"])
                except TypeError as error:
                    result = f"error: bad arguments for {call['name']}: {error}"
            if verbose:
                print(f"[tool] {call['name']}({call['arguments']}) -> {result}")
            messages.append({"role": "user", "content": f"<tool_response>{result}</tool_response>"})

    return "error: exceeded max turns without a final answer"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniCPM5-2B tool-calling agent, served via TensorRT-Model-Connect"
    )
    parser.add_argument("task", help="The task to give the agent")
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

    answer = run_agent(
        model_dir=args.model_dir,
        binary=Path(args.binary),
        bundle=Path(args.bundle),
        runtime_root=Path(args.runtime_root),
        task=args.task,
        max_new_tokens=args.max_new_tokens,
    )
    print("\n=== final answer ===")
    print(answer)


if __name__ == "__main__":
    main()
