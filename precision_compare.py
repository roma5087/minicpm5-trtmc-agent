#!/usr/bin/env python3
"""Measure MiniCPM5-2B's actual bf16 vs fp16 timing through TensorRT-Model-Connect.

Requires two bundles of the *same* checkpoint, built at different precisions
via the family's own build() (families/llama/model.py), e.g.:

    build(BuildRequest(..., precision="bf16", ...))  -> minicpm5-2b-bf16.bundle
    build(BuildRequest(..., precision="fp16", ...))  -> minicpm5-2b-fp16.bundle

Both bundles must be built with the same max_sequence_length and other
build settings -- this script has no way to verify that from the outside,
so a mismatch would silently confound the comparison. Verify it yourself
against the build commands you used.

This script doesn't build anything itself -- it just runs the identical
prompt through each pre-built bundle a few times and reports real measured
numbers (prefill_ms, decode_ms, ms/token), not an estimate. Uses the same
render.py path as agent.py (the model's own Jinja template via
transformers), not trtmc's `--use-chat-template true` -- the rest of this
project documents that C++ template-detection path as unreliable, so a
precision comparison built on top of it would be measuring a possibly
different prompt structure per precision, undermining the comparison.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

from render import load_tokenizer, render_prompt


def run_once(binary: Path, bundle: Path, runtime_root: Path, prompt: str, max_new_tokens: int) -> dict:
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


def measure(
    binary: Path, bundle: Path, runtime_root: Path, prompt: str, max_new_tokens: int, repeats: int
) -> dict:
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    # Discard the first invocation: it pays one-time costs (page cache, any
    # lazy initialization) that the steady-state numbers below shouldn't
    # include. If repeats == 1, there's nothing left to average after
    # discarding the warmup, so skip the discard in that case.
    if repeats > 1:
        run_once(binary, bundle, runtime_root, prompt, max_new_tokens)

    prefill_ms, decode_ms, ms_per_token = [], [], []
    for i in range(repeats):
        payload = run_once(binary, bundle, runtime_root, prompt, max_new_tokens)
        try:
            tokens = len(payload.get("token_ids", [])) or 1
            prefill_ms.append(payload["prefill_ms"])
            decode_ms.append(payload["decode_ms"])
            ms_per_token.append(payload["decode_ms"] / tokens)
        except KeyError as error:
            # A malformed/unexpected trtmc payload mid-run (e.g. an error
            # response instead of the expected timing fields) should say
            # which of the `repeats` invocations it was, not surface as a
            # bare KeyError with no context about which run produced it.
            raise RuntimeError(
                f"run {i + 1}/{repeats}: trtmc payload missing {error}: {payload!r}"
            ) from error

    return {
        "prefill_ms_avg": statistics.mean(prefill_ms),
        "decode_ms_avg": statistics.mean(decode_ms),
        "ms_per_token_avg": statistics.mean(ms_per_token),
        "ms_per_token_stdev": statistics.stdev(ms_per_token) if repeats > 1 else 0.0,
        "runs": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MiniCPM5-2B bf16 vs fp16 timing via trtmc")
    parser.add_argument("--model-dir", required=True, help="Local HF checkpoint dir (for the chat template)")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--bf16-bundle", required=True)
    parser.add_argument("--fp16-bundle", required=True)
    parser.add_argument("--prompt", default="Explain what a KV cache does in one paragraph.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5, help="Runs per precision, averaged (>=1)")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    tokenizer = load_tokenizer(args.model_dir)
    messages = [{"role": "user", "content": args.prompt}]
    rendered_prompt = render_prompt(tokenizer, messages, tools=[])

    binary = Path(args.binary)
    runtime_root = Path(args.runtime_root)

    results = {}
    for label, bundle_path in (("bf16", args.bf16_bundle), ("fp16", args.fp16_bundle)):
        results[label] = measure(
            binary, Path(bundle_path), runtime_root, rendered_prompt, args.max_new_tokens, args.repeats
        )

    print(f"prompt: {args.prompt!r}  max_new_tokens={args.max_new_tokens}  repeats={args.repeats}")
    if args.repeats > 1:
        print("(first invocation per precision discarded as warmup)\n")
    else:
        print("(repeats=1: no warmup discard, no stdev)\n")

    print(f"{'':10}{'prefill_ms':>14}{'decode_ms':>14}{'ms/token':>14}{'stdev':>10}")
    for label, r in results.items():
        print(
            f"{label:10}{r['prefill_ms_avg']:>14.2f}{r['decode_ms_avg']:>14.2f}"
            f"{r['ms_per_token_avg']:>14.3f}{r['ms_per_token_stdev']:>10.3f}"
        )

    bf16_ms, fp16_ms = results["bf16"]["ms_per_token_avg"], results["fp16"]["ms_per_token_avg"]
    if bf16_ms <= 0 or fp16_ms <= 0:
        print("\ncan't compute a speedup ratio: one precision measured ~0 ms/token (degenerate run)")
        return
    speedup = fp16_ms / bf16_ms
    faster = "bf16" if speedup > 1 else "fp16"
    print(f"\n{faster} is {max(speedup, 1 / speedup):.2f}x faster per token in this run.")


if __name__ == "__main__":
    main()
