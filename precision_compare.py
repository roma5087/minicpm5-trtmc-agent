#!/usr/bin/env python3
"""Measure MiniCPM5-2B's actual bf16 vs fp16 timing through TensorRT-Model-Connect.

Requires two bundles of the *same* checkpoint, built at different precisions
via the family's own build() (families/llama/model.py), e.g.:

    build(BuildRequest(..., precision="bf16", ...))  -> minicpm5-2b-bf16.bundle
    build(BuildRequest(..., precision="fp16", ...))  -> minicpm5-2b-fp16.bundle

This script doesn't build anything itself -- it just runs the identical
prompt through each pre-built bundle a few times and reports real measured
numbers (prefill_ms, decode_ms, ms/token), not an estimate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path


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
        "true",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=True)
    return json.loads(completed.stdout)


def measure(binary: Path, bundle: Path, runtime_root: Path, prompt: str, max_new_tokens: int, repeats: int) -> dict:
    prefill_ms, decode_ms, ms_per_token = [], [], []
    for _ in range(repeats):
        payload = run_once(binary, bundle, runtime_root, prompt, max_new_tokens)
        tokens = len(payload.get("token_ids", [])) or 1
        prefill_ms.append(payload["prefill_ms"])
        decode_ms.append(payload["decode_ms"])
        ms_per_token.append(payload["decode_ms"] / tokens)
    return {
        "prefill_ms_avg": statistics.mean(prefill_ms),
        "decode_ms_avg": statistics.mean(decode_ms),
        "ms_per_token_avg": statistics.mean(ms_per_token),
        "runs": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MiniCPM5-2B bf16 vs fp16 timing via trtmc")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--bf16-bundle", required=True)
    parser.add_argument("--fp16-bundle", required=True)
    parser.add_argument("--prompt", default="Explain what a KV cache does in one paragraph.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5, help="Runs per precision, averaged")
    args = parser.parse_args()

    binary = Path(args.binary)
    runtime_root = Path(args.runtime_root)

    results = {}
    for label, bundle_path in (("bf16", args.bf16_bundle), ("fp16", args.fp16_bundle)):
        results[label] = measure(
            binary, Path(bundle_path), runtime_root, args.prompt, args.max_new_tokens, args.repeats
        )

    print(f"prompt: {args.prompt!r}  max_new_tokens={args.max_new_tokens}  repeats={args.repeats}\n")
    print(f"{'':10}{'prefill_ms':>14}{'decode_ms':>14}{'ms/token':>14}")
    for label, r in results.items():
        print(f"{label:10}{r['prefill_ms_avg']:>14.2f}{r['decode_ms_avg']:>14.2f}{r['ms_per_token_avg']:>14.3f}")

    speedup = results["fp16"]["ms_per_token_avg"] / results["bf16"]["ms_per_token_avg"]
    faster = "bf16" if speedup > 1 else "fp16"
    print(f"\n{faster} is {max(speedup, 1 / speedup):.2f}x faster per token in this run.")


if __name__ == "__main__":
    main()
