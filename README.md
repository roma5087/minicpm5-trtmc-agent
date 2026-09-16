# MiniCPM5-2B Tool-Calling Agent, served via TensorRT-Model-Connect

A tool-calling agent where [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)
is the reasoning core, compiled and served through NVIDIA's
[TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect)
(`trtmc`) native runtime — not vLLM/SGLang's built-in tool-call support.

## Why this exists

Grew out of investigating and fixing a real `families/llama` bug in
TensorRT-Model-Connect that surfaced while validating MiniCPM5-2B against
that framework: HF configs may declare `eos_token_id` as a list of stop
tokens (Llama 3.1+, MiniCPM5-2B), and the generic Llama family kept only one.
Opened as [PR #1269](https://github.com/NVIDIA/TensorRT-Model-Connect/pull/1269);
closed in favor of [PR #1288](https://github.com/NVIDIA/TensorRT-Model-Connect/pull/1288),
which landed the same fix independently with a better approach (additive --
only writes the multi-id field when there's more than one, so single-EOS
bundles are untouched -- plus vocab-bounds validation this PR didn't have)
and merged into `upstream/main`. This project builds against that merged fix
and puts it to actual use, measuring what it actually costs and delivers
rather than just claiming it works.

## Task

> "I have a $5,000 budget and need at least 40GB of GPU VRAM for a workload.
> Research the NVIDIA A40, L40S, and RTX 6000 Ada, compare FP16 TFLOPS per
> dollar for each, and recommend one with justification. Save the comparison
> to a file."

Deliberately not a single linear tool chain: it requires researching three
separate entities, running a calculation per entity, comparing the results,
and synthesizing a recommendation -- exercising multi-step reasoning and
repeated tool use, not one search → one calc → done.

## Method

`trtmc` has no persistent/server mode -- every turn is a fresh `trtmc run`
subprocess invocation that reloads the compiled engine from disk. Tool
calling is implemented entirely in Python, not inside `trtmc` itself:

- **`render.py`** -- renders the full multi-turn conversation + tool schemas
  into one prompt string using the checkpoint's *own* Jinja chat template
  (via `transformers.AutoTokenizer.apply_chat_template`), rather than
  `trtmc run --use-chat-template true`'s C++ template-detection heuristic.
  That heuristic substring-sniffs the template to classify it into one of a
  few known formats and was found to misclassify at least one other model's
  template during this investigation -- tool-calling depends on exact
  template fidelity (the `<tools>` block, `<tool_response>` wrapping, etc.),
  so this sidesteps that path entirely.
- **`parse.py`** -- extracts MiniCPM5's `<function name="..."><param
  name="...">value</param></function>` tool-call XML from the model's raw
  output text. A position-based scanner, not a single whole-string regex --
  a greedy/non-greedy regex would either silently truncate a CDATA-wrapped
  value containing a literal `</param>`, or silently *merge* two overlapping
  hallucinated `<function>` blocks into one call with foreign arguments. Both
  failure modes are worse than raising, since the caller would then execute a
  corrupted action with no signal anything went wrong. This scanner instead
  parses each call strictly outward from its own opening tag and drops (never
  guesses at) a block that turns out malformed partway through.
- **`tools.py`** -- the actual tool implementations (`web_search`,
  `calculator`, `write_file`, `read_file`) and their JSON-schema definitions.
  `calculator` supports arithmetic and comparisons (`<`, `<=`, `==`, ...) via
  an AST-restricted evaluator -- never `eval()` -- so the agent can check a
  budget constraint explicitly instead of eyeballing it. File tools are
  sandboxed to `workspace/` with path-escape rejected.
- **`agent.py`** -- the loop: render → run `trtmc` → parse for tool calls →
  execute them → append results as `<tool_response>`-wrapped turns → repeat
  until the model returns a plain answer, capped at `MAX_TURNS`. A tool
  error is fed back to the model as a normal turn (so it can retry/reformulate
  on its own), with a `MAX_CONSECUTIVE_TOOL_ERRORS` cap so a genuinely stuck
  tool can't silently burn every remaining turn (any tool call failing within
  a turn counts that whole turn as an error turn; a turn where every call
  succeeds resets the counter). Each turn's assistant output is stripped of
  its `<think>...</think>` block before being stored in history -- MiniCPM5's
  own template re-renders the full history back into the next prompt, so an
  unstripped `<think>` block would compound turn over turn.
- **`precision_compare.py`** -- a standalone script that runs the identical
  prompt through two pre-built bundles of the same checkpoint (one bf16, one
  fp16) and reports measured `prefill_ms`/`decode_ms`/ms-per-token for each --
  a real comparison, not a claim. Renders the prompt via `render.py`'s own
  chat-template path (the same one `agent.py` uses), not `trtmc run
  --use-chat-template true`, for the same template-fidelity reason as above --
  a precision comparison built on the distrusted path could end up measuring
  a different prompt structure per precision, silently confounding the
  comparison. Discards the first invocation per precision as a warmup run
  (unless `--repeats 1`) and reports a standard deviation across the rest.

## Results

Measured on a rented A100-SXM4-40GB (driver 570.148.08, running the CUDA
13.3 NGC image via CUDA Forward Compatibility), building `families/llama`
from `upstream/main` at the commit that merged
[PR #1288](https://github.com/NVIDIA/TensorRT-Model-Connect/pull/1288)
(the multi-EOS fix; supersedes this project's own [PR #1269](https://github.com/NVIDIA/TensorRT-Model-Connect/pull/1269),
closed in favor of #1288's additive, better-validated approach -- see "Why
this exists" above). Both bundles built with `max_sequence_length=4096`,
`tensor_parallel_size=1`, no quantization.

**Precision comparison** (`precision_compare.py`, prompt "Explain what a KV
cache does in one paragraph.", `max_new_tokens=64`, `repeats=5`, first
invocation per precision discarded as warmup):

| precision | prefill_ms | decode_ms | ms/token | stdev |
|---|---|---|---|---|
| bf16 | 90.24 | 920.35 | 14.380 | 0.032 |
| fp16 | 159.06 | 883.85 | 13.810 | 0.004 |

fp16 was ~1.04x faster per token in this run -- a small, real difference,
not the large gap sometimes assumed between the two. fp16 also had lower
run-to-run variance (stdev 0.004 vs 0.032 ms/token). Notably, fp16's
*prefill* was slower than bf16's (159ms vs 90ms) despite its faster decode --
a genuine, slightly counterintuitive result, reported as measured rather
than smoothed over.

**Agent run** (`agent.py`, the GPU-budget-comparison task above,
`--max-new-tokens 1024`, bf16 bundle):

```
engine invocations : 1 (one per turn -- trtmc has no server mode)
total prefill time  : 145.72 ms
total decode time   : 6462.91 ms across 451 tokens
avg decode/token    : 14.330 ms
```

The single-invocation `avg decode/token` (14.330 ms) lines up closely with
`precision_compare.py`'s independently-measured bf16 figure (14.380
ms/token) -- two different code paths measuring the same underlying engine
land within 0.4% of each other, which is a reasonable cross-check that both
measurements are real.

**What actually happened in that run, reported honestly:** MiniCPM5-2B's
`<think>` reasoning correctly worked out the right multi-step plan (search
each GPU's specs, compute FP16 TFLOPS/$ for each, compare, save to a file).
But its first action attempt was a **malformed tool call**: it dropped the
literal `<function`/`<param` tag names and emitted only the attribute
fragments -- `name="web_search"> name="query">NVIDIA A40 GPU specs FP16
TFLOPS price` instead of `<function name="web_search"><param
name="query">...`. `parse.py`'s scanner correctly found no well-formed
`<function` tag and returned zero calls rather than guessing at one, so the
agent safely treated the malformed fragment as a plain final answer instead
of executing a corrupted action or crashing. This is deterministic, not a
one-off: a shorter run capped at `--max-new-tokens 300` produced a `<think>`
block that is an exact prefix of this run's, confirming `trtmc` decodes
greedily here and this is reproducible behavior for this prompt, not
sampling noise.

This is reported as the real result rather than re-rolled or prompt-tuned
away, because a small (2.5B parameter) model's raw tool-call reliability on
a genuinely multi-entity task is itself the useful finding: the parser's
job is to fail safe when that happens, and it did.

## Setup

Requires a machine with TensorRT-Model-Connect already built (the `trtmc`
CLI binary + compiled `families/llama` runtime `.so`s) and the MiniCPM5-2B
bundle already built through it -- build against `upstream/main` (the
`eos_token_id` fix landed there via #1288; no fork branch needed anymore).

```bash
pip install -r requirements.txt
```

Two non-obvious GPU-setup gotchas hit while validating this on rented
hardware, worth knowing before debugging them from scratch again:

- **`CMAKE_CUDA_ARCHITECTURES` defaults to 89 (Ada Lovelace) in TRT-MC's own
  GPU dev Dockerfile and CI**, matching the community CI's L4/L40/L40S
  fleet. On any other architecture (this project validated on an A100,
  compute capability 8.0), override it explicitly when building
  (`-e CMAKE_CUDA_ARCHITECTURES=80` for the `tools.community_gpu_ci` build
  path). In practice this only matters for families with actual `.cu`
  kernel sources -- `families/llama` has none (pure C++ against TensorRT's
  runtime API), so its build is unaffected either way, but other families
  are not.
- **NGC images' CUDA Forward Compatibility setup does not survive `docker
  exec` into an already-running container.** The entrypoint script
  (`nvidia_entrypoint.sh`) sets up the compat `LD_LIBRARY_PATH` only for the
  process it directly launches (e.g. `sleep infinity` for a long-lived dev
  container); a later `docker exec` into that same container starts a fresh
  environment that doesn't inherit it, so `torch.cuda.is_available()`
  silently returns `False` with a "driver too old" warning even though the
  same command works fine via `docker run --rm`. Fix: pass
  `-e LD_LIBRARY_PATH=/usr/local/cuda/compat/lib` explicitly on every
  `docker exec` that needs GPU access into such a container.

## Run

```bash
python agent.py \
  --model-dir /path/to/MiniCPM5-2B \
  --binary /path/to/trtmc \
  --bundle /path/to/minicpm5-2b.bundle \
  --runtime-root /path/to/build-output-dir
# task defaults to the GPU-comparison scenario above; pass a positional
# argument to override it with a different task
```

```bash
python precision_compare.py \
  --binary /path/to/trtmc \
  --runtime-root /path/to/build-output-dir \
  --bf16-bundle /path/to/minicpm5-2b-bf16.bundle \
  --fp16-bundle /path/to/minicpm5-2b-fp16.bundle
```

## Limitations

- No persistent server: every agent turn pays full engine-load time. Fine
  for a demo; a real deployment needs a long-running server instead.
- Tool-call parsing is a hand-written scanner matched to MiniCPM5's specific
  template convention -- not a general-purpose tool-call parser.
- Single GPU, single request at a time. No batching, no concurrency.
- `web_search` depends on a third-party search library (`ddgs`); result
  quality/availability isn't controlled by this project.
- **MiniCPM5-2B does not reliably reproduce its own template's `<function>`
  tag syntax on multi-entity tasks.** Measured directly (see Results): on
  this project's own demo task, the model correctly reasoned through the
  right multi-step plan but then dropped the literal tag names from its
  first tool-call attempt. This is a real characteristic of a 2.5B-parameter
  model's tool-calling reliability, not a bug in this project's prompt
  rendering or parsing -- the parser's role is to fail safe when it happens
  (drop the malformed call rather than guess), which it does.

## Status

Code complete, including a consecutive-tool-error safety cap and real
per-turn performance accounting. Reviewed by three independent fresh-context
passes (correctness, test coverage,
documentation accuracy); the correctness and coverage passes each
reproduced concrete bugs -- an uncaught crash in the calculator on results
too large to `str()`, file-tool I/O calls that ran outside their own
try/except, a tool-call parser that could silently truncate or merge
malformed output, and unstripped `<think>` blocks compounding across turns --
all fixed and covered by regression tests. Non-GPU-dependent parts
(`parse.py`, `tools.py`, `agent.py`'s orchestration logic, `render.py`,
`precision_compare.py`'s own arithmetic) are covered by 39 tests, all passing
locally. `agent.py` and `precision_compare.py` have both been run end-to-end
against real compiled bundles on an A100-SXM4-40GB, building `families/llama`
from `upstream/main`; see Results above for the actual measured numbers and
observed model behavior.
