# MiniCPM5-2B Tool-Calling Agent, served via TensorRT-Model-Connect

A tool-calling agent where [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)
is the reasoning core, compiled and served through NVIDIA's
[TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect)
(`trtmc`) native runtime — not vLLM/SGLang's built-in tool-call support.

## Why this exists

Grew out of investigating and fixing a real `families/llama` bug in
TensorRT-Model-Connect ([PR #1269](https://github.com/NVIDIA/TensorRT-Model-Connect/pull/1269))
that surfaced while validating MiniCPM5-2B against that framework. This
project puts the fixed build path to actual use, and measures what it
actually costs and delivers rather than just claiming it works.

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

`trtmc`'s own JSON output already reports `setup_ms`/`prefill_ms`/`decode_ms`
per invocation. `agent.py` accumulates these across the run instead of
discarding them and prints a summary alongside the final answer.

*Pending a live GPU session -- this section will be replaced with the actual
measured numbers from a real run (engine invocation count, cumulative
prefill/decode time, ms/token, and the bf16-vs-fp16 comparison table) once
captured. No fabricated numbers belong here in the meantime.*

## Setup

Requires a machine with TensorRT-Model-Connect already built (the `trtmc`
CLI binary + compiled `families/llama` runtime `.so`s) and the MiniCPM5-2B
bundle already built through it -- see the parent repo's own build docs.

```bash
pip install -r requirements.txt
```

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
- Tool-call parsing is XML-regex-based, matched to MiniCPM5's specific
  template convention -- not a general-purpose tool-call parser.
- Single GPU, single request at a time. No batching, no concurrency.
- `web_search` depends on a third-party search library (`ddgs`); result
  quality/availability isn't controlled by this project.

## Status

Code complete, including a consecutive-tool-error safety cap and real
per-turn performance accounting. Reviewed by three independent fresh-context
passes acting as NVIDIA senior AI SWEs (correctness, test coverage,
portfolio/interview readiness); the correctness and coverage passes each
reproduced concrete bugs -- an uncaught crash in the calculator on results
too large to `str()`, file-tool I/O calls that ran outside their own
try/except, a tool-call parser that could silently truncate or merge
malformed output, and unstripped `<think>` blocks compounding across turns --
all fixed and covered by regression tests. Non-GPU-dependent parts
(`parse.py`, `tools.py`, `agent.py`'s orchestration logic, `render.py`,
`precision_compare.py`'s own arithmetic) are covered by 39 tests, all passing
locally. Full agent-loop and `precision_compare.py` runs against a real
compiled bundle are pending a GPU session.
