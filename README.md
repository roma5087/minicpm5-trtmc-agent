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
> Research the NVIDIA A40, L40S, and RTX 6000 Ada online for background, then
> use these figures for your comparison: dense FP16 TFLOPS are 149.7 for the
> A40, 183 for the L40S, and 91.1 for the RTX 6000 Ada; all three have 48GB
> VRAM. Using approximate list prices of $5,500 for the A40, $8,600 for the
> L40S, and $6,800 for the RTX 6000 Ada, use the calculator to compute FP16
> TFLOPS per dollar for each. Then use the calculator again to directly check
> which ratio is greatest (e.g. a '>' comparison between two of the ratios at
> a time) rather than judging by eye, since the highest ratio is the one to
> recommend. Save the comparison to a file, including the recommendation and
> why."

(Verbatim `DEFAULT_TASK` in `agent.py` -- not paraphrased.)

Deliberately not a single linear tool chain: it requires researching three
separate entities, running a calculation per entity, comparing the results,
and synthesizing a recommendation -- exercising multi-step reasoning and
repeated tool use, not one search → one calc → done. The reference TFLOPS,
VRAM, and price figures are supplied directly rather than left for the agent
to search for -- see Results below for why: earlier runs showed a small
model can spiral indefinitely when the only path to an answer is resolving
ambiguous/conflicting numbers pulled from noisy search snippets. Live search
is still exercised (for background context on each card), just not as the
sole source of the numbers the recommendation depends on.

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
  unstripped `<think>` block would compound turn over turn. Every turn's raw
  output is decoded from `trtmc`'s own `token_ids` via the real HF tokenizer,
  **not** `trtmc`'s own `text` field -- see Results below for the bug this
  works around. An optional one-shot example (`ONE_SHOT_EXAMPLE`, on by
  default; `--no-example` to disable) shows one full tool-call round trip in
  the exact expected syntax, a standard technique for small-model tool-call
  reliability.
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

**Agent run, final working state** (`agent.py`, the task above,
`--max-new-tokens 1024`, bf16 bundle):

```
engine invocations : 4 (one per turn -- trtmc has no server mode)
total prefill time  : 616.96 ms
total decode time   : 15860.76 ms across 1108 tokens
avg decode/token    : 14.315 ms
```

`avg decode/token` (14.315 ms) again lines up with `precision_compare.py`'s
independently-measured bf16 figure (14.380 ms/token, ~0.5% apart) -- a
repeated cross-check that the accounting is real. The agent completed all
four turns for real: three parallel `calculator` calls to compute FP16
TFLOPS/$ for each GPU, two explicit `>` comparisons to determine the
highest ratio, a `write_file` call, and a final answer -- correctly
recommending the NVIDIA A40 (0.0272 TFLOPS/$, the true highest of the
three) with correct justification, verified against the real file it wrote:

```
FP16 TFLOPS per Dollar Comparison:
- NVIDIA A40: 0.0272 TFLOPS/$
- NVIDIA L40S: 0.0201 TFLOPS/$
- RTX 6K Ada: 0.0134 TFLOPS/$

Recommendation: NVIDIA A40
Reason: The A40 has the highest FP16 TFLOPS per dollar ratio (0.0272),
making it the most efficient choice for workloads prioritizing GPU
performance per unit cost.
```

(One residual, minor, and honestly-reported imperfection: it transcribed
the L40S ratio into that file as 0.0201 rather than its own correctly
calculated 0.0213 -- a cosmetic slip on a non-winning number that doesn't
affect the recommendation, which is backed by the actual `>` comparison
tool calls, not eyeballed.)

**Getting here took two real, diagnosed fixes, not prompt-tuning a lucky
roll -- both are the actual engineering content of this project:**

1. **A genuine bug in `trtmc`'s native detokenizer, not a MiniCPM5-2B
   reliability problem.** Earlier runs showed the model reliably producing
   *malformed* tool calls -- dropping the literal `<function`/`<param` tag
   names, keeping only `name="..."` attribute fragments (e.g.
   `name="web_search"> name="query">...`). This looked like a small-model
   tool-calling limitation. It wasn't: comparing `trtmc`'s own `"text"`
   field against `tokenizer.decode()` of the exact same `token_ids` proved
   the model was correctly generating the special tokens for
   `<function`/`<param`/`</param>`/`</function>` the whole time --
   `trtmc`'s own text-rendering path was silently dropping them. Confirmed
   with both bf16 and fp16 bundles (identical failure, ruling out a
   precision/quantization cause) and with a trivial single-tool prompt
   (ruling out "long reasoning corrupts the format"). The fix required no
   changes to TensorRT-Model-Connect itself: `agent.py` now decodes
   `token_ids` directly via the real HF tokenizer instead of trusting
   `trtmc`'s `text` field, which immediately unblocked well-formed tool
   calls.
2. **A real synthesis-reliability gap once tool calls were unblocked.**
   With correct tool-call syntax, the agent reliably executed real
   `web_search`/`calculator` calls and got correct individual numbers -- but
   the model twice failed to correctly identify the *largest* of three
   computed ratios by eye (once concluding a demonstrably smaller number was
   "highest"). The fix: the task instructs the model to also use the
   `calculator` tool's comparison support (`>`) to check pairwise which
   ratio is greatest, rather than trust its own mental comparison of three
   numbers -- offloading the specific step it was getting wrong to the same
   AST-restricted tool already used for arithmetic. This produced the
   correct recommendation, backed by verified tool calls rather than
   coincidence.

Both issues were real, reproduced, and fixed with real GPU runs at each
step -- not asserted or smoothed over. The failed runs are preserved above
in spirit (see git history) rather than deleted, because a diagnosed and
fixed bug is a stronger result than a demo that happened to work on the
first try.

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
  --model-dir /path/to/MiniCPM5-2B \
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
- **`trtmc`'s native runtime silently drops certain special tokens
  (`<function`, `<param`, `</param>`, `</function>`, `<|im_end|>`) when
  detokenizing.** Diagnosed and worked around at the `agent.py` level (see
  Results) by decoding `token_ids` via the real HF tokenizer instead of
  trusting `trtmc`'s own `text` field. Not fixed inside TensorRT-Model-Connect
  itself -- that would need a patch to `families/llama/runtime/bpe_tokenizer.cpp`
  and is out of this project's scope, though it's a well-diagnosed, reproducible
  finding (token-id-level proof included) that could become one.
- **Reference numbers (TFLOPS, VRAM, prices) are supplied in the task rather
  than sourced entirely by the agent's own web searches.** Earlier runs
  showed a small model can spiral indefinitely trying to resolve ambiguous
  or conflicting numbers pulled from noisy search snippets (e.g. a
  datasheet listing two different TFLOPS figures side by side without
  labeling which is which). Live search is still used for background
  context; the numbers the final recommendation depends on are given
  directly to keep the demo's success bounded by the model's tool-use and
  synthesis reliability, not by search-result quality.
- **The model cannot reliably compare multiple computed numbers by
  eye.** Measured directly: given three independently-correct calculator
  results, it twice concluded a non-maximal one was the largest. Worked
  around by instructing it to use the calculator's own comparison support
  (`>`) to check pairwise rather than judge visually -- a real, reproducible
  small-model limitation in numeric synthesis, not in tool execution.
- **`web_search` results are neutralized against known marker substrings
  (`<|im_end|>`, `<function`, CDATA delimiters, etc.) before being stored
  in conversation history, but this is pattern-based, not a formal
  guarantee.** A security review confirmed the underlying mechanism a
  poisoned search result could exploit -- HF fast tokenizers match
  registered special tokens as substrings anywhere in text, not just where
  a template placed them -- but did not have GPU access to validate whether
  MiniCPM5-2B itself is actually steerable this way end-to-end. Treat the
  current mitigation as a real fix for the mechanism, not a proven bound on
  what a sufficiently adversarial search result could still attempt.

## Status

**The agent completes its full task end-to-end on real GPU hardware and
gives a correct, verified recommendation** -- see Results above for the full
run and the two real bugs (one in `trtmc`, one in the model's numeric
synthesis) that had to be found and fixed to get there. Code has been
through two independent multi-reviewer rounds (three fresh-context passes
each, acting as NVIDIA senior AI SWEs). Round one (correctness, test
coverage, documentation accuracy) reproduced concrete bugs -- an uncaught
crash in the calculator on results too large to `str()`, file-tool I/O calls
that ran outside their own try/except, a tool-call parser that could
silently truncate or merge malformed output, and unstripped `<think>` blocks
compounding across turns. Round two, after the `trtmc`-detokenizer fix
landed, found: no error handling around the per-turn `trtmc` call/decode
path (an infrastructure failure -- a crash, timeout, or malformed payload --
would have taken down the whole process uncaught); a real contract gap
between `tools.py`'s calculator schema (which advertises unwrapped `<=`/`<`
comparisons) and `parse.py`'s CDATA requirement for any value containing a
literal `<` (an unwrapped comparison was silently dropped, not evaluated);
a fragile string-suffix match for the `<|im_end|>` turn marker instead of a
token-id-level strip; an unbounded-exponent cost in the calculator evaluated
before the existing result-size guard ever ran; and a shared-by-reference
one-shot example list. All fixed (verified against real GPU hardware after
the fix, not just locally) and covered by regression tests; also caught and
fixed in this round: a stale `precision_compare.py` example command missing
a required flag, and a quoted task string that had silently drifted from
the actual code.

**Round three** (correctness re-verification, a mutation-testing-style test
audit, and a dedicated security pass) found and fixed further real issues:
the round-two `trtmc` error handling still didn't catch a wrong-*shaped*
payload (valid JSON that's the wrong type, or a `token_ids` field that
isn't a list -- both crashed uncaught with a `TypeError`); the exponent
guard bounded only the exponent, not the base, so a small, allowed exponent
on an enormous base (or a chain of nested `**` calls, each individually
within the cap) was just as expensive to compute as the huge-exponent case
it was written to stop -- replaced with a guard that estimates the actual
result size; `web_search` results were fed into the next turn's prompt
completely unsanitized, and HF fast tokenizers match registered special
tokens as substrings anywhere in text, so a poisoned search result could in
principle inject fake turn boundaries or fake tool-call XML -- `web_search`
now neutralizes those marker substrings before they ever reach `messages`;
an oversized `write_file` (read back via `read_file` into a later prompt)
could grow a rendered prompt past the OS's `ARG_MAX`, crashing the
subprocess call with an uncaught `OSError` -- `write_file` now caps content
size, and `agent.py` catches `OSError` too. The path-traversal sandboxing
was independently re-attacked (symlinks, absolute-path joins, percent- and
Unicode-encoded traversal, null bytes, oversized filenames) and held in
every case; `subprocess` shell-injection was independently re-attempted and
confirmed not exploitable (list-form `subprocess.run`, no `shell=True`).
Re-verified against real GPU hardware after every fix in this round too --
byte-identical correct output to the pre-fix run, confirming no regression.
Non-GPU-dependent parts (`parse.py`, `tools.py`, `agent.py`'s orchestration
logic, `render.py`, `precision_compare.py`'s own arithmetic) are covered by
84 tests, all passing locally -- up from 56, including tests added via a
mutation-testing pass (deliberately introducing small real bugs one at a
time and confirming the suite actually catches each one, not just that it
currently passes).
