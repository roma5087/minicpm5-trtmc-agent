# MiniCPM5-2B tool-calling agent on TensorRT-Model-Connect

A tool-calling agent whose reasoning core is
[MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B), compiled and served
through NVIDIA's
[TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect)
(`trtmc`) native runtime. Tool calling is implemented in Python around the
runtime, not with an inference server's built-in tool-call support.

It started as a way to exercise the multi-EOS fix in `families/llama` (HF
configs may declare `eos_token_id` as a list; MiniCPM5-2B does). That fix
landed upstream as
[#1288](https://github.com/NVIDIA/TensorRT-Model-Connect/pull/1288), which
superseded an earlier PR of mine
([#1269](https://github.com/NVIDIA/TensorRT-Model-Connect/pull/1269), closed).

## How it works

`trtmc` has no server mode: every turn is a fresh `trtmc run` subprocess that
reloads the engine from disk.

| file | role |
|---|---|
| `render.py` | Renders the whole conversation plus tool schemas with the checkpoint's own Jinja chat template (via `transformers`) and passes the finished text to `trtmc run --use-chat-template false`, instead of relying on the runtime's template detection. |
| `parse.py` | Position-based scanner for MiniCPM5's `<function name=…><param name=…>…</param></function>` calls. A block that is malformed part-way is dropped, never guessed at; a whole-string regex would silently truncate a CDATA value containing `</param>` or merge two overlapping calls. |
| `tools.py` | `web_search` (ddgs), `calculator`, `write_file`, `read_file`. The calculator is an AST-restricted evaluator (never `eval`) supporting arithmetic and single comparisons, with every integer intermediate capped at 4,096 bits. File tools are confined to `workspace/`. |
| `engine.py` | One `trtmc run` call: argv, timeout, JSON parsing. Failures raise `EngineError` carrying the exit status and stderr tail, not the command line (which embeds the whole prompt). |
| `agent.py` | The loop: render, run, parse, execute tools, append `<tool_response>` turns, repeat. |
| `precision_compare.py` | Runs one prompt through two prebuilt bundles (bf16, fp16) of the same checkpoint and reports `prefill_ms`, `decode_ms` and ms/token. |

Behaviours of the loop worth knowing:

- **Output is decoded from `token_ids` with the real HF tokenizer**, not from
  `trtmc`'s `text` field (see the finding below).
- **Only text outside `<think>…</think>` counts.** Tool calls inside reasoning
  are not executed, and reasoning is not stored in history (the template
  re-renders history every turn). An unclosed `<think>` is discarded.
- **Truncation is an error.** Output that reaches `--max-new-tokens` without an
  end marker returns an error instead of being treated as an answer.
- **A call the parser rejects is reported back to the model**, not returned as
  the final answer.
- **Tool arguments are checked against the tool's schema** before the call.
- **Every tool result is sanitized where it enters the prompt.** `trtmc`'s
  encoder matches every added token as a substring anywhere in the prompt, so
  the token list is read from the tokenizer and each match is broken with a
  zero-width space; NUL bytes and lone surrogates are removed.
- **The prompt is length-checked** against `--max-sequence-length` (default
  4096, must equal the bundle's build setting) before `trtmc` is invoked.
- Tool errors are fed back so the model can retry; three consecutive error
  turns stop the run.
- An optional one-shot example of a full tool-call round trip is on by default
  (`--no-example` to disable).

## The trtmc findings

Two separate, real issues were found in `trtmc`'s own tokenizer code this
project depends on -- one on the decode side (found first, worked around from
the start), one on the encode side (found later, on 2026-09-22, not yet
worked around). They are independent: fixing one says nothing about the
other.

### Decode: special tokens are silently dropped from `text`

Early runs produced malformed tool calls: the literal `<function` / `<param`
tags were missing from the text, leaving only `name="…"` fragments. Comparing
`trtmc`'s `text` field with `tokenizer.decode()` of the same `token_ids` showed
the model had emitted the tag tokens all along; they were absent from `text`.

The cause is in `families/llama/runtime/bpe_tokenizer.cpp`: decode skips every
id whose `tokenizer.json` entry has `"special": true`, and that flag is
described in the source as controlling "decode filtering". MiniCPM5 flags
`<function`, `</function>`, `<param`, `</param>` (ids 18–21) and `<|im_end|>`
(130073) as special; `<think>` and `</think>` (8, 9) are not flagged, which is
why reasoning survives in `text` and tool calls do not. Encode, by contrast,
matches all added tokens, special or not, as substrings (see below).

So this is a design default with no opt-out (the equivalent of
`skip_special_tokens=True`), not corruption. For a model whose tool-call tags
are special tokens, `token_ids` is the interface to use. `agent.py` does that
and needs no change to TensorRT-Model-Connect. It is not filed upstream.

### Encode: `trtmc`'s BPE encoder under-merges relative to the real tokenizer

`agent.py` renders the prompt with HF's own tokenizer and hands `trtmc` the
finished text via `--prompt`, on the assumption that `trtmc`'s own encoder
then tokenizes that text the same way HF would. That assumption was checked
directly on 2026-09-22 and is false.

**Method:** a one-line debug print was added locally to `encode()` in
`families/llama/runtime/bpe_tokenizer.cpp` (env-var gated, dumping the
returned id vector to stderr as JSON), and just `trtmc_model_llama` was
rebuilt with it. This is instance-local instrumentation, never committed or
pushed anywhere -- it exists only to answer the question, not as a proposed
change. The real 826-token first-turn prompt from `DEFAULT_TASK` (rendered by
`render.py`, exactly what `agent.py` sends) was run through both `trtmc run
--prompt "$PROMPT"` (with the debug env var set) and, separately,
`tokenizer(prompt, add_special_tokens=False).input_ids` in Python, and the two
id sequences were aligned with `difflib.SequenceMatcher`.

**Result:** HF encodes the prompt to 826 ids; `trtmc` encodes the identical
text to 851. The alignment shows two distinct problems:

1. **A genuine double leading BOS.** HF: `[0, 130072, ...]`. `trtmc`: `[0, 0,
   130072, ...]`. The rendered prompt already contains a literal `<s>` from
   the template's own `{{- bos_token }}`; `trtmc`'s encoder matches that
   substring as its own token *and* separately prepends a BOS by default,
   producing two.
2. **31 places where HF merges two characters into one vocab token and
   `trtmc` does not**, scattered through the whole prompt, not clustered
   anywhere in particular. Representative examples (decoded):

   | text | HF (1 token) | trtmc (2 tokens) |
   |---|---|---|
   | `.` + newline | `350` | `35` (`.`) + `220` (newline) |
   | `:` + newline | `990` | `47` (`:`) + `220` |
   | `_search` | `72903` | `84` (`_`) + `14875` (`search`) |
   | `.g` | `2587` | `35` (`.`) + `92` (`g`) |
   | `-name` | `35768` | `34` (`-`) + `2075` (`name`) |
   | `}}}` + newline | `113927` | `16570` (`}}}`) + `220` |

   Two of the 31 mismatches are a different shape: HF and `trtmc` both use two
   tokens for the same text (e.g. `-wrapped`, `6000`), just split at a
   different point (`3248`+`39996` vs `34`+`112403` for `-wrapped`). Not a
   clean merge loss like the six above, but still a real divergence in what
   ids the model actually receives.

`trtmc`'s BPE encoder is not just mishandling special tokens (the decode
finding above) -- it is systematically **under-merging plain text** relative
to the checkpoint's real tokenizer. This means every prompt this project has
ever sent to `trtmc`, on every turn, differs from what MiniCPM5-2B's own
tokenizer would have produced for the same text. The root cause has not been
diagnosed (a pre-tokenization split-boundary mismatch is consistent with the
pattern -- every example above is a merge across what looks like a
regex-driven pretokenizer split point -- but this is not confirmed by reading
the encoder's pretokenization code, only inferred from the symptom). Not
fixed here, not filed upstream. Whether it measurably changes model behavior
(versus being a difference TensorRT-Model-Connect's own detokenization of the
*output* happens not to expose) is also unverified -- the agent still
completes its task correctly despite it, in every run so far.

## Results

Two runs exist, on different GPUs, and are reported separately rather than
merged, since the hardware and the code both differ between them.

**Current run (2026-09-22), post-hardening code, A100-SXM4-80GB** (driver
580.126.09, CUDA 13.3), `families/llama` built from `upstream/main` at the
commit that merged #1288, both bundles `max_sequence_length=4096`,
`tensor_parallel_size=1`, no quantization.

`precision_compare.py` ("Explain what a KV cache does in one paragraph.",
`max_new_tokens=64`, 5 runs after one discarded warmup):

| precision | prefill_ms | decode_ms | ms/token | stdev |
|---|---|---|---|---|
| bf16 | 69.52 | 880.56 | 13.759 | 0.113 |
| fp16 | 91.02 | 843.85 | 13.185 | 0.046 |

fp16 ~1.04x faster per token, the same ratio as the original 40GB run below,
on different hardware. Both precisions generated exactly 64 tokens per run (no
token-count confound). The caveats below still apply: this is two separate
engine builds run in sequence, not shown to isolate a precision effect, and
`prefill_ms` on a ~15-token prompt in a fresh process is dominated by
start-up cost, not prefill compute (see the note the script itself prints).

Agent run (`--max-new-tokens 1024`, bf16), full output:

```
engine invocations : 4 (one per turn -- trtmc has no server mode)
total setup time   : 0.00 ms
total prefill time : 594.37 ms
total decode time  : 16342.30 ms across 1191 tokens
avg decode/token   : 13.721 ms
total wall time    : 48441.17 ms
outside prefill/decode : 31504.51 ms (process start, engine load, JSON)
```

**This answers the previously-open question about per-turn reload cost: it is
the dominant cost.** 31.5 of 48.4 seconds of wall time (65%) was spent outside
prefill and decode -- process start, engine deserialization, CUDA context
setup -- not model compute. The per-turn-subprocess architecture, not the
model, is what a real deployment would need to fix first.

Four turns, no crashes, no malformed call, no fallback to the max-turns or
consecutive-error paths: three parallel `calculator` calls, two `>`
comparisons (the model CDATA-wrapped both, which parse.py accepts even though
only `<`/`<=` require it), a `write_file`, then a final answer correctly
recommending the A40. As before, the model made no `web_search` calls despite
the task asking it to research the three cards online first -- `web_search`
and its sanitization remain unexercised on real GPU hardware, in either run.
The saved file has its own new transcription glitch: `calculator` returned
the L40S ratio as the correct `0.02127906976744186`, and the model wrote it to
the file as `"0.02 1279069767"` -- a stray space inserted mid-digit-string
when composing the file content, not a tool or parser fault (the tool result
itself was correct). Same family of issue as the original run's slip, still
cosmetic, still doesn't touch the recommendation. It is one more data point
that this model's transcription of a correct number, not its arithmetic or
tool use, is the least reliable part of the pipeline.

Not done on this run: comparing the model's own budget reasoning (it never
explicitly checked the $10,000 figure against any price, though all three now
fit). Build-to-build variance was checked as a separate follow-up -- see
"Further checks" below.

**Original run (2026-09-16), pre-hardening code, A100-SXM4-40GB** (driver
570.148.08, CUDA 13.3 NGC image via forward compatibility). Kept for context;
not merged with the numbers above because the code, GPU and task text all
differ.

| precision | prefill_ms | decode_ms | ms/token | stdev |
|---|---|---|---|---|
| bf16 | 90.24 | 920.35 | 14.380 | 0.032 |
| fp16 | 159.06 | 883.85 | 13.810 | 0.004 |

```
engine invocations : 4
total prefill time : 616.96 ms
total decode time  : 15860.76 ms across 1108 tokens
avg decode/token   : 14.315 ms
```

This run's task had an unsatisfiable budget ($5,000 against three cards priced
at $5,500/$8,600/$6,800); the agent recommended the A40 without flagging that
it exceeded the budget. `DEFAULT_TASK` was changed to $10,000 afterward, which
the current run above used. This run also made no `web_search` calls, and its
saved file had its own transcription slip (0.0201 instead of the computed
0.0213).

Across both runs: the throughput figures in the task are supplied, not
sourced or checked against datasheets, and are labeled "dense FP16 TFLOPS" in
the task text without confirming all three cards are being compared on the
same execution path -- the recommendation is only as good as those inputs.

## Further checks (2026-09-22)

Three more checks were run on the same A100-SXM4-80GB, following up on items
the Results and Limitations sections above had left open. (The fourth
follow-up, encode-side parity, produced the encoder finding above and is
documented there, not here.)

**Build-to-build variance, to check whether the ~4% bf16/fp16 gap is a real
precision effect.** Each precision was built a second time from the same
checkpoint and compared against its own first build:

| comparison | ms/token (v1) | ms/token (v2) | ratio |
|---|---|---|---|
| bf16 vs bf16 | 13.718 | 13.879 | 1.01x |
| fp16 vs fp16 | 13.245 | 13.147 | 1.01x |

Same-precision build variance is ~1%, well under the ~4% bf16/fp16 gap seen
on both GPUs in the Results above. That is evidence the precision gap is a
real effect, not two engine builds that happen to differ -- though it is
still only two builds per precision, not a distribution.

**An HF baseline** (not vLLM/SGLang -- scoped down to a plain
`transformers.generate()` call for cost and time; the vLLM/SGLang comparison
is still not done): bf16, greedy, the same prompt and GPU as
`precision_compare.py`, prefill timed separately from decode the same way
`trtmc` reports the two:

| | ms/token (decode-only) |
|---|---|
| HF `generate()`, eager, bf16, greedy | 19.150 |
| `trtmc`, this session's runs | 13.147 - 13.879 |

`trtmc` is meaningfully faster than naive HF eager mode (roughly 1.4x), which
is a real result this project didn't have before. It does not answer whether
13-14 ms/token is *good* -- that needs the memory-bandwidth-floor estimate and
profiling the Limitations section below still asks for, and neither HF eager
nor `trtmc` here uses CUDA graphs or a KV-cache-optimized serving stack, so
neither number is a ceiling on what's achievable.

**The native `tool`-role format, read directly from
`chat_template.jinja`:** `role: "tool"` messages merge consecutive results
into one `<|im_start|>user...<|im_end|>` block, with `\n` around each
`<tool_response>...</tool_response>`, only opening/closing that block at a
run of consecutive tool messages. The shipped code's `role: "user"` +
manually-wrapped `<tool_response>` produces one such block per call instead.
A 2-line variant using `role: "tool"` (not shipped; instance-local only) ran
the real default task once: completed correctly in 4 turns, same A40
recommendation, no crash, no malformed call. Turn-by-turn prompt token counts
were close to the shipped format's (within a few dozen tokens either way,
confounded by the model's own output length differing turn to turn once the
input format changes). One run each is not enough to say whether either
format is more reliable -- it says only that the native format also works.

## Task

`DEFAULT_TASK` in `agent.py` asks for a three-GPU comparison under a budget:
compute FP16 TFLOPS per dollar with the calculator, check which ratio is
greatest with explicit `>` comparisons (a small model misjudged the largest of
three numbers by eye in earlier runs), and save the recommendation to a file.
Figures and prices are supplied so the outcome does not hinge on noisy search
snippets.

## Limitations and open questions

- **A plain HF eager baseline exists (see Further checks); vLLM/SGLang do
  not.** `trtmc` is ~1.4x faster than naive HF `generate()`, but nothing here
  compares against a real serving stack, so the numbers still say little about
  how competitive the runtime actually is.
- **The ~13-14 ms/token decode figure (both GPUs, both runs) has not been
  explained.** It has not been compared with the memory-bandwidth floor for
  this model's weights on either GPU, and nothing has been profiled, so it is
  unknown whether the time is in the engine or in `trtmc`'s decode loop.
  `decode_ms / len(token_ids)` may also be off by one if the first token comes
  from the prefill pass.
- **Per-turn reload cost is measured and is the dominant cost** (see Results):
  65% of one run's wall time was outside prefill/decode. A persistent-server
  runtime, not available for `trtmc` today, would be the fix.
- **Encode-side parity is checked and broken** (see "The trtmc findings"
  above): `trtmc`'s encoder produces 851 ids for a prompt HF encodes to 826,
  including a double leading BOS and 31 places where a merge HF applies is
  missing. Root cause not diagnosed, not fixed, not filed upstream. Whether it
  changes model behavior, not just token count, is unverified.
- **Tool results are formatted as one `user` turn per call**
  (`<tool_response>…</tool_response>`), confirmed to differ from the
  template's own `tool` role (see Further checks); a single run with the
  native format also worked, but one run each does not show which is more
  reliable.
- Sampling parameters are not passed to `trtmc`; whether it decodes greedily
  by default is not documented here.
- Tool-call parsing is matched to MiniCPM5's template, not general.
- One GPU, one request at a time, no batching.
- `web_search` depends on `ddgs`; result quality is outside this project.
- Sanitization is pattern-based and validated only against HF's tokenizer
  behaviour and `trtmc`'s source, not end to end on a GPU.
- `precision_compare.py` runs all bf16 invocations before all fp16 ones rather
  than interleaving them. Each precision has now been built twice (see Further
  checks): same-precision variance is ~1%, well under the ~4% bf16/fp16 gap,
  which supports the gap being a real precision effect -- but it is still only
  two builds per precision, not a distribution.
- The model has never used the calculator to check the task's own budget
  figure against a price; the current task's budget happens to be satisfied
  by all three cards, so this has not yet mattered to the answer.

## Setup

Requires a machine with TensorRT-Model-Connect built (the `trtmc` binary and
the `families/llama` runtime `.so`s) and a MiniCPM5-2B bundle built through it,
against `upstream/main` (which contains #1288).

```bash
pip install -r requirements.txt        # runtime
pip install -r requirements-dev.txt    # plus pytest
```

TRT-MC's own documented path
([`source-build.md`](https://nvidia.github.io/TensorRT-Model-Connect/getting-started/source-build))
worked as written, verified 2026-09-22 on an A100 (compute capability 8.0):
build `Dockerfile.dev.x86`, run it with `--gpus`, then inside the container:

```bash
cmake -S . -B build-sm80 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80-real \
  -DTRTMC_BUILD_BACKEND_RTX=OFF -DTRTMC_BUILD_TESTS=OFF -DTRTMC_BUILD_EXAMPLES=OFF
cmake --build build-sm80 --parallel "$(nproc)" \
  --target trtmc trtmc_backend_trt trtmc_model_llama
```

(`-real` after the SM number, and no build for families this project doesn't
use.) Then build both bundles:

```bash
export PYTHONPATH=core/builder:apps/benchmark:$PWD  # see gotcha below
python -m tensorrt_model_connect build openbmb/MiniCPM5-2B \
  --precision bf16 --max-sequence-length 4096 --output minicpm5-2b-bf16.bundle
python -m tensorrt_model_connect build openbmb/MiniCPM5-2B \
  --precision fp16 --max-sequence-length 4096 --output minicpm5-2b-fp16.bundle
```

Gotchas hit on rented hardware:

- **`source-build.md`'s own recommended
  `pip install --no-deps -e . -C py-only=true` is currently broken upstream**
  (a Conan build error, regardless of that flag, as of 2026-09-22). Worked
  around by skipping the pip install and setting `PYTHONPATH` directly to
  `core/builder`, `apps/benchmark`, and the repo root instead -- the same
  approach TRT-MC's own `tools/community_gpu_ci.py` uses internally. Only the
  Python build CLI needs this; the native `trtmc`/`trtmc_backend_trt`/
  `trtmc_model_llama` build above is unaffected.
- `CMAKE_CUDA_ARCHITECTURES` defaults to 89 (Ada) in TRT-MC's GPU dev
  Dockerfile and CI, matching the community CI's L4/L40/L40S fleet. The
  `source-build.md` path above derives the right value automatically from
  `nvidia-smi --query-gpu=compute_cap`; only the separate
  `tools.community_gpu_ci` CI path needs a manual
  `-e CMAKE_CUDA_ARCHITECTURES=80` override on non-Ada hardware. Moot for
  `families/llama` either way, since it has no `.cu` sources.
- NGC images set up CUDA Forward Compatibility only for the entrypoint's own
  process, so a later `docker exec` sees `torch.cuda.is_available() == False`.
  Pass `-e LD_LIBRARY_PATH=/usr/local/cuda/compat/lib` on each `docker exec`
  that needs the GPU. Not reproduced during the 2026-09-22 verification (which
  did `docker exec` into a long-lived container repeatedly and never saw
  `torch.cuda.is_available() == False` from this cause) -- possibly specific
  to an older image/driver combination than the one used here; listed as a
  known gotcha, not confirmed against this exact setup.

## Run

```bash
python agent.py \
  --model-dir /path/to/MiniCPM5-2B \
  --binary /path/to/trtmc \
  --bundle /path/to/minicpm5-2b.bundle \
  --runtime-root /path/to/build-output-dir
# a positional argument overrides the default task;
# --max-sequence-length must match the bundle's build setting (default 4096)
```

```bash
python precision_compare.py \
  --model-dir /path/to/MiniCPM5-2B \
  --binary /path/to/trtmc \
  --runtime-root /path/to/build-output-dir \
  --bf16-bundle /path/to/minicpm5-2b-bf16.bundle \
  --fp16-bundle /path/to/minicpm5-2b-fp16.bundle
```

## Tests

```bash
python -m pytest -q
```

The suite needs no GPU. It covers the parser, tools, loop control flow, and the
`trtmc` subprocess boundary (against a fake `trtmc` script, including exact
argv, timeout and malformed output). The tokenizer is stubbed, so the real chat
template and real special-token decoding are not exercised by the tests.
