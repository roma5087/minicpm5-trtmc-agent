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

## The trtmc finding

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
matches all added tokens, special or not, as substrings.

So this is a design default with no opt-out (the equivalent of
`skip_special_tokens=True`), not corruption. For a model whose tool-call tags
are special tokens, `token_ids` is the interface to use. `agent.py` does that
and needs no change to TensorRT-Model-Connect. It is not filed upstream.

## Results

One run each, on a rented A100-SXM4-40GB (driver 570.148.08, CUDA 13.3 NGC
image via forward compatibility), `families/llama` built from `upstream/main`
at the commit that merged #1288, both bundles `max_sequence_length=4096`,
`tensor_parallel_size=1`, no quantization. **These were recorded before the
loop changes described above and have not been re-run.**

**bf16 vs fp16** (`precision_compare.py`, "Explain what a KV cache does in one
paragraph.", `max_new_tokens=64`, 5 runs after one discarded warmup):

| precision | prefill_ms | decode_ms | ms/token | stdev |
|---|---|---|---|---|
| bf16 | 90.24 | 920.35 | 14.380 | 0.032 |
| fp16 | 159.06 | 883.85 | 13.810 | 0.004 |

How much weight to put on this: not much.

- The two bundles are separate engine builds, run one precision after the
  other. The 4% ms/token gap is far outside run-to-run noise but is not shown
  to be a precision effect; build-to-build variance was not measured (build
  each precision twice and compare).
- Each invocation is a fresh process, so `prefill_ms` on a ~40-token prompt is
  mostly per-process start-up cost, not prefill compute. The bf16/fp16 prefill
  difference should not be read as a property of either precision.
- bf16 and fp16 are both 2 bytes wide, so a small decode difference is what
  one would expect if decode is dominated by streaming weights.

**Agent run** (task below, `--max-new-tokens 1024`, bf16), excerpt:

```
engine invocations : 4
total prefill time : 616.96 ms
total decode time  : 15860.76 ms across 1108 tokens
avg decode/token   : 14.315 ms
```

Four turns: three parallel `calculator` calls, two `>` comparisons, a
`write_file`, then a final answer recommending the A40 (0.0272 TFLOPS/$, the
highest of the three). The saved file transcribed the L40S ratio as 0.0201
instead of the 0.0213 it had computed; the recommendation did not depend on it.

Caveats on this run:

- **The task it ran had an unsatisfiable budget.** It said $5,000, but the
  three cards are priced at $5,500 / $8,600 / $6,800. The agent recommended the
  A40 without flagging that it exceeds the budget. The task now says $10,000.
- **The throughput figures were supplied, not sourced, and were not checked
  against datasheets.** The task called them "dense FP16 TFLOPS"; they may not
  be the same precision or execution path across the three cards. The
  recommendation is only as good as those inputs, so it says nothing about
  which card is faster.
- **It made no `web_search` calls**, so `web_search` and the result
  sanitization have not been exercised on a GPU.
- It is a single run. Nothing here measures how reliably the model does this.

## Task

`DEFAULT_TASK` in `agent.py` asks for a three-GPU comparison under a budget:
compute FP16 TFLOPS per dollar with the calculator, check which ratio is
greatest with explicit `>` comparisons (a small model misjudged the largest of
three numbers by eye in earlier runs), and save the recommendation to a file.
Figures and prices are supplied so the outcome does not hinge on noisy search
snippets.

## Limitations and open questions

- **No baseline.** Nothing here compares against HF `generate`, vLLM or SGLang
  on the same GPU, so the numbers say nothing about how the runtime compares.
- **14.3 ms/token has not been explained.** It has not been compared with the
  memory-bandwidth floor for this model's weights on this GPU, and nothing has
  been profiled, so it is unknown whether the time is in the engine or in
  `trtmc`'s decode loop. `decode_ms / len(token_ids)` may also be off by one if
  the first token comes from the prefill pass.
- **Per-turn reload cost** is now measured (wall time is reported next to
  prefill and decode) but has no recorded run yet.
- **Encode-side parity is unchecked.** The prompt is rendered by HF but
  tokenized by `trtmc`'s C++ BPE. Whether both produce identical ids (a
  double BOS, special-token matching, pre-tokenizer differences) has not been
  verified; the fix for decode says nothing about encode.
- **Tool results are formatted as one `user` turn per call**
  (`<tool_response>…</tool_response>`), which differs from the template's own
  `tool` role (newlines around the content, consecutive results grouped in one
  turn). The model may have been trained on the latter; the effect is
  unmeasured.
- Sampling parameters are not passed to `trtmc`; whether it decodes greedily
  by default is not documented here.
- Tool-call parsing is matched to MiniCPM5's template, not general.
- One GPU, one request at a time, no batching.
- `web_search` depends on `ddgs`; result quality is outside this project.
- Sanitization is pattern-based and validated only against HF's tokenizer
  behaviour and `trtmc`'s source, not end to end on a GPU.
- `precision_compare.py` runs all bf16 invocations before all fp16 ones rather
  than interleaving them.

## Setup

Requires a machine with TensorRT-Model-Connect built (the `trtmc` binary and
the `families/llama` runtime `.so`s) and a MiniCPM5-2B bundle built through it,
against `upstream/main` (which contains #1288).

```bash
pip install -r requirements.txt        # runtime
pip install -r requirements-dev.txt    # plus pytest
```

Two GPU-setup gotchas hit on rented hardware:

- `CMAKE_CUDA_ARCHITECTURES` defaults to 89 (Ada) in TRT-MC's GPU dev
  Dockerfile and CI. On other architectures (this was validated on an A100,
  compute capability 8.0) override it when building, e.g.
  `-e CMAKE_CUDA_ARCHITECTURES=80` for the `tools.community_gpu_ci` path. It
  only matters for families with `.cu` sources; `families/llama` has none.
- NGC images set up CUDA Forward Compatibility only for the entrypoint's own
  process, so a later `docker exec` sees `torch.cuda.is_available() == False`.
  Pass `-e LD_LIBRARY_PATH=/usr/local/cuda/compat/lib` on each `docker exec`
  that needs the GPU.

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
