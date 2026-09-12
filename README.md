# MiniCPM5-2B Tool-Calling Agent, served via TensorRT-Model-Connect

A small tool-calling agent where [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)
is the reasoning core, compiled and served through NVIDIA's
[TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect)
(`trtmc`) native runtime rather than vLLM/SGLang's built-in tool-call support.

## Why this exists

Grew out of investigating and fixing a real `families/llama` bug in
TensorRT-Model-Connect ([PR #1269](https://github.com/NVIDIA/TensorRT-Model-Connect/pull/1269))
that surfaced while validating MiniCPM5-2B against that framework. This
project puts the fixed build path to actual use: MiniCPM5-2B's real strength
is tool-use (97.1 on τ²-Bench Telecom), so the demo is a small agent that
looks up GPU specs, computes a derived metric, and saves the result.

## Architecture

`trtmc` has no persistent/server mode -- every turn is a fresh `trtmc run`
subprocess invocation that reloads the compiled engine from disk. That's
acceptable for a demo (each turn costs a few seconds of engine-load time);
a real deployment would need a long-running server instead.

Tool calling is implemented entirely in Python, not inside `trtmc` itself:

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
  output text.
- **`tools.py`** -- the actual tool implementations (`web_search`,
  `calculator`, `write_file`, `read_file`) and their JSON-schema definitions.
  `calculator` uses an AST-restricted evaluator, never `eval()`; file tools
  are sandboxed to `workspace/` with path-escape rejected.
- **`agent.py`** -- the loop: render → run `trtmc` → parse for tool calls →
  execute them → append results as `<tool_response>`-wrapped turns → repeat
  until the model returns a plain answer (capped at `MAX_TURNS`).

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
  "Look up the NVIDIA A40's FP16 TFLOPS and price, calculate TFLOPS per dollar, and save the result." \
  --model-dir /path/to/MiniCPM5-2B \
  --binary /path/to/trtmc \
  --bundle /path/to/minicpm5-2b.bundle \
  --runtime-root /path/to/build-output-dir
```

## Status

Code complete; not yet run end-to-end against a live GPU (written while no
GPU instance was up). Non-GPU-dependent parts (`parse.py`, `tools.py`) are
covered by `test_parse.py`/`test_tools.py`. Full agent-loop validation is
pending a GPU session.
