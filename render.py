"""Render the full multi-turn + tools prompt ourselves via the checkpoint's
own chat template, instead of relying on trtmc's C++ chat-template detection
(`--use-chat-template true`). That detection logic substring-sniffs the Jinja
template to classify it into a handful of known formats (Llama/Phi/etc.) --
see the TensorRT-Model-Connect issue filed against families/llama/runtime/
chat_templates.cpp for a case where that misclassifies a real template. Tool
calling depends on exact template fidelity (tool schemas, <tool_response>
wrapping, etc.), so we render with the real HF tokenizer/template instead and
feed trtmc the finished text directly.
"""

from __future__ import annotations

from transformers import AutoTokenizer


def load_tokenizer(model_dir: str):
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)


def render_prompt(tokenizer, messages: list[dict], tools: list[dict]) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    # The template's own {{- bos_token }} puts a literal "<s>" at the start of
    # the text. trtmc's CLI adds its own BOS id by default on every --prompt
    # invocation, on top of whatever `trtmc`'s encoder matches in the text
    # itself -- so a literal "<s>" here becomes a genuine double BOS in the
    # ids trtmc actually runs (confirmed directly: see README "The trtmc
    # findings"). Stripping our own copy leaves trtmc's one auto-added BOS,
    # matching HF's own encoding of this text. This fixes one of the encode
    # mismatches found there; the other (BPE under-merging elsewhere in the
    # prompt) is not fixable from this side -- it needs a trtmc source change.
    bos_token = getattr(tokenizer, "bos_token", None)
    if bos_token and prompt.startswith(bos_token):
        prompt = prompt[len(bos_token):]
    return prompt
