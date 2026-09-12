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
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
