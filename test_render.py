"""render_prompt() coverage using a minimal stub tokenizer, not a real HF
checkpoint -- this only needs to verify render_prompt() calls
apply_chat_template with the right arguments and returns its result
verbatim, not that any particular template renders a particular way."""

from __future__ import annotations

import render
from render import render_prompt


class _StubTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tools=None, tokenize=None, add_generation_prompt=None):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return "RENDERED"


def test_render_prompt_returns_the_template_output_verbatim():
    tokenizer = _StubTokenizer()
    result = render_prompt(tokenizer, [{"role": "user", "content": "hi"}], tools=[])
    assert result == "RENDERED"


def test_render_prompt_forwards_messages_and_tools_untouched():
    tokenizer = _StubTokenizer()
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"name": "calculator"}]
    render_prompt(tokenizer, messages, tools)
    call = tokenizer.calls[0]
    assert call["messages"] == messages
    assert call["tools"] == tools


def test_render_prompt_requests_untokenized_text_with_generation_prompt():
    # tokenize=False and add_generation_prompt=True aren't incidental --
    # tokenize=True would return ids instead of the text trtmc expects, and
    # add_generation_prompt=False would omit the assistant-turn opening tag
    # the model needs to start generating from.
    tokenizer = _StubTokenizer()
    render_prompt(tokenizer, [], tools=[])
    call = tokenizer.calls[0]
    assert call["tokenize"] is False
    assert call["add_generation_prompt"] is True


def test_load_tokenizer_disables_trust_remote_code(monkeypatch):
    # trust_remote_code=False is a deliberate security choice (no arbitrary
    # code execution from a model repo's own tokenizer code) -- load_tokenizer()
    # had zero test coverage before this, so flipping it to True (or
    # dropping model_dir) would have gone unnoticed.
    captured = {}

    class _StubAutoTokenizer:
        @staticmethod
        def from_pretrained(model_dir, trust_remote_code=None):
            captured["model_dir"] = model_dir
            captured["trust_remote_code"] = trust_remote_code
            return "TOKENIZER"

    monkeypatch.setattr(render, "AutoTokenizer", _StubAutoTokenizer)
    result = render.load_tokenizer("/some/model/dir")
    assert result == "TOKENIZER"
    assert captured == {"model_dir": "/some/model/dir", "trust_remote_code": False}
