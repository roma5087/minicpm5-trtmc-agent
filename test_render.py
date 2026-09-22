"""render_prompt() coverage using a minimal stub tokenizer, not a real HF
checkpoint -- this only needs to verify render_prompt() calls
apply_chat_template with the right arguments and returns its result
verbatim, not that any particular template renders a particular way."""

from __future__ import annotations

import render
from render import render_prompt


class _StubTokenizer:
    def __init__(self, template_output="RENDERED", bos_token=None):
        self.calls = []
        self.bos_token = bos_token
        self._template_output = template_output

    def apply_chat_template(self, messages, tools=None, tokenize=None, add_generation_prompt=None):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return self._template_output


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


def test_render_prompt_strips_one_leading_bos_token():
    # trtmc's own CLI adds one BOS id by default on every --prompt call; a
    # literal bos_token left in the text becomes a second, genuine BOS in the
    # ids trtmc actually runs (see README "The trtmc findings"). Stripping our
    # own copy leaves exactly the one trtmc adds.
    tokenizer = _StubTokenizer(template_output="<s><|im_start|>system\nhi", bos_token="<s>")
    result = render_prompt(tokenizer, [], tools=[])
    assert result == "<|im_start|>system\nhi"


def test_render_prompt_leaves_text_alone_when_it_does_not_start_with_bos_token():
    tokenizer = _StubTokenizer(template_output="<|im_start|>system\nhi", bos_token="<s>")
    result = render_prompt(tokenizer, [], tools=[])
    assert result == "<|im_start|>system\nhi"


def test_render_prompt_leaves_text_alone_when_tokenizer_has_no_bos_token():
    tokenizer = _StubTokenizer(template_output="<s><|im_start|>system\nhi", bos_token=None)
    result = render_prompt(tokenizer, [], tools=[])
    assert result == "<s><|im_start|>system\nhi"


def test_render_prompt_only_strips_the_leading_bos_token_not_others_later_in_the_text():
    # A literal "<s>" that appears later (e.g. quoted inside message content)
    # must not be touched -- only the one the template puts at the very start.
    tokenizer = _StubTokenizer(template_output="<s>a<s>b", bos_token="<s>")
    result = render_prompt(tokenizer, [], tools=[])
    assert result == "a<s>b"


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
