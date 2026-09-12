"""Parse MiniCPM5's <function name="..."><param name="...">value</param></function>
tool-call XML out of raw generated text (its own chat template's convention --
see chat_template.jinja on the model repo). No external XML parser: the model's
output isn't guaranteed to be well-formed, so a permissive regex is more robust
than a strict parser that would throw on a malformed tag.
"""

from __future__ import annotations

import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FUNCTION_RE = re.compile(r'<function\s+name="([^"]+)">(.*?)</function>', re.DOTALL)
_PARAM_RE = re.compile(r'<param\s+name="([^"]+)">(.*?)</param>', re.DOTALL)
_CDATA_RE = re.compile(r"^\s*<!\[CDATA\[(.*)\]\]>\s*$", re.DOTALL)


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text)


def parse_tool_calls(text: str) -> list[dict]:
    """Return a list of {"name": str, "arguments": dict[str, str]}."""
    calls = []
    for match in _FUNCTION_RE.finditer(text):
        name = match.group(1)
        body = match.group(2)
        arguments: dict[str, str] = {}
        for param_match in _PARAM_RE.finditer(body):
            param_name = param_match.group(1)
            param_value = param_match.group(2)
            cdata_match = _CDATA_RE.match(param_value)
            arguments[param_name] = (
                cdata_match.group(1) if cdata_match else param_value.strip()
            )
        calls.append({"name": name, "arguments": arguments})
    return calls
