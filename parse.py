"""Parse MiniCPM5's <function name="..."><param name="...">value</param></function>
tool-call XML out of raw generated text (its own chat template's convention --
see chat_template.jinja on the model repo).

Deliberately NOT a single greedy/non-greedy regex over the whole string, and
NOT a strict XML parser (the model's output isn't guaranteed well-formed, and
a strict parser would just throw on the first hallucinated tag). Instead this
is a small position-based scanner: each function call is parsed strictly
outward from its own opening tag, expecting <param>...</param> entries then
its own </function> in sequence. Two failure modes a single whole-string
regex gets wrong, that this avoids:

  1. Overlapping/nested <function> tags (a model hallucination) would make a
     single non-greedy regex silently *merge* two calls into one wrong call
     with foreign arguments -- worse than raising, since the caller then
     executes a corrupted action with no signal anything went wrong.
  2. A CDATA-wrapped value containing a literal "</param>" substring would
     make a regex keyed off "</param>" truncate the value mid-CDATA-block,
     again silently, again with no signal.

Here, if a function block turns out malformed partway through (an expected
<param> or </function> isn't where it should be), that whole call is
dropped rather than guessed at -- silently executing a corrupted call is
worse than silently skipping a malformed one, which at least fails safe.
"""

from __future__ import annotations

import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FUNCTION_START_RE = re.compile(r'<function\s+name="([^"]*)"\s*>')
_FUNCTION_CLOSE_RE = re.compile(r"\s*</function>")
_PARAM_START_RE = re.compile(r'\s*<param\s+name="([^"]*)"\s*>')
_PARAM_CLOSE_RE = re.compile(r"\s*</param>")
_CDATA_OPEN = "<![CDATA["
_CDATA_CLOSE = "]]>"


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks. An unclosed <think> (output cut off
    mid-reasoning) is removed through the end of the text: reasoning that never
    finished is not an answer and must not leak into one."""
    text = _THINK_RE.sub("", text)
    unclosed = text.find("<think>")
    return text if unclosed == -1 else text[:unclosed]


def _parse_one_param(text: str, pos: int) -> tuple[str, str, int] | None:
    """Try to parse one <param>...</param> starting at pos. Returns
    (name, value, next_pos) or None if pos isn't the start of a param."""
    match = _PARAM_START_RE.match(text, pos)
    if not match:
        return None
    name = match.group(1)
    value_start = match.end()

    if text.startswith(_CDATA_OPEN, value_start):
        cdata_start = value_start + len(_CDATA_OPEN)
        cdata_end = text.find(_CDATA_CLOSE, cdata_start)
        if cdata_end == -1:
            return None
        value = text[cdata_start:cdata_end]
        after_value = cdata_end + len(_CDATA_CLOSE)
    else:
        # A well-formed non-CDATA value never contains a literal "<" -- the
        # template's own contract requires CDATA-wrapping whenever a value
        # contains <, &, or a newline. So the value ends at the FIRST "<" we
        # see, not at the first "</param>" found anywhere ahead: searching
        # unboundedly for "</param>" would walk straight past a malformed or
        # overlapping tag and merge unrelated content into this value.
        next_lt = text.find("<", value_start)
        if next_lt == -1 or not text.startswith("</param>", next_lt):
            return None
        value = text[value_start:next_lt].strip()
        after_value = next_lt

    close_match = _PARAM_CLOSE_RE.match(text, after_value)
    if not close_match:
        return None
    return name, value, close_match.end()


def _parse_one_function(text: str, pos: int) -> tuple[dict, int] | None:
    """Try to parse one <function>...</function> starting at pos (which must
    already be the start of a <function ...> tag). Returns (call, next_pos)
    or None if the block is malformed."""
    start_match = _FUNCTION_START_RE.match(text, pos)
    if not start_match:
        return None
    name = start_match.group(1).strip()
    scan_pos = start_match.end()
    arguments: dict[str, str] = {}

    while True:
        close_match = _FUNCTION_CLOSE_RE.match(text, scan_pos)
        if close_match:
            return {"name": name, "arguments": arguments}, close_match.end()
        parsed_param = _parse_one_param(text, scan_pos)
        if parsed_param is None:
            return None  # neither a param nor the closing tag -- malformed
        pname, pvalue, scan_pos = parsed_param
        arguments[pname] = pvalue


def parse_tool_calls(text: str) -> list[dict]:
    """Return a list of {"name": str, "arguments": dict[str, str]}."""
    calls = []
    pos = 0
    while True:
        start_match = _FUNCTION_START_RE.search(text, pos)
        if not start_match:
            break
        parsed = _parse_one_function(text, start_match.start())
        if parsed is None:
            # Malformed block: skip past this opening tag and keep scanning
            # for the next one, rather than merging into it or looping.
            pos = start_match.end()
            continue
        call, pos = parsed
        calls.append(call)
    return calls
