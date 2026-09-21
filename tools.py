"""Tool implementations and their JSON-schema definitions for the agent.

Each schema follows the OpenAI-style function-calling convention MiniCPM5's
own chat template expects (a dict with "type": "function" and a nested
"function" object) -- see the tools_definitions/*.json emitted into the
rendered prompt by chat_template.jinja.
"""

from __future__ import annotations

import ast
import math
import operator
from pathlib import Path

from ddgs import DDGS

WORKSPACE = (Path(__file__).resolve().parent / "workspace").resolve()

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return short text snippets for the top results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression (+, -, *, /, **, parentheses) or a single comparison (<, <=, >, >=, ==, !=) between two numbers/expressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "An expression, e.g. '150 * 3.7'. A comparison using < or <= "
                            "must be CDATA-wrapped (e.g. <![CDATA[4200 <= 5000]]>), same as "
                            "any value containing a literal '<'; >, >=, ==, != need no "
                            "wrapping."
                        ),
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Save text content to a file in the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File name, no path separators."},
                    "content": {"type": "string", "description": "Text content to write."},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read text content back from a file in the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "File name, no path separators."},
                },
                "required": ["filename"],
            },
        },
    },
]

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
# Comparisons let the agent explicitly check constraints (e.g. "is this price
# within budget?") instead of silently eyeballing numbers in its own text.
_ALLOWED_COMPAREOPS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


# Every integer intermediate is capped by size, not just the result of **.
# Bounding only the exponent is not enough (a huge base, or a chain of nested
# operations that each stay under an exponent cap, is just as expensive), and
# ** is not the only quadratic operation: chained multiplication feeding % or
# // is too. Capping the bit length of every integer that can exist during
# evaluation bounds the cost of every operation, and the cap is checked before
# ** and * run (from operand sizes), so no oversized value is ever computed.
_MAX_INT_BITS = 4_096  # ~1,230 decimal digits


def _is_plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_int_size(value, what: str):
    if _is_plain_int(value) and value.bit_length() > _MAX_INT_BITS:
        raise ValueError(f"{what} too large: ~{value.bit_length()} bits (max {_MAX_INT_BITS})")
    return value


def _reject_bool(value):
    # A comparison's True/False must not silently become 1/0 in arithmetic
    # (e.g. "(1 < 2) + 1" == 2) -- that is never what the caller meant.
    if isinstance(value, bool):
        raise ValueError("a comparison result cannot be used as a number")
    return value


def _eval_node(node: ast.AST) -> float | bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return _check_int_size(node.value, "number")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _reject_bool(_eval_node(node.left))
        right = _reject_bool(_eval_node(node.right))
        if _is_plain_int(left) and _is_plain_int(right):
            if isinstance(node.op, ast.Pow) and right > 1 and abs(left) > 1:
                estimated_bits = left.bit_length() * right
                if estimated_bits > _MAX_INT_BITS:
                    raise ValueError(f"estimated result too large: ~{estimated_bits} bits")
            elif isinstance(node.op, ast.Mult):
                estimated_bits = left.bit_length() + right.bit_length()
                if estimated_bits > _MAX_INT_BITS:
                    raise ValueError(f"estimated result too large: ~{estimated_bits} bits")
        result = _ALLOWED_BINOPS[type(node.op)](left, right)
        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("result is not a finite number")
        return _check_int_size(result, "result")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_reject_bool(_eval_node(node.operand)))
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _ALLOWED_COMPAREOPS:
        left = _reject_bool(_eval_node(node.left))
        right = _reject_bool(_eval_node(node.comparators[0]))
        return _ALLOWED_COMPAREOPS[type(node.ops[0])](left, right)
    raise ValueError(f"disallowed expression element: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Safely evaluate arithmetic (and simple comparisons) without eval().

    str(result) is inside the try too: a huge exponent (e.g. 10**200000)
    evaluates fine but raises on int-to-str conversion (Python's own
    integer-string-conversion limit), which must not escape as an
    uncaught exception -- every tool must return an "error: ..." string
    on failure, never raise, since the agent loop's safety net only
    recognizes failures that come back that way.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as error:
        return f"error: could not evaluate {expression!r}: {error}"


# Search-result text is untrusted: it flows straight into a <tool_response>
# turn that gets re-rendered into the next prompt. A poisoned page could
# embed these exact marker substrings to try to forge fake turn boundaries
# or fake tool-call XML that a later turn's model output might reproduce --
# HF fast tokenizers match a registered special token as a substring
# anywhere in the text, not only where a template placed it, so a literal
# "<|im_end|>" inside search-result text is not inert. Neutralizing them
# here (not in parse.py, which has no way to know where text originated)
# is the actual trust boundary: nothing this project doesn't already
# control should be able to inject a marker that means something structural
# downstream.
_UNSAFE_MARKER_SUBSTRINGS = (
    "<|im_start|>",
    "<|im_end|>",
    "<tool_response>",
    "</tool_response>",
    "<function",
    "</function>",
    "<param",
    "</param>",
    "<![CDATA[",
    "]]>",
)


def _neutralize_markers(text: str) -> str:
    for marker in _UNSAFE_MARKER_SUBSTRINGS:
        text = text.replace(marker, marker.replace("<", "[").replace(">", "]"))
    return text


def sanitize_tool_result(text: str, added_tokens=()) -> str:
    """Make untrusted text safe to place inside a <tool_response> turn.

    Applied to every tool result at the one place they enter the prompt, not
    inside individual tools. trtmc's encoder matches *every* added token
    (special or not) as a substring anywhere in the prompt text, so the token
    list comes from the tokenizer itself rather than a hand-kept list. A
    zero-width space after the first character breaks the match without
    changing how the text reads. Also removes what would crash the subprocess
    call (NUL bytes, lone surrogates).
    """
    text = _neutralize_markers(text)
    for token in sorted(added_tokens, key=len, reverse=True):
        if len(token) > 1 and token in text:
            text = text.replace(token, token[0] + "\u200b" + token[1:])
    text = text.replace("\x00", "")
    return text.encode("utf-8", "replace").decode("utf-8")


def web_search(query: str, max_results: int = 4) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as error:
        return f"error: web search failed: {error}"
    if not results:
        return "no results found"
    lines = []
    for item in results:
        title = _neutralize_markers(item.get("title", ""))
        body = _neutralize_markers(item.get("body", ""))
        lines.append(f"- {title}: {body}")
    return "\n".join(lines)


def _resolve_in_workspace(filename: str) -> Path:
    filename = filename.strip()
    if filename.startswith("."):
        # Keeps the tracked workspace/.gitkeep (and any dotfile) out of reach.
        raise ValueError(f"filenames may not start with '.': {filename!r}")
    candidate = (WORKSPACE / filename).resolve()
    # candidate == WORKSPACE (e.g. filename="." or "") is rejected too: a
    # tool call must always name a file *within* the workspace, never the
    # workspace directory itself (writing to it raises IsADirectoryError).
    if candidate == WORKSPACE or WORKSPACE not in candidate.parents:
        raise ValueError(f"path escapes workspace: {filename!r}")
    return candidate


# Bounds how much any single write can grow what gets fed back into the
# *next* rendered prompt: write_file's content has no other size limit, and
# a write followed by a read_file of it in the same or a later turn puts
# the whole blob into a <tool_response> turn with no windowing anywhere in
# agent.py -- an oversized prompt can crash trtmc's subprocess invocation
# with an OSError past agent.py's own except clause (ARG_MAX), so bounding
# it here is the cheapest place to actually prevent that.
_MAX_WRITE_FILE_CONTENT_BYTES = 200_000


def write_file(filename: str, content: str) -> str:
    try:
        encoded = content.encode("utf-8")
    except UnicodeEncodeError as error:
        return f"error: content is not valid text: {error.reason}"
    if len(encoded) > _MAX_WRITE_FILE_CONTENT_BYTES:
        return f"error: content too large ({len(encoded)} bytes, max {_MAX_WRITE_FILE_CONTENT_BYTES})"
    try:
        path = _resolve_in_workspace(filename)
        WORKSPACE.mkdir(exist_ok=True)
        path.write_bytes(encoded)
    except ValueError as error:
        return f"error: {error}"
    except OSError as error:
        # str(OSError) embeds the full resolved host path (e.g. "File name
        # too long: '/Users/.../workspace/...'") -- that's the caller's own
        # absolute filesystem layout, not something that belongs in text fed
        # back into the model's context. Report the OS-level reason and the
        # filename the caller actually asked for, not the resolved path.
        return f"error: could not write {filename!r}: {error.strerror or error}"
    return f"saved {len(encoded)} bytes to {path.name}"


def read_file(filename: str) -> str:
    try:
        path = _resolve_in_workspace(filename)
        if not path.is_file():
            return f"error: {filename!r} does not exist in workspace"
        return path.read_text(encoding="utf-8")
    except ValueError as error:
        return f"error: {error}"
    except OSError as error:
        return f"error: could not read {filename!r}: {error.strerror or error}"


TOOL_IMPLEMENTATIONS = {
    "web_search": web_search,
    "calculator": calculator,
    "write_file": write_file,
    "read_file": read_file,
}
