"""Tool implementations and their JSON-schema definitions for the agent.

Each schema follows the OpenAI-style function-calling convention MiniCPM5's
own chat template expects (a dict with "type": "function" and a nested
"function" object) -- see the tools_definitions/*.json emitted into the
rendered prompt by chat_template.jinja.
"""

from __future__ import annotations

import ast
import operator
from pathlib import Path

from ddgs import DDGS

WORKSPACE = (Path(__file__).resolve().parent / "workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)

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


# Caps the cost of ** before it runs, not after: computing a huge-exponent
# int result is expensive in itself, well before the existing int-to-str
# guard (see calculator()'s docstring) ever gets a chance to reject the
# *result* -- this bounds the *operation*, not just its output.
#
# Bounding just the immediate exponent isn't enough on its own -- confirmed
# two independent ways in review: (1) a chain of nested ** calls, each
# individually within an exponent-only cap, still blows up, since each
# level's real result feeds in as the next level's base; (2) a single **
# with a small, allowed exponent but an enormous *base* (e.g. a ~4300-digit
# literal) is just as expensive, and an exponent-only check never looks at
# the base at all. Estimating the *result* size (base's bit length *
# exponent) instead catches both: it reflects whatever the base actually is,
# however it was produced, and multiplying by the next exponent gives an
# accurate cost estimate for the operation about to run.
_MAX_POW_RESULT_BITS = 20_000  # ~6,000 decimal digits -- comfortably below
# the point where the existing int-to-str guard would trigger anyway, so
# this guard fires before real computational cost is spent, not after.


def _eval_node(node: ast.AST) -> float | bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            # Only int**int can actually blow up like this: float exponents
            # (or a float base) go through IEEE-754 arithmetic, which raises
            # OverflowError on its own long before consuming pathological
            # memory -- Python bignums are the only side with no built-in cap.
            is_plain_int = lambda v: isinstance(v, int) and not isinstance(v, bool)
            if is_plain_int(left) and is_plain_int(right) and right > 1 and abs(left) > 1:
                estimated_bits = left.bit_length() * right
                if estimated_bits > _MAX_POW_RESULT_BITS:
                    raise ValueError(f"estimated result too large: ~{estimated_bits} bits")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _ALLOWED_COMPAREOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.comparators[0])
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
        tree = ast.parse(expression, mode="eval")
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
    if len(content.encode("utf-8")) > _MAX_WRITE_FILE_CONTENT_BYTES:
        return f"error: content too large ({len(content)} chars, max {_MAX_WRITE_FILE_CONTENT_BYTES} bytes)"
    try:
        path = _resolve_in_workspace(filename)
        path.write_text(content, encoding="utf-8")
    except ValueError as error:
        return f"error: {error}"
    except OSError as error:
        # str(OSError) embeds the full resolved host path (e.g. "File name
        # too long: '/Users/.../workspace/...'") -- that's the caller's own
        # absolute filesystem layout, not something that belongs in text fed
        # back into the model's context. Report the OS-level reason and the
        # filename the caller actually asked for, not the resolved path.
        return f"error: could not write {filename!r}: {error.strerror or error}"
    return f"saved {len(content)} bytes to {path.name}"


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
