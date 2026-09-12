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
                        "description": "An expression, e.g. '150 * 3.7' or '4200 <= 5000'.",
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


def _eval_node(node: ast.AST) -> float | bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
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
        title = item.get("title", "")
        body = item.get("body", "")
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


def write_file(filename: str, content: str) -> str:
    try:
        path = _resolve_in_workspace(filename)
        path.write_text(content, encoding="utf-8")
    except (ValueError, OSError) as error:
        return f"error: {error}"
    return f"saved {len(content)} bytes to {path.name}"


def read_file(filename: str) -> str:
    try:
        path = _resolve_in_workspace(filename)
        if not path.is_file():
            return f"error: {filename!r} does not exist in workspace"
        return path.read_text(encoding="utf-8")
    except (ValueError, OSError) as error:
        return f"error: {error}"


TOOL_IMPLEMENTATIONS = {
    "web_search": web_search,
    "calculator": calculator,
    "write_file": write_file,
    "read_file": read_file,
}
