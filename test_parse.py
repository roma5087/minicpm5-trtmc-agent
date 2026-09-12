from parse import parse_tool_calls, strip_thinking


def test_single_function_call():
    text = (
        '<function name="web_search"><param name="query">NVIDIA A40 FP16 TFLOPS'
        "</param></function>"
    )
    calls = parse_tool_calls(text)
    assert calls == [{"name": "web_search", "arguments": {"query": "NVIDIA A40 FP16 TFLOPS"}}]


def test_multiple_params():
    text = (
        '<function name="write_file">'
        '<param name="filename">report.txt</param>'
        '<param name="content">hello world</param>'
        "</function>"
    )
    calls = parse_tool_calls(text)
    assert calls == [
        {"name": "write_file", "arguments": {"filename": "report.txt", "content": "hello world"}}
    ]


def test_cdata_wrapped_value_preserved_verbatim():
    text = (
        '<function name="write_file">'
        '<param name="filename">report.txt</param>'
        '<param name="content"><![CDATA[line one\n<b>line two</b> & more]]></param>'
        "</function>"
    )
    calls = parse_tool_calls(text)
    assert calls[0]["arguments"]["content"] == "line one\n<b>line two</b> & more"


def test_no_function_call_returns_empty_list():
    assert parse_tool_calls("just a plain final answer, no tools used") == []


def test_multiple_function_calls_in_one_turn():
    text = (
        '<function name="web_search"><param name="query">a</param></function>'
        '<function name="calculator"><param name="expression">1+1</param></function>'
    )
    calls = parse_tool_calls(text)
    assert [c["name"] for c in calls] == ["web_search", "calculator"]


def test_strip_thinking_removes_think_block_only():
    text = "<think>internal reasoning</think>the actual answer"
    assert strip_thinking(text) == "the actual answer"
