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


def test_cdata_value_containing_literal_closing_param_tag_not_truncated():
    # A value that itself contains the literal text "</param>" must not be
    # cut short by a naive "find the next </param>" scan -- the CDATA
    # boundary (]]>) is the real end of the value, not that substring.
    text = (
        '<function name="write_file">'
        '<param name="filename">report.txt</param>'
        '<param name="content"><![CDATA[before </param> after]]></param>'
        "</function>"
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["arguments"]["content"] == "before </param> after"


def test_overlapping_function_tags_do_not_silently_merge():
    # A hallucinated second <function> opening before the first one's
    # </function> must not be silently merged into one call with foreign
    # arguments -- the malformed block should just be dropped, not guessed.
    text = (
        '<function name="a"><param name="x">1</param>'
        '<function name="b"><param name="y">2</param></function></function>'
    )
    calls = parse_tool_calls(text)
    # Neither call should come out looking like a valid merged call with
    # both x and y as arguments to "a".
    assert not any(c["name"] == "a" and "y" in c["arguments"] for c in calls)


def test_malformed_function_block_is_dropped_not_guessed():
    text = '<function name="broken"><param name="x">no closing param tag'
    assert parse_tool_calls(text) == []


def test_well_formed_call_still_parses_after_a_malformed_one():
    text = (
        '<function name="broken"><param name="x">no closing param tag'
        '<function name="ok"><param name="y">1</param></function>'
    )
    calls = parse_tool_calls(text)
    assert any(c["name"] == "ok" and c["arguments"] == {"y": "1"} for c in calls)
