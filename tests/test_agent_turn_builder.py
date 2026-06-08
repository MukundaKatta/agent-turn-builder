"""Tests for agent_turn_builder."""

from __future__ import annotations

import pytest

from agent_turn_builder import AgentTurnBuilder, TurnError

# ---------------------------------------------------------------------------
# Constructor / repr
# ---------------------------------------------------------------------------


def test_repr():
    b = AgentTurnBuilder()
    assert "count=0" in repr(b)
    assert "last_role=None" in repr(b)


def test_repr_after_add():
    b = AgentTurnBuilder().user("hi")
    assert "count=1" in repr(b)
    assert "last_role='user'" in repr(b)


def test_initial_empty():
    b = AgentTurnBuilder()
    assert b.is_empty() is True
    assert b.count() == 0
    assert b.last_role() is None
    assert b.roles() == []


# ---------------------------------------------------------------------------
# system / user / assistant
# ---------------------------------------------------------------------------


def test_system():
    msgs = AgentTurnBuilder().system("Be helpful.").build()
    assert msgs == [{"role": "system", "content": "Be helpful."}]


def test_user():
    msgs = AgentTurnBuilder().user("Hello").build()
    assert msgs == [{"role": "user", "content": "Hello"}]


def test_assistant():
    msgs = AgentTurnBuilder().assistant("Hi!").build()
    assert msgs == [{"role": "assistant", "content": "Hi!"}]


def test_three_turn_conversation():
    msgs = (
        AgentTurnBuilder()
        .system("You are helpful.")
        .user("What is 2+2?")
        .assistant("4")
        .build()
    )
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"


def test_chaining_returns_self():
    b = AgentTurnBuilder()
    assert b.user("x") is b
    assert b.assistant("y") is b
    assert b.system("s") is b


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


def test_add_known_role():
    b = AgentTurnBuilder()
    b.add("user", "hello")
    assert b.last_role() == "user"


def test_add_unknown_role_raises():
    with pytest.raises(ValueError, match="Unknown role"):
        AgentTurnBuilder().add("bot", "text")


def test_add_tool_role():
    b = AgentTurnBuilder()
    b.add("tool", "tool output")
    assert b.last_role() == "tool"


# ---------------------------------------------------------------------------
# content as list
# ---------------------------------------------------------------------------


def test_user_with_content_list():
    content = [{"type": "text", "text": "hi"}]
    msgs = AgentTurnBuilder().user(content).build()
    assert msgs[0]["content"] == content


def test_build_deepcopies_list_content():
    content = [{"type": "text", "text": "hi"}]
    b = AgentTurnBuilder().user(content)
    msgs = b.build()
    msgs[0]["content"][0]["text"] = "changed"
    # Original should be unaffected
    assert b.build()[0]["content"][0]["text"] == "hi"


# ---------------------------------------------------------------------------
# tool_use / tool_result
# ---------------------------------------------------------------------------


def test_tool_use_block():
    msgs = AgentTurnBuilder().tool_use("tu_001", "search", {"query": "python"}).build()
    assert msgs[0]["role"] == "assistant"
    block = msgs[0]["content"][0]
    assert block["type"] == "tool_use"
    assert block["id"] == "tu_001"
    assert block["name"] == "search"
    assert block["input"] == {"query": "python"}


def test_tool_result_block():
    msgs = AgentTurnBuilder().tool_result("tu_001", "result text").build()
    assert msgs[0]["role"] == "user"
    block = msgs[0]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "tu_001"
    assert block["content"] == "result text"


def test_tool_use_deepcopies_inputs():
    inputs = {"q": "hello"}
    b = AgentTurnBuilder().tool_use("id1", "fn", inputs)
    inputs["q"] = "changed"
    block = b.build()[0]["content"][0]
    assert block["input"]["q"] == "hello"


def test_tool_use_custom_role():
    msgs = AgentTurnBuilder().tool_use("id1", "fn", {}, role="user").build()
    assert msgs[0]["role"] == "user"


def test_tool_result_custom_role():
    msgs = AgentTurnBuilder().tool_result("id1", "out", role="assistant").build()
    assert msgs[0]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def test_count():
    b = AgentTurnBuilder().user("a").assistant("b").user("c")
    assert b.count() == 3


def test_roles():
    b = AgentTurnBuilder().system("s").user("u").assistant("a")
    assert b.roles() == ["system", "user", "assistant"]


def test_has_system_true():
    assert AgentTurnBuilder().system("s").has_system() is True


def test_has_system_false():
    assert AgentTurnBuilder().user("u").has_system() is False


def test_last_role():
    b = AgentTurnBuilder().user("a").assistant("b")
    assert b.last_role() == "assistant"


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_empty():
    assert AgentTurnBuilder().validate() == []


def test_validate_valid():
    b = AgentTurnBuilder().system("s").user("u").assistant("a")
    assert b.validate() == []


def test_validate_system_not_first():
    b = AgentTurnBuilder().user("u").system("s")
    errors = b.validate()
    assert any("system" in e for e in errors)


def test_validate_consecutive_user():
    b = AgentTurnBuilder().user("a").user("b")
    errors = b.validate()
    assert any("user" in e for e in errors)


def test_validate_consecutive_assistant():
    b = AgentTurnBuilder().assistant("a").assistant("b")
    errors = b.validate()
    assert any("assistant" in e for e in errors)


# ---------------------------------------------------------------------------
# build strict
# ---------------------------------------------------------------------------


def test_build_strict_valid():
    msgs = AgentTurnBuilder(strict=True).user("hi").assistant("hey").build()
    assert len(msgs) == 2


def test_build_strict_raises():
    with pytest.raises(TurnError):
        AgentTurnBuilder(strict=True).user("a").user("b").build()


def test_build_strict_override_false():
    b = AgentTurnBuilder(strict=True)
    b.user("a").user("b")
    # override to non-strict for this call
    msgs = b.build(strict=False)
    assert len(msgs) == 2


def test_build_strict_override_true():
    b = AgentTurnBuilder(strict=False)
    b.user("a").user("b")
    with pytest.raises(TurnError):
        b.build(strict=True)


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def test_reset_clears():
    b = AgentTurnBuilder().user("hi").assistant("hey")
    b.reset()
    assert b.is_empty()
    assert b.count() == 0


def test_reset_returns_self():
    b = AgentTurnBuilder()
    assert b.reset() is b


# ---------------------------------------------------------------------------
# from_list
# ---------------------------------------------------------------------------


def test_from_list():
    source = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    b = AgentTurnBuilder.from_list(source)
    assert b.count() == 2
    assert b.roles() == ["user", "assistant"]


def test_from_list_deepcopy():
    source = [{"role": "user", "content": "hi"}]
    b = AgentTurnBuilder.from_list(source)
    source[0]["content"] = "changed"
    assert b.build()[0]["content"] == "hi"


def test_from_list_deepcopies_mutable_content():
    # In-place mutation of the source content must not leak into the builder.
    source = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    b = AgentTurnBuilder.from_list(source)
    source[0]["content"][0]["text"] = "changed"
    assert b.build()[0]["content"][0]["text"] == "hi"


# ---------------------------------------------------------------------------
# to_anthropic
# ---------------------------------------------------------------------------


def test_to_anthropic_with_system():
    b = AgentTurnBuilder().system("Be helpful.").user("Hello").assistant("Hi!")
    payload = b.to_anthropic()
    assert payload["system"] == "Be helpful."
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "user"


def test_to_anthropic_without_system():
    b = AgentTurnBuilder().user("Hello")
    payload = b.to_anthropic()
    assert payload["system"] is None
    assert len(payload["messages"]) == 1


def test_to_anthropic_deepcopy():
    b = AgentTurnBuilder().user("hi")
    payload = b.to_anthropic()
    payload["messages"][0]["content"] = "changed"
    assert b.build()[0]["content"] == "hi"


# ---------------------------------------------------------------------------
# to_openai
# ---------------------------------------------------------------------------


def test_to_openai():
    b = AgentTurnBuilder().system("sys").user("u").assistant("a")
    msgs = b.to_openai()
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"


# ---------------------------------------------------------------------------
# build returns deep copy
# ---------------------------------------------------------------------------


def test_build_returns_deep_copy():
    b = AgentTurnBuilder().user("hi")
    m1 = b.build()
    m1[0]["content"] = "changed"
    m2 = b.build()
    assert m2[0]["content"] == "hi"
