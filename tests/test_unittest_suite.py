"""Standard-library ``unittest`` suite for :mod:`agent_turn_builder`.

This suite mirrors the pytest-based tests but uses only the Python
standard library so it can run in environments without third-party test
dependencies::

    python3 -m unittest discover -s tests

It imports and exercises the real :class:`AgentTurnBuilder` code rather
than re-implementing any of its behaviour.
"""

from __future__ import annotations

import os
import sys
import unittest

# Make ``src`` importable when running from a source checkout (e.g. CI that
# has not installed the package).  When the package is already installed this
# is a harmless no-op.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agent_turn_builder import AgentTurnBuilder, TurnError  # noqa: E402


class ConstructorReprTests(unittest.TestCase):
    def test_repr_empty(self) -> None:
        b = AgentTurnBuilder()
        self.assertIn("count=0", repr(b))
        self.assertIn("last_role=None", repr(b))

    def test_repr_after_add(self) -> None:
        b = AgentTurnBuilder().user("hi")
        self.assertIn("count=1", repr(b))
        self.assertIn("last_role='user'", repr(b))

    def test_initial_empty(self) -> None:
        b = AgentTurnBuilder()
        self.assertTrue(b.is_empty())
        self.assertEqual(b.count(), 0)
        self.assertIsNone(b.last_role())
        self.assertEqual(b.roles(), [])


class RoleMethodTests(unittest.TestCase):
    def test_system(self) -> None:
        msgs = AgentTurnBuilder().system("Be helpful.").build()
        self.assertEqual(msgs, [{"role": "system", "content": "Be helpful."}])

    def test_user(self) -> None:
        msgs = AgentTurnBuilder().user("Hello").build()
        self.assertEqual(msgs, [{"role": "user", "content": "Hello"}])

    def test_assistant(self) -> None:
        msgs = AgentTurnBuilder().assistant("Hi!").build()
        self.assertEqual(msgs, [{"role": "assistant", "content": "Hi!"}])

    def test_three_turn_conversation(self) -> None:
        msgs = (
            AgentTurnBuilder()
            .system("You are helpful.")
            .user("What is 2+2?")
            .assistant("4")
            .build()
        )
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant"])

    def test_chaining_returns_self(self) -> None:
        b = AgentTurnBuilder()
        self.assertIs(b.user("x"), b)
        self.assertIs(b.assistant("y"), b)
        self.assertIs(b.system("s"), b)


class AddTests(unittest.TestCase):
    def test_add_known_role(self) -> None:
        b = AgentTurnBuilder().add("user", "hello")
        self.assertEqual(b.last_role(), "user")

    def test_add_unknown_role_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            AgentTurnBuilder().add("bot", "text")
        self.assertIn("Unknown role", str(ctx.exception))

    def test_add_tool_role(self) -> None:
        b = AgentTurnBuilder().add("tool", "tool output")
        self.assertEqual(b.last_role(), "tool")


class ContentListTests(unittest.TestCase):
    def test_user_with_content_list(self) -> None:
        content = [{"type": "text", "text": "hi"}]
        msgs = AgentTurnBuilder().user(content).build()
        self.assertEqual(msgs[0]["content"], content)

    def test_build_deepcopies_list_content(self) -> None:
        content = [{"type": "text", "text": "hi"}]
        b = AgentTurnBuilder().user(content)
        msgs = b.build()
        msgs[0]["content"][0]["text"] = "changed"
        self.assertEqual(b.build()[0]["content"][0]["text"], "hi")

    def test_mutating_source_list_does_not_leak(self) -> None:
        content = [{"type": "text", "text": "hi"}]
        b = AgentTurnBuilder().user(content)
        # Appending content takes a defensive deep copy, so mutating the
        # caller's original list afterwards must not affect builder state.
        first = b.build()
        content.append({"type": "text", "text": "later"})
        self.assertEqual(b.build(), first)

    def test_mutating_source_dict_in_list_does_not_leak(self) -> None:
        content = [{"type": "text", "text": "hi"}]
        b = AgentTurnBuilder().user(content)
        content[0]["text"] = "MUTATED"
        self.assertEqual(b.build()[0]["content"][0]["text"], "hi")

    def test_add_with_list_content_is_isolated(self) -> None:
        content = [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]
        b = AgentTurnBuilder().add("user", content)
        content[0]["content"] = "changed"
        self.assertEqual(b.build()[0]["content"][0]["content"], "ok")


class ToolBlockTests(unittest.TestCase):
    def test_tool_use_block(self) -> None:
        msgs = (
            AgentTurnBuilder()
            .tool_use("tu_001", "search", {"query": "python"})
            .build()
        )
        self.assertEqual(msgs[0]["role"], "assistant")
        block = msgs[0]["content"][0]
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["id"], "tu_001")
        self.assertEqual(block["name"], "search")
        self.assertEqual(block["input"], {"query": "python"})

    def test_tool_result_block(self) -> None:
        msgs = AgentTurnBuilder().tool_result("tu_001", "result text").build()
        self.assertEqual(msgs[0]["role"], "user")
        block = msgs[0]["content"][0]
        self.assertEqual(block["type"], "tool_result")
        self.assertEqual(block["tool_use_id"], "tu_001")
        self.assertEqual(block["content"], "result text")

    def test_tool_use_deepcopies_inputs(self) -> None:
        inputs = {"q": "hello"}
        b = AgentTurnBuilder().tool_use("id1", "fn", inputs)
        inputs["q"] = "changed"
        block = b.build()[0]["content"][0]
        self.assertEqual(block["input"]["q"], "hello")

    def test_tool_use_custom_role(self) -> None:
        msgs = AgentTurnBuilder().tool_use("id1", "fn", {}, role="user").build()
        self.assertEqual(msgs[0]["role"], "user")

    def test_tool_result_custom_role(self) -> None:
        msgs = AgentTurnBuilder().tool_result("id1", "out", role="assistant").build()
        self.assertEqual(msgs[0]["role"], "assistant")

    def test_tool_use_then_result_pairing(self) -> None:
        msgs = (
            AgentTurnBuilder()
            .user("search for cats")
            .tool_use("tu_1", "search", {"q": "cats"})
            .tool_result("tu_1", "found cats")
            .build()
        )
        self.assertEqual(msgs[1]["content"][0]["id"], "tu_1")
        self.assertEqual(msgs[2]["content"][0]["tool_use_id"], "tu_1")


class IntrospectionTests(unittest.TestCase):
    def test_count(self) -> None:
        b = AgentTurnBuilder().user("a").assistant("b").user("c")
        self.assertEqual(b.count(), 3)

    def test_roles(self) -> None:
        b = AgentTurnBuilder().system("s").user("u").assistant("a")
        self.assertEqual(b.roles(), ["system", "user", "assistant"])

    def test_has_system_true(self) -> None:
        self.assertTrue(AgentTurnBuilder().system("s").has_system())

    def test_has_system_false(self) -> None:
        self.assertFalse(AgentTurnBuilder().user("u").has_system())

    def test_last_role(self) -> None:
        b = AgentTurnBuilder().user("a").assistant("b")
        self.assertEqual(b.last_role(), "assistant")


class ValidateTests(unittest.TestCase):
    def test_validate_empty(self) -> None:
        self.assertEqual(AgentTurnBuilder().validate(), [])

    def test_validate_valid(self) -> None:
        b = AgentTurnBuilder().system("s").user("u").assistant("a")
        self.assertEqual(b.validate(), [])

    def test_validate_system_not_first(self) -> None:
        errors = AgentTurnBuilder().user("u").system("s").validate()
        self.assertTrue(any("system" in e for e in errors))

    def test_validate_consecutive_user(self) -> None:
        errors = AgentTurnBuilder().user("a").user("b").validate()
        self.assertTrue(any("user" in e for e in errors))

    def test_validate_consecutive_assistant(self) -> None:
        errors = AgentTurnBuilder().assistant("a").assistant("b").validate()
        self.assertTrue(any("assistant" in e for e in errors))

    def test_validate_ignores_tool_role_for_alternation(self) -> None:
        # A tool message between two assistant turns should not be flagged as a
        # consecutive-assistant violation by itself, but the two assistant
        # turns surrounding it are still adjacent in the conversation stream.
        b = AgentTurnBuilder().user("u").assistant("a").add("tool", "t").user("u2")
        self.assertEqual(b.validate(), [])


class BuildStrictTests(unittest.TestCase):
    def test_build_strict_valid(self) -> None:
        msgs = AgentTurnBuilder(strict=True).user("hi").assistant("hey").build()
        self.assertEqual(len(msgs), 2)

    def test_build_strict_raises(self) -> None:
        with self.assertRaises(TurnError):
            AgentTurnBuilder(strict=True).user("a").user("b").build()

    def test_build_strict_override_false(self) -> None:
        b = AgentTurnBuilder(strict=True).user("a").user("b")
        msgs = b.build(strict=False)
        self.assertEqual(len(msgs), 2)

    def test_build_strict_override_true(self) -> None:
        b = AgentTurnBuilder(strict=False).user("a").user("b")
        with self.assertRaises(TurnError):
            b.build(strict=True)

    def test_turn_error_is_exception(self) -> None:
        self.assertTrue(issubclass(TurnError, Exception))


class ResetTests(unittest.TestCase):
    def test_reset_clears(self) -> None:
        b = AgentTurnBuilder().user("hi").assistant("hey")
        b.reset()
        self.assertTrue(b.is_empty())
        self.assertEqual(b.count(), 0)

    def test_reset_returns_self(self) -> None:
        b = AgentTurnBuilder()
        self.assertIs(b.reset(), b)

    def test_reset_allows_reuse(self) -> None:
        b = AgentTurnBuilder().user("first")
        b.build()
        b.reset().user("second")
        self.assertEqual(b.build(), [{"role": "user", "content": "second"}])


class FromListTests(unittest.TestCase):
    def test_from_list(self) -> None:
        source = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        b = AgentTurnBuilder.from_list(source)
        self.assertEqual(b.count(), 2)
        self.assertEqual(b.roles(), ["user", "assistant"])

    def test_from_list_deepcopy(self) -> None:
        source = [{"role": "user", "content": "hi"}]
        b = AgentTurnBuilder.from_list(source)
        source[0]["content"] = "changed"
        self.assertEqual(b.build()[0]["content"], "hi")

    def test_from_list_empty(self) -> None:
        b = AgentTurnBuilder.from_list([])
        self.assertTrue(b.is_empty())

    def test_from_list_roundtrip_with_build(self) -> None:
        source = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
        b = AgentTurnBuilder.from_list(source)
        self.assertEqual(b.build(), source)


class ToAnthropicTests(unittest.TestCase):
    def test_to_anthropic_with_system(self) -> None:
        b = AgentTurnBuilder().system("Be helpful.").user("Hello").assistant("Hi!")
        payload = b.to_anthropic()
        self.assertEqual(payload["system"], "Be helpful.")
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "user")

    def test_to_anthropic_without_system(self) -> None:
        payload = AgentTurnBuilder().user("Hello").to_anthropic()
        self.assertIsNone(payload["system"])
        self.assertEqual(len(payload["messages"]), 1)

    def test_to_anthropic_deepcopy(self) -> None:
        b = AgentTurnBuilder().user("hi")
        payload = b.to_anthropic()
        payload["messages"][0]["content"] = "changed"
        self.assertEqual(b.build()[0]["content"], "hi")

    def test_to_anthropic_excludes_system_from_messages(self) -> None:
        b = AgentTurnBuilder().system("s").user("u")
        payload = b.to_anthropic()
        self.assertTrue(all(m["role"] != "system" for m in payload["messages"]))


class ToOpenAITests(unittest.TestCase):
    def test_to_openai(self) -> None:
        b = AgentTurnBuilder().system("sys").user("u").assistant("a")
        msgs = b.to_openai()
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[0]["role"], "system")

    def test_to_openai_is_deep_copy(self) -> None:
        b = AgentTurnBuilder().user("hi")
        msgs = b.to_openai()
        msgs[0]["content"] = "changed"
        self.assertEqual(b.build()[0]["content"], "hi")


class BuildDeepCopyTests(unittest.TestCase):
    def test_build_returns_deep_copy(self) -> None:
        b = AgentTurnBuilder().user("hi")
        m1 = b.build()
        m1[0]["content"] = "changed"
        self.assertEqual(b.build()[0]["content"], "hi")

    def test_builds_are_independent(self) -> None:
        b = AgentTurnBuilder().user("hi")
        self.assertIsNot(b.build(), b.build())


if __name__ == "__main__":
    unittest.main()
