"""Fluent builder for multi-turn LLM conversation message arrays.

Builds the ``messages`` list expected by Anthropic and OpenAI APIs.
Enforces basic role-alternation rules and provides helpers for
common patterns (tool_use / tool_result pairing).

Example::

    from agent_turn_builder import AgentTurnBuilder

    msgs = (
        AgentTurnBuilder()
        .system("You are a helpful assistant.")
        .user("What is 2+2?")
        .assistant("4")
        .user("Thanks!")
        .build()
    )
    # [
    #   {"role": "system",    "content": "You are a helpful assistant."},
    #   {"role": "user",      "content": "What is 2+2?"},
    #   {"role": "assistant", "content": "4"},
    #   {"role": "user",      "content": "Thanks!"},
    # ]
"""

from __future__ import annotations

import copy
from typing import Any


class TurnError(Exception):
    """Raised when a message sequence is invalid."""


# Valid role strings
_ROLES = frozenset({"system", "user", "assistant", "tool"})


class AgentTurnBuilder:
    """Build a list of LLM chat messages with a fluent API.

    The builder tracks message order and can optionally enforce
    role-alternation rules (user/assistant must alternate after system).

    Args:
        strict: When ``True``, :meth:`build` raises :class:`TurnError` if
                role-alternation rules are violated.  Default ``False``.
    """

    def __init__(self, *, strict: bool = False) -> None:
        self._messages: list[dict[str, Any]] = []
        self._strict = strict

    # ------------------------------------------------------------------
    # Convenience role methods
    # ------------------------------------------------------------------

    def system(self, content: str) -> AgentTurnBuilder:
        """Append a ``system`` message."""
        return self._append("system", content)

    def user(self, content: str | list[Any]) -> AgentTurnBuilder:
        """Append a ``user`` message.

        *content* may be a plain string or a content-block list (e.g. an
        Anthropic multi-modal content array).
        """
        return self._append("user", content)

    def assistant(self, content: str | list[Any]) -> AgentTurnBuilder:
        """Append an ``assistant`` message."""
        return self._append("assistant", content)

    def tool_result(
        self,
        tool_use_id: str,
        content: str,
        *,
        role: str = "user",
    ) -> AgentTurnBuilder:
        """Append a tool-result message (Anthropic style).

        Builds a ``user`` role message whose content is a list containing a
        single ``tool_result`` block.

        Args:
            tool_use_id: The ``id`` of the corresponding ``tool_use`` block.
            content:     The text result from the tool call.
            role:        Override the wrapper role (default ``"user"``).
        """
        block = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        return self._append(role, [block])

    def tool_use(
        self,
        tool_use_id: str,
        name: str,
        inputs: dict[str, Any],
        *,
        role: str = "assistant",
    ) -> AgentTurnBuilder:
        """Append a tool-use message (Anthropic style).

        Builds an ``assistant`` role message whose content is a list containing
        a single ``tool_use`` block.

        Args:
            tool_use_id: Unique id for this tool call.
            name:        Tool name.
            inputs:      Argument dict for the tool.
            role:        Override the wrapper role (default ``"assistant"``).
        """
        block = {
            "type": "tool_use",
            "id": tool_use_id,
            "name": name,
            "input": copy.deepcopy(inputs),
        }
        return self._append(role, [block])

    def add(
        self,
        role: str,
        content: str | list[Any],
    ) -> AgentTurnBuilder:
        """Append a raw message with an explicit *role*.

        Args:
            role:    One of ``"system"``, ``"user"``, ``"assistant"``,
                     ``"tool"``.
            content: String or content-block list.

        Raises:
            ValueError: For unknown roles.
        """
        if role not in _ROLES:
            raise ValueError(f"Unknown role {role!r}; expected one of {sorted(_ROLES)}")
        return self._append(role, content)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of messages accumulated so far."""
        return len(self._messages)

    def last_role(self) -> str | None:
        """Return the role of the last message, or ``None`` if empty."""
        if not self._messages:
            return None
        return self._messages[-1]["role"]

    def roles(self) -> list[str]:
        """Return the role sequence of all accumulated messages."""
        return [m["role"] for m in self._messages]

    def has_system(self) -> bool:
        """Return ``True`` if a system message has been added."""
        return any(m["role"] == "system" for m in self._messages)

    def is_empty(self) -> bool:
        """Return ``True`` if no messages have been added."""
        return len(self._messages) == 0

    # ------------------------------------------------------------------
    # Build / validation
    # ------------------------------------------------------------------

    def build(self, *, strict: bool | None = None) -> list[dict[str, Any]]:
        """Return a deep copy of the accumulated message list.

        Args:
            strict: Override instance-level ``strict`` flag for this call.
                    When ``True``, validates role-alternation before returning.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts suitable for
            passing to Anthropic or OpenAI chat APIs.

        Raises:
            TurnError: When ``strict=True`` and the message sequence is
                       invalid.
        """
        _strict = self._strict if strict is None else strict
        if _strict:
            self._validate()
        return copy.deepcopy(self._messages)

    def validate(self) -> list[str]:
        """Return a list of validation error strings (empty = valid).

        Does not raise; call :meth:`build` with ``strict=True`` to get an
        exception.
        """
        errors: list[str] = []
        msgs = self._messages
        if not msgs:
            return errors
        # System must come first if present
        for i, m in enumerate(msgs):
            if m["role"] == "system" and i != 0:
                errors.append(f"system message at index {i} must be first")
        # user/assistant should alternate (ignoring system/tool)
        conv = [m["role"] for m in msgs if m["role"] not in ("system", "tool")]
        for i in range(1, len(conv)):
            if conv[i] == conv[i - 1]:
                errors.append(
                    f"consecutive {conv[i]!r} turns at conversation positions "
                    f"{i - 1} and {i}"
                )
        return errors

    def reset(self) -> AgentTurnBuilder:
        """Clear all accumulated messages and return ``self`` for chaining."""
        self._messages.clear()
        return self

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_list(cls, messages: list[dict[str, Any]]) -> AgentTurnBuilder:
        """Create a builder pre-loaded with *messages*.

        A deep copy is taken so the original list is not mutated.
        """
        b = cls()
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            b._append(role, content)
        return b

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_openai(self) -> list[dict[str, Any]]:
        """Return messages in OpenAI format (same as :meth:`build`).

        All roles are passed through as-is; ``tool`` role messages are
        included verbatim.
        """
        return self.build()

    def to_anthropic(self) -> dict[str, Any]:
        """Return an Anthropic-compatible payload fragment.

        Returns a dict with:

        - ``"system"`` — the system message content (str) if present, else
          ``None``.
        - ``"messages"`` — the remaining messages (system excluded).
        """
        system_content: str | None = None
        messages: list[dict[str, Any]] = []
        for m in self._messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                messages.append(copy.deepcopy(m))
        return {"system": system_content, "messages": messages}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append(
        self,
        role: str,
        content: str | list[Any],
    ) -> AgentTurnBuilder:
        self._messages.append({"role": role, "content": content})
        return self

    def _validate(self) -> None:
        errors = self.validate()
        if errors:
            raise TurnError("; ".join(errors))

    def __repr__(self) -> str:
        return f"AgentTurnBuilder(count={self.count()}, last_role={self.last_role()!r})"
