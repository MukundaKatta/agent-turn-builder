# agent-turn-builder

Fluent builder for multi-turn LLM conversation message arrays.

```python
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
```

## Install

```bash
pip install agent-turn-builder
```

## Features

- Fluent `.system()` / `.user()` / `.assistant()` methods for clean message building
- `.tool_use()` and `.tool_result()` helpers for Anthropic tool-call pairs
- `.add(role, content)` for arbitrary roles (`tool`, custom)
- `content` can be a string **or** a content-block list (multi-modal)
- `strict=True` enforces role-alternation rules (raises `TurnError`)
- `.validate()` returns a list of error strings without raising
- `.to_anthropic()` splits out `system` prompt from messages dict
- `.to_openai()` returns all messages in a flat list
- `.from_list(messages)` loads an existing messages list
- `.reset()` reuses the builder for the next call
- All builds return deep copies — the builder state stays intact
- Zero dependencies

## API

```python
b = AgentTurnBuilder(*, strict=False)

b.system(content)                          -> self
b.user(content)                            -> self
b.assistant(content)                       -> self
b.tool_use(id, name, inputs, *, role=...)  -> self
b.tool_result(id, content, *, role=...)    -> self
b.add(role, content)                       -> self
b.reset()                                  -> self

b.build(*, strict=None)  -> list[dict]
b.validate()             -> list[str]   # error strings
b.to_anthropic()         -> {"system": str|None, "messages": [...]}
b.to_openai()            -> list[dict]

b.count()       -> int
b.last_role()   -> str | None
b.roles()       -> list[str]
b.has_system()  -> bool
b.is_empty()    -> bool

AgentTurnBuilder.from_list(messages)  -> AgentTurnBuilder
```

## License

MIT
