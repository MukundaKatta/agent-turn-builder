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

## Tool calls and provider payloads

The builder also models Anthropic-style tool-call turns and can emit the exact
payload shape each provider expects:

```python
from agent_turn_builder import AgentTurnBuilder

b = (
    AgentTurnBuilder()
    .system("You can search the web.")
    .user("What is the capital of France?")
    .tool_use("toolu_1", "web_search", {"query": "capital of France"})
    .tool_result("toolu_1", "Paris")
    .assistant("The capital of France is Paris.")
)

# Anthropic: system prompt is a separate kwarg, not a message.
payload = b.to_anthropic()
# payload["system"]   -> "You can search the web."
# payload["messages"] -> [user, assistant(tool_use), user(tool_result), assistant]

# OpenAI: everything (including the system message) is one flat list.
messages = b.to_openai()
```

Pass `strict=True` (at construction or per `build`/`validate` call) to reject
sequences that break role-alternation rules:

```python
from agent_turn_builder import AgentTurnBuilder, TurnError

try:
    AgentTurnBuilder(strict=True).user("a").user("b").build()
except TurnError as exc:
    print(exc)  # consecutive 'user' turns at conversation positions 0 and 1
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
- Content is deep-copied on the way **in** and on the way **out**, so neither
  the object you pass nor the list you build can corrupt builder state
- Fully type-hinted and ships a `py.typed` marker (PEP 561)
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

## Development

Run the full test suite (requires the dev extras for `pytest` + `ruff`):

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src tests
```

A standard-library `unittest` suite is also included so the package can be
verified with **no third-party dependencies installed**:

```bash
python -m unittest discover -s tests -p "test_unittest_suite.py" -v
```

## License

MIT
