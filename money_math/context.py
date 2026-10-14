"""Long-conversation context management for multi-turn chat.

Without this, every turn resends the whole SQLite history (old code, tool outputs,
tracebacks), so cost grows each turn and a long chat eventually overflows the model's
context window. That error repeats on every later turn of the session.

Two layers, both plugged in through RunConfig:

1. compact_history (session_input_callback): runs once per turn when history is loaded.
   Earlier turns keep only the customer's messages and the agent's replies; their
   run_python calls/outputs are dropped. The replies already state the numbers, so
   follow-ups still work. The most recent KEEP_TOOL_TURNS turns keep their tool items.
2. enforce_token_budget (call_model_input_filter): runs before every LLM call, so it also
   covers growth inside a turn. If the input is still over budget it drops the oldest
   whole turns (the first customer message and the current turn are always kept), then
   shortens older tool outputs in the current turn.

The full history stays in SQLite; this only changes what is sent to the model.
"""

import json
import os

from agents import RunConfig
from agents.run_config import CallModelData, ModelInputData

# Rough budget for the model input. Leaves room for long reasoning output: gpt-oss-120b
# has a 131K window, and Qwen3.6 can spend tens of thousands of tokens reasoning.
CONTEXT_TOKEN_BUDGET = int(os.environ.get("CONTEXT_TOKEN_BUDGET", "60000"))
KEEP_TOOL_TURNS = 1  # previous turns whose tool calls/outputs are still sent
SHORT_TOOL_OUTPUT_CHARS = 500
TRIM_NOTE = (
    "[Context note: {n} earlier conversation turn(s) were omitted to fit the context window. "
    "If you need a detail from them, ask the customer to restate it.]"
)


def _is_user_msg(item) -> bool:
    return item.get("role") == "user" and item.get("type", "message") == "message"


def _is_chat_msg(item) -> bool:
    return item.get("role") in ("user", "assistant") and item.get("type", "message") == "message"


def _turn_starts(items) -> list[int]:
    return [i for i, it in enumerate(items) if _is_user_msg(it)]


def estimate_tokens(items, instructions: str | None = None) -> int:
    """~4 chars per token. Close enough for a budget guard; no tokenizer dependency."""
    chars = len(json.dumps(items, default=str)) + len(instructions or "")
    return chars // 4


def compact_history(history: list, new_input: list) -> list:
    starts = _turn_starts(history)
    cut = starts[-KEEP_TOOL_TURNS] if len(starts) >= KEEP_TOOL_TURNS > 0 else len(history)
    # Return the same item objects (no copies) so the SDK knows they are history and
    # doesn't persist them again as new input.
    kept = [it for it in history[:cut] if _is_chat_msg(it)] + history[cut:]
    return kept + new_input


def enforce_token_budget(data: CallModelData, budget: int = CONTEXT_TOKEN_BUDGET) -> ModelInputData:
    items = list(data.model_data.input)
    instructions = data.model_data.instructions
    if estimate_tokens(items, instructions) <= budget:
        return data.model_data

    # 1) Drop the oldest whole turns, keeping turn 0 (usually the loan details) and the
    #    current turn. Whole turns keep tool call/output pairs intact.
    dropped = 0
    starts = _turn_starts(items)
    while len(starts) > 2 and estimate_tokens(items, instructions) > budget:
        del items[starts[1] : starts[2]]
        dropped += 1
        starts = _turn_starts(items)
    if dropped:
        note = {"role": "user", "content": TRIM_NOTE.format(n=dropped)}
        items.insert(starts[1] if len(starts) > 1 else 0, note)

    # 2) Still over (a long tool loop inside one turn): shorten every tool output except
    #    the latest one, which the model is about to read.
    if estimate_tokens(items, instructions) > budget:
        outputs = [i for i, it in enumerate(items) if it.get("type") == "function_call_output"]
        for i in outputs[:-1]:
            out = items[i].get("output")
            if isinstance(out, str) and len(out) > SHORT_TOOL_OUTPUT_CHARS:
                items[i] = {**items[i], "output": out[:SHORT_TOOL_OUTPUT_CHARS] + "\n[...truncated]"}

    return ModelInputData(input=items, instructions=instructions)


def chat_run_config(budget: int = CONTEXT_TOKEN_BUDGET, **kwargs) -> RunConfig:
    """RunConfig for multi-turn chat with history compaction + a token budget."""
    return RunConfig(
        session_input_callback=compact_history,
        call_model_input_filter=lambda data: enforce_token_budget(data, budget),
        **kwargs,
    )
