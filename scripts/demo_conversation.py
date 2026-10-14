"""Scripted multi-turn conversation (demo backup + memory test).

    python scripts/demo_conversation.py [--version v2_sandbox] [--model MODEL_ID]

Turn 2 only makes sense if the agent remembers turn 1 (session memory).
"""

import argparse
import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from money_math.config import init_weave  # noqa: E402

TURNS = [
    "I'm taking a $400,000 mortgage at 6.5% for 30 years. What's my monthly payment?",
    "What if I pay an extra $500 every month? How many years until it's paid off, and how much interest do I save?",
]


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v2_sandbox")
    p.add_argument("--model", default=None)
    args = p.parse_args()

    init_weave()
    from agents import Runner, SQLiteSession, trace

    from money_math.agents_def import SANDBOX_PACKAGES, build_agent
    from money_math.context import chat_run_config
    from money_math.sandbox_tool import SandboxSession

    session_id = f"demo-{uuid.uuid4().hex[:8]}"
    memory = SQLiteSession(session_id)  # in-memory for the scripted demo
    agent = build_agent(args.version, args.model, interactive=True)
    sandbox = SandboxSession(packages=SANDBOX_PACKAGES.get(args.version, ()))
    try:
        for q in TURNS:
            print(f"\nyou> {q}")
            with trace("money_math_chat", group_id=session_id, metadata={"version": args.version}):
                result = await Runner.run(agent, q, session=memory, context=sandbox, max_turns=12, run_config=chat_run_config())
            print(f"agent> {result.final_output}")
        print(
            f"\n[sandbox] cold start {sandbox.cold_start_seconds or 0:.1f}s, "
            f"{sandbox.exec_count} execs, {sandbox.exec_errors} errors"
        )
    finally:
        await sandbox.close()


if __name__ == "__main__":
    asyncio.run(main())
