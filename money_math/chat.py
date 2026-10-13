"""Interactive multi-turn chat for the live demo.

    python -m money_math.chat [--version v2_sandbox] [--session demo1]

Memory: an Agents SDK SQLiteSession stores the conversation history, so
"what if I pay $500 extra?" reuses the loan details from the previous turn.
All turns are grouped as one conversation in the Weave Agents view (group_id).
"""

import argparse
import asyncio
import uuid

from .config import init_weave


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v2_sandbox")
    p.add_argument("--session", default=None, help="reuse a session id to resume memory")
    p.add_argument("--model", default=None)
    args = p.parse_args()

    init_weave()
    from agents import Runner, SQLiteSession, trace

    from .agents_def import SANDBOX_PACKAGES, build_agent
    from .sandbox_tool import SandboxSession

    session_id = args.session or f"chat-{uuid.uuid4().hex[:8]}"
    memory = SQLiteSession(session_id, "conversations.db")
    agent = build_agent(args.version, args.model, interactive=True)
    sandbox = SandboxSession(packages=SANDBOX_PACKAGES.get(args.version, ()))

    print(f"Money Math ({args.version}) | session {session_id} | type 'exit' to quit\n")
    try:
        while True:
            q = (await asyncio.to_thread(input, "you> ")).strip()
            if q.lower() in {"exit", "quit"}:
                break
            if not q:
                continue
            with trace("money_math_chat", group_id=session_id, metadata={"version": args.version}):
                result = await Runner.run(agent, q, session=memory, context=sandbox, max_turns=12)
            print(f"\nagent> {result.final_output}\n")
    finally:
        await sandbox.close()


if __name__ == "__main__":
    asyncio.run(main())
