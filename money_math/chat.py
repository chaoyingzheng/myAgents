"""Interactive multi-turn chat for the live demo.

    python -m money_math.chat [--version v2_sandbox] [--session demo1]

Memory: an Agents SDK SQLiteSession stores the conversation history, so
"what if I pay $500 extra?" reuses the loan details from the previous turn.
All turns are grouped as one conversation in the Weave Agents view (group_id).
Long chats: context.py compacts old turns and enforces a token budget (see there).
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
    from agents.exceptions import AgentsException
    from openai import APIError

    from .agents_def import SANDBOX_PACKAGES, build_agent
    from .context import chat_run_config
    from .sandbox_tool import SandboxSession

    session_id = args.session or f"chat-{uuid.uuid4().hex[:8]}"
    memory = SQLiteSession(session_id, "conversations.db")
    agent = build_agent(args.version, args.model, interactive=True)
    sandbox = SandboxSession(packages=SANDBOX_PACKAGES.get(args.version, ()))
    run_config = chat_run_config()

    print(f"Money Math ({args.version}) | session {session_id} | type 'exit' to quit\n")
    try:
        while True:
            q = (await asyncio.to_thread(input, "you> ")).strip()
            if q.lower() in {"exit", "quit"}:
                break
            if not q:
                continue
            try:
                with trace("money_math_chat", group_id=session_id, metadata={"version": args.version}):
                    result = await Runner.run(
                        agent, q, session=memory, context=sandbox, max_turns=12, run_config=run_config
                    )
            except (AgentsException, APIError) as e:
                # One failed turn (timeout, context overflow, max turns) shouldn't end the chat.
                print(f"\nagent> Sorry, I couldn't answer that ({type(e).__name__}). Please try again.\n")
                continue
            print(f"\nagent> {result.final_output}\n")
    finally:
        await sandbox.close()


if __name__ == "__main__":
    asyncio.run(main())
