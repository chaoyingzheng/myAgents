"""`run_python` tool: executes LLM-written code in a CoreWeave Serverless Sandbox.

Why a sandbox: the code is written by a model, so it is untrusted. It runs in an
isolated, disposable container instead of on the host running the agent.

Design choices (good interview talking points):
- One sandbox per conversation, created lazily on the first tool call and reused,
  so cold start is paid once per conversation, not once per tool call.
- A hard per-exec timeout so an infinite loop can't hang the agent.
- stderr + exit code are returned to the LLM so it can read the error and retry.
- Output is truncated so a huge print can't blow up the context window.
"""

import time
from dataclasses import dataclass, field

from agents import RunContextWrapper, function_tool
from cwsandbox import AuthStrategy, Sandbox
from cwsandbox.exceptions import SandboxTimeoutError

SANDBOX_IMAGE = "python:3.11-slim"
EXEC_TIMEOUT_SECONDS = 20
MAX_OUTPUT_CHARS = 4000


@dataclass
class SandboxSession:
    """Per-conversation sandbox handle, passed to tools via the Agents SDK run context."""

    packages: tuple[str, ...] = ()  # pip packages installed once when the sandbox starts
    sandbox: Sandbox | None = None
    exec_count: int = 0
    exec_errors: int = 0
    cold_start_seconds: float | None = None
    setup_seconds: float | None = None  # package install time, on top of cold start
    exec_seconds: list[float] = field(default_factory=list)

    async def ensure(self) -> Sandbox:
        if self.sandbox is None:
            t0 = time.perf_counter()
            self.sandbox = Sandbox.run(
                container_image=SANDBOX_IMAGE,
                auth=AuthStrategy.WANDB,
                resources={"cpu": "1", "memory": "1Gi"},
                max_lifetime_seconds=900,
                tags=["money-math-agent"],
            )
            await self.sandbox  # wait until RUNNING
            self.cold_start_seconds = time.perf_counter() - t0
            if self.packages:
                # Installed per sandbox for simplicity. Production: bake a custom image
                # or keep a warm pool so this cost isn't paid per conversation.
                t1 = time.perf_counter()
                await self.sandbox.exec(
                    ["pip", "install", "-q", "--disable-pip-version-check", "--root-user-action=ignore", *self.packages],
                    timeout_seconds=120,
                )
                self.setup_seconds = time.perf_counter() - t1
        return self.sandbox

    async def run(self, code: str) -> str:
        sb = await self.ensure()
        self.exec_count += 1
        t0 = time.perf_counter()
        try:
            result = await sb.exec(["python", "-c", code], timeout_seconds=EXEC_TIMEOUT_SECONDS)
        except SandboxTimeoutError:
            self.exec_errors += 1
            return f"exit_code: TIMEOUT\nerror: code did not finish within {EXEC_TIMEOUT_SECONDS}s (infinite loop?)"
        finally:
            self.exec_seconds.append(time.perf_counter() - t0)
        if result.returncode != 0:
            self.exec_errors += 1
        out = (
            f"exit_code: {result.returncode}\n"
            f"stdout:\n{result.stdout[-MAX_OUTPUT_CHARS:]}\n"
            f"stderr:\n{result.stderr[-MAX_OUTPUT_CHARS:]}"
        )
        # Bug found via Weave trace: the model printed f"${{M:.2f}}" (escaped braces),
        # stdout contained no number, and the model invented one. Flag it explicitly.
        if result.returncode == 0 and not any(ch.isdigit() for ch in result.stdout):
            self.exec_errors += 1
            out += (
                "\nWARNING: stdout contains no numbers, so nothing was computed. Check for "
                "escaped f-string braces like {{x}} and print the raw values, then run again."
            )
        return out

    async def close(self):
        if self.sandbox is not None:
            try:
                await self.sandbox.stop(missing_ok=True)
            finally:
                self.sandbox = None


@function_tool
async def run_python(ctx: RunContextWrapper[SandboxSession], code: str) -> str:
    """Run a Python 3.11 script in an isolated sandbox and return its exit code, stdout and stderr.

    Use this for ALL arithmetic. The standard library is available (math, decimal, datetime,
    etc), plus any extra packages named in your instructions. The script must print() the
    values you need.

    Args:
        code: A complete Python script.
    """
    return await ctx.context.run(code)
