"""Agent definitions. Four versions so evals can compare them:

  v1_no_tool        LLM does the math "in its head" (baseline).
  v2_sandbox        LLM must write Python and run it in a CoreWeave Sandbox.
  v3_sandbox_check  v2 + a `checker` sub-agent (agent-as-tool) that independently
                    recomputes the answer before the solver replies.
  v4_clarify        v2 + numpy-financial in the sandbox + an ambiguity policy: ask a
                    clarifying question in chat, or state assumptions + the alternative
                    answer when it can't ask (single-turn evals).

Each version has a distinct agent name so they are listed separately in the Agents view.
"""

from agents import Agent, ModelSettings, OpenAIChatCompletionsModel

from .config import MODEL_ID, inference_client
from .sandbox_tool import SandboxSession, run_python

ANSWER_FORMAT = (
    "End every reply with a final line in exactly this format:\n"
    "FINAL ANSWER: <number>\n"
    "Use a plain number with no units, commas or currency symbols (e.g. 2528.27). "
    "Round money to 2 decimals unless the question asks otherwise."
)

BASE_PERSONA = (
    "You are Money Math, a careful personal-finance assistant for a retail bank's customers. "
    "You answer questions about loans, mortgages, savings, interest and everyday math. "
    "Use details the customer gave earlier in the conversation when they ask follow-ups. "
    "Briefly explain the method in plain language, then give the result."
)

SOLVER_TOOL_RULES = (
    "Never do arithmetic in your head. ALWAYS call run_python to compute numbers, "
    "printing the values you need. If the tool returns a non-zero exit_code or an error, "
    "read the stderr, fix the code and call run_python again. Only report numbers that "
    "actually appear in a run_python output; never estimate or invent a number."
)


FINANCE_LIB_RULES = (
    "The sandbox also has numpy_financial installed (import numpy_financial as npf). Prefer its "
    "vetted functions over hand-written formulas: npf.pmt, npf.nper, npf.ipmt, npf.ppmt, "
    "npf.fv, npf.pv, npf.rate, npf.npv, npf.irr. Mind the sign convention (cash paid out is negative)."
)

AMBIGUITY_RULES_CHAT = (
    "If the question can reasonably be read in more than one way AND the readings give materially "
    "different answers (more than about 1% apart), do not guess: ask ONE short clarifying question "
    "that names the options, and stop. If the difference is negligible, answer and state the "
    "assumption you made."
)

# v4 smoke test: a conditional "if ambiguous, state assumptions" was ignored. Models are
# overconfident in their own reading. So the check is a mandatory step on every answer;
# the flags_ambiguity judge penalizes flagging clear questions, so hedging can't game it.
AMBIGUITY_RULES_EVAL = (
    "This is a one-shot question; you cannot ask the customer anything. Before answering, "
    "re-read the question and list every interpretation choice you had to make (e.g. what a "
    "percentage applies to, whether a discount also reduces fees, rounding direction, sign "
    "convention). Then, just before the final line, ALWAYS write:\n"
    "ASSUMPTIONS: <each choice that changes the answer by more than ~1%, or 'none'>\n"
    "ALTERNATIVE: <the other reading> -> <its number>   (only if ASSUMPTIONS is not 'none'; "
    "compute it with run_python too)\n"
    "Use the most likely reading for the FINAL ANSWER."
)

# Packages each version needs inside its sandbox (installed once per sandbox).
SANDBOX_PACKAGES = {"v4_clarify": ("numpy-financial",)}


def _model(model_id: str | None = None) -> OpenAIChatCompletionsModel:
    return OpenAIChatCompletionsModel(model=model_id or MODEL_ID, openai_client=inference_client())


def build_agent(
    version: str = "v2_sandbox", model_id: str | None = None, interactive: bool = False
) -> Agent[SandboxSession]:
    """interactive=True for live chat (v4 may ask clarifying questions);
    False for single-turn evals (v4 states assumptions instead)."""
    settings = ModelSettings(temperature=0.0)
    # Bug found via Weave trace: with a chatty persona the model *described* the
    # computation ("we will now compute...") and ended the turn with 0 tool calls.
    # Fix: tool rule first in the prompt + force a tool call on the first model step.
    # The SDK resets tool_choice to auto after a tool runs (reset_tool_choice=True),
    # so this can't loop forever.
    # Exception: gpt-oss-120b on the inference endpoint returns an EMPTY message when
    # tool_choice="required" (works with "auto"), so the forcing is model-aware.
    model_name = model_id or MODEL_ID
    tool_choice = "auto" if model_name.startswith("openai/gpt-oss") else "required"
    tool_settings = ModelSettings(temperature=0.0, tool_choice=tool_choice)

    if version == "v1_no_tool":
        return Agent(
            name="money_math_v1_no_tool",
            instructions=f"{BASE_PERSONA}\n\n{ANSWER_FORMAT}",
            model=_model(model_id),
            model_settings=settings,
        )

    if version == "v2_sandbox":
        return Agent(
            name="money_math_v2_sandbox",
            instructions=f"{SOLVER_TOOL_RULES}\n\n{BASE_PERSONA}\n\n{ANSWER_FORMAT}",
            tools=[run_python],
            model=_model(model_id),
            model_settings=tool_settings,
        )

    if version == "v3_sandbox_check":
        checker = Agent(
            name="money_math_v3_checker",
            instructions=(
                "You are an independent verifier. You receive a math/finance question and a proposed "
                "answer. Recompute the answer from scratch with run_python, using a different method "
                "where possible. Reply 'VERIFIED: <number>' if it matches (within 0.5%), otherwise "
                "'MISMATCH: <your number>' plus a one-line reason."
            ),
            tools=[run_python],
            model=_model(model_id),
            model_settings=tool_settings,
        )
        return Agent(
            name="money_math_v3_solver",
            instructions=(
                f"{SOLVER_TOOL_RULES}\n\n{BASE_PERSONA}\n\n"
                "Before replying, call verify_answer with the question and your proposed number. "
                "If it reports MISMATCH, re-examine your work and fix it.\n\n"
                f"{ANSWER_FORMAT}"
            ),
            tools=[
                run_python,
                checker.as_tool(
                    tool_name="verify_answer",
                    tool_description="Independently recompute and verify a proposed numeric answer.",
                ),
            ],
            model=_model(model_id),
            model_settings=tool_settings,
        )

    if version == "v4_clarify":
        ambiguity = AMBIGUITY_RULES_CHAT if interactive else AMBIGUITY_RULES_EVAL
        return Agent(
            name="money_math_v4_clarify",
            instructions=(
                f"{SOLVER_TOOL_RULES}\n\n{FINANCE_LIB_RULES}\n\n{BASE_PERSONA}\n\n"
                f"{ambiguity}\n\n{ANSWER_FORMAT}"
                + ("\nIf you are asking a clarifying question, omit the FINAL ANSWER line." if interactive else "")
            ),
            tools=[run_python],
            model=_model(model_id),
            model_settings=tool_settings,
        )

    raise ValueError(f"unknown version: {version}")


VERSIONS = ["v1_no_tool", "v2_sandbox", "v3_sandbox_check", "v4_clarify"]
