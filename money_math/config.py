"""Shared setup: env vars, Weave, and the CoreWeave Serverless Inference model client."""

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"
WEAVE_PROJECT = os.environ.get("WEAVE_PROJECT", "money-math-agent")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.6-35B-A3B")

_initialized = False


def init_weave():
    """Init Weave once. Weave auto-patches the OpenAI Agents SDK so every run,
    LLM call and tool call lands in the Agents view."""
    global _initialized
    if _initialized:
        return
    from agents import set_trace_processors

    # Drop the SDK's default exporter (it ships traces to OpenAI's dashboard).
    # Weave registers its own processor in weave.init, so Weave is the only sink.
    set_trace_processors([])

    import weave

    weave.init(WEAVE_PROJECT)
    _initialized = True


def inference_client() -> AsyncOpenAI:
    """OpenAI-compatible client pointed at CoreWeave Serverless Inference.
    Auth is the same W&B API key used for Weave and Sandboxes."""
    return AsyncOpenAI(
        base_url=INFERENCE_BASE_URL,
        api_key=os.environ["WANDB_API_KEY"],
        # Bug found in eval: one request hung for the SDK's 600s default timeout and
        # stalled the whole eval run (614s vs ~40s). Fail fast and retry instead.
        timeout=180,  # reasoning models can need >90s for long answers (seen in eval run 2)
        max_retries=2,
        default_headers={"OpenAI-Project": WEAVE_PROJECT},  # usage attribution
    )
