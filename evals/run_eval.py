"""Runs Weave Evaluations of one or more agent versions on the same dataset.

    python -m evals.run_eval --versions v1_no_tool v2_sandbox --limit 10
    python -m evals.run_eval --versions v2_sandbox --parallel 1     # serial baseline
    python -m evals.run_eval --versions v2_sandbox --parallel 20    # fan out: 1 sandbox per case

Each example gets its own fresh sandbox, so cases are fully isolated from each
other and can run concurrently; --parallel controls how many run at once.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--versions", nargs="+", default=["v1_no_tool", "v2_sandbox"])
    p.add_argument("--model", default=None, help="override MODEL_ID")
    p.add_argument("--limit", type=int, default=None, help="only the first N rows (save credits)")
    p.add_argument("--parallel", type=int, default=10, help="concurrent examples (1 = serial)")
    p.add_argument("--dataset", default="money_math_eval:latest")
    p.add_argument("--no-judge", action="store_true", help="skip the flags_ambiguity LLM-judge scorer")
    return p.parse_args()


args = parse_args()
# Weave reads its concurrency setting from the environment.
os.environ["WEAVE_PARALLELISM"] = str(args.parallel)

import weave  # noqa: E402
from agents import Runner, trace  # noqa: E402

from evals.scorers import flags_ambiguity, followed_format, numeric_match, tool_usage  # noqa: E402
from money_math.agents_def import SANDBOX_PACKAGES, build_agent  # noqa: E402
from money_math.config import MODEL_ID, init_weave  # noqa: E402
from money_math.sandbox_tool import SandboxSession  # noqa: E402


class MoneyMathAgent(weave.Model):
    """Weave Model wrapper: version + model_id are tracked as the model's config,
    so the Evaluations comparison view shows exactly what changed between runs."""

    version: str
    model_id: str

    @weave.op
    async def predict(self, question: str) -> dict:
        agent = build_agent(self.version, self.model_id)
        sandbox = SandboxSession(packages=SANDBOX_PACKAGES.get(self.version, ()))  # fresh, isolated sandbox per example
        t0 = time.perf_counter()
        try:
            with trace("money_math_eval", metadata={"version": self.version}):
                result = await Runner.run(agent, question, context=sandbox, max_turns=12)
            answer = str(result.final_output)
        except Exception as e:  # keep the eval running; the failure is visible in the trace
            answer = f"AGENT_ERROR: {type(e).__name__}: {e}"
        finally:
            await sandbox.close()
        return {
            "answer": answer,
            "sandbox_execs": sandbox.exec_count,
            "sandbox_errors": sandbox.exec_errors,
            "sandbox_cold_start_s": sandbox.cold_start_seconds,
            "sandbox_setup_s": sandbox.setup_seconds,
            "latency_s": time.perf_counter() - t0,
        }


async def main():
    init_weave()
    full = weave.ref(args.dataset).get()
    if args.limit:
        # Small slices for cheap smoke runs: round-robin across sources so each is represented.
        by_source: dict[str, list] = {}
        for r in full.rows:
            by_source.setdefault(r["source"], []).append(dict(r))
        mixed = [r for group in zip(*by_source.values()) for r in group]
        dataset = mixed[: args.limit]
    else:
        dataset = full  # pass the published Dataset itself so Weave links eval -> dataset version

    summaries = {}
    for version in args.versions:
        model = MoneyMathAgent(name=f"money_math_{version}", version=version, model_id=args.model or MODEL_ID)
        evaluation = weave.Evaluation(
            name="money_math_benchmark",
            dataset=dataset,
            scorers=[numeric_match, followed_format, tool_usage] + ([] if args.no_judge else [flags_ambiguity]),
        )
        t0 = time.perf_counter()
        summary = await evaluation.evaluate(model, __weave={"display_name": f"{version} | {model.model_id.split('/')[-1]} | p={args.parallel}"})
        wall = time.perf_counter() - t0
        summaries[version] = {"wall_clock_s": round(wall, 1), "parallel": args.parallel, "n": len(dataset.rows) if hasattr(dataset, "rows") else len(dataset), **summary}
        acc = summary.get("numeric_match", {}).get("correct", {}).get("true_fraction")
        print(f"\n== {version}: accuracy={acc} wall_clock={wall:.1f}s (parallel={args.parallel}, n={summaries[version]['n']})")

    out = Path("result_backup") / f"eval_{int(time.time())}.json"  # local copy; Weave holds the source of truth
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"\nSaved summary -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
