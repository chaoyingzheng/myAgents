"""Long-conversation eval: does context management keep a 20-turn chat correct and cheap?

    python -m evals.long_conversation --model openai/gpt-oss-120b
    python -m evals.long_conversation --modes compact tight_budget --tight-budget 2500 --trials 2
    python -m evals.long_conversation --dry-run          # print turns + expected answers only

One scripted conversation with one customer. Every expected answer is computed below
from closed-form formulas, so there are no hand labels. Each turn has a `kind`:

  self       self-contained question
  prev       needs the previous turn (layer 1 keeps the previous turn's tool items)
  pinned     needs only turn 1 (the budget filter always keeps the first message)
  mid        needs a middle turn (dropped first when the budget filter trims)

Context modes (one Weave Model each, same agent version and model):

  full_history   no context management (how chat.py worked before context.py)
  compact        production setting: compaction + the default token budget
  tight_budget   compaction + a small budget, to force the budget filter to drop turns

Expected: compact matches full_history on accuracy with flatter input tokens per turn;
tight_budget keeps self/prev/pinned turns right and loses or asks about `mid` turns.
"""

import argparse
import asyncio
import json
import math
import re
import time
from pathlib import Path

import weave
from agents import RunConfig, Runner, SQLiteSession, trace
from agents.exceptions import AgentsException
from openai import APIError

from evals.scorers import FINAL_RE, extract_number
from money_math.agents_def import SANDBOX_PACKAGES, build_agent
from money_math.config import MODEL_ID, init_weave
from money_math.context import (
    CONTEXT_TOKEN_BUDGET,
    compact_history,
    enforce_token_budget,
    estimate_tokens,
)
from money_math.sandbox_tool import SandboxSession

MODES = ["full_history", "compact", "tight_budget"]
DATASET_NAME = "money_math_long_conversation"
REL_TOL = 0.005  # loose on purpose: this eval measures forgetting (big errors), not rounding


# ---- ground truth --------------------------------------------------------------------------

def pmt(principal: float, annual_rate: float, months: int) -> float:
    r = annual_rate / 12
    return principal * r / (1 - (1 + r) ** -months)


def months_to_payoff(balance: float, annual_rate: float, payment: float) -> int:
    r = annual_rate / 12
    return math.ceil(-math.log(1 - r * balance / payment) / math.log(1 + r))


def build_turns() -> list[dict]:
    mort = pmt(400_000, 0.065, 360)
    car = pmt(32_000, 0.079, 60)
    cagr = ((19_500 / 12_000) ** (1 / 6) - 1) * 100
    r = 0.07 / 12
    t = [
        ("I'm taking a $400,000 mortgage at 6.5% annual interest for 30 years, monthly payments. "
         "What's my monthly payment?", mort, "self", "rel"),
        ("Separately, I have $25,000 in savings earning 4.2% APY, compounded annually. "
         "How much will it be in 5 years?", 25_000 * 1.042 ** 5, "self", "rel"),
        ("And in 10 years?", 25_000 * 1.042 ** 10, "prev", "rel"),
        ("My index fund grew from $12,000 to $19,500 over 6 years. What's its CAGR, in percent?",
         cagr, "self", "rel"),
        ("I'm also considering a $32,000 car loan at 7.9% APR for 60 months. What's the monthly payment?",
         car, "self", "rel"),
        ("What's the total interest on that car loan over the 60 months?", car * 60 - 32_000, "prev", "rel"),
        ("My emergency fund target is 6 months of expenses, and my expenses are $4,850 a month. "
         "How much is the target?", 6 * 4_850, "self", "rel"),
        ("If I save $900 a month toward that target with no interest, how many whole months until I reach it?",
         math.ceil(6 * 4_850 / 900), "prev", "exact"),
        ("If I invest $500 at the end of every month for 25 years at 7% annual return compounded monthly, "
         "what will it be worth?", 500 * ((1 + r) ** 300 - 1) / r, "self", "rel"),
        ("My rent is $2,100 a month and goes up 3.5% each year. What will it be after 4 increases?",
         2_100 * 1.035 ** 4, "self", "rel"),
        ("I have a $6,200 credit card balance at 22.9% APR compounded monthly. Paying $250 a month, "
         "how many months to pay it off? Round up to a whole month.",
         months_to_payoff(6_200, 0.229, 250), "self", "exact"),
        ("With 3% annual inflation, what is $100,000 twenty years from now worth in today's dollars?",
         100_000 / 1.03 ** 20, "self", "rel"),
        ("Back to my mortgage: if the rate were 5.75% instead, what would the monthly payment be?",
         pmt(400_000, 0.0575, 360), "pinned", "rel"),
        ("How much is that per month compared with the payment at my original rate? Give the difference.",
         mort - pmt(400_000, 0.0575, 360), "prev", "rel"),
        ("At the original 6.5% rate, if I pay $500 extra every month, how many months until the "
         "mortgage is paid off? Round up to a whole month.",
         months_to_payoff(400_000, 0.065, mort + 500), "pinned", "exact"),
        ("A quick one: what is 18% of $3,450?", 0.18 * 3_450, "self", "rel"),
        ("What was the CAGR of my index fund again, in percent?", cagr, "mid", "rel"),
        ("For the car loan I mentioned, what would the monthly payment be over 48 months instead of 60?",
         pmt(32_000, 0.079, 48), "mid", "rel"),
        ("For my savings account, what would the balance be after 5 years if the APY were 4.5% instead?",
         25_000 * 1.045 ** 5, "mid", "rel"),
        ("What's my total monthly debt payment: the mortgage at the original rate plus the 60-month car loan?",
         mort + car, "mid", "rel"),
    ]
    return [{"turn": i + 1, "question": q, "expected": round(e, 6), "kind": k, "match": m}
            for i, (q, e, k, m) in enumerate(t)]


# ---- agent ---------------------------------------------------------------------------------

NOTE_RE = re.compile(r"\[Context note: (\d+) earlier")
LAST_OUTPUT: dict[str, dict] = {}  # context_mode -> latest predict output, for the terminal table


class LongConversationAgent(weave.Model):
    """One whole conversation per predict call: same session + sandbox across all turns."""

    version: str
    model_id: str
    context_mode: str
    budget: int

    def _run_config(self, stats: dict) -> RunConfig:
        if self.context_mode == "full_history":
            return RunConfig()

        def budget_filter(data):
            out = enforce_token_budget(data, self.budget)
            stats["prompt_tokens_est"] = max(stats["prompt_tokens_est"], estimate_tokens(out.input, out.instructions))
            for it in out.input:
                m = NOTE_RE.search(str(it.get("content", ""))) if it.get("role") == "user" else None
                if m:
                    stats["dropped_turns"] = max(stats["dropped_turns"], int(m.group(1)))
            return out

        return RunConfig(session_input_callback=compact_history, call_model_input_filter=budget_filter)

    @weave.op
    async def predict(self, turns: list[dict]) -> dict:
        agent = build_agent(self.version, self.model_id, interactive=True)
        sandbox = SandboxSession(packages=SANDBOX_PACKAGES.get(self.version, ()))
        memory = SQLiteSession(f"long-{self.context_mode}-{time.time_ns()}")  # in-memory
        results = []
        try:
            for t in turns:
                stats = {"prompt_tokens_est": 0, "dropped_turns": 0}
                t0 = time.perf_counter()
                usage = None
                try:
                    with trace("money_math_long_conversation", group_id=memory.session_id,
                               metadata={"version": self.version, "context_mode": self.context_mode}):
                        result = await Runner.run(agent, t["question"], session=memory, context=sandbox,
                                                  max_turns=12, run_config=self._run_config(stats))
                    answer = str(result.final_output)
                    usage = result.context_wrapper.usage
                except (AgentsException, APIError) as e:  # like chat.py: one bad turn doesn't end the chat
                    answer = f"AGENT_ERROR: {type(e).__name__}: {e}"
                results.append({
                    "turn": t["turn"],
                    "answer": answer,
                    "input_tokens": usage.input_tokens if usage else None,
                    "output_tokens": usage.output_tokens if usage else None,
                    "llm_calls": usage.requests if usage else None,
                    "latency_s": round(time.perf_counter() - t0, 1),
                    **stats,
                })
        finally:
            await sandbox.close()
        output = {"turns": results, "sandbox_execs": sandbox.exec_count}
        LAST_OUTPUT[self.context_mode] = output
        return output


# ---- scorers -------------------------------------------------------------------------------

def grade(turn: dict, answer: str) -> str:
    """correct | wrong | asked (a question back with no FINAL ANSWER) | error."""
    if answer.startswith("AGENT_ERROR"):
        return "error"
    if not FINAL_RE.search(answer) and "?" in answer:
        return "asked"
    got = extract_number(answer)
    if got is None:
        return "wrong"
    exp = turn["expected"]
    ok = abs(got - exp) <= 0.5 if turn["match"] == "exact" else abs(got - exp) <= REL_TOL * abs(exp)
    return "correct" if ok else "wrong"


@weave.op
def turn_accuracy(turns: list[dict], output: dict) -> dict:
    by_turn = {r["turn"]: r for r in output["turns"]}
    grades = {t["turn"]: grade(t, by_turn[t["turn"]]["answer"]) for t in turns}
    out = {"accuracy": sum(g == "correct" for g in grades.values()) / len(turns)}
    for kind in ("self", "prev", "pinned", "mid"):
        ks = [t["turn"] for t in turns if t["kind"] == kind]
        if ks:
            out[f"acc_{kind}"] = sum(grades[k] == "correct" for k in ks) / len(ks)
    out["asked"] = sum(g == "asked" for g in grades.values())
    out["errors"] = sum(g == "error" for g in grades.values())
    out["grades"] = grades
    return out


@weave.op
def context_growth(output: dict) -> dict:
    """Token growth across the conversation. Flat is good; full_history grows every turn."""
    toks = [r["input_tokens"] for r in output["turns"] if r["input_tokens"] is not None]
    if not toks:
        return {}
    k = max(1, len(toks) // 4)
    first, last = sum(toks[:k]) / k, sum(toks[-k:]) / k
    return {
        "total_input_tokens": sum(toks),
        "first_quarter_avg_input": round(first),
        "last_quarter_avg_input": round(last),
        "growth_ratio": round(last / first, 2) if first else None,
        "turns_with_trimming": sum(r["dropped_turns"] > 0 for r in output["turns"]),
        "max_dropped_turns": max(r["dropped_turns"] for r in output["turns"]),
    }


# ---- main ----------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v2_sandbox")
    p.add_argument("--model", default=None, help="override MODEL_ID")
    p.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    p.add_argument("--tight-budget", type=int, default=2500,
                   help="token budget for tight_budget mode (small enough to force trimming)")
    p.add_argument("--trials", type=int, default=1, help="repeat the conversation N times per mode")
    p.add_argument("--dry-run", action="store_true", help="print turns + expected answers, no agent calls")
    return p.parse_args()


def print_table(name: str, turns: list[dict], output: dict):
    by_turn = {r["turn"]: r for r in output["turns"]}
    print(f"\n== {name}")
    print(f"{'turn':>4} {'kind':<6} {'grade':<7} {'expected':>12} {'got':>12} {'in_tok':>7} {'calls':>5} {'dropped':>7}")
    for t in turns:
        r = by_turn[t["turn"]]
        got = extract_number(r["answer"])
        print(f"{t['turn']:>4} {t['kind']:<6} {grade(t, r['answer']):<7} {t['expected']:>12.2f} "
              f"{(f'{got:.2f}' if got is not None else '-'):>12} {r['input_tokens'] or '-':>7} "
              f"{r['llm_calls'] or '-':>5} {r['dropped_turns']:>7}")


async def main():
    args = parse_args()
    turns = build_turns()
    if args.dry_run:
        for t in turns:
            print(f"{t['turn']:>2} {t['kind']:<6} {t['expected']:>14.4f}  {t['question']}")
        return

    init_weave()
    dataset = weave.Dataset(name=DATASET_NAME, rows=[{"scenario": "household_finances", "turns": turns}])
    weave.publish(dataset)
    model_id = args.model or MODEL_ID

    async def run_mode(mode: str):
        model = LongConversationAgent(
            name=f"money_math_{args.version}_{mode}", version=args.version, model_id=model_id,
            context_mode=mode, budget=args.tight_budget if mode == "tight_budget" else CONTEXT_TOKEN_BUDGET,
        )
        evaluation = weave.Evaluation(name="money_math_long_conversation_eval", dataset=dataset,
                                      scorers=[turn_accuracy, context_growth], trials=args.trials)
        summary = await evaluation.evaluate(
            model, __weave={"display_name": f"long | {mode} | {model_id.split('/')[-1]}"})
        return mode, summary

    # Modes are independent conversations (own session + sandbox), so run them concurrently.
    results = await asyncio.gather(*(run_mode(m) for m in args.modes))

    summaries = {mode: summary for mode, summary in results}
    for mode in summaries:
        if mode in LAST_OUTPUT:
            print_table(f"{mode} (last trial)", turns, LAST_OUTPUT[mode])
    print()
    for mode, summary in summaries.items():
        acc = summary.get("turn_accuracy", {})
        growth = summary.get("context_growth", {})
        print(f"{mode:<13} accuracy={acc.get('accuracy', {}).get('mean')} "
              f"mid={acc.get('acc_mid', {}).get('mean')} "
              f"total_input_tokens={growth.get('total_input_tokens', {}).get('mean')} "
              f"growth_ratio={growth.get('growth_ratio', {}).get('mean')}")

    out = Path("result_backup") / f"long_conversation_{int(time.time())}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"\nSaved summary -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
