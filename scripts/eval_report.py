"""Summarize recent evaluation runs from Weave into one comparison table:
model x version -> accuracy, latency, tokens, $ cost, sandbox usage, wrong answers.

    python scripts/eval_report.py [--hours 2] [--wrong]

Data sources (all Weave):
  - Evaluation calls (get_calls): accuracy + per-example latency
  - Agent spans (get_agent_spans): tokens per eval run, linked by eval_run_id
  - Agent turn view (get_agent_turn): per-token $ price per model (Weave's cost fields)
"""

import argparse
import datetime as dt
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from money_math.config import WEAVE_PROJECT  # noqa: E402

import weave  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--hours", type=float, default=2)
p.add_argument("--wrong", action="store_true", help="list wrong answers")
args = p.parse_args()

client = weave.init(WEAVE_PROJECT)
since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=args.hours)


def _ts(x):
    t = x if isinstance(x, dt.datetime) else dt.datetime.fromisoformat(str(x))
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


# 1) Evaluations + their per-example rows
evals, rows = {}, defaultdict(list)
for c in client.get_calls(sort_by=[{"field": "started_at", "direction": "asc"}], limit=5000):
    if _ts(c.started_at) < since:
        continue
    op = c.op_name.split("/")[-1].split(":")[0]
    if op == "Evaluation.evaluate":
        evals[c.id] = {"name": c.display_name or c.id, "done": c.ended_at is not None, "model": None}
    elif op == "Evaluation.predict_and_score" and c.parent_id in evals:
        rows[c.parent_id].append(c)

# 2) Tokens per eval run from agent spans; sample $/token per model from the turn view
tokens = defaultdict(lambda: [0, 0])  # eval_run_id -> [input, output]
sample_trace = {}  # model -> trace_id of one chat span
for s in client.get_agent_spans(limit=20000):
    if _ts(s.started_at) < since:
        continue
    if s.operation_name == "chat" and s.eval_run_id in evals:
        tokens[s.eval_run_id][0] += s.input_tokens or 0
        tokens[s.eval_run_id][1] += s.output_tokens or 0
        if s.request_model:
            sample_trace.setdefault(s.request_model, s.trace_id)
            evals[s.eval_run_id]["model"] = s.request_model

price = {}  # model -> ($/input token, $/output token)
for model, trace_id in sample_trace.items():
    for m in client.get_agent_turn(trace_id=trace_id).messages:
        am = getattr(m, "assistant_message", None)
        if am and am.input_tokens and am.input_cost_usd is not None:
            price[model] = (am.input_cost_usd / am.input_tokens, (am.output_cost_usd or 0) / max(1, am.output_tokens or 1))
            break

# 3) Table
hdr = f"{'model':28s} {'eval':24s} {'acc':>9s} {'p50 s':>6s} {'p90 s':>6s} {'wall*':>6s} {'in tok':>8s} {'out tok':>8s} {'$ total':>8s} {'$/q':>8s} {'execs/q':>7s}"
print(hdr)
print("-" * len(hdr))
for eid, e in evals.items():
    rs = [r for r in rows[eid] if r.ended_at is not None and r.output and r.output.get("output")]
    if not rs:
        continue
    ok = sum(1 for r in rs if r.output["scores"]["numeric_match"]["correct"])
    lat = sorted(r.output["output"]["latency_s"] for r in rs)
    execs = st.mean(r.output["output"]["sandbox_execs"] for r in rs)
    tin, tout = tokens[eid]
    pin, pout = price.get(e["model"], (None, None))
    cost = tin * pin + tout * pout if pin is not None else None
    wall = max(_ts(r.ended_at) for r in rs) - min(_ts(r.started_at) for r in rs)
    print(
        f"{(e['model'] or '?')[:28]:28s} {e['name'][:24]:24s} {ok:>3d}/{len(rs):<3d}{ok/len(rs):>3.0%} "
        f"{st.median(lat):6.1f} {lat[int(0.9 * (len(lat) - 1))]:6.1f} {wall.total_seconds():6.0f} "
        f"{tin:8d} {tout:8d} {('%.4f' % cost) if cost is not None else '   n/a':>8s} "
        f"{('%.5f' % (cost / len(rs))) if cost is not None else '   n/a':>8s} {execs:7.2f}"
        + ("" if e["done"] else "  (INCOMPLETE)")
    )
    amb = [r.output["scores"].get("flags_ambiguity") for r in rs]
    amb = [a for a in amb if a and "appropriate" in a]
    if amb:
        n_amb = sum(a["question_ambiguous"] for a in amb)
        caught = sum(a["question_ambiguous"] and a["flagged"] for a in amb)
        false_alarm = sum((not a["question_ambiguous"]) and a["flagged"] for a in amb)
        print(f"{'':28s}   ambiguity: flagged {caught}/{n_amb} ambiguous | false alarms {false_alarm}/{len(amb) - n_amb} clear"
              f" | appropriate {sum(a['appropriate'] for a in amb)}/{len(amb)}")
print("\n* wall = first example start -> last example end (seconds)")

if args.wrong:
    for eid, e in evals.items():
        bad = [r for r in rows[eid] if r.output and r.output.get("output") and not r.output["scores"]["numeric_match"]["correct"]]
        if bad:
            print(f"\n--- wrong: {e['model']} / {e['name']}")
            for r in bad:
                ex = r.inputs["example"]
                ans = r.output["output"]["answer"][-120:].replace("\n", " ")
                print(f"  {ex['id']:28s} expected {ex['expected']:<12g} ...{ans}")
