"""Re-score existing eval answers with the CURRENT scorer version; no agent rerun.

    python scripts/rescore.py            # dry run: old vs new accuracy + tolerance sweep
    python scripts/rescore.py --apply    # also log the new scores to Weave

How it is logged: `call.apply_scorer(numeric_match)` runs the scorer on each saved
MoneyMathAgent.predict call. Weave records the scorer as a versioned op (numeric_match:vN)
and attaches its result to that call as feedback, so the new score sits next to the original
answer and the old score. The original evaluation summaries are left untouched.

Target: the latest 50-question run (dataset v2) for each agent version x model.
"""

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from money_math.config import WEAVE_PROJECT  # noqa: E402

import weave  # noqa: E402

from evals.scorers import _rounding_unit, extract_number, numeric_match  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--apply", action="store_true", help="log the new scores to Weave")
p.add_argument("--days", type=float, default=7)
args = p.parse_args()

client = weave.init(WEAVE_PROJECT)
since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.days)


def _ts(x):
    return x if x.tzinfo else x.replace(tzinfo=dt.timezone.utc)


def _op(c):
    return c.op_name.split("/")[-1].split(":")[0]


# 1) Pull eval calls and their rows/predict calls in one pass
evals, pas, predicts = {}, defaultdict(list), {}
for c in client.get_calls(sort_by=[{"field": "started_at", "direction": "asc"}], limit=20000):
    if _ts(c.started_at) < since:
        continue
    op = _op(c)
    if op == "Evaluation.evaluate":
        evals[c.id] = c
    elif op == "Evaluation.predict_and_score":
        pas[c.parent_id].append(c)
    elif op == "MoneyMathAgent.predict":
        predicts[c.parent_id] = c  # parent = the predict_and_score call


def config_of(ev):
    m = ev.inputs.get("model")
    m = m.get() if hasattr(m, "get") and not hasattr(m, "version") else m
    return getattr(m, "version", "?"), getattr(m, "model_id", "?")


# 2) Latest complete dataset-v2 run per (version, model)
latest = {}
for eid, ev in evals.items():
    rows = [r for r in pas[eid] if r.output and r.output.get("output")]
    if len(rows) != 50 or not str(rows[0].inputs["example"]["id"]).startswith(("finreason", "gsmp2")):
        continue
    latest[config_of(ev)] = (ev, rows)  # ascending order -> last wins


def v0_correct(exp, got):
    err = abs(got - exp)
    rel = err / abs(exp) if exp else err
    return rel <= 0.005 or err <= _rounding_unit(exp) + 1e-9


def rel_ok(exp, got, tol):
    return abs(got - exp) / abs(exp) <= tol if exp else got == exp


# 3) Dry-run comparison + tolerance sweep
table, flips, todo = [], [], []
for (version, model), (ev, rows) in sorted(latest.items()):
    n_v0 = n_v1 = n_01 = n_05 = n_1 = 0
    for r in rows:
        ex = r.inputs["example"]
        out = r.output["output"]
        got = extract_number(out.get("answer", ""))
        # resolve_fn = the plain function: comparing locally must NOT log a Weave call per row
        new = numeric_match.resolve_fn(expected=ex["expected"], output=out, question=ex["question"])
        old = r.output["scores"]["numeric_match"]["correct"]
        n_v0 += old
        n_v1 += new["correct"]
        if got is not None:
            n_01 += rel_ok(ex["expected"], got, 0.001)
            n_05 += v0_correct(ex["expected"], got)
            n_1 += rel_ok(ex["expected"], got, 0.01) or v0_correct(ex["expected"], got)
        if old != new["correct"]:
            flips.append((version, model.split("/")[-1], ex["id"], ex["expected"], got, old, new["correct"]))
        if r.id in predicts:
            todo.append((predicts[r.id], ex))
    table.append((version, model.split("/")[-1], n_v0, n_v1, n_01, n_05, n_1))

print(f"\n{'version':18s} {'model':16s} {'v0 (logged)':>11s} {'v1 strict':>10s} {'rel 0.1%':>9s} {'rel 0.5%':>9s} {'rel 1%':>7s}")
for v, m, a, b, c, d, e in table:
    print(f"{v:18s} {m:16s} {a:>8d}/50 {b:>7d}/50 {c:>6d}/50 {d:>6d}/50 {e:>4d}/50")
print("\nFlips (v0 -> v1):")
for f in flips:
    print(f"  {f[0]:18s} {f[1]:16s} {f[2]:22s} expected {f[3]:<10g} got {f[4]!s:<12} {f[5]} -> {f[6]}")

out_dir = Path("result_backup")
out_dir.mkdir(exist_ok=True)
(out_dir / f"rescore_{int(time.time())}.json").write_text(json.dumps({"table": table, "flips": flips}, default=str, indent=1))


# 4) Log to Weave
async def apply_all():
    sem = asyncio.Semaphore(16)
    done = errors = skipped = 0
    scorer_uri = weave.publish(numeric_match).uri()  # the current scorer version's ref

    async def one(call, ex):
        nonlocal done, errors, skipped
        async with sem:
            # idempotent: don't attach the same scorer version twice
            existing = await asyncio.to_thread(lambda: list(client.get_call(call.id).feedback))
            if any(getattr(f, "runnable_ref", None) == scorer_uri for f in existing):
                skipped += 1
                return
            try:
                # Weave stores predict's `question` input as a ref to the dataset row, so pass the
                # plain text explicitly (otherwise the scorer receives an ObjectRef).
                await call.apply_scorer(
                    numeric_match,
                    additional_scorer_kwargs={"expected": ex["expected"], "question": ex["question"]},
                )
                done += 1
            except Exception as e:  # keep going; report at the end
                errors += 1
                print("  apply failed:", ex["id"], type(e).__name__, str(e)[:120])

    await asyncio.gather(*(one(c, ex) for c, ex in todo))
    print(f"\nLogged {scorer_uri} to {done}/{len(todo)} predict calls; "
          f"{skipped} already had it; {errors} errors")


if args.apply:
    asyncio.run(apply_all())
else:
    print(f"\nDry run. {len(todo)} predict calls would get the new score. Re-run with --apply to log to Weave.")
