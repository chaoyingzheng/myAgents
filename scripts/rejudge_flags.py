"""Re-judge "did the reply surface an ambiguity?" (flags_ambiguity v2) on SAVED answers, scored
against the HUMAN labels. No agent rerun; results are logged to Weave as versioned feedback.

    python scripts/rejudge_flags.py [--versions v2_sandbox v4_clarify]
"""

import argparse
import asyncio
import datetime as dt
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from money_math.config import WEAVE_PROJECT  # noqa: E402

import weave  # noqa: E402

from evals.build_dataset import DATASET_NAME  # noqa: E402
from evals.scorers import flags_ambiguity  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--versions", nargs="+", default=["v2_sandbox", "v4_clarify"])
p.add_argument("--days", type=float, default=7)
args = p.parse_args()

client = weave.init(WEAVE_PROJECT)
since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.days)
human = {r["id"]: bool(r["human_ambiguous"]) for r in weave.ref(f"{DATASET_NAME}:latest").get().rows}


def _ts(x):
    return x if x.tzinfo else x.replace(tzinfo=dt.timezone.utc)


evals, pas, predicts = {}, defaultdict(list), {}
for c in client.get_calls(sort_by=[{"field": "started_at", "direction": "asc"}], limit=20000):
    if _ts(c.started_at) < since:
        continue
    op = c.op_name.split("/")[-1].split(":")[0]
    if op == "Evaluation.evaluate":
        evals[c.id] = c
    elif op == "Evaluation.predict_and_score":
        pas[c.parent_id].append(c)
    elif op == "MoneyMathAgent.predict":
        predicts[c.parent_id] = c

latest = {}
for eid, ev in evals.items():
    rows = [r for r in pas[eid] if r.output and r.output.get("output")]
    if len(rows) != 50 or not str(rows[0].inputs["example"]["id"]).startswith(("finreason", "gsmp2")):
        continue
    m = ev.inputs.get("model")
    version, model = getattr(m, "version", "?"), getattr(m, "model_id", "?")
    if version in args.versions:
        latest[(version, model)] = rows


async def main():
    sem = asyncio.Semaphore(16)
    results = defaultdict(list)

    async def one(key, r):
        ex = r.inputs["example"]
        call = predicts.get(r.id)
        if call is None:
            return
        async with sem:
            res = await call.apply_scorer(
                flags_ambiguity,
                additional_scorer_kwargs={"question": ex["question"], "human_ambiguous": human[ex["id"]]},
            )
        results[key].append((ex["id"], res.result))

    await asyncio.gather(*(one(k, r) for k, rows in latest.items() for r in rows))

    print(f"\n{'version':14s} {'model':16s} {'flagged':>8s} {'caught (of human-ambiguous)':>28s} {'false alarms (of clear)':>24s} {'appropriate':>12s}")
    for (version, model), res in sorted(results.items()):
        ok = [x for _, x in res if "flagged" in x]
        n_amb = sum(x["human_ambiguous"] for x in ok)
        print(f"{version:14s} {model.split('/')[-1]:16s} {sum(x['flagged'] for x in ok):>5d}/{len(ok)}"
              f" {sum(x['caught'] for x in ok):>22d}/{n_amb}"
              f" {sum(x['false_alarm'] for x in ok):>20d}/{len(ok) - n_amb}"
              f" {sum(x['appropriate'] for x in ok):>9d}/{len(ok)}")
    print(f"\nLogged {weave.publish(flags_ambiguity).uri()}")


asyncio.run(main())
