"""Post-hoc metrics from existing eval data (no rerun):
  - final-or-alternative: is the correct answer either the FINAL ANSWER or a stated ALTERNATIVE?
  - timeouts: how many misses were API timeouts (infra), and accuracy excluding them.

    python scripts/alt_metric.py [--hours 2]
"""

import argparse
import datetime as dt
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from money_math.config import WEAVE_PROJECT  # noqa: E402

import weave  # noqa: E402

from evals.scorers import _rounding_unit  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--hours", type=float, default=2)
args = p.parse_args()

ALT_RE = re.compile(r"ALTERNATIVE:.*?(?:->|→)\s*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)


def close(got: float, exp: float) -> bool:
    err = abs(got - exp)
    return (err / abs(exp) if exp else err) <= 0.005 or err <= _rounding_unit(exp) + 1e-9


client = weave.init(WEAVE_PROJECT)
since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=args.hours)
evals, rows = {}, {}
for c in client.get_calls(sort_by=[{"field": "started_at", "direction": "asc"}], limit=5000):
    started = c.started_at if c.started_at.tzinfo else c.started_at.replace(tzinfo=dt.timezone.utc)
    if started < since:
        continue
    op = c.op_name.split("/")[-1].split(":")[0]
    if op == "Evaluation.evaluate":
        evals[c.id] = c.display_name or c.id
        rows[c.id] = []
    elif op == "Evaluation.predict_and_score" and c.parent_id in rows and c.output and c.output.get("output"):
        rows[c.parent_id].append(c)

for eid, name in evals.items():
    rs = rows[eid]
    if len(rs) < 20:  # skip smoke runs
        continue
    final_ok = alt_ok = timeouts = 0
    for r in rs:
        ans = r.output["output"]["answer"]
        exp = r.inputs["example"]["expected"]
        if r.output["scores"]["numeric_match"]["correct"]:
            final_ok += 1
            continue
        if "TimeoutError" in ans:
            timeouts += 1
        alts = []
        for x in ALT_RE.findall(ans):
            try:
                alts.append(float(x.replace(",", "")))
            except ValueError:
                pass
        if any(close(a, exp) for a in alts):
            alt_ok += 1
    n = len(rs)
    print(
        f"{name[:42]:42s} final {final_ok}/{n} | final-or-alt {final_ok + alt_ok}/{n} | timeouts {timeouts}"
        f" | excl. timeouts {final_ok}/{n - timeouts} = {final_ok / max(1, n - timeouts):.0%}"
    )
