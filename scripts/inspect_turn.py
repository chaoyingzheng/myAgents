"""Debug helper: print the latest agent turn(s) from Weave (LLM messages + tool calls).

    python scripts/inspect_turn.py [--n 2]

Uses Weave's Agents API (the same data as the Agents view), handy for debugging
from the terminal or from a coding agent.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from money_math.config import WEAVE_PROJECT  # noqa: E402  (loads .env)

import weave  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=1, help="number of most recent turns")
args = p.parse_args()

client = weave.init(WEAVE_PROJECT)
seen = []
for s in client.get_agent_spans(limit=200):
    if s.trace_id not in seen:
        seen.append(s.trace_id)
for trace_id in reversed(seen[: args.n]):
    print(f"\n######## turn {trace_id}")
    for m in client.get_agent_turn(trace_id=trace_id).messages:
        print(json.dumps(m.model_dump(exclude_none=True), default=str, indent=1)[:1500])
