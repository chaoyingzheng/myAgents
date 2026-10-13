"""Builds the eval dataset and publishes it to Weave as a versioned `weave.Dataset`.

    python -m evals.build_dataset [--finreason 30] [--gsm-p2 20] [--seed 7]

Sources (both public, recent, and designed to resist training-data contamination):
  - FinanceReasoning "hard" split (ACL 2025, BUPT-Reasoning-Lab/FinanceReasoning): 238 finance
    questions, many over company-report tables, each with a numeric answer + Python reference.
  - GSM-Symbolic P2 (Apple, 2024, apple/GSM-Symbolic): GSM8K templates re-instantiated with new
    names/numbers plus 2 extra reasoning clauses; models that ace GSM8K drop on P2.

History: v1 of this dataset was 40 random GSM8K-test + 15 hand-written finance questions. Dropped
because GSM8K is likely memorized (no-tool baseline scored 98%) and hand-written labels were
ambiguous (fin-08). See docs/RESULTS.md.
"""

import argparse
import json
import random

import weave
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from money_math.config import init_weave

DATASET_NAME = "money_math_eval"


def finance_reasoning_rows(n: int, seed: int) -> list[dict]:
    path = hf_hub_download("BUPT-Reasoning-Lab/FinanceReasoning", "FinanceReasoning/hard.json", repo_type="dataset")
    data = json.load(open(path, encoding="utf-8"))
    # Numeric answers only (3 of 238 are booleans).
    data = [d for d in data if isinstance(d["ground_truth"], (int, float)) and not isinstance(d["ground_truth"], bool)]
    rows = []
    for d in random.Random(seed).sample(data, n):
        q = d["question"].strip()
        if d.get("context"):
            q = f"{d['context'].strip()}\n\nQuestion: {q}"
        rows.append({
            "id": f"finreason-{d['question_id']}",
            "source": "finreason_hard",
            "question": q,
            "expected": float(d["ground_truth"]),
            "difficulty": float(d["difficulty"]),
        })
    return rows


def gsm_symbolic_rows(n: int, seed: int) -> list[dict]:
    ds = load_dataset("apple/GSM-Symbolic", "p2", split="test")
    rows = []
    for i in random.Random(seed).sample(range(len(ds)), n):
        ex = ds[i]
        rows.append({
            "id": f"gsmp2-{ex['id']}-{ex['instance']}",
            "source": "gsm_symbolic_p2",
            "question": ex["question"],
            "expected": float(ex["answer"].split("####")[-1].strip().replace(",", "")),
            "difficulty": float(ex["answer"].count("\n") + 1),  # rough: solution lines
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--finreason", type=int, default=30)
    p.add_argument("--gsm-p2", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    init_weave()
    rows = finance_reasoning_rows(args.finreason, args.seed) + gsm_symbolic_rows(args.gsm_p2, args.seed)
    ref = weave.publish(weave.Dataset(name=DATASET_NAME, rows=rows))
    print(f"Published {len(rows)} rows -> {ref.uri()}")


if __name__ == "__main__":
    main()
