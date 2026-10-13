"""Add human ambiguity labels (docs/ambiguity_review.xlsx) to the eval dataset as a NEW version.

    python -m evals.add_human_labels [--sheet docs/ambiguity_review.xlsx]

Blank 'Ambiguous?' cells are treated as clear (N), per the reviewer's instruction.
Rows, ids, questions and expected answers are unchanged; only columns are added, so results
on the previous version stay comparable.
"""

import argparse

import openpyxl
import weave

from evals.build_dataset import DATASET_NAME
from money_math.config import init_weave


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sheet", default="docs/ambiguity_review.xlsx")
    p.add_argument("--base", default=f"{DATASET_NAME}:latest")
    args = p.parse_args()

    ws = openpyxl.load_workbook(args.sheet)["Review"]
    labels = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row[1]:
            continue
        mark = str(row[6] or "").strip().upper()
        labels[row[1]] = {
            "human_ambiguous": mark == "Y",
            "human_reviewed": mark in ("Y", "N"),  # False = blank, defaulted to clear
            "human_alt_reading": row[7] or "",
            "human_notes": row[9] or "",
        }

    init_weave()
    base = weave.ref(args.base).get()
    rows = []
    for r in base.rows:
        r = dict(r)
        if r["id"] not in labels:
            raise SystemExit(f"row {r['id']} missing from the review sheet")
        rows.append({**r, **labels[r["id"]]})

    ref = weave.publish(weave.Dataset(name=DATASET_NAME, rows=rows))
    n_amb = sum(r["human_ambiguous"] for r in rows)
    n_rev = sum(r["human_reviewed"] for r in rows)
    print(f"Published {len(rows)} rows ({n_amb} ambiguous, {n_rev} explicitly reviewed) -> {ref.uri()}")


if __name__ == "__main__":
    main()
