"""Judge alignment: how well does the question-only LLM judge agree with HUMAN ambiguity labels?
Logged as a Weave Evaluation (the judge is the "model", human labels are the ground truth).

    python -m evals.judge_alignment
"""

import asyncio

import weave

from evals.build_dataset import DATASET_NAME
from evals.scorers import JUDGE_MODEL, judge_question_ambiguity
from money_math.config import init_weave


class QuestionAmbiguityJudge(weave.Model):
    judge_model: str

    @weave.op
    async def predict(self, question: str) -> dict:
        return await judge_question_ambiguity(question)


@weave.op
def agrees_with_human(human_ambiguous: bool, output: dict) -> dict:
    judge = bool(output.get("ambiguous"))
    human = bool(human_ambiguous)
    return {
        "agree": judge == human,
        "true_positive": judge and human,
        "false_positive": judge and not human,
        "false_negative": human and not judge,
        "true_negative": not judge and not human,
    }


async def main():
    init_weave()
    dataset = weave.ref(f"{DATASET_NAME}:latest").get()
    ev = weave.Evaluation(name="judge_alignment_question_ambiguity", dataset=dataset, scorers=[agrees_with_human])
    summary = await ev.evaluate(QuestionAmbiguityJudge(name="question_ambiguity_judge", judge_model=JUDGE_MODEL))
    s = summary["agrees_with_human"]
    tp, fp, fn, tn = (s[k]["true_count"] for k in ("true_positive", "false_positive", "false_negative", "true_negative"))
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"\nJudge vs human (n={n}): agreement {po:.0%} | TP {tp} FP {fp} FN {fn} TN {tn} | Cohen's kappa {kappa:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
