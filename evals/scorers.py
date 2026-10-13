"""Scorers. Each receives the dataset row fields it names plus `output` (the model's return dict)."""

import re

import weave

FINAL_RE = re.compile(r"FINAL ANSWER:\s*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_number(text: str) -> float | None:
    """Prefer the 'FINAL ANSWER:' line; fall back to the last number in the reply."""
    if not text:
        return None
    m = FINAL_RE.findall(text)
    raw = m[-1] if m else (NUMBER_RE.findall(text) or [None])[-1]
    if raw is None:
        return None
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _rounding_unit(x: float) -> float:
    """Half a unit in the last decimal place of the reference, e.g. 0.035 -> 0.0005, 1152 -> 0.5."""
    s = repr(float(x))
    decimals = 0 if s.endswith(".0") else len(s.split(".")[1]) if "." in s else 0
    return 0.5 * 10 ** (-decimals)


_NUM_WORDS = {"zero": 0, "one": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_DECIMALS_RE = re.compile(r"\b(zero|one|single|two|three|four|five|\d)\s+decimal\s+places?\b", re.IGNORECASE)
_NEAREST = [  # (pattern, decimals)
    (re.compile(r"nearest\s+cent", re.I), 2),
    (re.compile(r"nearest\s+(dollar|integer|whole\s+number|unit)|whole\s+number", re.I), 0),
    (re.compile(r"nearest\s+tenth", re.I), 1),
    (re.compile(r"nearest\s+hundredth", re.I), 2),
    (re.compile(r"nearest\s+thousandth", re.I), 3),
]


def requested_decimals(question: str) -> int | None:
    """The precision the question asks for ('three decimal places', 'nearest dollar', ...), if any."""
    q = question or ""
    m = _DECIMALS_RE.search(q)
    if m:
        w = m.group(1).lower()
        return int(w) if w.isdigit() else _NUM_WORDS[w]
    for pat, d in _NEAREST:
        if pat.search(q):
            return d
    return None


# v1 (Oct 13): precision-aware. v0 accepted anything within 0.5% relative, which let 851.313
# pass for an expected 854.466 on a "three decimal places" question (found in the ARIA report).
@weave.op
def numeric_match(expected: float, output: dict, question: str = "") -> dict:
    """Custom scorer. Boolean `correct` = the answer matches the reference AT THE PRECISION THE
    QUESTION ASKS FOR (half a unit of that precision, or of the reference's own precision if the
    label is coarser). Questions with no stated precision fall back to half a unit of the
    reference's last digit, or 0.01% relative (float noise only). Raw abs/rel errors are kept
    for tolerance sweeps."""
    got = extract_number(output.get("answer", ""))
    decimals = requested_decimals(question)
    if got is None:
        return {"correct": False, "parsed": False, "rel_error": None, "abs_error": None,
                "requested_decimals": decimals}
    err = abs(got - expected)
    rel = err / abs(expected) if expected else err
    if decimals is not None:
        tol = max(0.5 * 10 ** (-decimals), _rounding_unit(expected)) + 1e-9
        correct = err <= tol
    else:
        correct = rel <= 0.0001 or err <= _rounding_unit(expected) + 1e-9
    return {"correct": correct, "parsed": True, "rel_error": rel, "abs_error": err,
            "requested_decimals": decimals}


@weave.op
def followed_format(output: dict) -> dict:
    """Did the reply contain the required machine-readable 'FINAL ANSWER:' line?"""
    return {"has_final_line": bool(FINAL_RE.search(output.get("answer", "") or ""))}


# --- Ambiguity judging (v2, Oct 13) -------------------------------------------------------
# v1 asked ONE judge call "is the question ambiguous?" AND "did the reply flag it?" while showing
# it the reply. The ambiguity verdict was contaminated by the reply (24/50 vs 34/50 "ambiguous"
# on identical questions). v2 splits it:
#   judge_question_ambiguity(question)      -> question ONLY, never sees a reply
#   flags_ambiguity(question, output, ...)  -> only "did the reply surface an ambiguity?"
# Ground truth for ambiguity is the human label (dataset column `human_ambiguous`,
# docs/ambiguity_review.xlsx); the question-only judge is evaluated against it separately
# (evals/judge_alignment.py).

JUDGE_MODEL = "openai/gpt-oss-120b"  # cheap, fast, and a different model family from Qwen

QUESTION_JUDGE_PROMPT = """You are labeling a finance/math question for AMBIGUITY.

Question:
<<<{question}>>>

Is the question materially ambiguous? Answer true only if a careful reader could reasonably
interpret it in two or more ways that give answers more than ~1% apart (e.g. what a percentage
applies to, whether a discount also reduces fees, rounding direction, sign convention).
Ignore far-fetched readings. A question that is merely hard, or needs domain knowledge, is NOT
ambiguous.

Reply with JSON only: {{"ambiguous": true|false, "reason": "<one sentence>"}}"""

# Flag judge v2 (Oct 13): v1 counted "the reply states the interpretation it used" as flagging,
# so a v2 answer that silently picked the WRONG reading ("meaning a total reduction of 45%")
# was credited with catching the ambiguity. Now flagging requires acknowledging ANOTHER reading.
FLAG_JUDGE_PROMPT = """Does this assistant reply explicitly acknowledge that the question could be
read in MORE THAN ONE way? Count it as flagged ONLY if the reply does at least one of:
  (a) names an alternative interpretation it did not use (e.g. "if instead the fees apply to the
      list price...", "ALTERNATIVE: ..."), or
  (b) asks the user a clarifying question about what they meant.
Do NOT count: merely stating or explaining the single interpretation/method it used, listing
standard assumptions (30-day month, compounding convention) without an alternative, or
"ASSUMPTIONS: none" / "ALTERNATIVE: none".

Reply:
<<<{answer}>>>

Reply with JSON only: {{"flagged": true|false, "reason": "<one sentence>"}}"""


async def _judge_json(prompt: str) -> dict | None:
    import json

    from money_math.config import inference_client

    resp = await inference_client().chat.completions.create(
        model=JUDGE_MODEL, temperature=0, messages=[{"role": "user", "content": prompt}]
    )
    text = resp.choices[0].message.content or ""
    try:
        return json.loads(text[text.find("{"): text.rfind("}") + 1])
    except ValueError:
        return None


@weave.op
async def judge_question_ambiguity(question: str) -> dict:
    """LLM judge that sees the QUESTION ONLY. Evaluated against human labels before use."""
    v = await _judge_json(QUESTION_JUDGE_PROMPT.format(question=question[-4000:]))
    if v is None:
        return {"judge_parse_error": True}
    return {"ambiguous": bool(v.get("ambiguous")), "reason": v.get("reason", "")}


@weave.op
async def flags_ambiguity(question: str, output: dict, human_ambiguous: bool | None = None) -> dict:
    """Did the reply surface an ambiguity? `appropriate` = flagged == human label, so flagging a
    clear question (hedging noise) is penalized as much as missing a real ambiguity."""
    v = await _judge_json(FLAG_JUDGE_PROMPT.format(answer=(output.get("answer", "") or "")[-3000:]))
    if v is None:
        return {"judge_parse_error": True}
    flagged = bool(v.get("flagged"))
    res = {"flagged": flagged, "reason": v.get("reason", "")}
    if human_ambiguous is not None:
        res.update({
            "human_ambiguous": bool(human_ambiguous),
            "appropriate": flagged == bool(human_ambiguous),
            "caught": flagged and bool(human_ambiguous),
            "false_alarm": flagged and not human_ambiguous,
        })
    return res


@weave.op
def tool_usage(output: dict) -> dict:
    """Sandbox behaviour: did the agent compute with code, and did any execution fail?"""
    n = output.get("sandbox_execs", 0)
    return {
        "used_sandbox": n > 0,
        "sandbox_execs": n,
        "had_exec_error": output.get("sandbox_errors", 0) > 0,
    }
