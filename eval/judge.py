"""
LLM-as-Judge for grading strategy answers.

Uses a DIFFERENT model from the solver to avoid self-collusion.
The judge decides if `predicted_answer` is equivalent to `ground_truth`
for a given `problem`.

Sanity-check: hand-grade >= 8 examples and report judge-human agreement
before trusting these scores. See eval/judge_sanity_check.py.

Rubric:
  CORRECT   - the predicted answer is numerically or semantically equivalent
  INCORRECT - the predicted answer is wrong or missing
  UNCLEAR   - the prediction is ambiguous (treated as incorrect in scoring)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from observability.llm_client import call_llm

_JUDGE_SYSTEM = """\
You are a strict math grader. Your job is to judge whether a student's answer
is correct given the problem and the official ground truth answer.

Rules:
1. Compare the numerical value only - ignore units, currency symbols, and formatting.
   For example: "$18", "18 dollars", "18.0" all equal "18".
2. Accept rounding within 0.5% relative error for non-integer answers.
3. If the prediction is empty, None, or clearly a failure message: INCORRECT.
4. Do not give partial credit.

Respond with EXACTLY one of these lines as your first line:
VERDICT: CORRECT
VERDICT: INCORRECT
VERDICT: UNCLEAR

Then optionally add one sentence of explanation.
"""

_JUDGE_PROMPT = """\
Problem: {problem}

Ground truth answer: {ground_truth}

Student's predicted answer: {prediction}

Is the student's answer correct?
"""


@dataclass
class JudgeResult:
    verdict: str
    is_correct: bool
    explanation: str
    tokens_in: int
    tokens_out: int
    latency_ms: float


class LLMJudge:
    def __init__(self, model: str, cache: Optional[dict] = None):
        self.model = model
        self.cache = cache

    def grade(
        self,
        problem: str,
        ground_truth: str,
        prediction: Optional[str],
    ) -> JudgeResult:
        if prediction is None or not str(prediction).strip() or str(prediction).strip() == "None":
            return JudgeResult(
                verdict="INCORRECT",
                is_correct=False,
                explanation="Empty prediction",
                tokens_in=0,
                tokens_out=0,
                latency_ms=0.0,
            )

        prompt = _JUDGE_PROMPT.format(
            problem=problem,
            ground_truth=ground_truth,
            prediction=prediction,
        )

        text, tok_in, tok_out, latency = call_llm(
            model=self.model,
            system=_JUDGE_SYSTEM,
            prompt=prompt,
            max_tokens=128,
            cache=self.cache,
        )

        verdict, explanation = self._parse_verdict(text)
        is_correct = verdict == "CORRECT"

        return JudgeResult(
            verdict=verdict,
            is_correct=is_correct,
            explanation=explanation,
            tokens_in=tok_in,
            tokens_out=tok_out,
            latency_ms=latency,
        )

    @staticmethod
    def _parse_verdict(text: str) -> tuple[str, str]:
        lines = text.strip().splitlines()
        verdict = "UNCLEAR"

        for i, line in enumerate(lines):
            m = re.match(r"VERDICT:\s*(CORRECT|INCORRECT|UNCLEAR)", line.strip(), re.IGNORECASE)
            if m:
                verdict = m.group(1).upper()
                explanation = " ".join(l.strip() for l in lines[i + 1:] if l.strip())
                return verdict, explanation

        explanation = " ".join(l.strip() for l in lines[1:] if l.strip()) if len(lines) > 1 else text.strip()
        return verdict, explanation

    def grade_batch(
        self,
        problems: list[str],
        ground_truths: list[str],
        predictions: list[Optional[str]],
    ) -> list[JudgeResult]:
        return [
            self.grade(p, g, pred)
            for p, g, pred in zip(problems, ground_truths, predictions)
        ]


def sanity_check_agreement(
    judge_results: list[JudgeResult],
    human_labels: list[bool],
) -> dict:
    assert len(judge_results) == len(human_labels)
    n = len(judge_results)
    matches = [jr.is_correct == hl for jr, hl in zip(judge_results, human_labels)]
    accuracy = sum(matches) / n

    p0 = accuracy
    judge_pos = sum(jr.is_correct for jr in judge_results) / n
    human_pos = sum(human_labels) / n
    pe = judge_pos * human_pos + (1 - judge_pos) * (1 - human_pos)
    kappa = (p0 - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

    disagreements = [
        {
            "idx": i,
            "judge": judge_results[i].verdict,
            "human": human_labels[i],
            "explanation": judge_results[i].explanation,
        }
        for i, match in enumerate(matches)
        if not match
    ]

    return {
        "n": n,
        "accuracy": round(accuracy, 4),
        "cohen_kappa": round(kappa, 4),
        "disagreements": disagreements,
    }