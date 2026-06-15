"""
Evaluation metrics for the Agentic Reasoning Lab.

Primary metric: numerical-tolerance match for math word problems.
  - Strip units and punctuation from both prediction and ground truth.
  - Compare as floats within a relative tolerance of 1e-3.
  - Falls back to exact string match for non-numeric answers.

Why this metric: GSM8K answers are always integers or simple decimals.
  Exact string match would penalise "18.0" vs "18" and similar safe variants.
  A 0.1% relative tolerance handles floating-point formatting differences
  without being loose enough to accept wrong answers.
"""

from __future__ import annotations

import math
import re
from typing import Optional


_REL_TOL = 1e-3   # 0.1% relative tolerance for numeric comparison
_ABS_TOL = 1e-6   # absolute tolerance for near-zero values


def _extract_number(text: str) -> Optional[float]:
    """Return the first parseable float from a string, or None."""
    if text is None:
        return None
    text = str(text).strip()
    # Remove common units and currency symbols
    text = re.sub(r"[$€£¥%,]", "", text)
    # Remove trailing period
    text = text.rstrip(".")
    # Try direct parse
    try:
        return float(text)
    except ValueError:
        pass
    # Extract first number-like token
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if nums:
        try:
            return float(nums[-1])  # last number tends to be the final answer
        except ValueError:
            pass
    return None


def is_correct(prediction: Optional[str], ground_truth: str,
               rel_tol: float = _REL_TOL) -> bool:
    """
    Return True if prediction matches ground_truth.
    Tries numeric comparison first, falls back to case-insensitive string match.
    """
    if prediction is None:
        return False

    pred_num = _extract_number(prediction)
    gt_num = _extract_number(ground_truth)

    if pred_num is not None and gt_num is not None:
        if gt_num == 0:
            return abs(pred_num - gt_num) <= _ABS_TOL
        return math.isclose(pred_num, gt_num, rel_tol=rel_tol, abs_tol=_ABS_TOL)

    # String fallback
    return prediction.strip().lower() == ground_truth.strip().lower()


def score_batch(
    predictions: list[Optional[str]],
    ground_truths: list[str],
) -> dict:
    """
    Score a list of predictions against ground truths.
    Returns dict with: correct, total, accuracy, per_problem.
    """
    assert len(predictions) == len(ground_truths)
    results = [is_correct(p, g) for p, g in zip(predictions, ground_truths)]
    n = len(results)
    k = sum(results)
    return {
        "correct": k,
        "total": n,
        "accuracy": k / n if n > 0 else 0.0,
        "per_problem": results,
    }


# ─────────────────────────────── Confidence Intervals ────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score interval for a proportion k/n at confidence level z.
    Returns (lower, upper) as fractions in [0, 1].
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_ci(
    correct_flags: list[bool],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for accuracy.
    Returns (lower, upper) as fractions.
    """
    import random
    rng = random.Random(seed)
    n = len(correct_flags)
    means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(correct_flags) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - confidence) / 2
    lo = means[int(alpha * n_bootstrap)]
    hi = means[int((1 - alpha) * n_bootstrap)]
    return lo, hi


# ─────────────────────────────── Win Matrix ──────────────────────────────────

def win_matrix(
    strategy_names: list[str],
    per_problem_correct: dict[str, list[bool]],
) -> dict[str, dict[str, int]]:
    """
    Build a strategy × strategy win matrix.
    matrix[A][B] = number of problems where A was correct and B was not.
    """
    matrix = {a: {b: 0 for b in strategy_names} for a in strategy_names}
    n_problems = len(next(iter(per_problem_correct.values())))

    for i in range(n_problems):
        for a in strategy_names:
            for b in strategy_names:
                if a == b:
                    continue
                a_correct = per_problem_correct[a][i]
                b_correct = per_problem_correct[b][i]
                if a_correct and not b_correct:
                    matrix[a][b] += 1
    return matrix


def mcnemar_p(correct_a: list[bool], correct_b: list[bool]) -> float:
    """
    McNemar's test p-value for two paired classifiers.
    Tests H0: the two strategies have equal error rates.
    Returns p-value (< 0.05 = significant difference).
    """
    from scipy.stats import binom
    b = sum(1 for a, b_ in zip(correct_a, correct_b) if a and not b_)
    c = sum(1 for a, b_ in zip(correct_a, correct_b) if not a and b_)
    n = b + c
    if n == 0:
        return 1.0
    # Exact binomial test: p = 2 * P(X <= min(b,c)) for X~Bin(n, 0.5)
    p = 2 * binom.cdf(min(b, c), n, 0.5)
    return min(p, 1.0)