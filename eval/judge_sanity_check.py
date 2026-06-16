
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.judge import LLMJudge, sanity_check_agreement


SANITY_EXAMPLES = [
    {
        "problem": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4. She sells the rest at $2 each. How much does she make?",
        "ground_truth": "18",
        "predicted": "18",
        "human_correct": True,
        "note": "Exact match",
    },
    {
        "problem": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4. She sells the rest at $2 each. How much does she make?",
        "ground_truth": "18",
        "predicted": "$18.00",
        "human_correct": True,
        "note": "Currency formatting — same value",
    },
    {
        "problem": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4. She sells the rest at $2 each. How much does she make?",
        "ground_truth": "18",
        "predicted": "18 dollars per day",
        "human_correct": True,
        "note": "Units appended — still correct",
    },
    {
        "problem": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?",
        "ground_truth": "3",
        "predicted": "2",
        "human_correct": False,
        "note": "Arithmetic error — only counted blue fiber",
    },
    {
        "problem": "Josh buys a $80K house, puts in $50K repairs, value increases 150%. What is his profit?",
        "ground_truth": "70000",
        "predicted": "70,000",
        "human_correct": True,
        "note": "Comma formatting — same value",
    },
    {
        "problem": "Josh buys a $80K house, puts in $50K repairs, value increases 150%. What is his profit?",
        "ground_truth": "70000",
        "predicted": "200000",
        "human_correct": False,
        "note": "Forgot to subtract costs",
    },
    {
        "problem": "Josh buys a $80K house, puts in $50K repairs, value increases 150%. What is his profit?",
        "ground_truth": "70000",
        "predicted": None,
        "human_correct": False,
        "note": "No answer produced",
    },
    {
        "problem": "James runs 3 sprints 3 times a week at 60m each. Total meters per week?",
        "ground_truth": "540",
        "predicted": "540 meters",
        "human_correct": True,
        "note": "Correct value with unit label",
    },
    {
        "problem": "Mark has 3 tanks with 4 pregnant fish each, each having 20 babies. Total babies?",
        "ground_truth": "240",
        "predicted": "240",
        "human_correct": True,
        "note": "Exact correct",
    },
    {
        "problem": "A train travels at 60mph for 180 miles. How many minutes?",
        "ground_truth": "180",
        "predicted": "3",
        "human_correct": False,
        "note": "Answered in hours instead of minutes",
    },
]


def run_sanity_check(
    judge_model: str = "claude-sonnet-4-6",
    verbose: bool = True,
) -> dict:
    cache: dict = {}
    judge = LLMJudge(model=judge_model, cache=cache)

    problems = [e["problem"] for e in SANITY_EXAMPLES]
    gts = [e["ground_truth"] for e in SANITY_EXAMPLES]
    preds = [e["predicted"] for e in SANITY_EXAMPLES]
    human_labels = [e["human_correct"] for e in SANITY_EXAMPLES]

    if verbose:
        print(f"Running judge sanity check ({len(SANITY_EXAMPLES)} examples)...")
        print(f"Judge model: {judge_model}\n")

    judge_results = judge.grade_batch(problems, gts, preds)
    report = sanity_check_agreement(judge_results, human_labels)

    if verbose:
        print(f"{'#':<4} {'Problem snippet':<40} {'GT':>6} {'Pred':>12} {'Human':>6} {'Judge':>10} {'Match':>6}")
        print("-" * 90)
        for i, (ex, jr) in enumerate(zip(SANITY_EXAMPLES, judge_results)):
            prob_snip = ex["problem"][:38]
            human_str = "✓" if ex["human_correct"] else "✗"
            judge_str = "✓" if jr.is_correct else "✗"
            match_str = "=" if (jr.is_correct == ex["human_correct"]) else "≠"
            gt = str(ex["ground_truth"])[:6]
            pred = str(ex["predicted"] or "None")[:12]
            print(f"{i+1:<4} {prob_snip:<40} {gt:>6} {pred:>12} {human_str:>6} {judge_str:>10} {match_str:>6}")

        print(f"\nJudge-human accuracy: {report['accuracy']:.1%} ({int(report['accuracy'] * len(SANITY_EXAMPLES))}/{len(SANITY_EXAMPLES)})")
        print(f"Cohen's kappa:        {report['cohen_kappa']:.3f}")

        if report["disagreements"]:
            print(f"\nDisagreements ({len(report['disagreements'])}):")
            for d in report["disagreements"]:
                ex = SANITY_EXAMPLES[d["idx"]]
                print(f"  [{d['idx']+1}] Judge={d['judge']}, Human={'CORRECT' if d['human'] else 'INCORRECT'}")
                print(f"       Note: {ex['note']}")
                print(f"       Explanation: {d['explanation']}")

        threshold = 0.70
        if report["accuracy"] >= threshold:
            print(f"\n✓ Judge meets {threshold:.0%} agreement threshold — safe to use.")
        else:
            print(f"\n✗ Judge BELOW {threshold:.0%} threshold — fix rubric before trusting scores!")

    return report


if __name__ == "__main__":
    judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
    result = run_sanity_check(judge_model=judge_model)
    print(f"\nFinal: accuracy={result['accuracy']:.1%}, kappa={result['cohen_kappa']:.3f}")