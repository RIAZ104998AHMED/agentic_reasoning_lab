"""
Failure mode taxonomy classifier.

Hand-classify failures into categories and produce a distribution report.
Run this after eval to analyse what went wrong.

Usage:
    python -m observability.failure_analysis --run results/run_<ts>.json
    python -m observability.failure_analysis --interactive   # guided CLI
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from observability.logger import TraceLogger

# Failure taxonomy (based on common LLM agent failure modes)
FAILURE_CATEGORIES = {
    "arithmetic_error":    "Model computed the arithmetic incorrectly without using a tool",
    "wrong_tool_call":     "Tool was called with malformed or wrong arguments",
    "tool_error":          "Tool returned an error or malformed output the model didn't handle",
    "plan_abandoned":      "Plan-and-Execute abandoned steps mid-execution",
    "infinite_loop":       "Strategy hit the step limit without converging",
    "extraction_failure":  "Answer was in the reasoning but not correctly extracted",
    "setup_error":         "Misread the problem or set up the wrong equation",
    "judge_disagreement":  "Programmatic metric and LLM judge disagreed",
    "empty_answer":        "Strategy returned no answer (None or empty string)",
    "other":               "Doesn't fit the above categories",
}


@dataclass
class FailureRecord:
    trace_id: str
    problem_id: str
    strategy: str
    problem: str
    predicted_answer: Optional[str]
    ground_truth: str
    category: str
    notes: str
    trace_excerpt: str


def auto_classify(trace_dict: dict, ground_truth: str) -> str:
    """
    Heuristic auto-classification — always hand-review afterwards.
    """
    pred = trace_dict.get("final_answer")
    error = trace_dict.get("error", "")

    if pred is None or pred == "" or pred == "None":
        return "empty_answer"

    if error and "step" in error.lower():
        return "infinite_loop"

    # Look at events for clues
    events = trace_dict.get("events", [])
    for ev in events:
        if ev.get("step_type") == "tool_call":
            out = ev.get("outputs", {})
            if out.get("error"):
                return "tool_error"
            inp = ev.get("inputs", {})
            arg = str(inp.get("arg", ""))
            if "expression" in arg or "(" in arg:
                # Check if arg is parseable as a math expression
                try:
                    import ast
                    ast.parse(arg, mode="eval")
                except SyntaxError:
                    return "wrong_tool_call"

    # Check for plan abandonment
    if trace_dict.get("strategy") == "plan_execute":
        n_exec = sum(1 for ev in events if ev.get("step_type") == "execution")
        n_planned = trace_dict.get("extra", {}).get("n_steps", 99)
        if n_exec < n_planned and n_exec > 0:
            return "plan_abandoned"

    return "arithmetic_error"  # most common catch-all for math problems


def build_failure_report(
    results_path: Path,
    golden_set_path: Path = Path("eval/golden_set.json"),
    output_path: Optional[Path] = None,
    auto_only: bool = False,
) -> dict:
    """
    Load a results JSON, find all failures, classify them, and print a report.
    """
    results = json.loads(results_path.read_text())
    golden = {p["id"]: p for p in json.loads(golden_set_path.read_text())}
    all_traces = {t["problem_id"] + "|" + t["strategy"]: t for t in TraceLogger.list_traces()}

    failures: list[FailureRecord] = []

    for strat_name, strat_data in results.get("strategies", {}).items():
        # We need per-problem predictions — stored in run traces
        strat_traces = [
            t for t in TraceLogger.list_traces()
            if t["strategy"] == strat_name
        ]
        for tr in strat_traces:
            pid = tr["problem_id"]
            if pid not in golden:
                continue
            gt = golden[pid]["answer"]
            pred = tr.get("final_answer")

            from eval.metrics import is_correct
            if is_correct(pred, gt):
                continue   # only failures

            category = auto_classify(tr, gt)
            excerpt = _make_excerpt(tr)

            failures.append(FailureRecord(
                trace_id=tr["trace_id"],
                problem_id=pid,
                strategy=strat_name,
                problem=tr["problem"][:150],
                predicted_answer=pred,
                ground_truth=gt,
                category=category,
                notes="(auto-classified)",
                trace_excerpt=excerpt,
            ))

    # Distribution table
    from collections import Counter
    dist = Counter(f.category for f in failures)

    print("\n" + "="*60)
    print("Failure Mode Taxonomy")
    print("="*60)
    print(f"Total failures: {len(failures)}\n")
    print(f"{'Category':<25} {'Count':>6} {'%':>6}")
    print("-"*40)
    for cat, count in dist.most_common():
        pct = count / len(failures) * 100 if failures else 0
        desc = FAILURE_CATEGORIES.get(cat, "")
        print(f"{cat:<25} {count:>6} {pct:>5.1f}%")
        print(f"  └─ {desc}")

    print("\nWorked examples:")
    shown = set()
    for f in failures:
        if f.category not in shown and len(shown) < 4:
            shown.add(f.category)
            print(f"\n[{f.category}] {f.problem_id} ({f.strategy})")
            print(f"  Problem: {f.problem[:100]}...")
            print(f"  Predicted: {f.predicted_answer}")
            print(f"  Ground truth: {f.ground_truth}")
            print(f"  Trace excerpt:\n{f.trace_excerpt}")

    report = {
        "timestamp": int(time.time()),
        "total_failures": len(failures),
        "distribution": dict(dist),
        "failures": [
            {
                "trace_id": f.trace_id,
                "problem_id": f.problem_id,
                "strategy": f.strategy,
                "predicted": f.predicted_answer,
                "ground_truth": f.ground_truth,
                "category": f.category,
                "notes": f.notes,
            }
            for f in failures
        ],
    }

    if output_path:
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\nFailure report saved to {output_path}")

    return report


def _make_excerpt(trace_dict: dict) -> str:
    """Pull a short trace excerpt for display."""
    events = trace_dict.get("events", [])
    lines = []
    for ev in events[:5]:
        step_type = ev.get("step_type", "?")
        if step_type == "llm_call":
            out = ev.get("outputs", {}).get("text", "")[:120]
            lines.append(f"    LLM → {out!r}")
        elif step_type == "tool_call":
            tool = ev.get("inputs", {}).get("tool", "?")
            arg = str(ev.get("inputs", {}).get("arg", ""))[:60]
            obs = str(ev.get("outputs", {}).get("observation", ""))[:60]
            err = ev.get("outputs", {}).get("error")
            lines.append(f"    TOOL {tool}({arg}) → {obs}" + (f" [ERR: {err}]" if err else ""))
    return "\n".join(lines) or "    (no events)"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Failure mode taxonomy analysis")
    parser.add_argument("--run", required=True, help="Path to results JSON")
    parser.add_argument("--output", help="Save report to this JSON file")
    parser.add_argument("--golden", default="eval/golden_set.json")
    args = parser.parse_args()

    build_failure_report(
        results_path=Path(args.run),
        golden_set_path=Path(args.golden),
        output_path=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()