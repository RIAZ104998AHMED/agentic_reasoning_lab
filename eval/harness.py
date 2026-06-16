"""
Eval Harness — the main entry point.

Usage:
    python -m eval.harness                  # run all strategies on full golden set
    python -m eval.harness --strategy react  # run one strategy
    python -m eval.harness --limit 5         # run first N problems (dev mode)
    python -m eval.harness --diff            # compare against baseline.json

Produces:
    - Per-strategy accuracy with 95% Wilson CIs
    - Strategy × strategy win matrix
    - Cost / latency table
    - Saves results/run_<timestamp>.json
    - Updates baseline.json if --save-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

from eval.judge import LLMJudge
from eval.metrics import (
    is_correct, score_batch, wilson_ci, win_matrix, mcnemar_p
)
from observability.logger import TraceLogger
from strategies.base import Strategy
from strategies.plan_execute import PlanAndExecuteStrategy
from strategies.react import ReActStrategy
from strategies.self_consistency import SelfConsistencyStrategy
from strategies.tree_of_thoughts import TreeOfThoughtsStrategy
from tools.registry import ToolRegistry

_GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
_RESULTS_DIR = Path("results")
_BASELINE_PATH = Path("baseline.json")

# Pricing estimates (per 1M tokens) — update for your model
_COST_PER_1M_IN = 3.0    # USD, claude-3-5-haiku or similar
_COST_PER_1M_OUT = 15.0


def load_golden_set(path: Path = _GOLDEN_SET_PATH) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def build_strategies(model: str, judge_model: str, cache: dict) -> dict[str, Strategy]:
    tools = ToolRegistry()
    return {
        "react": ReActStrategy(model=model, tool_registry=tools, cache=cache),
        "plan_execute": PlanAndExecuteStrategy(model=model, tool_registry=tools, cache=cache),
        "self_consistency": SelfConsistencyStrategy(model=model, tool_registry=tools, cache=cache, n_samples=5),
        "tree_of_thoughts": TreeOfThoughtsStrategy(model=model, tool_registry=tools, cache=cache, beam_width=2, max_depth=3),
    }


def run_eval(
    strategy_names: Optional[list[str]] = None,
    limit: Optional[int] = None,
    model: str = "claude-haiku-4-5-20251001",
    judge_model: str = "claude-sonnet-4-6",
    save_baseline: bool = False,
    diff_baseline: bool = False,
    verbose: bool = True,
) -> dict:
    problems = load_golden_set()
    if limit:
        problems = problems[:limit]

    cache: dict = {}
    judge_cache: dict = {}
    logger = TraceLogger()
    judge = LLMJudge(model=judge_model, cache=judge_cache)

    all_strategies = build_strategies(model, judge_model, cache)
    if strategy_names:
        all_strategies = {k: v for k, v in all_strategies.items() if k in strategy_names}

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Run each strategy over all problems ──────────────────────────────────
    results: dict[str, dict] = {}

    for strat_name, strategy in all_strategies.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Strategy: {strat_name.upper()}")
            print(f"{'='*60}")

        strat_results = {
            "predictions": [],
            "correct_flags": [],
            "traces": [],
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_wall_ms": 0.0,
        }

        for prob in problems:
            pid = prob["id"]
            ground_truth = prob["answer"]

            if verbose:
                print(f"  [{pid}] {prob['problem'][:70]}...", end=" ", flush=True)

            try:
                trace = strategy.solve(prob["problem"], pid)
            except Exception as e:
                # Never crash the harness
                from strategies.base import Trace
                trace = Trace.empty(strat_name, pid, prob["problem"])
                trace.error = str(e)

            logger.log_trace_events(trace)
            logger.save_trace(trace)

            prediction = trace.final_answer
            correct = is_correct(prediction, ground_truth)

            strat_results["predictions"].append(prediction)
            strat_results["correct_flags"].append(correct)
            strat_results["traces"].append(trace)
            strat_results["total_tokens_in"] += trace.total_tokens_in
            strat_results["total_tokens_out"] += trace.total_tokens_out
            strat_results["total_wall_ms"] += trace.wall_time_ms

            if verbose:
                status = "✓" if correct else "✗"
                print(f"{status}  (pred={prediction}, gt={ground_truth})")

        n = len(problems)
        k = sum(strat_results["correct_flags"])
        lo, hi = wilson_ci(k, n)
        cost = (
            strat_results["total_tokens_in"] / 1e6 * _COST_PER_1M_IN
            + strat_results["total_tokens_out"] / 1e6 * _COST_PER_1M_OUT
        )
        cost_per_correct = cost / k if k > 0 else float("inf")

        strat_results.update({
            "n": n,
            "correct": k,
            "accuracy": k / n,
            "ci_low": lo,
            "ci_high": hi,
            "cost_usd": round(cost, 4),
            "cost_per_correct_usd": round(cost_per_correct, 4),
            "avg_wall_ms": strat_results["total_wall_ms"] / n,
        })
        results[strat_name] = strat_results

        if verbose:
            print(f"\n  Accuracy: {k}/{n} = {k/n:.1%}  95%CI [{lo:.1%}, {hi:.1%}]")
            print(f"  Tokens: {strat_results['total_tokens_in']}in / {strat_results['total_tokens_out']}out")
            print(f"  Cost: ${cost:.4f}  (${cost_per_correct:.4f}/correct)")

    # ── Win matrix ───────────────────────────────────────────────────────────
    strat_names = list(results.keys())
    per_problem = {s: results[s]["correct_flags"] for s in strat_names}
    wmatrix = win_matrix(strat_names, per_problem)

    # ── McNemar p-values ─────────────────────────────────────────────────────
    pvalues = {}
    for i, a in enumerate(strat_names):
        for b in strat_names[i+1:]:
            p = mcnemar_p(per_problem[a], per_problem[b])
            pvalues[f"{a}_vs_{b}"] = round(p, 4)

    # ── Print summary tables ─────────────────────────────────────────────────
    if verbose:
        _print_results_table(results)
        _print_win_matrix(strat_names, wmatrix)
        _print_pvalues(pvalues)

    # ── Save results ─────────────────────────────────────────────────────────
    run_summary = {
        "timestamp": int(time.time()),
        "model": model,
        "n_problems": len(problems),
        "strategies": {
            s: {
                "accuracy": r["accuracy"],
                "correct": r["correct"],
                "n": r["n"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "total_tokens_in": r["total_tokens_in"],
                "total_tokens_out": r["total_tokens_out"],
                "cost_usd": r["cost_usd"],
                "cost_per_correct_usd": r["cost_per_correct_usd"],
                "avg_wall_ms": round(r["avg_wall_ms"], 1),
            }
            for s, r in results.items()
        },
        "win_matrix": wmatrix,
        "mcnemar_pvalues": pvalues,
    }

    out_path = _RESULTS_DIR / f"run_{run_summary['timestamp']}.json"
    out_path.write_text(json.dumps(run_summary, indent=2))
    if verbose:
        print(f"\nResults saved to {out_path}")

    if save_baseline:
        _BASELINE_PATH.write_text(json.dumps(run_summary, indent=2))
        print(f"Baseline saved to {_BASELINE_PATH}")

    if diff_baseline and _BASELINE_PATH.exists():
        _diff_against_baseline(run_summary)

    return run_summary


# ─────────────────────────────── Printing helpers ────────────────────────────

def _print_results_table(results: dict):
    print("\n" + "="*80)
    print(f"{'Strategy':<20} {'Acc':>6} {'95% CI':>18} {'Tokens':>10} {'Cost':>8} {'$/correct':>10} {'ms/prob':>9}")
    print("-"*80)
    for s, r in results.items():
        ci = f"[{r['ci_low']:.1%}, {r['ci_high']:.1%}]"
        tokens = r["total_tokens_in"] + r["total_tokens_out"]
        print(
            f"{s:<20} {r['accuracy']:>6.1%} {ci:>18} {tokens:>10,} "
            f"{r['cost_usd']:>7.4f}  {r['cost_per_correct_usd']:>9.4f} {r['avg_wall_ms']:>8.0f}"
        )
    print("="*80)


def _print_win_matrix(names: list[str], matrix: dict):
    print("\nWin matrix (row beats column):")
    w = 14
    header = " " * w + "".join(f"{n[:w]:>{w}}" for n in names)
    print(header)
    for a in names:
        row = f"{a:<{w}}" + "".join(f"{matrix[a][b]:>{w}}" for b in names)
        print(row)


def _print_pvalues(pvalues: dict):
    if not pvalues:
        return
    print("\nMcNemar p-values (< 0.05 = significant difference):")
    for pair, p in pvalues.items():
        sig = " *" if p < 0.05 else ""
        print(f"  {pair}: p={p:.4f}{sig}")


def _diff_against_baseline(current: dict):
    baseline = json.loads(_BASELINE_PATH.read_text())
    print("\nDelta vs baseline:")
    for s in current["strategies"]:
        if s not in baseline.get("strategies", {}):
            continue
        cur_acc = current["strategies"][s]["accuracy"]
        base_acc = baseline["strategies"][s]["accuracy"]
        delta = cur_acc - base_acc
        sign = "+" if delta >= 0 else ""
        print(f"  {s}: {base_acc:.1%} → {cur_acc:.1%}  ({sign}{delta:.1%})")


# ─────────────────────────────── CLI ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Agentic Reasoning Lab — Eval Harness")
    parser.add_argument("--strategy", nargs="+", help="Which strategies to run")
    parser.add_argument("--limit", type=int, help="Limit to first N problems")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Solver model")
    parser.add_argument("--judge-model", default="claude-sonnet-4-6", help="Judge model")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--diff", action="store_true", dest="diff_baseline")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_eval(
        strategy_names=args.strategy,
        limit=args.limit,
        model=args.model,
        judge_model=args.judge_model,
        save_baseline=args.save_baseline,
        diff_baseline=args.diff_baseline,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()