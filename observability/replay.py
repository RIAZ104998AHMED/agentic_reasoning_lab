"""
Replay tool — re-run a single problem given a trace ID (or problem ID).

Usage:
    python -m observability.replay --trace-id abc123
    python -m observability.replay --problem-id gsm_001 --strategy react
    python -m observability.replay --problem-id gsm_005 --strategy tree_of_thoughts --diff

The --diff flag prints a side-by-side comparison of original vs replay answers
and highlights any events that changed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from observability.logger import TraceLogger


def replay(
    trace_id: str | None = None,
    problem_id: str | None = None,
    strategy_override: str | None = None,
    model_override: str | None = None,
    show_diff: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Re-run a problem from a stored trace.
    Returns the new trace dict.
    """
    # ── Look up original trace ────────────────────────────────────────────────
    original_trace = None
    if trace_id:
        original_trace = TraceLogger.load_trace(trace_id)
        if not original_trace:
            raise ValueError(f"Trace {trace_id} not found")
    elif problem_id:
        all_traces = TraceLogger.list_traces()
        matches = [t for t in all_traces if t["problem_id"] == problem_id]
        if strategy_override:
            matches = [t for t in matches if t["strategy"] == strategy_override]
        if not matches:
            raise ValueError(f"No traces found for problem_id={problem_id}")
        original_trace = matches[-1]  # use most recent
    else:
        raise ValueError("Provide --trace-id or --problem-id")

    strat_name = strategy_override or original_trace["strategy"]
    problem = original_trace["problem"]
    pid = original_trace["problem_id"]
    model = model_override or original_trace.get("model", "claude-haiku-4-5-20251001")

    if verbose:
        print(f"Replaying: problem_id={pid}, strategy={strat_name}, model={model}")
        print(f"Problem: {problem[:100]}...")
        print(f"Original answer: {original_trace.get('final_answer')}")
        print()

    # ── Re-run ────────────────────────────────────────────────────────────────
    from tools.registry import ToolRegistry
    from strategies.react import ReActStrategy
    from strategies.plan_execute import PlanAndExecuteStrategy
    from strategies.self_consistency import SelfConsistencyStrategy
    from strategies.tree_of_thoughts import TreeOfThoughtsStrategy

    strategy_map = {
        "react": ReActStrategy,
        "plan_execute": PlanAndExecuteStrategy,
        "self_consistency": SelfConsistencyStrategy,
        "tree_of_thoughts": TreeOfThoughtsStrategy,
    }

    StratClass = strategy_map.get(strat_name)
    if not StratClass:
        raise ValueError(f"Unknown strategy: {strat_name}")

    tools = ToolRegistry()
    cache: dict = {}   # fresh cache for replay (no cache hits from original run)

    strategy = StratClass(model=model, tool_registry=tools, cache=cache)
    t_start = time.perf_counter()
    new_trace = strategy.solve(problem, pid)
    wall_ms = (time.perf_counter() - t_start) * 1000

    # ── Save replay trace ─────────────────────────────────────────────────────
    logger = TraceLogger(run_id=f"replay_{int(time.time())}")
    logger.log_trace_events(new_trace)
    logger.save_trace(new_trace)

    if verbose:
        print(f"Replay answer:   {new_trace.final_answer}")
        print(f"Wall time: {wall_ms:.0f}ms")
        print(f"Tokens: {new_trace.total_tokens_in}in / {new_trace.total_tokens_out}out")

    if show_diff:
        _print_diff(original_trace, new_trace)

    return {
        "original_trace_id": original_trace["trace_id"],
        "replay_trace_id": new_trace.trace_id,
        "original_answer": original_trace.get("final_answer"),
        "replay_answer": new_trace.final_answer,
        "answer_changed": original_trace.get("final_answer") != new_trace.final_answer,
        "replay_events": len(new_trace.events),
    }


def _print_diff(original: dict, new_trace) -> None:
    print("\n── Diff ─────────────────────────────────────────────────")
    orig_ans = original.get("final_answer")
    new_ans = new_trace.final_answer
    if orig_ans == new_ans:
        print(f"  Answer: SAME ({orig_ans})")
    else:
        print(f"  Answer CHANGED: {orig_ans!r} → {new_ans!r}")

    orig_steps = original.get("n_events", "?")
    new_steps = len(new_trace.events)
    print(f"  Events: original={orig_steps}, replay={new_steps}")

    orig_tok = (original.get("total_tokens_in", 0) + original.get("total_tokens_out", 0))
    new_tok = new_trace.total_tokens
    print(f"  Tokens: original={orig_tok}, replay={new_tok}")

    if hasattr(new_trace, "extra") and "tree" in new_trace.extra:
        print(f"  ToT tree nodes: {new_trace.extra.get('nodes_expanded')}")
    print("─────────────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser(description="Replay a problem from a stored trace")
    parser.add_argument("--trace-id", help="Specific trace ID to replay")
    parser.add_argument("--problem-id", help="Problem ID to replay (uses most recent trace)")
    parser.add_argument("--strategy", dest="strategy_override", help="Override the strategy")
    parser.add_argument("--model", dest="model_override", help="Override the model")
    parser.add_argument("--diff", action="store_true", dest="show_diff")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    replay(
        trace_id=args.trace_id,
        problem_id=args.problem_id,
        strategy_override=args.strategy_override,
        model_override=args.model_override,
        show_diff=args.show_diff,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()