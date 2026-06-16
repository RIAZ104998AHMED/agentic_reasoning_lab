

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_demo(problem: str = None, problem_id: str = None):
    from tools.registry import ToolRegistry
    from strategies.react import ReActStrategy
    from strategies.plan_execute import PlanAndExecuteStrategy
    from strategies.self_consistency import SelfConsistencyStrategy
    from strategies.tree_of_thoughts import TreeOfThoughtsStrategy
    from observability.logger import TraceLogger

    if problem is None:
        golden = json.loads(Path("eval/golden_set.json").read_text())
        pid = problem_id or "gsm_007"
        prob = next((p for p in golden if p["id"] == pid), golden[0])
        problem = prob["problem"]
        ground_truth = prob["answer"]
        problem_id = prob["id"]
    else:
        ground_truth = "?"
        problem_id = problem_id or "custom"

    model = os.environ.get("SOLVER_MODEL", "openai/gpt-4o-mini")
    cache: dict = {}
    tools = ToolRegistry()
    logger = TraceLogger(run_id="demo")

    strategies = [
        ("ReAct", ReActStrategy(model=model, tool_registry=tools, cache=cache)),
        ("Plan-and-Execute", PlanAndExecuteStrategy(model=model, tool_registry=tools, cache=cache)),
        ("Self-Consistency", SelfConsistencyStrategy(model=model, tool_registry=tools, cache=cache, n_samples=3)),
        ("Tree-of-Thoughts", TreeOfThoughtsStrategy(model=model, tool_registry=tools, cache=cache, beam_width=2, max_depth=3)),
    ]

    print("\n" + "╔" + "═" * 70 + "╗")
    print(f"║  DEMO PROBLEM [{problem_id}]" + " " * (70 - len(problem_id) - 18) + "║")
    print("╠" + "═" * 70 + "╣")

    words = problem.split()
    line = "║  "
    for word in words:
        if len(line) + len(word) > 72:
            print(line + " " * (73 - len(line)) + "║")
            line = "║  " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line + " " * (73 - len(line)) + "║")

    print(f"║  Ground truth: {ground_truth}" + " " * (70 - len(str(ground_truth)) - 16) + "║")
    print("╚" + "═" * 70 + "╝\n")

    for display_name, strategy in strategies:
        print(f"\n{'─' * 70}")
        print(f"  ▶ {display_name}")
        print(f"{'─' * 70}")

        trace = strategy.solve(problem, problem_id)
        logger.log_trace_events(trace)
        logger.save_trace(trace)

        for ev in trace.events:
            st = ev.step_type
            if st == "planning":
                plan_lines = ev.outputs.get("plan", "").strip().splitlines()[:6]
                print("  [PLAN]")
                for l in plan_lines:
                    print(f"    {l}")
            elif st == "thought_generation":
                thought = ev.outputs.get("thought", "")[:200]
                print(f"  [THOUGHT {ev.metadata.get('node_id', '')}] {thought[:120]}...")
            elif st == "llm_call":
                text = ev.outputs.get("text", "")[:200]
                print(f"  [LLM step {ev.metadata.get('step', '')}] {text[:120]}...")
            elif st == "tool_call":
                tool = ev.inputs.get("tool", "")
                arg = str(ev.inputs.get("arg", ""))[:60]
                obs = str(ev.outputs.get("observation", ""))[:60]
                err = ev.outputs.get("error")
                print(f"  [TOOL] {tool}({arg}) -> {obs}" + (f"  WARNING: {err}" if err else ""))
            elif st == "voting":
                dist = ev.inputs.get("all_answers", [])
                winner = ev.outputs.get("winner")
                print(f"  [VOTE] {dist} -> winner: {winner}")

        print()
        status = "✓ CORRECT" if trace.final_answer == ground_truth else (
            "~ CLOSE" if _close(trace.final_answer, ground_truth) else "✗ WRONG"
        )
        print(f"  Answer: {trace.final_answer}  [{status}]")
        print(f"  Events: {len(trace.events)} | Tokens: {trace.total_tokens_in}in/{trace.total_tokens_out}out | {trace.wall_time_ms:.0f}ms")

        if display_name == "Tree-of-Thoughts" and "tree" in trace.extra:
            print("\n  Search tree:")
            _print_tree(trace.extra["tree"], indent=4)


def _close(pred, gt) -> bool:
    try:
        return abs(float(pred) - float(gt)) < 0.5
    except (TypeError, ValueError):
        return False


def _print_tree(node: dict, indent: int = 0):
    prefix = " " * indent
    score = node.get("score", 0)
    pa = node.get("partial_answer", "")
    nid = node.get("id", "")
    print(f"{prefix}[{nid}] score={score:.2f}  partial={pa}")
    for child in node.get("children", []):
        _print_tree(child, indent + 4)


def main():
    parser = argparse.ArgumentParser(description="Single-problem demo across all strategies")
    parser.add_argument("--problem-id", default=None)
    parser.add_argument("--custom", default=None, dest="custom_problem")
    args = parser.parse_args()
    run_demo(problem=args.custom_problem, problem_id=args.problem_id)


if __name__ == "__main__":
    main()