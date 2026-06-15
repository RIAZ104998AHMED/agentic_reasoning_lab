"""
Tree-of-Thoughts (ToT) Strategy.
Paper: Yao et al. 2023 (https://arxiv.org/abs/2305.10601)

Algorithm:
1. Generate B=2 alternative continuations from the current state.
2. Score each continuation with a value LLM call (sure/likely/impossible).
3. Keep top-B states (beam search).
4. Repeat for max_depth steps.
5. Return the highest-scoring leaf's answer.

The full search tree is logged in trace.extra["tree"] for at least one example.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from strategies.base import Strategy, Trace, TraceEvent

_THOUGHT_SYSTEM = """\
You are solving a math problem step-by-step. Each response is ONE reasoning step.
Show your work for this single step, then state what you've established so far.
End with: PARTIAL ANSWER: <number or 'unknown'>
"""

_VALUE_SYSTEM = """\
Evaluate this partial solution to a math problem.
Reply with exactly one word on the first line: sure / likely / impossible
Then optionally explain briefly.
- sure: this path definitely leads to the correct final answer
- likely: this path seems on track but isn't certain
- impossible: this path has an error or is going nowhere
"""

_SCORE_MAP = {"sure": 1.0, "likely": 0.5, "impossible": 0.0}


@dataclass
class ToTNode:
    node_id: str
    depth: int
    state: str          # accumulated reasoning so far
    partial_answer: Optional[str]
    score: float = 0.0
    parent_id: Optional[str] = None
    children: list["ToTNode"] = field(default_factory=list)
    is_leaf: bool = False


class TreeOfThoughtsStrategy(Strategy):
    name = "tree_of_thoughts"

    def __init__(self, *args, beam_width: int = 2, max_depth: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.beam_width = beam_width
        self.max_depth = max_depth

    def solve(self, problem: str, problem_id: str) -> Trace:
        trace = Trace.empty(self.name, problem_id, problem)
        t_start = time.perf_counter()

        root = ToTNode(
            node_id="root", depth=0,
            state=f"Problem: {problem}",
            partial_answer=None, score=1.0,
        )
        all_nodes: dict[str, ToTNode] = {"root": root}
        frontier: list[ToTNode] = [root]
        node_counter = 0

        for depth in range(1, self.max_depth + 1):
            if not frontier:
                break

            candidates: list[ToTNode] = []

            for parent in frontier:
                for branch in range(self.beam_width):
                    node_counter += 1
                    node_id = f"n{node_counter}"

                    # ── Generate thought ────────────────────────────────────
                    gen_prompt = (
                        f"{parent.state}\n\n"
                        f"Continue solving. This is step {depth}, branch {branch + 1}. "
                        f"Show ONE reasoning step."
                    )
                    gen_text, tok_in, tok_out, latency = self._timed_llm(
                        prompt=gen_prompt,
                        system=_THOUGHT_SYSTEM,
                        problem_id=problem_id,
                        max_tokens=400,
                    )
                    trace.total_tokens_in += tok_in
                    trace.total_tokens_out += tok_out

                    partial = self._extract_partial(gen_text)
                    new_state = parent.state + f"\n\n[Step {depth}.{branch+1}]\n{gen_text}"

                    node = ToTNode(
                        node_id=node_id, depth=depth,
                        state=new_state, partial_answer=partial,
                        parent_id=parent.node_id,
                    )
                    parent.children.append(node)
                    all_nodes[node_id] = node

                    trace.events.append(TraceEvent.make(
                        strategy=self.name, problem_id=problem_id,
                        step_type="thought_generation",
                        inputs={"parent_id": parent.node_id, "depth": depth, "branch": branch},
                        outputs={"thought": gen_text, "partial_answer": partial},
                        tokens_in=tok_in, tokens_out=tok_out, latency_ms=latency,
                        node_id=node_id,
                    ))

                    # ── Score thought ───────────────────────────────────────
                    val_prompt = (
                        f"Problem: {problem}\n\n"
                        f"Partial solution:\n{gen_text}\n\n"
                        f"Is this on track?"
                    )
                    val_text, v_in, v_out, v_latency = self._timed_llm(
                        prompt=val_prompt,
                        system=_VALUE_SYSTEM,
                        problem_id=problem_id,
                        max_tokens=60,
                    )
                    trace.total_tokens_in += v_in
                    trace.total_tokens_out += v_out

                    score = self._parse_score(val_text)
                    node.score = parent.score * score   # propagate

                    trace.events.append(TraceEvent.make(
                        strategy=self.name, problem_id=problem_id,
                        step_type="value_scoring",
                        inputs={"node_id": node_id},
                        outputs={"value_text": val_text, "score": score, "cumulative": node.score},
                        tokens_in=v_in, tokens_out=v_out, latency_ms=v_latency,
                        node_id=node_id,
                    ))

                    candidates.append(node)

            # ── Beam selection ──────────────────────────────────────────────
            candidates.sort(key=lambda n: -n.score)
            frontier = [n for n in candidates[:self.beam_width] if n.score > 0.0]
            if not frontier:
                break

        # ── Extract answer from best leaf ───────────────────────────────────
        all_leaves = [n for n in all_nodes.values() if not n.children]
        if not all_leaves:
            all_leaves = list(all_nodes.values())
        best = max(all_leaves, key=lambda n: n.score)

        # Final answer extraction — ask model to conclude
        if best.partial_answer and best.partial_answer.lower() != "unknown":
            trace.final_answer = best.partial_answer
            trace.success = True
        else:
            conclude_prompt = (
                f"{best.state}\n\n"
                "Based on your reasoning above, state ONLY the final numerical answer."
            )
            conc_text, c_in, c_out, c_lat = self._timed_llm(
                prompt=conclude_prompt,
                system="Reply with only the final answer — a number or short phrase.",
                problem_id=problem_id,
                max_tokens=64,
            )
            trace.total_tokens_in += c_in
            trace.total_tokens_out += c_out
            nums = re.findall(r"[-+]?\d*\.?\d+", conc_text)
            if nums:
                trace.final_answer = nums[-1]
                trace.success = True
            else:
                trace.final_answer = conc_text.strip()
                trace.success = bool(conc_text.strip())

        # Store tree in trace for analysis
        trace.extra["tree"] = self._tree_to_dict(root)
        trace.extra["nodes_expanded"] = node_counter
        trace.extra["best_node"] = best.node_id
        trace.extra["best_score"] = best.score

        trace.wall_time_ms = (time.perf_counter() - t_start) * 1000
        return trace

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_partial(text: str) -> Optional[str]:
        m = re.search(r"PARTIAL ANSWER:\s*(.+)", text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    @staticmethod
    def _parse_score(text: str) -> float:
        first_word = text.strip().split()[0].lower().rstrip(".,:")
        return _SCORE_MAP.get(first_word, 0.5)

    def _tree_to_dict(self, node: ToTNode) -> dict:
        return {
            "id": node.node_id,
            "depth": node.depth,
            "score": round(node.score, 4),
            "partial_answer": node.partial_answer,
            "children": [self._tree_to_dict(c) for c in node.children],
        }