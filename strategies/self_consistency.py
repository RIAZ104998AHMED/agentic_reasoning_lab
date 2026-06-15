"""
Self-Consistency Strategy.
Paper: Wang et al. 2022 (https://arxiv.org/abs/2203.11171)

Sample N ≥ 3 independent chain-of-thought reasoning paths in parallel,
then majority-vote on the extracted final answers.

For math problems the "answer" is extracted as a number and
voted on with numeric equality (within tolerance).
"""

from __future__ import annotations

import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from strategies.base import Strategy, Trace, TraceEvent

_SYSTEM = """\
Solve the following math or reasoning problem step by step.
Show all your work clearly.
At the end, write exactly:
#### <your numerical answer>
(just the number, no units, no text after it)
"""

_DEFAULT_N = 5
_NUMERIC_TOL = 1e-6


class SelfConsistencyStrategy(Strategy):
    name = "self_consistency"

    def __init__(self, *args, n_samples: int = _DEFAULT_N, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_samples = n_samples

    def solve(self, problem: str, problem_id: str) -> Trace:
        trace = Trace.empty(self.name, problem_id, problem)
        t_start = time.perf_counter()

        prompt = f"Problem: {problem}"

        # Sample N paths (parallelised with threads)
        paths: list[dict] = []

        def sample_one(idx: int) -> dict:
            t0 = time.perf_counter()
            # Add slight variation to prompt to break cache — index in prompt
            varied_prompt = f"{prompt}\n\n[Attempt {idx + 1}]"
            text, tok_in, tok_out, latency = self._timed_llm(
                prompt=varied_prompt,
                system=_SYSTEM,
                problem_id=problem_id,
                max_tokens=768,
            )
            answer = self._extract_answer(text)
            return {
                "idx": idx,
                "text": text,
                "answer": answer,
                "tok_in": tok_in,
                "tok_out": tok_out,
                "latency_ms": latency,
            }

        with ThreadPoolExecutor(max_workers=self.n_samples) as pool:
            futures = [pool.submit(sample_one, i) for i in range(self.n_samples)]
            for fut in as_completed(futures):
                paths.append(fut.result())

        # Sort by index for reproducible logging
        paths.sort(key=lambda x: x["idx"])

        for p in paths:
            trace.total_tokens_in += p["tok_in"]
            trace.total_tokens_out += p["tok_out"]
            trace.events.append(TraceEvent.make(
                strategy=self.name, problem_id=problem_id,
                step_type="llm_call",
                inputs={"sample_idx": p["idx"]},
                outputs={"text": p["text"], "extracted_answer": p["answer"]},
                tokens_in=p["tok_in"], tokens_out=p["tok_out"],
                latency_ms=p["latency_ms"],
                sample_idx=p["idx"],
            ))

        # ── Majority vote ────────────────────────────────────────────────────
        answers = [p["answer"] for p in paths if p["answer"] is not None]
        winner, votes = self._majority_vote(answers)

        vote_event = TraceEvent.make(
            strategy=self.name, problem_id=problem_id,
            step_type="voting",
            inputs={"all_answers": answers},
            outputs={"winner": winner, "vote_count": votes, "n_valid": len(answers)},
        )
        trace.events.append(vote_event)

        if winner is not None:
            trace.final_answer = str(winner)
            trace.success = True
        else:
            trace.error = "No valid answers extracted from any path"

        trace.extra["vote_distribution"] = dict(Counter(str(a) for a in answers))
        trace.wall_time_ms = (time.perf_counter() - t_start) * 1000
        return trace

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_answer(text: str) -> str | None:
        """Extract the answer after #### marker, or last plain number."""
        m = re.search(r"####\s*([-+]?\d*\.?\d+)", text)
        if m:
            return m.group(1).strip()
        # Fallback: last number in the text
        nums = re.findall(r"[-+]?\d*\.?\d+", text)
        return nums[-1] if nums else None

    @staticmethod
    def _majority_vote(answers: list[str]) -> tuple[str | None, int]:
        """
        Numeric-aware majority vote.
        Groups answers within _NUMERIC_TOL of each other.
        Returns (winning_answer_string, vote_count).
        """
        if not answers:
            return None, 0

        # Try numeric comparison first
        numeric: list[float] = []
        for a in answers:
            try:
                numeric.append(float(a))
            except ValueError:
                numeric.append(None)

        if all(n is not None for n in numeric):
            # Group by numeric proximity
            groups: list[list[int]] = []
            used = [False] * len(numeric)
            for i, val in enumerate(numeric):
                if used[i]:
                    continue
                group = [i]
                for j in range(i + 1, len(numeric)):
                    if not used[j] and abs(val - numeric[j]) <= _NUMERIC_TOL:
                        group.append(j)
                        used[j] = True
                used[i] = True
                groups.append(group)
            best_group = max(groups, key=len)
            rep_idx = best_group[0]
            return answers[rep_idx], len(best_group)

        # Fallback: exact string match
        counter = Counter(answers)
        winner, count = counter.most_common(1)[0]
        return winner, count