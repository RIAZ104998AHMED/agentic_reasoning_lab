"""
Plan-and-Execute Strategy.
Paper: Wang et al. 2023 (https://arxiv.org/abs/2305.04091)

Phase 1 — Planner LLM: produce a numbered step-by-step plan.
Phase 2 — Executor LLM: run each step sequentially, using tools where needed.
          Accumulates a scratchpad of completed steps before each new step.
"""

from __future__ import annotations

import re
import time

from strategies.base import Strategy, Trace, TraceEvent

_PLANNER_SYSTEM = """\
You are a meticulous planner. Given a math or reasoning problem, produce a
numbered, step-by-step plan that another agent will execute. Each step should
be a single, concrete action.

Output format (strictly follow this):
Step 1: <what to do>
Step 2: <what to do>

Step N: State the final answer.

Do not solve the problem — only plan it. Use tools when arithmetic is needed.
Available tools: calculator(expression), python_executor(code), retriever(query)
"""

_EXECUTOR_SYSTEM = """\
You are a precise executor. You will be given one step of a plan to carry out.
Use the provided tool if arithmetic or computation is needed.

Output format:
Thought: <brief reasoning>
Action: tool_name(argument)   ← include this line ONLY if you need a tool
Result: <what you computed or concluded>

If no tool is needed:
Thought: <brief reasoning>
Result: <what you concluded>

For the final step, write:
Final Answer: <the answer as a plain number or short phrase>
"""

_MAX_STEPS = 12


class PlanAndExecuteStrategy(Strategy):
    name = "plan_execute"

    def solve(self, problem: str, problem_id: str) -> Trace:
        trace = Trace.empty(self.name, problem_id, problem)
        t_start = time.perf_counter()

        # ── Phase 1: Plan ────────────────────────────────────────────────────
        plan_prompt = f"Problem: {problem}\n\nProduce a step-by-step plan."
        plan_text, tok_in, tok_out, latency = self._timed_llm(
            prompt=plan_prompt,
            system=_PLANNER_SYSTEM,
            problem_id=problem_id,
            max_tokens=512,
        )
        trace.total_tokens_in += tok_in
        trace.total_tokens_out += tok_out

        trace.events.append(TraceEvent.make(
            strategy=self.name, problem_id=problem_id,
            step_type="planning",
            inputs={"problem": problem},
            outputs={"plan": plan_text},
            tokens_in=tok_in, tokens_out=tok_out, latency_ms=latency,
        ))

        # Parse steps
        steps = self._parse_steps(plan_text)
        if not steps:
            trace.error = "Planner produced no steps"
            trace.wall_time_ms = (time.perf_counter() - t_start) * 1000
            return trace

        # ── Phase 2: Execute each step ───────────────────────────────────────
        scratchpad: list[str] = []

        for i, step_desc in enumerate(steps[:_MAX_STEPS], start=1):
            context = "\n".join(scratchpad)
            exec_prompt = (
                f"Problem: {problem}\n\n"
                f"Plan so far:\n{context}\n\n" if scratchpad else
                f"Problem: {problem}\n\n"
            ) + f"Now execute step {i}: {step_desc}"

            exec_text, tok_in, tok_out, latency = self._timed_llm(
                prompt=exec_prompt,
                system=_EXECUTOR_SYSTEM,
                problem_id=problem_id,
                max_tokens=512,
            )
            trace.total_tokens_in += tok_in
            trace.total_tokens_out += tok_out

            trace.events.append(TraceEvent.make(
                strategy=self.name, problem_id=problem_id,
                step_type="execution",
                inputs={"step": i, "step_desc": step_desc},
                outputs={"exec_text": exec_text},
                tokens_in=tok_in, tokens_out=tok_out, latency_ms=latency,
                plan_step=i,
            ))

            # Run tool if requested
            action_match = re.search(
                r"Action:\s*(\w+)\(([^)]*)\)", exec_text, re.IGNORECASE
            )
            observation = None
            if action_match:
                tool_name = action_match.group(1).strip().lower()
                tool_arg = action_match.group(2).strip()
                t_tool = time.perf_counter()
                if tool_name == "calculator":
                    result = self.tools.run("calculator", expression=tool_arg)
                elif tool_name in ("python_executor", "python"):
                    result = self.tools.run("python_executor", code=tool_arg)
                elif tool_name == "retriever":
                    result = self.tools.run("retriever", query=tool_arg)
                else:
                    result = self.tools.run(tool_name)
                tool_latency = (time.perf_counter() - t_tool) * 1000
                observation = result.output if result.ok else f"Error: {result.error}"

                trace.events.append(TraceEvent.make(
                    strategy=self.name, problem_id=problem_id,
                    step_type="tool_call",
                    inputs={"tool": tool_name, "arg": tool_arg},
                    outputs={"observation": observation},
                    latency_ms=tool_latency, plan_step=i,
                ))

            # Extract result
            result_match = re.search(r"Result:\s*(.+)", exec_text, re.IGNORECASE | re.DOTALL)
            step_result = result_match.group(1).strip() if result_match else exec_text.strip()
            if observation:
                step_result = f"{step_result} (tool: {observation})"
            scratchpad.append(f"Step {i} ({step_desc}): {step_result}")

            # Check for Final Answer
            final_match = re.search(r"Final Answer:\s*(.+)", exec_text, re.IGNORECASE)
            if final_match:
                trace.final_answer = final_match.group(1).strip()
                trace.success = True
                break

        if not trace.success:
            # Try to extract from last scratchpad entry
            if scratchpad:
                last = scratchpad[-1]
                nums = re.findall(r"[-+]?\d*\.?\d+", last)
                if nums:
                    trace.final_answer = nums[-1]
                    trace.success = True
                else:
                    trace.error = "Executor did not produce Final Answer"
            else:
                trace.error = "No steps executed"

        trace.wall_time_ms = (time.perf_counter() - t_start) * 1000
        return trace

    @staticmethod
    def _parse_steps(plan_text: str) -> list[str]:
        steps = []
        for line in plan_text.splitlines():
            m = re.match(r"Step\s+\d+[:.)]\s*(.+)", line.strip(), re.IGNORECASE)
            if m:
                steps.append(m.group(1).strip())
        return steps