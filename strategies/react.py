"""
ReAct Strategy — Reason, Act, Observe loop.
Paper: Yao et al. 2022 (https://arxiv.org/abs/2210.03629)

The model alternates between:
  Thought: <free-form reasoning>
  Action: <tool_name>(<args>)
  Observation: <tool result>
  ...
  Answer: <final answer>

We parse each turn and route tool calls to the shared ToolRegistry.
"""

from __future__ import annotations

import re
import time

from strategies.base import Strategy, Trace, TraceEvent
from tools.registry import ToolRegistry

_SYSTEM = """\
You are a precise reasoning agent. Solve problems using this exact format:

Thought: reason about what to do next
Action: tool_name(argument)
Observation: [result will be provided]
(repeat as needed)
Answer: <your final answer as a plain number or short phrase>

Available tools:
{tools}

Rules:
- Always show your Thought before any Action.
- Use exactly one Action per turn.
- After receiving an Observation, continue with the next Thought.
- When you have enough information, write Answer: followed by just the answer.
- For math problems, Answer must be a number only (no units, no explanation).
- Never guess — use the calculator for any arithmetic.
"""

_MAX_STEPS = 10


class ReActStrategy(Strategy):
    name = "react"

    def solve(self, problem: str, problem_id: str) -> Trace:
        trace = Trace.empty(self.name, problem_id, problem)
        t_start = time.perf_counter()

        system = _SYSTEM.format(tools=self.tools.describe())
        conversation = f"Problem: {problem}\n\nBegin."

        step = 0
        while step < _MAX_STEPS:
            step += 1
            t0 = time.perf_counter()
            text, tok_in, tok_out, latency = self._timed_llm(
                prompt=conversation,
                system=system,
                problem_id=problem_id,
                max_tokens=512,
            )
            trace.total_tokens_in += tok_in
            trace.total_tokens_out += tok_out

            # Log LLM call
            event = TraceEvent.make(
                strategy=self.name, problem_id=problem_id,
                step_type="llm_call",
                inputs={"prompt_tail": conversation[-300:]},
                outputs={"text": text},
                tokens_in=tok_in, tokens_out=tok_out, latency_ms=latency,
                step=step,
            )
            trace.events.append(event)
            from observability.logger import TraceLogger
            # logger is injected at run time if present — see harness.py

            conversation += f"\n\n{text}"

            # Check for Answer
            answer_match = re.search(r"Answer:\s*(.+)", text, re.IGNORECASE)
            if answer_match:
                trace.final_answer = answer_match.group(1).strip()
                trace.success = True
                break

            # Parse Action
            action_match = re.search(
                r"Action:\s*(\w+)\(([^)]*)\)", text, re.IGNORECASE
            )
            if not action_match:
                # Model didn't produce an action or answer — prompt it
                conversation += "\nObservation: (no action detected — please use Action: tool_name(args) or Answer: your_answer)"
                continue

            tool_name = action_match.group(1).strip().lower()
            tool_arg = action_match.group(2).strip()

            # Route to tool
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

            tool_event = TraceEvent.make(
                strategy=self.name, problem_id=problem_id,
                step_type="tool_call",
                inputs={"tool": tool_name, "arg": tool_arg},
                outputs={"observation": observation, "error": result.error},
                latency_ms=tool_latency, step=step,
            )
            trace.events.append(tool_event)

            conversation += f"\nObservation: {observation}"

        if not trace.success:
            trace.error = f"Exceeded {_MAX_STEPS} steps without Answer"

        trace.wall_time_ms = (time.perf_counter() - t_start) * 1000
        return trace