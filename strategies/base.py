"""
Base Strategy interface and Trace dataclass.
All reasoning strategies must implement this interface so they can be
compared apples-to-apples in the eval harness.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TraceEvent:
    """A single structured event in a reasoning trace."""
    event_id: str
    timestamp: float
    strategy: str
    problem_id: str
    step_type: str          # llm_call | tool_call | reasoning | observation | final_answer
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make(strategy: str, problem_id: str, step_type: str,
             inputs: dict, outputs: dict, tokens_in: int = 0,
             tokens_out: int = 0, latency_ms: float = 0.0,
             **metadata) -> "TraceEvent":
        return TraceEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            strategy=strategy,
            problem_id=problem_id,
            step_type=step_type,
            inputs=inputs,
            outputs=outputs,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            metadata=metadata,
        )


@dataclass
class Trace:
    """Complete trace for one strategy solving one problem."""
    trace_id: str
    strategy: str
    problem_id: str
    problem: str
    final_answer: Optional[str]
    events: list[TraceEvent]
    wall_time_ms: float
    total_tokens_in: int
    total_tokens_out: int
    success: bool
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)   # e.g. ToT search tree

    @property
    def total_tokens(self) -> int:
        return self.total_tokens_in + self.total_tokens_out

    @staticmethod
    def empty(strategy: str, problem_id: str, problem: str) -> "Trace":
        return Trace(
            trace_id=str(uuid.uuid4()),
            strategy=strategy,
            problem_id=problem_id,
            problem=problem,
            final_answer=None,
            events=[],
            wall_time_ms=0.0,
            total_tokens_in=0,
            total_tokens_out=0,
            success=False,
        )


class Strategy(ABC):
    """
    Abstract base class for all reasoning strategies.

    Subclasses must implement `solve()`. They receive a shared ToolRegistry
    so every strategy uses exactly the same tools - ensuring fair comparisons.
    """

    name: str = "base"

    def __init__(self, model: str, tool_registry=None, cache=None):
        self.model = model
        self.tools = tool_registry      # shared ToolRegistry instance
        self.cache = cache              # optional LLM call cache

    @abstractmethod
    def solve(self, problem: str, problem_id: str) -> Trace:
        """
        Solve `problem` and return a fully-populated Trace.
        Must never raise - catch exceptions and set trace.error instead.
        """
        ...

    def _timed_llm(self, prompt: str, system: str = "",
                   problem_id: str = "", max_tokens: int = 1024) -> tuple[str, int, int, float]:
        """
        Call the LLM (via anthropic client), return (text, tok_in, tok_out, latency_ms).
        Uses cache keyed on (model, system, prompt).
        """
        from observability.llm_client import call_llm
        return call_llm(
            model=self.model,
            system=system,
            prompt=prompt,
            max_tokens=max_tokens,
            cache=self.cache,
        )