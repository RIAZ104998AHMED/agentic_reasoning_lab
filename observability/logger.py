
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from strategies.base import Trace, TraceEvent

_TRACE_DIR = Path(os.environ.get("TRACE_DIR", "traces"))


class TraceLogger:
    def __init__(self, run_id: Optional[str] = None):
        _TRACE_DIR.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or f"run_{int(time.time())}"
        self._event_path = _TRACE_DIR / f"{self.run_id}_events.jsonl"
        self._trace_path = _TRACE_DIR / f"{self.run_id}_traces.jsonl"

    def log(self, event: TraceEvent) -> None:
        with self._event_path.open("a") as f:
            f.write(json.dumps(asdict(event), default=str) + "\n")

    def log_trace_events(self, trace: Trace) -> None:
        for event in trace.events:
            self.log(event)

    def save_trace(self, trace: Trace) -> None:
        d = asdict(trace)
        d["n_events"] = len(d.pop("events"))
        with self._trace_path.open("a") as f:
            f.write(json.dumps(d, default=str) + "\n")

    @staticmethod
    def load_trace(trace_id: str) -> Optional[dict]:
        for path in sorted(_TRACE_DIR.glob("*_traces.jsonl")):
            with path.open() as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if obj.get("trace_id") == trace_id:
                            return obj
                    except json.JSONDecodeError:
                        pass
        return None

    @staticmethod
    def load_events_for_trace(trace_id: str) -> list[dict]:
        trace = TraceLogger.load_trace(trace_id)
        if not trace:
            return []
        strategy = trace["strategy"]
        problem_id = trace["problem_id"]
        events = []
        for path in sorted(_TRACE_DIR.glob("*_events.jsonl"), reverse=True):
            with path.open() as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if obj.get("strategy") == strategy and obj.get("problem_id") == problem_id:
                            events.append(obj)
                    except json.JSONDecodeError:
                        pass
        return events

    @staticmethod
    def list_traces() -> list[dict]:
        traces = []
        for path in sorted(_TRACE_DIR.glob("*_traces.jsonl"), reverse=True):
            with path.open() as f:
                for line in f:
                    try:
                        traces.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return traces