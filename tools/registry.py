"""
Shared tool implementations used by ALL strategies.
Keeping tools here ensures every strategy uses identical tools — 
fair comparisons require identical instruments.
"""

from __future__ import annotations

import ast
import math
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ToolResult:
    tool: str
    inputs: dict
    output: str
    error: Optional[str] = None
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


# ─────────────────────────────── Calculator ──────────────────────────────────

class Calculator:
    """
    Safe arithmetic evaluator using Python's ast module.
    Supports: +, -, *, /, //, %, **, sqrt, abs, round, floor, ceil, log.
    No exec/eval — parse-tree whitelist only.
    """
    name = "calculator"

    _ALLOWED_NODES = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
        ast.Pow, ast.USub, ast.UAdd,
    }
    _ALLOWED_FUNCS = {
        "sqrt": math.sqrt, "abs": abs, "round": round,
        "floor": math.floor, "ceil": math.ceil, "log": math.log,
        "log10": math.log10, "exp": math.exp,
    }

    def _safe_eval(self, node: ast.AST) -> float:
        if type(node) not in self._ALLOWED_NODES:
            raise ValueError(f"Disallowed AST node: {type(node).__name__}")
        if isinstance(node, ast.Expression):
            return self._safe_eval(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            left = self._safe_eval(node.left)
            right = self._safe_eval(node.right)
            ops = {
                ast.Add: lambda a, b: a + b,
                ast.Sub: lambda a, b: a - b,
                ast.Mult: lambda a, b: a * b,
                ast.Div: lambda a, b: a / b,
                ast.FloorDiv: lambda a, b: a // b,
                ast.Mod: lambda a, b: a % b,
                ast.Pow: lambda a, b: a ** b,
            }
            return ops[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            operand = self._safe_eval(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            return operand
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only named function calls allowed")
            fname = node.func.id
            if fname not in self._ALLOWED_FUNCS:
                raise ValueError(f"Unknown function: {fname}")
            args = [self._safe_eval(a) for a in node.args]
            return self._ALLOWED_FUNCS[fname](*args)
        raise ValueError(f"Cannot evaluate: {ast.dump(node)}")

    def run(self, expression: str) -> ToolResult:
        t0 = time.perf_counter()
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            result = self._safe_eval(tree)
            # Return clean numeric string
            if result == int(result):
                output = str(int(result))
            else:
                output = f"{result:.6g}"
            return ToolResult(
                tool=self.name, inputs={"expression": expression},
                output=output, latency_ms=(time.perf_counter() - t0) * 1000
            )
        except Exception as e:
            return ToolResult(
                tool=self.name, inputs={"expression": expression},
                output="", error=str(e),
                latency_ms=(time.perf_counter() - t0) * 1000
            )


# ─────────────────────────────── Python Executor ─────────────────────────────

class PythonExecutor:
    """
    Runs Python code in a subprocess with a strict timeout.
    Only stdout is captured; the last non-empty line is treated as the answer.
    """
    name = "python_executor"

    def __init__(self, timeout_seconds: int = 10):
        self.timeout = timeout_seconds

    def run(self, code: str) -> ToolResult:
        t0 = time.perf_counter()
        # Wrap code to catch exceptions and print them
        wrapped = textwrap.dedent(f"""\
import math, sys
try:
{textwrap.indent(code, '    ')}
except Exception as _e:
    print(f"ERROR: {{_e}}")
""")
        try:
            result = subprocess.run(
                [sys.executable, "-c", wrapped],
                capture_output=True, text=True,
                timeout=self.timeout,
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            latency = (time.perf_counter() - t0) * 1000

            if result.returncode != 0 or stdout.startswith("ERROR:"):
                return ToolResult(
                    tool=self.name, inputs={"code": code},
                    output=stdout, error=stderr or stdout,
                    latency_ms=latency
                )
            # Last non-empty line = answer
            lines = [l for l in stdout.splitlines() if l.strip()]
            answer = lines[-1] if lines else ""
            return ToolResult(
                tool=self.name, inputs={"code": code},
                output=answer, latency_ms=latency
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool=self.name, inputs={"code": code},
                output="", error=f"Timeout after {self.timeout}s",
                latency_ms=(time.perf_counter() - t0) * 1000
            )


# ─────────────────────────────── Simple Retriever ────────────────────────────

class SimpleRetriever:
    """
    Keyword-based retriever over a list of (title, text) documents.
    In production, swap this out for a vector store (e.g. from Task 3).
    Returns top-k snippets scored by term overlap.
    """
    name = "retriever"

    def __init__(self, docs: Optional[list[dict]] = None):
        # Each doc: {"id": ..., "title": ..., "text": ...}
        self.docs = docs or []

    def run(self, query: str, top_k: int = 3) -> ToolResult:
        t0 = time.perf_counter()
        if not self.docs:
            return ToolResult(
                tool=self.name, inputs={"query": query},
                output="No documents loaded.", latency_ms=0.0
            )
        tokens = set(re.sub(r"[^\w\s]", "", query.lower()).split())
        scored = []
        for doc in self.docs:
            text_tokens = set(doc["text"].lower().split())
            score = len(tokens & text_tokens) / (len(tokens) + 1)
            scored.append((score, doc))
        scored.sort(key=lambda x: -x[0])
        snippets = []
        for _, doc in scored[:top_k]:
            snippet = doc["text"][:400].replace("\n", " ")
            snippets.append(f"[{doc['title']}] {snippet}")
        output = "\n\n".join(snippets) if snippets else "No relevant documents found."
        return ToolResult(
            tool=self.name, inputs={"query": query, "top_k": top_k},
            output=output,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


# ─────────────────────────────── Tool Registry ───────────────────────────────

class ToolRegistry:
    """Single registry shared across all strategies."""

    def __init__(self, docs: Optional[list[dict]] = None):
        self.calculator = Calculator()
        self.executor = PythonExecutor()
        self.retriever = SimpleRetriever(docs=docs)
        self._map = {
            "calculator": self.calculator,
            "python_executor": self.executor,
            "retriever": self.retriever,
        }

    def run(self, tool_name: str, **kwargs) -> ToolResult:
        tool = self._map.get(tool_name)
        if tool is None:
            return ToolResult(
                tool=tool_name, inputs=kwargs,
                output="", error=f"Unknown tool: {tool_name}"
            )
        # Route kwargs by tool type
        if tool_name == "calculator":
            return tool.run(kwargs.get("expression", ""))
        if tool_name == "python_executor":
            return tool.run(kwargs.get("code", ""))
        if tool_name == "retriever":
            return tool.run(kwargs.get("query", ""), top_k=kwargs.get("top_k", 3))
        return ToolResult(tool=tool_name, inputs=kwargs, output="", error="Unrouted tool")

    def describe(self) -> str:
        return (
            "Available tools:\n"
            "- calculator(expression): evaluate a math expression, e.g. calculator(12 * 34 + 5)\n"
            "- python_executor(code): run Python code, print the answer on the last line\n"
            "- retriever(query): fetch relevant document snippets"
        )