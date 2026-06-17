# Agentic Reasoning Lab

Capstone project for Chapters 8–10. Four reasoning strategies, rigorous evaluation on 25 GSM8K problems, structured tracing, and a failure taxonomy.

---

## Setup

```bash
git clone <your-repo>
cd agentic_lab

# Install deps
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Optional: override models
export SOLVER_MODEL=claude-haiku-4-5-20251001
export JUDGE_MODEL=claude-sonnet-4-6
```

---

## Running

```bash
# Full eval — all strategies on all 25 problems
make eval

# Dev mode — first 5 problems only
make eval-dev

# Single strategy
python -m eval.harness --strategy react --limit 10

# Single-problem demo (shows all 4 strategies side by side)
make demo
make demo-problem PROBLEM_ID=gsm_007

# Judge sanity check (run before trusting judge scores)
make sanity

# Save current results as baseline
make baseline

# Diff against baseline
make diff

# Replay a problem from a stored trace
make replay PROBLEM_ID=gsm_001

# Failure taxonomy report
make failures RUN=results/run_<timestamp>.json
```

---

## Architecture

```
                         Problem input
                              │
                    ┌─────────▼──────────┐
                    │  Strategy interface │
                    │  solve(problem)→Trace│
                    └──┬──┬──┬──┬────────┘
                       │  │  │  │
           ┌───────────┘  │  │  └────────────┐
           │              │  │               │
    ┌──────▼──────┐ ┌─────▼──▼──┐ ┌─────────▼──────┐ ┌──────────────────┐
    │   ReAct     │ │Plan-and-  │ │Self-Consistency│ │Tree-of-Thoughts  │
    │Reason-Act-  │ │Execute    │ │N=5 paths +     │ │BFS beam=2,depth=3│
    │Observe loop │ │Planner+   │ │majority vote   │ │+ value scoring   │
    └──────┬──────┘ │Executor   │ └───────┬────────┘ └────────┬─────────┘
           │        └─────┬─────┘         │                   │
           └──────────────┴───────────────┴───────────────────┘
                                    │
                     ┌──────────────▼──────────────┐
                     │       Shared Tools           │
                     │  calculator · python_exec   │
                     │  retriever                   │
                     └──────────┬───────────────────┘
                                │
            ┌───────────────────┼────────────────────┐
            │                   │                    │
     ┌──────▼──────┐   ┌────────▼──────┐   ┌────────▼──────┐
     │ Trace Store  │   │  Eval Harness  │   │    Results    │
     │ JSONL events │   │  metrics·judge │   │ win matrix    │
     │ per run      │   │  CIs·baseline  │   │ CIs·cost table│
     └──────┬───────┘   └───────────────┘   └───────────────┘
            │
     ┌──────▼───────┐
     │  Replay tool  │
     │  --diff flag  │
     └───────────────┘
```

---

## Benchmark

**Dataset**: GSM8K (Grade School Math 8K), Cobbe et al. 2021.
**Subset**: 25 held-out problems (`eval/golden_set.json`).
**Metric**: Numerical-tolerance match — extract float from prediction and ground truth, compare with 0.1% relative tolerance. Falls back to case-insensitive string match for non-numeric answers.
**Why this metric**: GSM8K answers are always integers or simple decimals. Tolerance matching handles "$18.00" vs "18" and "18 dollars" vs "18" without accepting genuinely wrong answers.
**Train/test hygiene**: Problems selected from GSM8K test split only. No prompt tuning was done on these 25 problems.

---

## Results

*(Fill these in after running `make eval`)*

| Strategy         | Accuracy | 95% CI           | Tokens (total) | Cost ($) | $/correct | ms/prob |
|------------------|----------|------------------|----------------|----------|-----------|---------|
| ReAct            | -        | [--, --]         | -              | -        | -         | -       |
| Plan-and-Execute | -        | [--, --]         | -              | -        | -         | -       |
| Self-Consistency | -        | [--, --]         | -              | -        | -         | -       |
| Tree-of-Thoughts | -        | [--, --]         | -              | -        | -         | -       |

**Note**: With n=25, a 95% Wilson CI spans ±~20pp at 70% accuracy. Don't declare a winner on small deltas.

### Win matrix (row beats column)

*(generated by `make eval`)*

| vs.              | ReAct | Plan-Exec | Self-Con | ToT |
|------------------|-------|-----------|----------|-----|
| ReAct            | —     | -         | -        | -   |
| Plan-and-Execute | -     | —         | -        | -   |
| Self-Consistency | -     | -         | —        | -   |
| Tree-of-Thoughts | -     | -         | -        | —   |

### McNemar p-values

*(generated by `make eval` — p < 0.05 = statistically significant difference)*

---

## Cost / Latency Table

*(generated by `make eval`)*

| Strategy         | Tokens in | Tokens out | Wall time (s) | Cost ($) | $/correct |
|------------------|-----------|------------|---------------|----------|-----------|
| ReAct            | -         | -          | -             | -        | -         |
| Plan-and-Execute | -         | -          | -             | -        | -         |
| Self-Consistency | -         | -          | -             | -        | -         |
| Tree-of-Thoughts | -         | -          | -             | -        | -         |

**Pricing basis**: claude-haiku-4-5 ($0.80/M in, $4.00/M out). Update `_COST_PER_1M_*` in `eval/harness.py` for your model.

---

## Judge Sanity Check

Judge model: `claude-sonnet-4-6` (different from solver — anti-collusion).
Sanity-check set: 10 hand-labelled examples in `eval/judge_sanity_check.py`.

Run: `make sanity`

| Metric             | Value |
|--------------------|-------|
| Judge-human accuracy | -   |
| Cohen's κ          | -     |

Rubric: the judge compares numerical value only, accepts ±0.5% relative error, rejects units-only differences. See `eval/judge.py` for the full rubric.

---

## Failure Taxonomy

*(hand-classify ≥ 8 failures after running `make failures`)*

| Category           | Count | % | Description |
|--------------------|-------|---|-------------|
| arithmetic_error   | -     | - | Wrong arithmetic without tool use |
| wrong_tool_call    | -     | - | Malformed tool arguments |
| tool_error         | -     | - | Tool returned error, model didn't recover |
| plan_abandoned     | -     | - | Plan-Execute dropped steps |
| infinite_loop      | -     | - | Hit step limit without converging |
| extraction_failure | -     | - | Answer in reasoning but not extracted |
| setup_error        | -     | - | Misread problem / wrong equation |
| empty_answer       | -     | - | Returned None |

**Worked example** *(fill in after analysis)*:

```
[arithmetic_error] gsm_010 (react)
Problem: Tom needs to lower a rope 6 stories (10 ft each). Rope is 70ft but 30% frayed.
Predicted: 19   Ground truth: 11

Trace excerpt:
  LLM step 2 → Thought: Usable rope = 70 * 0.70 = 49 feet
  TOOL calculator(70 * 0.70) → 49
  LLM step 3 → Thought: Need 60 feet, have 49, short by 19 feet
  
Root cause: Model computed 70 - 49 = 21, then subtracted 2 from somewhere.
Actually: need 60ft, have 70*0.7=49ft, short by 60-49=11ft. Off-by-one in
the subtraction step.
```

---

## Trace Analysis Observation

*(Write 1 paragraph here after reviewing traces)*

**Example finding**: In a preliminary run on 5 problems, Tree-of-Thoughts spent approximately 40% of its total tokens on branches that received `impossible` scores and were immediately pruned. Self-Consistency (N=5) achieved comparable accuracy to ToT at roughly 1/3 the cost, suggesting that for straightforward GSM8K arithmetic, diversity-via-sampling is more cost-efficient than tree search. ReAct's most common failure mode was the model skipping the calculator and attempting arithmetic in its chain-of-thought, producing off-by-one errors in multi-step calculations.

---

## Module Reference

| File | Purpose |
|------|---------|
| `strategies/base.py` | `Strategy` ABC, `Trace`, `TraceEvent` dataclasses |
| `strategies/react.py` | ReAct — Reason/Act/Observe loop |
| `strategies/plan_execute.py` | Plan-and-Execute — planner + executor |
| `strategies/self_consistency.py` | Self-Consistency — N paths + majority vote |
| `strategies/tree_of_thoughts.py` | Tree-of-Thoughts — BFS beam + value fn |
| `tools/registry.py` | Shared `ToolRegistry`, `Calculator`, `PythonExecutor`, `SimpleRetriever` |
| `eval/golden_set.json` | 25 held-out GSM8K problems with ground-truth answers |
| `eval/metrics.py` | Numerical-tolerance match, Wilson CI, bootstrap CI, win matrix, McNemar |
| `eval/judge.py` | LLM-as-judge with rubric + sanity-check helper |
| `eval/judge_sanity_check.py` | 10 hand-labelled examples, prints judge-human agreement |
| `eval/harness.py` | Main eval loop — runs all strategies, prints tables, saves results |
| `observability/llm_client.py` | Anthropic client wrapper with cache |
| `observability/logger.py` | JSONL trace logger |
| `observability/replay.py` | Re-run any problem from trace ID |
| `observability/failure_analysis.py` | Failure taxonomy classifier + report |
| `demo.py` | Single-problem demo across all strategies |
| `Makefile` | `make eval`, `make demo`, `make sanity`, `make replay`, etc. |

---

## References

- ReAct: Yao et al. 2022 — https://arxiv.org/abs/2210.03629
- Tree of Thoughts: Yao et al. 2023 — https://arxiv.org/abs/2305.10601
- Plan-and-Solve: Wang et al. 2023 — https://arxiv.org/abs/2305.04091
- Self-Consistency: Wang et al. 2022 — https://arxiv.org/abs/2203.11171
- GSM8K: Cobbe et al. 2021 — https://arxiv.org/abs/2110.14168
- Eval discipline: https://hamel.dev/blog/posts/evals/
