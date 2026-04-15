# PRISM v2.1 Implementation Guide

## Kaggle Community Benchmark -- Metacognition Track

**Competition:** Measuring Progress Toward AGI: Cognitive Abilities Hackathon
**Track:** Metacognition
**Deadline:** April 16, 2026

---

## Table of Contents

1. [What This Benchmark Measures](#what-this-benchmark-measures)
2. [Architecture Overview](#architecture-overview)
3. [File Structure](#file-structure)
4. [Step-by-Step Kaggle Deployment](#step-by-step-kaggle-deployment)
5. [How the Pipeline Works](#how-the-pipeline-works)
6. [Task Scoring Details](#task-scoring-details)
7. [Problem Design](#problem-design)
8. [SDK Integration Notes](#sdk-integration-notes)
9. [Troubleshooting](#troubleshooting)
10. [Scientific Background](#scientific-background)

---

## 1. What This Benchmark Measures

PRISM (Prospective-Retrospective Introspective Self-Model) measures **metacognition** -- whether LLMs can accurately monitor their own reasoning. It operationalizes the Nelson & Narens (1990) monitoring/control framework across three temporal dimensions:

| Dimension | When | What the LLM Does |
|-----------|------|--------------------|
| D1: Prospective | Before solving | Predict which steps will be hardest, rate per-step confidence, place a bet |
| D2: Performance | During solving | Solve the problem step-by-step |
| D3: Retrospective | After solving | Report which steps were hardest, self-assess each step, describe a counterfactual approach |

Six tasks derive scores from comparing D1, D2, and D3:

| Task | Metric | What It Tests | Range |
|------|--------|--------------|-------|
| T1 | AUROC | Can overall confidence predict correctness? | 0-1 (chance=0.5) |
| T2 | Spearman rho | Do per-step confidences track step difficulty? | 0-1 (chance=0.5) |
| T3 | Proportion | Can the model accurately recognize which steps it got right vs. wrong after solving? | 0-1 |
| T4 | Weighted composite | Are before/after metacognitive reports coherent? | 0-1 |
| T5 | Utility ratio | Does the model act on its self-knowledge (accept/decline)? | 0-1 |
| T6 | L2/L1 ratio | Does metacognition survive novel operators? | 0-1 |

**Primary score = 0.40 × mean(T4_L1, T4_L2) + 0.30 × mean(T5_L1, T5_L2) + 0.30 × min(T6, 1)** — a weighted combination of coherence (the most novel construct), metacognitive control, and novelty robustness.

---

## 2. Architecture Overview

```
                     +-------------------+
                     |  Problem Generator |
                     |  (L1 + L2 sets)   |
                     +---------+---------+
                               |
                    +----------v----------+
                    |   Pipeline Engine    |
                    |  D1 -> D2 -> D3     |
                    |  per problem        |
                    +----------+----------+
                               |
              +----------------+----------------+
              |                |                |
        +-----v-----+   +-----v-----+   +-----v-----+
        |  Prompts   |   |  Scoring  |   |  Metrics  |
        |  (D1/D2/D3)|   |  (Parser, |   |  (AUROC,  |
        |            |   |   Scorer) |   |   Spearman)|
        +-----+------+   +-----+-----+   +-----+-----+
              |                |                |
              +----------------+----------------+
                               |
                    +----------v----------+
                    |   Task Modules      |
                    |  T1-T6 compute_*()  |
                    +----------+----------+
                               |
                    +----------v----------+
                    |   Kaggle Notebook   |
                    |   @kbench.task      |
                    |   %choose           |
                    +---------------------+
```

---

## 3. File Structure

```
prism_v2/
  __init__.py                              # Package init (v2.1.0)
  pipeline.py                              # Pipeline orchestrator
  notebook.py                              # Plain Python notebook code
  PRISM_v2_1_Benchmark.ipynb               # Jupyter notebook for Kaggle

  problems/
    __init__.py
    generator.py                           # Problem generation engine
    l1_problems.json                       # 20 pre-generated L1 problems
    l2_problems.json                       # 20 pre-generated L2 problems

  prompts/
    __init__.py
    system.py                              # System prompt
    prospective.py                         # D1 prompt builder
    solve.py                               # D2 prompt builder
    retrospective.py                       # D3 prompt builder
    decision.py                            # Decision prompt builder (Task 5)
    feedback.py                            # Feedback round prompt (reserved)

  scoring/
    __init__.py
    confidence_parser.py                   # Parse LLM metacognitive responses
    step_scorer.py                         # Extract and verify step answers
    decision_scorer.py                     # Parse accept/decline decisions
    metrics.py                             # All statistical metrics

  tasks/
    __init__.py
    task_01_prospective_calibration.py     # AUROC
    task_02_step_accuracy.py               # Spearman rho
    task_03_retrospective_accuracy.py      # Self-assessment accuracy
    task_04_coherence.py                   # Coherence composite
    task_05_adaptive_calibration.py        # Metacognitive control (accept/decline)
    task_06_novelty_robustness.py          # L2/L1 ratio
```

---

## 4. Step-by-Step Kaggle Deployment

### Step 1: Create a Kaggle Dataset

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) and click **New Dataset**
2. Name it `prism-v2-benchmark` (or `prism-v2`)
3. Upload the entire `prism_v2/` directory as a folder:
   - All `.py` files (including subdirectories)
   - Both `.json` problem files
   - Do NOT upload `notebook.py` or `PRISM_v2_1_Benchmark.ipynb` (these go in the notebook, not the dataset)
4. Publish the dataset

The directory structure in the dataset should be:
```
/kaggle/input/prism-v2-benchmark/
  prism_v2/
    __init__.py
    pipeline.py
    problems/
      __init__.py
      generator.py
      l1_problems.json
      l2_problems.json
    prompts/
      __init__.py
      system.py
      prospective.py
      solve.py
      retrospective.py
      decision.py
      feedback.py
    scoring/
      __init__.py
      confidence_parser.py
      step_scorer.py
      decision_scorer.py
      metrics.py
    tasks/
      __init__.py
      task_01_prospective_calibration.py
      task_02_step_accuracy.py
      task_03_retrospective_accuracy.py
      task_04_coherence.py
      task_05_adaptive_calibration.py
      task_06_novelty_robustness.py
```

### Step 2: Create the Benchmark Notebook

1. Go to [kaggle.com/benchmarks/tasks/new](https://www.kaggle.com/benchmarks/tasks/new) to create a new Benchmarks notebook
2. Attach your `prism-v2-benchmark` dataset to the notebook
3. Either:
   - **Option A:** Upload `PRISM_v2_1_Benchmark.ipynb` directly
   - **Option B:** Copy-paste cells from `notebook.py` into the Kaggle notebook editor
4. Run all cells to verify

### Step 3: Verify the Notebook

The notebook has 6 code cells:

| Cell | Purpose | Key Code |
|------|---------|----------|
| 1 | Imports | `import kaggle_benchmarks as kbench` + prism_v2 imports |
| 2 | Load problems | `load_problems()` finds JSON files or generates at runtime |
| 3 | Init pipeline | `PrismPipeline(l1_problems, l2_problems)` |
| 4 | Define task | `@kbench.task(name="prism_metacognition")` with full scoring |
| 5 | Run | `prism_metacognition.run(llm=kbench.llm)` |
| 6 | Select | `%choose prism_metacognition` |

### Step 4: Submit

- Cell 6 (`%choose prism_metacognition`) designates this task for the leaderboard
- The task returns the **primary composite score** as the leaderboard metric:
  `0.40 * mean(T4_L1, T4_L2) + 0.30 * mean(T5_L1, T5_L2) + 0.30 * min(T6, 1)`
- All six task scores are reported via `kbench.assertions.assert_true(...)` for visibility

---

## 5. How the Pipeline Works

### Execution Flow

For each of the active main-evaluation problems (default: 10 L1 + 10 L2):

```
1. Create isolated chat: kbench.chats.new("prism_{id}", system_instructions=SYSTEM_PROMPT)
2. D1: Send prospective prompt -> LLM predicts weakest step, rates confidence, places bet
3. D2: Send solve prompt -> LLM solves step-by-step (same conversation)
4. D3: Send retrospective prompt (with per-step correctness feedback) -> LLM self-assesses
5. Parse all three responses; score step answers against ground truth
6. Store ProblemResult in cache
```

For 10 decision problems (5 L1 + 5 L2) used by Task 5:

```
1. Create isolated chat per decision problem
2. Present problem with explicit payoff structure (+correct, -wrong, +decline)
3. Model decides ACCEPT or DECLINE, then attempts to solve regardless
4. Parse decision and evaluate solve attempt against ground truth
5. Store DecisionResult in cache
```

### Caching

- `pipeline.run_all(llm, kbench)` runs once and sets `_has_run = True`
- Subsequent calls return immediately
- `kbench.client.enable_cache()` wraps all LLM calls in Kaggle's response cache

---

## 6. Task Scoring Details

### Task 1: Prospective Calibration (AUROC)

- **Input:** Per-problem bet fractions (D1) + per-problem correctness (D2)
- **Metric:** Area Under the ROC Curve via Mann-Whitney U
- **Interpretation:** Can the model distinguish problems it will get right from problems it will get wrong?
- **Chance level:** 0.5

### Task 2: Step-Level Prospective Accuracy (Spearman rho)

- **Input:** Per-step confidence vectors (D1) + per-step correctness (D2)
- **Metric:** Average Spearman rank correlation, normalized to [0, 1] via `(rho + 1) / 2`
- **Interpretation:** Do per-step confidence ratings track actual step difficulty?
- **Chance level:** 0.5

### Task 3: Retrospective Self-Assessment

- **Input:** Per-step self-assessment labels (D3) + actual correctness (D2)
- **Expected labels:** "confident and correct", "uncertain and correct", "confident but wrong", "uncertain and wrong"
- **Metric:** Proportion of steps where the reported correct/wrong status matches reality
- **Interpretation:** Can the model accurately recognize its own step-level successes and failures after the fact?

### Task 4: Coherence (Composite)

Three sub-scores:

| Sub-score | Weight | What it measures |
|-----------|--------|-----------------|
| A: Location consistency | 0.35 | Predicted-weakest == reported-hardest? |
| B: Confidence consistency | 0.45 | Spearman rho between D1 confidence and D3 difficulty |
| C: Counterfactual plausibility | 0.20 | LLM-as-judge rates counterfactual quality (demoted; fluent reflection is weak evidence of genuine self-monitoring) |

- **Sub-score C** uses `kbench.assertions.assess_response_with_judge()` with 3 criteria:
  1. Is the alternative approach mathematically valid?
  2. Is the reason for rejecting it logically coherent?
  3. Is it substantively different from what was actually done?

### Task 5: Metacognitive Control (Utility Ratio)

- **Input:** Accept/decline decisions + solve outcomes + payoff structures
- **Problems:** 5 per novelty level (3 solvable at varying difficulty/risk + 2 unsolvable)
- **Payoff profiles:** low (+5/-3/+1), medium (+10/-15/+2), high (+20/-30/+3)
- **Metric:** `(model_utility - worst_utility) / (optimal_utility - worst_utility)`
- **Interpretation:** Does the model act on its self-knowledge by declining problems it can't solve and accepting problems it can?
- **Range:** 0.0 (worst possible decisions) to 1.0 (optimal oracle)

### Task 6: Novelty Robustness (L2/L1 ratio)

- **Input:** Tasks 1-4 scores for both L1 and L2
- **Metric:** `min(L2_composite / L1_composite, 1.0)`
- **Interpretation:** Does metacognition survive novel operators (Zeta, Star)?

---

## 7. Problem Design

### Problem Types

| Type | L1 (Familiar) | L2 (Novel) |
|------|--------------|------------|
| A | 3x3 linear systems (Gaussian elimination) | Zeta-operator computation chains |
| B | Multi-step word problems (discount, tax, shipping) | Word problems with Zeta arithmetic |
| C | Modular arithmetic chains | Modular arithmetic with Star-operators |

### Novelty Levels

- **L1 (Familiar):** Standard mathematical operations
- **L2 (Novel):** Custom in-context operators that override normal arithmetic
  - **Zeta operators:** `a (+) b = a + b + 1`, `a (-) b = a - b - 1`, `a (*) b = a*b + a + b`
  - **Star operators:** `a (+) b = 2a + b` (non-commutative), `a (-) b = 2a - b`

### Difficulty Levels

Each type has easy, medium, and hard variants with increasing coefficient/operand ranges.

### Problem Counts

- Default eval slice: 10 main problems per novelty level (sampled from a balanced bank)
- 5 decision problems per novelty level (3 solvable + 2 unsolvable, used for Task 5)
- Optional bank mode: 45 main problems per novelty level (5 per type/difficulty cell) plus 5 decision problems
- Feedback problems are legacy/reserved and not used in the current Kaggle path

---

## 8. SDK Integration Notes

### Key SDK APIs Used

```python
# Task definition
@kbench.task(name="prism_metacognition")
def prism_metacognition(llm) -> float:
    ...
    return score  # float returned = numeric leaderboard score

# LLM interaction
response = llm.prompt(message)  # returns str by default

# Chat management (multi-turn conversations)
with kbench.chats.new("chat_name", system_instructions="..."):
    r1 = llm.prompt("first message")
    r2 = llm.prompt("second message")  # has context from r1

# Response caching
with kbench.client.enable_cache():
    pipeline.run_all(llm, kbench)

# Score reporting (visible on leaderboard)
kbench.assertions.assert_true(True, expectation=f"T1 AUROC: {score:.3f}")

# LLM-as-judge assessment (creates its own chat internally)
assessment = kbench.assertions.assess_response_with_judge(
    criteria=["criterion 1", "criterion 2"],
    response_text="text to assess",
    judge_llm=kbench.judge_llm,
)

# Task selection for leaderboard
%choose prism_metacognition
```

### Important SDK Constraints

1. **`assess_response_with_judge` creates its own chat** -- do NOT wrap it in `kbench.chats.new()`
2. **`llm.prompt()` returns `str`** by default (or typed `T` if `schema` is specified)
3. **`kbench.llm`** is the model under test; **`kbench.judge_llm`** is the judge model
4. **No external dependencies** -- the benchmark uses only Python stdlib (no numpy, scipy, sklearn)

---

## 9. Troubleshooting

### Common Issues

**Problem: "ModuleNotFoundError: No module named 'prism_v2'"**
- Ensure the dataset is attached to the notebook
- Check that the `sys.path` setup in Cell 1 matches your dataset name
- The code tries three paths: `prism-v2-benchmark`, `prism-v2`, and `.`

**Problem: "Could not extract predicted weakest step" parse errors**
- The parser tries multiple patterns (WEAKEST STEP, HARDEST, step N)
- If the LLM uses an unexpected format, the parser defaults gracefully
- Parse error rate is reported in the pipeline summary assertion

**Problem: Task scores are all 0.5 (chance level)**
- Check that LLM responses are being parsed correctly
- Verify that ground truth step answers match the expected format
- If all confidences default to "uncertain" (3), Spearman rho becomes undefined

**Problem: Task 4 counterfactual score is 0.0**
- The counterfactual parser looks for "COUNTERFACTUAL:" header
- If the LLM doesn't produce this section, the score defaults to 0.0
- The LLM-as-judge call may fail silently (caught by try/except)

---

## 10. Scientific Background

### Theoretical Framework

PRISM v2.1 operationalizes the **Nelson & Narens (1990) metacognitive monitoring framework**, which distinguishes:

- **Prospective monitoring:** Judgments of learning (JOL) -- predictions about future performance
- **Concurrent monitoring:** Feeling of knowing (FOK) -- real-time confidence during task execution
- **Retrospective monitoring:** Retrospective confidence judgments (RCJ) -- post-hoc self-assessment

The benchmark extends this framework with a novel **coherence dimension** (Task 4) that measures whether prospective and retrospective judgments are internally consistent -- a property that genuine metacognition should exhibit but superficial response patterns may not.

### Key Hypotheses

1. **H1:** LLMs will show above-chance prospective calibration (AUROC > 0.5) on familiar problems
2. **H2:** Step-level accuracy tracking (Spearman rho) will be weaker than global calibration
3. **H3:** Retrospective accuracy will exceed prospective accuracy (hindsight advantage)
4. **H4:** Coherence scores will discriminate between models more sharply than individual metrics
5. **H5:** Novel operators (L2) will degrade metacognitive performance, revealing reliance on pattern matching

### References

- Nelson, T.O. & Narens, L. (1990). Metamemory: A theoretical framework and new findings.
- Fleming, S.M. & Lau, H.C. (2014). How to measure metacognition.
- Kadavath, S. et al. (2022). Language models (mostly) know what they know.
- Lin, S. et al. (2022). Teaching models to express their uncertainty in words.
