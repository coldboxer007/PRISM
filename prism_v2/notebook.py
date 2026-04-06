"""
PRISM v2.1: Prospective-Retrospective Introspective Self-Model
==============================================================

Kaggle Community Benchmark — Metacognition Track
Measuring Progress Toward AGI: Cognitive Abilities Hackathon

This notebook implements a metacognition benchmark that measures whether
LLMs can accurately monitor their own reasoning — before, during, and
after solving multi-step mathematical problems.

It operationalizes the Nelson & Narens (1990) monitoring/control framework
across six complementary tasks:

  Task 1: Prospective Calibration (AUROC)
  Task 2: Step-Level Prospective Accuracy (Spearman rho)
  Task 3: Retrospective Self-Assessment Accuracy
  Task 4: Prospective-Retrospective Coherence (composite)
  Task 5: Metacognitive Control (accept/decline utility)
  Task 6: Novelty Robustness (L2/L1 ratio)

Primary score = 0.40 * mean(T4_L1, T4_L2) + 0.30 * T5_L1 + 0.30 * min(T6, 1)

Usage on Kaggle:
  1. Upload the prism_v2/ package as a Kaggle dataset
  2. Create a new Benchmarks notebook at kaggle.com/benchmarks/tasks/new
  3. Paste these cells into the notebook
  4. Run all cells
"""

# ============================================================================
# CELL 1: Imports and Setup
# ============================================================================

import json
import os
import sys

import kaggle_benchmarks as kbench

# Add the prism_v2 package to path if running from dataset
# On Kaggle, the dataset is mounted at /kaggle/input/<dataset-name>/
PRISM_PATHS = [
    "/kaggle/input/prism-v2-benchmark",
    "/kaggle/input/prism-v2",
    ".",
]
for path in PRISM_PATHS:
    if os.path.isdir(os.path.join(path, "prism_v2")):
        sys.path.insert(0, path)
        break

from prism_v2.problems.generator import generate_all_problems
from prism_v2.pipeline import PrismPipeline
from prism_v2.tasks.task_01_prospective_calibration import compute_task_1
from prism_v2.tasks.task_02_step_accuracy import compute_task_2
from prism_v2.tasks.task_03_retrospective_accuracy import compute_task_3
from prism_v2.tasks.task_04_coherence import compute_task_4
from prism_v2.tasks.task_05_adaptive_calibration import compute_task_5
from prism_v2.tasks.task_06_novelty_robustness import compute_task_6

# ============================================================================
# CELL 2: Generate or Load Problem Sets
# ============================================================================


# Try to load pre-generated problems; fall back to runtime generation
def load_problems():
    """Load problem sets from JSON files or generate them."""
    problem_paths = [
        "/kaggle/input/prism-v2-benchmark/prism_v2/problems",
        "/kaggle/input/prism-v2/prism_v2/problems",
        "prism_v2/problems",
    ]

    for base_path in problem_paths:
        l1_path = os.path.join(base_path, "l1_problems.json")
        l2_path = os.path.join(base_path, "l2_problems.json")
        if os.path.exists(l1_path) and os.path.exists(l2_path):
            with open(l1_path) as f:
                l1 = json.load(f)
            with open(l2_path) as f:
                l2 = json.load(f)
            print(f"Loaded problems from {base_path}")
            return l1, l2

    # Fall back to generation
    print("Generating problems at runtime...")
    data = generate_all_problems(base_seed=42, num_main=10, num_feedback=5)
    return data["l1"], data["l2"]


l1_problems, l2_problems = load_problems()
l1_main = sum(
    1 for p in l1_problems if "feedback" not in p["id"] and "decision" not in p["id"]
)
l1_dec = sum(1 for p in l1_problems if "decision" in p["id"])
l2_main = sum(
    1 for p in l2_problems if "feedback" not in p["id"] and "decision" not in p["id"]
)
l2_dec = sum(1 for p in l2_problems if "decision" in p["id"])
print(f"L1 problems: {len(l1_problems)} (main: {l1_main}, decision: {l1_dec})")
print(f"L2 problems: {len(l2_problems)} (main: {l2_main}, decision: {l2_dec})")

# ============================================================================
# CELL 3: Initialize Pipeline
# ============================================================================

pipeline = PrismPipeline(l1_problems, l2_problems)

# ============================================================================
# CELL 4: Define the Main Benchmark Task
# ============================================================================


@kbench.task(name="prism_metacognition")
def prism_metacognition(llm) -> float:
    """
    PRISM v2.1: Prospective-Retrospective Introspective Self-Model

    A metacognition benchmark measuring whether LLMs can accurately
    monitor their own reasoning before, during, and after solving problems.

    Primary score = 0.40 * mean(T4_L1, T4_L2) + 0.30 * T5_L1 + 0.30 * min(T6, 1)
    """
    # Run the full pipeline (D1 -> D2 -> D3 for all problems)
    # Cache covers both the pipeline run AND all task score computations
    # (including judge LLM calls in Task 4) to avoid redundant API calls.
    with kbench.client.enable_cache():
        pipeline.run_all(llm, kbench)

        # --- Compute all six task scores ---

        # L1 (familiar) scores
        t1_l1 = compute_task_1(pipeline, novelty_level=1)
        t2_l1 = compute_task_2(pipeline, novelty_level=1)
        t3_l1 = compute_task_3(pipeline, novelty_level=1)
        t4_l1 = compute_task_4(
            pipeline, kbench, judge_llm=kbench.judge_llm, novelty_level=1
        )
        t5_l1 = compute_task_5(pipeline, novelty_level=1)

        # L2 (novel operator) scores
        t1_l2 = compute_task_1(pipeline, novelty_level=2)
        t2_l2 = compute_task_2(pipeline, novelty_level=2)
        t3_l2 = compute_task_3(pipeline, novelty_level=2)
        t4_l2 = compute_task_4(
            pipeline, kbench, judge_llm=kbench.judge_llm, novelty_level=2
        )
        t5_l2 = compute_task_5(pipeline, novelty_level=2)

        # Cross-level score (reads cached T4 scores — no extra judge calls)
        t6 = compute_task_6(pipeline, kbench, judge_llm=kbench.judge_llm)

    # --- Report all scores via assertions for leaderboard visibility ---
    kbench.assertions.assert_true(
        True,
        expectation=f"T1 Prospective Calibration (AUROC): L1={t1_l1:.3f}, L2={t1_l2:.3f}",
    )
    kbench.assertions.assert_true(
        True,
        expectation=f"T2 Step-Level Accuracy (Spearman): L1={t2_l1:.3f}, L2={t2_l2:.3f}",
    )
    kbench.assertions.assert_true(
        True,
        expectation=f"T3 Retrospective Accuracy: L1={t3_l1:.3f}, L2={t3_l2:.3f}",
    )
    kbench.assertions.assert_true(
        True,
        expectation=f"T4 Coherence (composite): L1={t4_l1:.3f}, L2={t4_l2:.3f}",
    )
    kbench.assertions.assert_true(
        True,
        expectation=f"T5 Metacognitive Control: L1={t5_l1:.3f}, L2={t5_l2:.3f}",
    )
    kbench.assertions.assert_true(
        True,
        expectation=f"T6 Novelty Robustness: {t6:.3f}",
    )

    # Report pipeline summary
    summary = pipeline.summary()
    cf_parse_rate = pipeline.get_counterfactual_parse_rate()
    kbench.assertions.assert_true(
        True,
        expectation=(
            f"Pipeline: {summary['l1_main_count']} L1 + {summary['l2_main_count']} L2 problems | "
            f"Decision: {summary['l1_decision_count']} L1 + {summary['l2_decision_count']} L2 | "
            f"Accuracy L1={summary['l1_overall_accuracy']:.1%} L2={summary['l2_overall_accuracy']:.1%} | "
            f"Parse errors={summary['parse_error_rate']:.1%}"
        ),
    )

    # Flag low counterfactual parse rates — if the model rarely produces
    # COUNTERFACTUAL: responses, sub-score C defaults to 0 and the composite
    # is unreliable.  This assertion makes the issue visible on the leaderboard.
    kbench.assertions.assert_true(
        cf_parse_rate >= 0.5,
        expectation=(
            f"Counterfactual parse rate >= 50% (actual: {cf_parse_rate:.0%}). "
            "Low rates mean sub-score C is unreliable."
        ),
    )

    # Primary score: weighted combination of coherence, control, and robustness
    # 0.40 * mean(T4_L1, T4_L2) + 0.30 * T5_L1 + 0.30 * min(T6, 1.0)
    t4_mean = (t4_l1 + t4_l2) / 2.0
    primary = 0.40 * t4_mean + 0.30 * t5_l1 + 0.30 * min(t6, 1.0)
    return primary


# ============================================================================
# CELL 5: Run the Benchmark
# ============================================================================

run = prism_metacognition.run(llm=kbench.llm)
run

# ============================================================================
# CELL 6: Select this task for the leaderboard
# ============================================================================

# %choose prism_metacognition
