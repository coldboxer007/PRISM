"""
Task 6: Novelty Robustness

Measures whether metacognitive ability survives when problems
use novel, in-context rules (L2 vs L1).

Score: min(L2_composite / L1_composite, 1.0) — capped at 1.0

Note: Task 4 (coherence) scores are read from the pipeline's score cache
when available, to avoid redundant LLM judge calls. The notebook computes
Task 4 for L1 and L2 before calling Task 6, so the cache is always warm.
"""


def compute_task_6(
    pipeline,
    kbench,
    judge_llm=None,
) -> float:
    """Compute novelty robustness score.

    Reads cached Task 4 scores from pipeline._task_score_cache when
    available, avoiding redundant judge LLM calls.
    """
    from prism_v2.scoring.metrics import compute_novelty_robustness
    from prism_v2.tasks.task_01_prospective_calibration import compute_task_1
    from prism_v2.tasks.task_02_step_accuracy import compute_task_2
    from prism_v2.tasks.task_03_retrospective_accuracy import compute_task_3
    from prism_v2.tasks.task_04_coherence import compute_task_4

    # L1 composite (average of Tasks 1-4 on L1)
    l1_scores = [
        compute_task_1(pipeline, novelty_level=1),
        compute_task_2(pipeline, novelty_level=1),
        compute_task_3(pipeline, novelty_level=1),
        # compute_task_4 checks the pipeline cache before recomputing
        compute_task_4(pipeline, kbench, judge_llm, novelty_level=1),
    ]
    l1_composite = sum(l1_scores) / len(l1_scores)

    # L2 composite (average of Tasks 1-4 on L2)
    l2_scores = [
        compute_task_1(pipeline, novelty_level=2),
        compute_task_2(pipeline, novelty_level=2),
        compute_task_3(pipeline, novelty_level=2),
        compute_task_4(pipeline, kbench, judge_llm, novelty_level=2),
    ]
    l2_composite = sum(l2_scores) / len(l2_scores)

    return compute_novelty_robustness(l1_composite, l2_composite)


def compute_task_6_details(
    pipeline,
    kbench,
    judge_llm=None,
) -> dict:
    """Compute novelty robustness with per-task breakdown.

    Uses cached Task 4 scores via compute_task_4's built-in caching.
    """
    from prism_v2.scoring.metrics import compute_novelty_robustness
    from prism_v2.tasks.task_01_prospective_calibration import compute_task_1
    from prism_v2.tasks.task_02_step_accuracy import compute_task_2
    from prism_v2.tasks.task_03_retrospective_accuracy import compute_task_3
    from prism_v2.tasks.task_04_coherence import compute_task_4

    l1_t1 = compute_task_1(pipeline, 1)
    l1_t2 = compute_task_2(pipeline, 1)
    l1_t3 = compute_task_3(pipeline, 1)
    l1_t4 = compute_task_4(pipeline, kbench, judge_llm, 1)

    l2_t1 = compute_task_1(pipeline, 2)
    l2_t2 = compute_task_2(pipeline, 2)
    l2_t3 = compute_task_3(pipeline, 2)
    l2_t4 = compute_task_4(pipeline, kbench, judge_llm, 2)

    l1_composite = (l1_t1 + l1_t2 + l1_t3 + l1_t4) / 4
    l2_composite = (l2_t1 + l2_t2 + l2_t3 + l2_t4) / 4

    return {
        "l1_task_scores": {
            "prospective_calibration": l1_t1,
            "step_accuracy": l1_t2,
            "retro_accuracy": l1_t3,
            "coherence": l1_t4,
        },
        "l2_task_scores": {
            "prospective_calibration": l2_t1,
            "step_accuracy": l2_t2,
            "retro_accuracy": l2_t3,
            "coherence": l2_t4,
        },
        "l1_composite": l1_composite,
        "l2_composite": l2_composite,
        "ratio": compute_novelty_robustness(l1_composite, l2_composite),
    }
