"""
Task 3: Retrospective Self-Assessment Accuracy

Measures whether, after being told which steps were right/wrong,
the model can accurately categorize its own experience.

Score: Proportion correct (0.0 to 1.0)
"""


def compute_task_3(pipeline, novelty_level: int = 1) -> float:
    """Compute retrospective self-assessment accuracy."""
    from prism_v2.scoring.metrics import compute_retro_accuracy

    retro, corr, conf = pipeline.get_retro_data(novelty_level)
    return compute_retro_accuracy(retro, corr, conf)


def compute_task_3_with_ci(
    pipeline, novelty_level: int = 1
) -> tuple[float, float, float]:
    """Compute retrospective accuracy with bootstrap CI."""
    from prism_v2.scoring.metrics import compute_retro_accuracy, bootstrap_ci

    retro, corr, conf = pipeline.get_retro_data(novelty_level)
    point = compute_retro_accuracy(retro, corr, conf)

    # Flatten to per-step binary accuracy for bootstrapping
    per_step_correct = []
    for retro_list, corr_list, conf_list in zip(retro, corr, conf):
        for r, c, p in zip(retro_list, corr_list, conf_list):
            is_confident = p >= 4
            is_correct = c
            if is_correct and is_confident:
                expected = "confident and correct"
            elif is_correct and not is_confident:
                expected = "uncertain and correct"
            elif not is_correct and is_confident:
                expected = "confident but wrong"
            else:
                expected = "uncertain and wrong"
            per_step_correct.append(1.0 if r.lower().strip() == expected else 0.0)

    if not per_step_correct:
        return 0.0, 0.0, 0.0

    _, lo, hi = bootstrap_ci(per_step_correct)
    return point, lo, hi
