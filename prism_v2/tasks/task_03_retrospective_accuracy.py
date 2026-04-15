"""
Task 3: Retrospective Self-Assessment Accuracy

Measures whether, after being told which steps were right/wrong,
the model can accurately recognize which steps it got right vs. wrong.

Score: Proportion correct (0.0 to 1.0)
"""


def compute_task_3(pipeline, novelty_level: int = 1) -> float:
    """Compute retrospective self-assessment accuracy."""
    from prism_v2.scoring.metrics import compute_retro_accuracy

    retro, corr, _ = pipeline.get_retro_data(novelty_level)
    return compute_retro_accuracy(retro, corr)


def compute_task_3_with_ci(
    pipeline, novelty_level: int = 1
) -> tuple[float, float, float]:
    """Compute retrospective accuracy with bootstrap CI."""
    from prism_v2.scoring.metrics import compute_retro_accuracy, bootstrap_ci

    retro, corr, _ = pipeline.get_retro_data(novelty_level)
    point = compute_retro_accuracy(retro, corr)

    # Flatten to per-step binary accuracy for bootstrapping
    per_step_correct = []
    for retro_list, corr_list in zip(retro, corr):
        for r, c in zip(retro_list, corr_list):
            label = r.lower().strip()
            if c:
                expected_ok = label in {
                    "confident and correct",
                    "uncertain and correct",
                }
            else:
                expected_ok = label in {
                    "confident but wrong",
                    "uncertain and wrong",
                }
            per_step_correct.append(1.0 if expected_ok else 0.0)

    if not per_step_correct:
        return 0.0, 0.0, 0.0

    _, lo, hi = bootstrap_ci(per_step_correct)
    return point, lo, hi
