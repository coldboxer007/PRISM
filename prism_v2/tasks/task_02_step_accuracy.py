"""
Task 2: Step-Level Prospective Accuracy (Spearman rho)

Measures whether the model's per-step confidence ranking
predicts which steps it actually gets right.

Score: (mean_rho + 1) / 2, normalized to 0-1 (chance = 0.5)
"""


def compute_task_2(pipeline, novelty_level: int = 1) -> float:
    """Compute step-level prospective accuracy score."""
    from prism_v2.scoring.metrics import compute_step_accuracy_score

    rhos = pipeline.get_step_rhos(novelty_level)
    return compute_step_accuracy_score(rhos)


def compute_task_2_with_ci(
    pipeline, novelty_level: int = 1
) -> tuple[float, float, float]:
    """Compute step accuracy with bootstrap CI."""
    from prism_v2.scoring.metrics import compute_step_accuracy_score, bootstrap_ci

    rhos = pipeline.get_step_rhos(novelty_level)
    valid = [r for r in rhos if r is not None]

    if not valid:
        return 0.5, 0.5, 0.5

    point = compute_step_accuracy_score(rhos)

    def norm_mean(vals):
        if not vals:
            return 0.5
        return (sum(vals) / len(vals) + 1.0) / 2.0

    _, lo, hi = bootstrap_ci(valid, stat_fn=norm_mean)
    return point, lo, hi
