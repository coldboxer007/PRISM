"""
Task 1: Prospective Calibration (AUROC)

Measures whether the model's stated confidence (bet fraction)
discriminates between problems it gets right vs. wrong.

Score: AUROC (0.0-1.0, chance = 0.5)
"""


def compute_task_1(pipeline, novelty_level: int = 1) -> float:
    """Compute AUROC for prospective calibration."""
    from prism_v2.scoring.metrics import compute_auroc

    bets, outcomes = pipeline.get_bet_fractions_and_outcomes(novelty_level)

    if len(bets) < 2:
        return 0.5

    return compute_auroc(bets, outcomes)


def compute_task_1_with_ci(
    pipeline, novelty_level: int = 1
) -> tuple[float, float, float]:
    """Compute AUROC with bootstrap confidence interval."""
    from prism_v2.scoring.metrics import compute_auroc, bootstrap_ci

    bets, outcomes = pipeline.get_bet_fractions_and_outcomes(novelty_level)

    if len(bets) < 2:
        return 0.5, 0.5, 0.5

    point = compute_auroc(bets, outcomes)

    # Bootstrap: resample (bet, outcome) pairs
    pairs = list(zip(bets, outcomes))

    def auroc_from_pairs(sample_pairs):
        b = [p[0] for p in sample_pairs]
        o = [p[1] for p in sample_pairs]
        return compute_auroc(b, o)

    _, lo, hi = bootstrap_ci(
        pairs,
        stat_fn=auroc_from_pairs,
    )
    return point, lo, hi
