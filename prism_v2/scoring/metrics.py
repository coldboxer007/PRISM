"""
Metrics Engine: Statistical computations for all six PRISM v2.1 benchmark tasks.

Implements:
  Task 1: AUROC for prospective calibration
  Task 2: Spearman rho for step-level prospective accuracy
  Task 3: Retrospective self-assessment accuracy
  Task 4: Prospective-retrospective coherence (composite)
  Task 5: Metacognitive control (accept/decline utility)
  Task 6: Novelty robustness (L2/L1 ratio)
"""

import math
from typing import Optional


# ---------------------------------------------------------------------------
# Task 1: AUROC (Prospective Calibration)
# ---------------------------------------------------------------------------


def compute_auroc(
    confidences: list[float],
    outcomes: list[int],
) -> float:
    """Compute Area Under the ROC Curve (type-2 metacognitive sensitivity).

    Uses the trapezoidal rule (Mann-Whitney U statistic formulation).

    Args:
        confidences: list of bet_fraction_correct values (0.0 to 1.0)
        outcomes: list of binary outcomes (1 = correct, 0 = incorrect)

    Returns:
        AUROC value (0.0 to 1.0). Returns 0.5 if undefined (all same class).
    """
    if len(confidences) != len(outcomes):
        raise ValueError("confidences and outcomes must have the same length")

    n = len(confidences)
    if n < 2:
        return 0.5

    positives = [(c, o) for c, o in zip(confidences, outcomes) if o == 1]
    negatives = [(c, o) for c, o in zip(confidences, outcomes) if o == 0]

    n_pos = len(positives)
    n_neg = len(negatives)

    if n_pos == 0 or n_neg == 0:
        return 0.5  # undefined; return chance

    # Mann-Whitney U formulation
    u_count = 0
    tie_count = 0
    for pc, _ in positives:
        for nc, _ in negatives:
            if pc > nc:
                u_count += 1
            elif pc == nc:
                tie_count += 1

    auroc = (u_count + 0.5 * tie_count) / (n_pos * n_neg)
    return auroc


# ---------------------------------------------------------------------------
# Task 2: Spearman Rank Correlation (Step-Level Prospective Accuracy)
# ---------------------------------------------------------------------------


def _rank(values: list[float]) -> list[float]:
    """Compute ranks with averaged ties."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1

    return ranks


def compute_spearman_rho(x: list[float], y: list[float]) -> Optional[float]:
    """Compute Spearman rank correlation coefficient.

    Returns None if either variable has zero variance (all same values).
    """
    n = len(x)
    if n != len(y) or n < 2:
        return None

    rx = _rank(x)
    ry = _rank(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    var_x = sum((rx[i] - mean_rx) ** 2 for i in range(n))
    var_y = sum((ry[i] - mean_ry) ** 2 for i in range(n))

    if var_x == 0 or var_y == 0:
        return None

    rho = cov / math.sqrt(var_x * var_y)
    return max(-1.0, min(1.0, rho))


def compute_step_accuracy_score(
    per_problem_rhos: list[Optional[float]],
) -> float:
    """Compute the normalized step-level accuracy score.

    Averages valid Spearman rho values and normalizes to 0-1 range.
    Score = (mean_rho + 1) / 2
    """
    valid = [r for r in per_problem_rhos if r is not None]
    if not valid:
        return 0.5  # no signal

    mean_rho = sum(valid) / len(valid)
    return (mean_rho + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Task 3: Retrospective Self-Assessment Accuracy
# ---------------------------------------------------------------------------


def compute_retro_accuracy(
    retro_assessments: list[list[str]],
    step_correctness: list[list[bool]],
) -> float:
    """Compute retrospective self-assessment accuracy.

    Task 3 should measure whether the model can correctly recognize, after the
    fact, which steps it got right or wrong. We therefore score the reported
    correctness component of each retrospective label and do not condition the
    metric on the earlier prospective confidence report.

    Accepted labels:
      - actual correct  -> "confident and correct" OR "uncertain and correct"
      - actual wrong    -> "confident but wrong" OR "uncertain and wrong"

    Args:
        retro_assessments: list of per-problem retrospective label lists
        step_correctness: list of per-problem step correctness lists

    Returns:
        Proportion of steps where the reported correctness status matches the
        actual outcome (0.0 to 1.0).
    """
    total = 0
    correct = 0

    for retro, corr in zip(retro_assessments, step_correctness):
        for r, c in zip(retro, corr):
            label = r.lower().strip()
            if c and label in {"confident and correct", "uncertain and correct"}:
                correct += 1
            elif (not c) and label in {"confident but wrong", "uncertain and wrong"}:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Task 4: Prospective-Retrospective Coherence
# ---------------------------------------------------------------------------


def compute_location_consistency(
    predicted_weakest: list[Optional[int]],
    reported_hardest: list[Optional[int]],
) -> float:
    """Sub-score A: Do predicted-weakest and reported-hardest match?

    Returns proportion of problems where they agree.
    """
    valid_pairs = [
        (p, r)
        for p, r in zip(predicted_weakest, reported_hardest)
        if p is not None and r is not None
    ]
    if not valid_pairs:
        return 0.0

    matches = sum(1 for p, r in valid_pairs if p == r)
    return matches / len(valid_pairs)


def compute_confidence_consistency(
    prospective_vectors: list[list[int]],
    retro_assessments: list[list[str]],
) -> float:
    """Sub-score B: Spearman rho between prospective confidence and retrospective difficulty.

    Maps retrospective assessments to difficulty scale:
      "uncertain and wrong" = 4 (hardest)
      "confident but wrong" = 3
      "uncertain and correct" = 2
      "confident and correct" = 1 (easiest)

    Returns normalized average rho: (rho + 1) / 2
    """
    from prism_v2.scoring.confidence_parser import RETRO_TO_DIFFICULTY

    rhos = []
    for conf_vec, retro_vec in zip(prospective_vectors, retro_assessments):
        # Map retrospective to difficulty
        difficulty_vec = [
            RETRO_TO_DIFFICULTY.get(r.lower().strip(), 2) for r in retro_vec
        ]

        # Prospective confidence should inversely correlate with difficulty
        # Higher confidence (5) -> lower difficulty expected (1)
        # So we compare confidence directly with difficulty
        rho = compute_spearman_rho(
            [float(c) for c in conf_vec],
            [float(d) for d in difficulty_vec],
        )
        if rho is not None:
            # Negate rho because we expect an *inverse* relationship:
            # high prospective confidence (5) should correspond to low
            # retrospective difficulty (1 = "confident and correct").
            # A well-calibrated model produces negative rho; negating
            # converts that into a positive score for the benchmark.
            rhos.append(-rho)

    if not rhos:
        return 0.5

    mean_rho = sum(rhos) / len(rhos)
    return (mean_rho + 1.0) / 2.0


def compute_coherence_composite(
    location_score: float,
    confidence_score: float,
    counterfactual_score: float,
    weights: tuple[float, float, float] = (0.35, 0.45, 0.20),
) -> float:
    """Compute the composite coherence score.

    Args:
        location_score: Sub-score A (location consistency)
        confidence_score: Sub-score B (confidence consistency)
        counterfactual_score: Sub-score C (counterfactual plausibility)
        weights: (w_A, w_B, w_C) summing to 1.0.
                 Counterfactual is weighted lower (0.20) because fluent
                 reflective text is not strong evidence of genuine
                 self-monitoring on its own.
    """
    return (
        weights[0] * location_score
        + weights[1] * confidence_score
        + weights[2] * counterfactual_score
    )


# ---------------------------------------------------------------------------
# Task 5: Metacognitive Control (Accept/Decline Utility)
# ---------------------------------------------------------------------------


def compute_metacognitive_control(
    decisions: list[str],
    is_solvable: list[bool],
    model_correct: list[bool],
    payoff_correct: list[int],
    payoff_wrong: list[int],
    payoff_decline: list[int],
) -> float:
    """Compute metacognitive control score from accept/decline decisions.

    For each problem the model chose ACCEPT or DECLINE.  We compute:
      - model_utility:   actual payoff from the model's decisions
      - worst_utility:   payoff if every decision were maximally bad
      - optimal_utility: payoff from an oracle with perfect self-knowledge

    Score = (model_utility - worst_utility) / (optimal_utility - worst_utility)
    Clamped to [0, 1].

    Design notes:
      - The oracle is relative to the model's actual correctness, not just
        solvability. A model that fails all 3 solvable problems but correctly
        declines everything would score 1.0. This is defensible: the oracle
        represents perfect *self-knowledge*, not perfect *problem-solving*.
        The metric rewards accurate self-assessment, not raw ability.
      - A "decline everything" strategy scores ~0.54 (not 0.0), partly
        because the decision prompt warns "Some problems may be unsolvable",
        which inflates decline rates. This is a deliberate design tradeoff:
        the warning is necessary so models can discover unsolvable problems,
        and the 0.46 gap to 1.0 still provides meaningful discrimination.

    The *optimal* decision for each problem is:
      - Solvable & oracle-correct: ACCEPT (get payoff_correct)
      - Solvable & oracle-incorrect: DECLINE or ACCEPT depending on EV,
        but since the oracle knows it can't solve it, DECLINE
      - Unsolvable: DECLINE (model_correct is always False for unsolvable)

    The *worst* decision for each problem is:
      - Solvable & correct:   DECLINE (missed reward)
      - Solvable & incorrect: ACCEPT  (took penalty)
      - Unsolvable:           ACCEPT  (took penalty)

    Args:
        decisions: list of "accept" or "decline" strings
        is_solvable: whether each problem has a valid solution
        model_correct: whether the model's attempted answer was correct
        payoff_correct: reward for correct acceptance per problem
        payoff_wrong: penalty for wrong acceptance per problem (negative)
        payoff_decline: safe reward for declining per problem

    Returns:
        Normalized utility ratio in [0, 1].  Returns 0.5 if optimal == worst.
    """
    n = len(decisions)
    if n == 0:
        return 0.0

    model_u = 0.0
    optimal_u = 0.0
    worst_u = 0.0

    for i in range(n):
        pc = payoff_correct[i]
        pw = payoff_wrong[i]
        pd = payoff_decline[i]
        correct = model_correct[i]

        # --- Model utility ---
        if decisions[i] == "accept":
            model_u += pc if correct else pw
        else:
            model_u += pd

        # --- Optimal utility (oracle knows correctness ahead of time) ---
        if correct:
            # Oracle would accept and get it right
            optimal_u += pc
        else:
            # Oracle knows it can't solve -> decline
            optimal_u += pd

        # --- Worst utility ---
        if correct:
            # Worst move: decline a problem you'd get right
            worst_u += pd
        else:
            # Worst move: accept a problem you'd get wrong
            worst_u += pw

    spread = optimal_u - worst_u
    if spread <= 0:
        return 0.5  # degenerate: no information

    score = (model_u - worst_u) / spread
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Task 6: Novelty Robustness
# ---------------------------------------------------------------------------


def compute_novelty_robustness(
    l1_composite: float,
    l2_composite: float,
) -> float:
    """Compute novelty robustness score.

    Ratio = L2_composite / L1_composite, capped at 1.0.
    """
    if l1_composite <= 0:
        return 0.0
    ratio = l2_composite / l1_composite
    return min(ratio, 1.0)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: list[float],
    stat_fn=None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        values: data to bootstrap
        stat_fn: function to compute statistic (default: mean)
        n_resamples: number of bootstrap resamples
        confidence: confidence level (e.g., 0.95 for 95% CI)
        seed: random seed

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    import random as _random

    rng = _random.Random(seed)

    if stat_fn is None:
        stat_fn = lambda x: sum(x) / len(x) if x else 0.0

    point = stat_fn(values)

    if len(values) < 2:
        return point, point, point

    boot_stats = []
    n = len(values)
    for _ in range(n_resamples):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        boot_stats.append(stat_fn(sample))

    boot_stats.sort()
    alpha = (1 - confidence) / 2
    lo_idx = max(0, int(alpha * n_resamples))
    hi_idx = min(n_resamples - 1, int((1 - alpha) * n_resamples))

    return point, boot_stats[lo_idx], boot_stats[hi_idx]
