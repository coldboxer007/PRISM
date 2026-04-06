"""
Metrics Engine: Statistical computations for all six PRISM v2.1 benchmark tasks.

Implements:
  Task 1: AUROC for prospective calibration
  Task 2: Spearman rho for step-level prospective accuracy
  Task 3: Retrospective self-assessment accuracy
  Task 4: Prospective-retrospective coherence (composite)
  Task 5: Adaptive calibration (confidence drift)
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
    prospective_confidences: list[list[int]],
    confidence_threshold: int = 4,
) -> float:
    """Compute retrospective self-assessment accuracy.

    For each step, the "correct" retrospective label is determined by combining
    ground truth correctness with prospective confidence:
      - correct + high confidence -> "confident and correct"
      - correct + low confidence  -> "uncertain and correct"
      - wrong + high confidence   -> "confident but wrong"
      - wrong + low confidence    -> "uncertain and wrong"

    Args:
        retro_assessments: list of per-problem retrospective label lists
        step_correctness: list of per-problem step correctness lists
        prospective_confidences: list of per-problem confidence vectors (1-5 scale)
        confidence_threshold: values >= this are "confident", below are "uncertain"

    Returns:
        Proportion of correct retrospective labels (0.0 to 1.0).
    """
    total = 0
    correct = 0

    for retro, corr, conf in zip(
        retro_assessments, step_correctness, prospective_confidences
    ):
        for r, c, p in zip(retro, corr, conf):
            is_confident = p >= confidence_threshold
            is_correct = c

            if is_correct and is_confident:
                expected = "confident and correct"
            elif is_correct and not is_confident:
                expected = "uncertain and correct"
            elif not is_correct and is_confident:
                expected = "confident but wrong"
            else:
                expected = "uncertain and wrong"

            if r.lower().strip() == expected:
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
            # We expect negative correlation (high confidence = low difficulty)
            # So negate rho for scoring purposes
            rhos.append(-rho)

    if not rhos:
        return 0.5

    mean_rho = sum(rhos) / len(rhos)
    return (mean_rho + 1.0) / 2.0


def compute_coherence_composite(
    location_score: float,
    confidence_score: float,
    counterfactual_score: float,
    weights: tuple[float, float, float] = (0.3, 0.4, 0.3),
) -> float:
    """Compute the composite coherence score.

    Args:
        location_score: Sub-score A (location consistency)
        confidence_score: Sub-score B (confidence consistency)
        counterfactual_score: Sub-score C (counterfactual plausibility)
        weights: (w_A, w_B, w_C) summing to 1.0
    """
    return (
        weights[0] * location_score
        + weights[1] * confidence_score
        + weights[2] * counterfactual_score
    )


# ---------------------------------------------------------------------------
# Task 5: Adaptive Calibration (Confidence Drift)
# ---------------------------------------------------------------------------


def _linear_slope(values: list[float]) -> float:
    """Compute the slope of a simple linear regression of values vs. index."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))

    return num / den if den != 0 else 0.0


def compute_pearson_r(x: list[float], y: list[float]) -> Optional[float]:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n != len(y) or n < 2:
        return None

    x_mean = sum(x) / n
    y_mean = sum(y) / n

    cov = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    var_x = sum((x[i] - x_mean) ** 2 for i in range(n))
    var_y = sum((y[i] - y_mean) ** 2 for i in range(n))

    if var_x == 0 or var_y == 0:
        return None

    r = cov / math.sqrt(var_x * var_y)
    return max(-1.0, min(1.0, r))


def compute_adaptive_calibration(
    round_confidences: list[list[int]],
    round_correctness: list[list[bool]],
    num_steps: int = 5,
) -> float:
    """Compute adaptive calibration score from feedback rounds.

    Tracks per-step confidence changes and accuracy changes across rounds,
    then correlates them.

    Args:
        round_confidences: list of confidence vectors, one per round
        round_correctness: list of correctness vectors, one per round
        num_steps: number of steps per problem

    Returns:
        max(0, Pearson_r) — clamped to 0-1 range.
    """
    num_rounds = len(round_confidences)
    if num_rounds < 2:
        return 0.0

    # Compute per-step slopes
    conf_slopes = []
    acc_slopes = []

    for step in range(num_steps):
        step_confs = [
            round_confidences[r][step] if step < len(round_confidences[r]) else 3
            for r in range(num_rounds)
        ]
        step_accs = [
            float(round_correctness[r][step])
            if step < len(round_correctness[r])
            else 0.0
            for r in range(num_rounds)
        ]

        conf_slopes.append(_linear_slope(step_confs))
        acc_slopes.append(_linear_slope(step_accs))

    # Correlate confidence slopes with accuracy slopes
    r = compute_pearson_r(conf_slopes, acc_slopes)

    if r is None:
        return 0.0

    return max(0.0, r)


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
