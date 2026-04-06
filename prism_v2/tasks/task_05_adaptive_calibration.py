"""
Task 5: Adaptive Calibration (Confidence Drift)

Measures whether the model appropriately updates its confidence
on subsequent problems after receiving step-level feedback.

Score: max(0, Pearson_r) — clamped to 0-1 range.
"""


def compute_task_5(pipeline, novelty_level: int = 1) -> float:
    """Compute adaptive calibration score from feedback rounds."""
    from prism_v2.scoring.metrics import compute_adaptive_calibration

    round_confs, round_corrs = pipeline.get_feedback_data(novelty_level)

    if len(round_confs) < 2:
        return 0.0

    # Determine num_steps from the first round
    num_steps = len(round_confs[0]) if round_confs[0] else 5

    return compute_adaptive_calibration(round_confs, round_corrs, num_steps)


def compute_task_5_details(pipeline, novelty_level: int = 1) -> dict:
    """Compute adaptive calibration with detailed round-by-round data."""
    from prism_v2.scoring.metrics import compute_adaptive_calibration

    round_confs, round_corrs = pipeline.get_feedback_data(novelty_level)

    if len(round_confs) < 2:
        return {
            "score": 0.0,
            "num_rounds": len(round_confs),
            "per_round_accuracy": [],
            "per_round_mean_confidence": [],
        }

    num_steps = len(round_confs[0]) if round_confs[0] else 5
    score = compute_adaptive_calibration(round_confs, round_corrs, num_steps)

    # Additional detail
    per_round_acc = [sum(c) / len(c) if c else 0.0 for c in round_corrs]
    per_round_conf = [sum(c) / len(c) if c else 3.0 for c in round_confs]

    return {
        "score": score,
        "num_rounds": len(round_confs),
        "per_round_accuracy": per_round_acc,
        "per_round_mean_confidence": per_round_conf,
    }
